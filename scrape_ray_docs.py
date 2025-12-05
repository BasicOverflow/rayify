# used to scrape ray docs, not part of agent conversion process

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import html2text
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

class RayDocsScraper:
    def __init__(self, base_url="https://docs.ray.io/en/latest/", output_dir="resources", max_workers=5, debug=True):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.visited_urls = set()
        self.visited_lock = threading.Lock()
        self.debug = debug
        self.max_workers = max_workers
        self.stats = {
            'scraped': 0,
            'saved': 0,
            'errors': 0,
            'skipped': 0,
            'links_found': 0
        }
        self.stats_lock = threading.Lock()
        
        # Create session per thread (thread-safe)
        self.h2m = html2text.HTML2Text()
        self.h2m.ignore_links = False
        self.h2m.ignore_images = False
        self.h2m.body_width = 0
        
        # Sections to focus on
        self.target_sections = [
            'use-cases',
            'examples',
            'ecosystem',
            'ray-core',
            'ray-data',
            'ray-train',
            'ray-tune',
            'ray-serve',
            'ray-rllib',
            'ray-clusters',
            'monitoring-and-debugging'
        ]
    
    def _log(self, message, level="INFO"):
        """Thread-safe logging"""
        if self.debug:
            thread_id = threading.current_thread().name
            print(f"[{level}] [{thread_id}] {message}")
    
    def _create_session(self):
        """Create a new session for a thread"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session
        
    def should_scrape(self, url):
        """Check if URL is within target sections"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check if it's a Ray docs URL
        if 'docs.ray.io' not in parsed.netloc:
            self._log(f"Rejecting {url}: not docs.ray.io", "DEBUG")
            return False
            
        # Must be under /en/latest/
        if '/en/latest/' not in path:
            self._log(f"Rejecting {url}: not under /en/latest/", "DEBUG")
            return False
            
        # Allow the main index page
        if path.endswith('/index.html') or path == '/en/latest/' or path.endswith('/en/latest'):
            self._log(f"Accepting {url}: index page", "DEBUG")
            return True
            
        # Check if it's in our target sections (more flexible matching)
        path_lower = path.lower()
        for section in self.target_sections:
            # Match section name in path
            if f'/{section}' in path_lower:
                self._log(f"Accepting {url}: matches section '{section}'", "DEBUG")
                return True
        
        self._log(f"Rejecting {url}: no matching section", "DEBUG")
        return False
    
    def clean_filename(self, url):
        """Convert URL to clean filename - returns just the filename, not the path"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        # Remove .html extension
        if path.endswith('.html'):
            path = path[:-5]
        if path.endswith('/index'):
            path = path[:-6] or 'index'
            
        parts = [p for p in path.split('/') if p]
        
        # Remove /en/latest/ prefix if present
        if len(parts) >= 2 and parts[0] == 'en' and parts[1] == 'latest':
            parts = parts[2:]
        
        if not parts:
            return 'index'
        
        # Return only the last part (filename)
        return parts[-1]
    
    def get_directory_path(self, url):
        """Get directory path matching URL structure"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if path.endswith('.html'):
            path = path[:-5]
        if path.endswith('/index'):
            path = path[:-6]
            
        parts = [p for p in path.split('/') if p]
        
        if not parts:
            return self.output_dir
        
        # Remove /en/latest/ prefix if present
        if len(parts) >= 2 and parts[0] == 'en' and parts[1] == 'latest':
            parts = parts[2:]
            
        if not parts:
            return self.output_dir
            
        # Remove 'index' from end if present
        if parts[-1] == 'index':
            parts = parts[:-1]
            
        if not parts:
            return self.output_dir
        
        # Return directory path (everything except the filename)
        # This matches the URL directory structure
        dir_parts = parts[:-1] if len(parts) > 1 else []
        return self.output_dir / '/'.join(dir_parts) if dir_parts else self.output_dir
    
    def extract_useful_content(self, soup, url):
        """Extract useful content from page"""
        # Remove navigation, headers, footers, scripts, styles
        for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'noscript']):
            element.decompose()
            
        # Remove common non-content elements
        for element in soup.find_all(['div'], class_=re.compile('sidebar|navigation|nav|toc|breadcrumb|footer|header', re.I)):
            element.decompose()
            
        # Find main content area - Ray docs typically use specific classes
        main_content = (soup.find('main') or 
                       soup.find('article') or 
                       soup.find('div', class_=re.compile('content|main|body|document', re.I)) or
                       soup.find('div', id=re.compile('content|main|body', re.I)))
        
        if not main_content:
            # Fallback to body but remove unwanted parts
            main_content = soup.find('body')
            if main_content:
                for unwanted in main_content.find_all(['nav', 'header', 'footer', 'aside']):
                    unwanted.decompose()
        
        if not main_content:
            return None
            
        # Remove sidebar/toc if still present
        for sidebar in main_content.find_all(['aside', 'div'], class_=re.compile('sidebar|toc|table-of-contents', re.I)):
            sidebar.decompose()
            
        # Keep code blocks, examples, and main content
        return main_content
    
    def scrape_page(self, url):
        """Scrape a single page (thread-safe)"""
        # Check if already visited (thread-safe)
        with self.visited_lock:
            if url in self.visited_urls:
                self._log(f"Skipping {url}: already visited", "DEBUG")
                with self.stats_lock:
                    self.stats['skipped'] += 1
                return None
            self.visited_urls.add(url)
        
        # For index page, always scrape to get links
        is_index = url.endswith('index.html') or url.endswith('/en/latest/') or url.endswith('/en/latest')
        
        if not is_index and not self.should_scrape(url):
            with self.stats_lock:
                self.stats['skipped'] += 1
            return None
        
        session = self._create_session()
        
        try:
            self._log(f"Fetching: {url}")
            start_time = time.time()
            response = session.get(url, timeout=15)
            response.raise_for_status()
            fetch_time = time.time() - start_time
            self._log(f"Fetched {url} in {fetch_time:.2f}s (status: {response.status_code}, size: {len(response.content)} bytes)")
            
            with self.stats_lock:
                self.stats['scraped'] += 1
            
            soup = BeautifulSoup(response.content, 'html.parser')
            self._log(f"Parsed HTML for {url}")
            
            content = self.extract_useful_content(soup, url)
            
            if not content:
                self._log(f"No content extracted from {url}", "WARN")
                return None
                
            # Convert to markdown
            markdown = self.h2m.handle(str(content))
            
            # Clean up markdown
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)
            markdown = markdown.strip()
            
            # For index pages, accept shorter content (they're navigation pages)
            min_length = 50 if is_index else 100
            self._log(f"Markdown length: {len(markdown)} chars (min: {min_length})")
            
            if not markdown or len(markdown) < min_length:
                # Still extract links even if we don't save the page
                if is_index:
                    self._log(f"Index page {url} has short content, will extract links only")
                else:
                    self._log(f"Content too short for {url}, skipping save", "WARN")
                    return None
                
            # Get file path
            dir_path = self.get_directory_path(url)
            filename = self.clean_filename(url)
            
            # Ensure filename ends with .md
            if not filename.endswith('.md'):
                filename = f"{filename}.md"
                
            file_path = dir_path / filename
            # Create directory structure matching URL
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add URL header
            title = soup.title.string if soup.title else filename
            header = f"# {title}\n\n"
            header += f"Source: {url}\n\n---\n\n"
            markdown = header + markdown
            
            # Only save if we have content
            if markdown and len(markdown) >= min_length:
                file_path.write_text(markdown, encoding='utf-8')
                self._log(f"Saved: {file_path} ({len(markdown)} chars)")
                with self.stats_lock:
                    self.stats['saved'] += 1
            else:
                self._log(f"Not saving {url}: content too short", "DEBUG")
            
            # Extract links to follow (prioritize content links)
            links = []
            all_links = soup.find_all('a', href=True)
            self._log(f"Found {len(all_links)} total links on {url}")
            
            for link in all_links:
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only follow internal Ray docs links
                if 'docs.ray.io' in full_url:
                    # Remove fragments and query params
                    full_url = full_url.split('#')[0].split('?')[0]
                    
                    # Check if already visited (thread-safe check)
                    with self.visited_lock:
                        if full_url not in self.visited_urls:
                            # For index pages, collect all links; for others, filter
                            if is_index or self.should_scrape(full_url):
                                links.append(full_url)
            
            # Deduplicate while preserving order
            seen = set()
            links = [x for x in links if not (x in seen or seen.add(x))]
            
            with self.stats_lock:
                self.stats['links_found'] += len(links)
            
            self._log(f"Extracted {len(links)} new links from {url}")
            
            return links
            
        except requests.exceptions.RequestException as e:
            self._log(f"Request error for {url}: {e}", "ERROR")
            with self.stats_lock:
                self.stats['errors'] += 1
            return None
        except Exception as e:
            self._log(f"Error scraping {url}: {type(e).__name__}: {e}", "ERROR")
            with self.stats_lock:
                self.stats['errors'] += 1
            return None
    
    def scrape(self, start_url=None):
        """Start scraping from index (threaded)"""
        if start_url is None:
            start_url = self.base_url + "index.html"
        
        self._log(f"Starting scraper with {self.max_workers} workers")
        self._log(f"Target sections: {', '.join(self.target_sections)}")
        self._log(f"Output directory: {self.output_dir}")
        
        # Thread-safe queue for URLs to process
        url_queue = Queue()
        url_queue.put(start_url)
        
        # Track pending URLs to avoid duplicates
        pending_urls = set([start_url])
        pending_lock = threading.Lock()
        
        def worker():
            """Worker thread function"""
            self._log(f"{threading.current_thread().name} started")
            while True:
                try:
                    url = url_queue.get(timeout=2)
                except:
                    # Queue empty, check if we're done
                    with pending_lock:
                        if len(pending_urls) == 0:
                            self._log(f"{threading.current_thread().name} exiting: no more work")
                            break
                    continue
                
                try:
                    self._log(f"Processing: {url}")
                    links = self.scrape_page(url)
                    
                    if links:
                        # Add new links to queue
                        new_count = 0
                        with pending_lock:
                            for link in links:
                                if link not in pending_urls:
                                    with self.visited_lock:
                                        if link not in self.visited_urls:
                                            pending_urls.add(link)
                                            url_queue.put(link)
                                            new_count += 1
                        if new_count > 0:
                            self._log(f"Queued {new_count} new URLs from {url}")
                    
                    with pending_lock:
                        pending_urls.discard(url)
                    
                    url_queue.task_done()
                    time.sleep(0.2)  # Be polite
                    
                except Exception as e:
                    self._log(f"Worker error processing {url}: {type(e).__name__}: {e}", "ERROR")
                    with pending_lock:
                        pending_urls.discard(url)
                    url_queue.task_done()
        
        # Start worker threads
        threads = []
        for i in range(self.max_workers):
            t = threading.Thread(target=worker, name=f"Worker-{i+1}", daemon=True)
            t.start()
            threads.append(t)
            self._log(f"Started worker thread: {t.name}")
        
        # Wait for all URLs to be processed
        url_queue.join()
        
        # Wait a bit for threads to finish
        time.sleep(2)
        
        # Print final stats
        with self.stats_lock:
            self._log("\n" + "="*60)
            self._log("SCRAPING COMPLETE")
            self._log("="*60)
            self._log(f"Pages scraped: {self.stats['scraped']}")
            self._log(f"Pages saved: {self.stats['saved']}")
            self._log(f"Pages skipped: {self.stats['skipped']}")
            self._log(f"Errors: {self.stats['errors']}")
            self._log(f"Total links found: {self.stats['links_found']}")
            self._log(f"Total URLs visited: {len(self.visited_urls)}")
            self._log(f"Output directory: {self.output_dir}")
            self._log("="*60)

if __name__ == "__main__":
    scraper = RayDocsScraper()
    scraper.scrape()

