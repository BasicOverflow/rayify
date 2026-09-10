# Lexis Markets — Proxmox LXC deploy

Public app repo. Runtime is **Python + uv + systemd** — not npm, not Node, not the
DeepSeek Harness package path, and not Docker-inside-the-LXC. Local Docker remains
only for laptop tests (`docker compose up --build supervisor`).

Paste-install (root on a Proxmox node, usually **home-media**):

```bash
bash -c "$(curl -fsSL http://10.0.0.52:3000/admin/lexis-markets/raw/branch/main/deploy/install.sh)"
```

Defaults: CT **117**, `10.0.0.58`, hostname `lexis-markets-supervisor`, Debian 12, unit `lexis-markets.service`.

## Secrets (host-only)

App code is public. Cluster `.env` stays in private `admin/rayify_real_infra_creds`.

On the Proxmox host (never commit this):

```bash
# /root/lexis-markets.secrets.env
GITEA_TOKEN=...
```

Install uses that token only to pull the private creds file into `/etc/lexis-markets.env`.

## Ops

```bash
pct status 117
pct exec 117 -- systemctl status lexis-markets
pct exec 117 -- journalctl -u lexis-markets -f
pct enter 117
```
