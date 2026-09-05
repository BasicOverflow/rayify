# Ray 2.56 wheels may not exist on Alpine; use slim Debian.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lexis_markets ./lexis_markets
ENV PYTHONPATH=/app
ENV MARKETS_SUPERVISOR_STATE=/app/.supervisor/state.db
CMD ["python", "-m", "lexis_markets.supervisor.main"]
