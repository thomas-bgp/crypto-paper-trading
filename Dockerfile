FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY data/ data/
COPY src/ src/
RUN mkdir -p results

# Persistent volume for paper trading state
RUN mkdir -p paper_trading
VOLUME /app/paper_trading

# Expose dashboard port
EXPOSE 5001

# Entrypoint script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
