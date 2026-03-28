FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY data/ data/
COPY src/ src/
RUN mkdir -p results

# Paper trading state directory (managed by Coolify persistent storage)
RUN mkdir -p paper_trading

# Seed data for first deploy (copied to volume if empty)
COPY paper_trading_seed/ paper_trading_seed/

# Expose dashboard port
EXPOSE 5001

# Entrypoint script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
