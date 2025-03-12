FROM python:3.9-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install required packages
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code
COPY dashboard/ .

# Create folders for data and assets if they don't exist
RUN mkdir -p assets

# Expose dashboard port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=120s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8050 || exit 1

# Command to run the dashboard
CMD ["python", "main.py"]