FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ .

# Copy the model wheel file
COPY dist/model_triage-1.0.0-py3-none-any.whl /app/dist/

# Install the model package (with dependencies)
RUN pip install /app/dist/model_triage-1.0.0-py3-none-any.whl

# Create models directory even if empty
RUN mkdir -p ./models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=120s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]