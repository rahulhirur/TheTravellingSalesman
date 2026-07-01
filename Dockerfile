# Use a lightweight python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user for HF Spaces security compliance
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Copy dependency files
COPY pyproject.toml ./

# Install uv and project dependencies for backend only
RUN pip install --no-cache-dir uv && \
    uv pip install --system ".[backend]"

# Copy remaining codebase
COPY --chown=user . .

# Switch to non-root user
USER user

# Expose port and run API
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
