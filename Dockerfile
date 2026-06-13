# Use lightweight Python runtime as base image
FROM python:3.12-slim

# Set application working directory
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code into container
COPY . .

# Make startup script executable
RUN chmod +x run.sh

# Execute ML pipeline when container starts
CMD ["sh", "run.sh"]