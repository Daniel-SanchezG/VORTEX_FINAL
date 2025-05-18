#  Python 3.10 as a base (pycaret compatible)
FROM python:3.10-slim

# Proyect Information
LABEL maintainer="Daniel Sanchez-Gomez <daniel-sanchez-gomez@edu.ulisboa.pt>"
LABEL description="VORTEX: Variscite ORigin Technology X-ray based"
LABEL version="1.0"

# Establishing the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Setting environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command to run the complete pipeline
ENTRYPOINT ["python3", "main.py"]
CMD ["--input", "DATA/raw/input_data.xlsx", "--output-dir", "outputs"]