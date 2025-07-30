# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire source and artifacts
COPY src/ src/
COPY artifacts/ artifacts/

# Run predict script
CMD ["python", "src/predict.py"]

