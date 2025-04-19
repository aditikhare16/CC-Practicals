# Use official Python image
FROM python:3.9-alpine

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install Pillow

# Copy the script
COPY resize.py .

# Command to run the script
ENTRYPOINT ["python", "resize.py"]
