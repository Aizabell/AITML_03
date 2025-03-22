# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first for better caching
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files from the 'app' folder only
COPY app/ .

# Expose Flask's port
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]
