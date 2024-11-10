# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir tensorflow==2.17.0 --timeout=999999

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Flask will run on
EXPOSE 5000

# Run the application
CMD [ "flask", "run", "--host=0.0.0.0", "--port=5000" ]
