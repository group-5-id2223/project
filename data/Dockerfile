# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install requests

# Create a volume for results
VOLUME /app/results

# Run extractor.py when the container launches
CMD ["python", "./extractor.py"]