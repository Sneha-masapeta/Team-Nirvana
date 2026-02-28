# Use the base image
FROM continuumio/anaconda3:main

# Install fluidsynth
RUN apt-get update && apt-get install -y fluidsynth

# Copy the current directory contents into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Command to run your application (adjust as needed)
CMD ["python", "app.py"]