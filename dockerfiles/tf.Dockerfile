# Use an official Tensorflow (+gpu) runtime as a parent image
FROM tensorflow/tensorflow:2.15.0-gpu

# Set the working directory in the container
WORKDIR /src

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /
RUN pip install -r /requirements.txt

# Install any needed packages specified in requirements.txt
# This is important for financial projects that rely on Python packages like pandas, numpy, etc.
