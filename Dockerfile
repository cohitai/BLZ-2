# Use the Python3.8 container image
FROM python:3.8

# Set the working directory to /bot
WORKDIR /bot

# Copy the current directory contents into the container at /bot
ADD ./BLZ-2 /bot 

# Install the dependecies
RUN apt-get update
RUN apt-get install libsnappy-dev -y
RUN apt-get install -y libkrb5-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#Build the trainbot - staging -
CMD python main.py -M 500 20 10 4 -A 30

#RUN the trainbot script in production.

#CMD python main.py -A 30
