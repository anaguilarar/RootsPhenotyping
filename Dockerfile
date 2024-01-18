FROM python:3.9

ADD main.py .
ADD requirements.txt .

RUN mkdir /root_counter
COPY root_counter/ /root_counter
RUN mkdir /configuration
COPY configuration/ /configuration
RUN mkdir /images
COPY images/ /images

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt


CMD ["python", "./main.py"]
