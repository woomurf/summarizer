FROM tensorflow/tensorflow:1.15.5-gpu-py3
WORKDIR /app 

ADD requirements.txt /app
RUN pip install -r /app/requirements.txt

ADD app.py /app

EXPOSE 8051
CMD ["opyrator", "launch-ui", "app:summerize"]