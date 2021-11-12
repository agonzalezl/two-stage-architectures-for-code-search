FROM tensorflow/tensorflow:2.4.1-gpu


COPY requirements.txt .

RUN pip install -r requirements.txt