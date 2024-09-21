FROM python:3.11.7

WORKDIR /usr/src/api

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app

CMD python app/main.py
