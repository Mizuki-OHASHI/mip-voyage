FROM python:3.11.7

WORKDIR /usr/src/app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./src ./src

CMD exec uvicorn src.main:app --host 0.0.0.0 --port 8000

