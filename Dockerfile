FROM python:3.6

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]
