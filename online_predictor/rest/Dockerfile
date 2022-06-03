FROM python:3.8

WORKDIR /app

COPY rest_requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt

COPY main.py models ./

EXPOSE 8560

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8560"]