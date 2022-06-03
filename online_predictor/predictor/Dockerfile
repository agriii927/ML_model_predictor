FROM python:3.8

WORKDIR /app

COPY predictor_requirements.txt ./requirements.txt
#RUN #pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

COPY cronjob logging_pnos.conf predictionApp.py ./

COPY models ./models

RUN apt-get update && apt-get -y install cron

COPY cronjob /etc/cron.d/cronjob

RUN chmod 0644 /etc/cron.d/cronjob

RUN crontab /etc/cron.d/cronjob

CMD ["cron", "-f"]