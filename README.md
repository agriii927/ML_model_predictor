# ML_model_predictor
Docker compose automation for deployment FastAPI-Clickhouse-Grafana ML model predictor system

## Requirements
1. Docker
2. Docker compose, version 2.5

## About
It is a ML model predictor system based on FastAPI model API and cron prediction app. Prediction is written to ClickHouse database, which is automatically configured by Docker Compose. Prediction data from ClickHouse is visualized in Grafana.

## NB
Now the system is perfectly working with the data mounted to docker containers. You should use your data or configure it manually.

## Two types of using this 
1. Build predictor images from files
```bash
docker-compose up -d
```
2. Load images from tar (If you additionally got it) 
```bash
docker load < perm_rest.tar
docker load < perm_predictor.tar

docker-compose -f docker-compose-from-loaded-images.yaml up -d
```
