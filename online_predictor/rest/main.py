from fastapi import FastAPI, HTTPException
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import json
import pandas as pd
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import logging.config

log_config = {
    "version": 1,
    "root": {
        "handlers": ["console", "file"],
        "level": "WARNING"
    },
    "handlers": {
        "console": {
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        },
        "file": {
            "formatter": "std_out",
            "class": "logging.FileHandler",
            "level": "WARNING",
            "filename": "warnings.log"
        }
    },
    "formatters": {
        "std_out": {
            "format": "%(levelname)s : %(module)s : %(funcName)s : %(message)s",
        }
    },
}

# logging.config.fileConfig('logging.conf')
logging.config.dictConfig(log_config)
logger = logging.getLogger("logs")

PATH = './models/'
class InputVector(BaseModel):
    input: List[float]


app = FastAPI()


def get_model_names(all_models_path):
    """Создает словарь из имени модели и ее пути расположения
    Args:
        path (str): путь поиска файла описаний
    Returns:
        names (list): список имен всех моделей выбранной папки
        paths (dict): словарь моделей имя-путь расположения
    """
    try:
        model_paths = os.listdir(all_models_path)
        model_paths = [all_models_path + path.replace("\\", "/") + "/" for path in model_paths]

        names = []
        for path in model_paths:
            with open(path + '/model.txt', mode='r', encoding='utf-8') as f:
                model_name = f.read()
                names.append(model_name)
        models = dict(zip(names, model_paths))
        logger.debug(f'Get model names {names}')
        return names, models
    except Exception as error:
        logger.error(f'Error is {error}')


def read_sensors(path) -> pd.DataFrame:
    """Читает параметры модели из файла описаний model.csv
    Args:
        path (str): путь поиска файла описаний
    Returns:
        pandas.DataFrame: датафрейм с параметрами модели
    """
    df = pd.read_csv(path + 'model' + '.csv', delimiter=';', encoding='utf-8')
    return df


def make_pred(x, model, sensors):
    """Читает параметры модели из файла описаний model.csv
    Args:
        x (str): путь поиска файла описаний
        model (model pickle): модель формата pickle
        sensors (pandas.DataFrame): датафрейм с параметрами входов модели
    Returns:
        prediction (str): предсказание модели
    """
    for_pred = np.array([x])
    scaler = StandardScaler()
    scaler.fit(for_pred)
    scaler.scale_ = np.array(sensors['Scale'])
    scaler.mean_ = np.array(sensors['Mean'])

    prediction = model.predict(scaler.transform(for_pred))[0]

    return str(prediction)


@app.get('/')
def get_models():
    model_names, model_paths_dict = get_model_names(PATH)
    json_compatible_data = jsonable_encoder({f'Model {i + 1}': model_names[i] for i in range(len(model_names))})
    return JSONResponse(content=json_compatible_data)


@app.get("/{model_id}")
async def get_model_data(model_id: int):
    model_names, model_paths_dict = get_model_names(PATH)
    model_path = model_paths_dict[model_names[int(model_id) - 1]]
    df = pd.read_csv(model_path + 'model' + '.csv', delimiter=';', encoding='utf-8')
    df['Min'].astype(int)
    df['Max'].astype(int)
    df['Value'].astype(int)
    df['Step'].astype(int)
    return df.to_json(orient="records", force_ascii=False)


@app.post('/{model_id}/predict')
def get_model_prediction(model_id: int, body: InputVector):
    model_names, model_paths_dict = get_model_names(PATH)
    model_path = model_paths_dict[model_names[int(model_id) - 1]]
    if 'model.cb' in os.listdir(model_path):
        model_file = model_path + 'model.cb'
        model = CatBoostRegressor().load_model(model_file)
    elif 'model.pkl' in os.listdir(model_path):
        model_file = model_path + 'model.pkl'
        model = pickle.load(open(model_file, 'rb'))
    elif 'model.pickle' in os.listdir(model_path):
        model_file = model_path + 'model.pickle'
        model = pickle.load(open(model_file, 'rb'))
    elif 'model.xgb' in os.listdir(model_path):
        model = XGBRegressor()
        model.load_model(model_path + 'model.xgb')

    try:
        sensors = read_sensors(model_path)
        return JSONResponse(content=jsonable_encoder({'pred': make_pred(body.input, model, sensors)}))

    except ValueError:
        try:
            for_pred = np.array([body.input])
            pred1 = str(model.predict(for_pred)[0])
            logger.debug(f'Made prediction {pred1}')
            return json.dumps({'pred': pred1}, ensure_ascii=False)
        except Exception as error2:
            logger.error(f'Error in writing prediction is {error2}')
