import datetime
import json
import logging.config
import warnings
from argparse import ArgumentParser
from datetime import timezone

import pandas as pd
import numpy as np
import requests
from numpy.core.numeric import NaN
from requests.auth import HTTPBasicAuth
# from tqdm import tqdm

from Clickhouse_load import ClickHouseConnect
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.config.fileConfig('/app/logging_pnos.conf')
logger = logging.getLogger("predictApp_pnos")

config = {}


def get_data(dt, max_lag, tags):
    """Функция получения данных из кликхауса

    Args:
        dt (datetime): Дата-время окончания периода в формате UTC
        max_lag (int): Максимальный лаг по датчикам, использующимся в предсказании
        tags (array of string): Список датчиков в том порядке каком надо вернуть

    Returns:
        pandas.DataFrame: набор данных из кликхауса
    """
    
    c = ClickHouseConnect(
        database=config['clickhouse']['database'], link=config['clickhouse']['intaddr'],
        user=config['clickhouse']['user'], password=config['clickhouse']['password'])

    # Расчет интервала дат для получения данных
    print(dt)
    to_dt = dt
    from_dt = dt - datetime.timedelta(minutes=max_lag)

    # Получение сырых данных
    logger.debug(f"Starting getting data from {from_dt} to {to_dt}")
    df_now = c.df_from_tags('indications', 'date_time_utc', tags=tags, freq=5,
                            start_datetime=from_dt.strftime('%Y-%m-%d %H:%M:%S'),
                            end_datetime=to_dt.strftime('%Y-%m-%d %H:%M:%S'))
    logger.debug(f"Got raw data from clickhouse")

    # Создание пустых колонок для тех данных которые не нашлись
    skipped_columns = set(tags) - set(df_now.columns)
    logger.debug(f"No data in clickhouse for cols {skipped_columns}")

    # Создание и применение эталонного индекса
    idx = pd.date_range(pd.Timestamp(from_dt).round(freq="5min"),
                        pd.Timestamp(to_dt).round(freq="5min"), freq='5min')
    df_now.index = pd.DatetimeIndex(df_now["DateTime"])
    df_now = df_now.reindex(idx.tz_localize('UTC'))
    df_now["DateTime"] = df_now.index
    logger.debug(f"Data after reindex {df_now}")

    if df_now.count == 0:
        logger.error(f"NULL current data DT= {dt}  Tags={tags}")

    # Заполнение пропусков интерполяцией
    if df_now.isnull().values.any():
        logger.debug('There are NaNs in raw data')
        df_now.set_index('DateTime', inplace=True)
        df_now.interpolate(method='linear', axis=0, limit=None, inplace=True, limit_direction='both', downcast=None)
        df_now.reset_index(inplace=True)
        logger.debug('NaNs are filled with interpolation')
    else:
        logger.debug('There are no NaNs in raw data')
    print(df_now)
    return df_now.drop(['DateTime'], axis=1)


def calc_lags(data, model_params):
    """
       Функция расчета лагов

       Args:
           data (pd.DataFrame): датафрейм с данными
           model_params (pd.DataFrame): датафрейм с именами и лагами датчиков, использующихся в предсказании
       Returns:
           model_input (list): список значений датчиков с лагами, подаваемый далее на вход модели
       """

    # Расчет значений переменных с лагами, формирование листа model_input, подаваемого на вход модели
    model_input = []
    for ind, row in model_params[model_params['Is key param'] == 1].iterrows():
        if row['Name'] in data.columns:
            # Добавляем усредненное значение с лагом - выбираем промежуток строк от min до max лага и берем среднее
            param_shifted = data[row['Name']].shift(int(row['Min lag']) // 5)
            average_param_from_min_to_max_lag = \
                param_shifted.iloc[((int(row['Min lag']) - int(row['Max lag'])) // 5) - 1:].mean()
            model_input.append(average_param_from_min_to_max_lag)
        else:
            model_input.append(NaN)
    logger.info(f"Append values with lags to model input list")

    # Проверка инпута на пустые значения
    if NaN in model_input:
        logger.error(f"NaN in model input {model_input.index(NaN)} model input is {model_input}")
    else:
        logger.info(f"Model input is successfully checked for nans")

    return model_input


def ask_rest(model_input, model_name):
    """
    Функция запроса предсказания через rest api

    Args:
        model_input (list): список значений датчиков с лагами, подаваемый далее на вход модели
        model_name (str): имя модели, использующееся для сопоставления с моделями в REST API
    Returns:
        float: результат предсказания
    """

    # Получение списка всех моделей для нахождения индекса нужной
    try:
        get_models_request = requests.request('GET', config['rest']['inturl'],
                             auth=HTTPBasicAuth(config['rest']['user'], config['rest']['password']))
        models = get_models_request.json()
        logger.info(f"Get models request is done")

    except Exception as error1:
        logger.error(f"Error in parsing response {error1}")

    model_num_in_rest = list(models.keys())[list(models.values()).index(model_name)]
    model_num_in_rest = int(model_num_in_rest.split()[-1])

    # Парсинг результата предсказания
    try:
        get_prediction_query = requests.request('POST',
                                            f'{config["rest"]["inturl"]}{int(model_num_in_rest)}/predict',
                                            auth=HTTPBasicAuth(config['rest']['user'], config['rest']['password']),
                                            json={"input": model_input})
    except Exception as error2:
        logger.error(f"Error in parsing response {error2}")

    prediction = float(get_prediction_query.json()['pred'])
    logger.info(f"Success prediction parsing")
    return prediction


def save_data(resdf):
    """
        Функция сохранения предсказания в ClickHouse

        Args:
            resdf (pd.DataFrame): датафрейм с данными предсказания
        """
    resdf['unit_id'] = resdf['unit_id'].astype(int)
    resdf['period_sec'] = resdf['period_sec'].astype(int)
    resdf['confidence'] = resdf['confidence'].astype(int)

    c = ClickHouseConnect(
        database=config['clickhouse']['database'], link=config['clickhouse']['intaddr'],
        user=config['clickhouse']['user'], password=config['clickhouse']['password'])
    
    logger.info(f"Starting writing prediction to clickhouse")

    try:
        c.client.execute(f"INSERT INTO indications VALUES",
                         resdf[['unit_id', 'sensor_name', 'date_time_utc',
                                'period_sec', 'indication', 'confidence' ]].to_dict('records'),
                         types_check=True)
    except Exception as err:
        logger.error(f"Error on writing to clickhouse: {err}")
    finally:
        logger.debug(f"Success writing prediction to ClickHouse")


def executeTask(task, dt):
    """Функция выполнения 1 задачи

    Args:
        task (dict): задача из конфигурации
    """
    logger.info(f"Execute task {task['name']}")

    model_name = open(f'/app/models/{task["folder"]}/model.txt', encoding='utf-8').read()
    model_params = pd.read_csv(f'/app/models/{task["folder"]}/model_new.csv', delimiter=';')
    max_lag = max(model_params['Max lag']) + 300
    tags = model_params['Name']

    # Время до которого брать интервал (UTC)
    # dt = datetime.datetime.utcnow()
    dt = dt + datetime.timedelta(0, -dt.second, -dt.microsecond)

    # Получение данных из ClickHouse
    data = get_data(dt, max_lag, tags)
    logger.info(f'Get raw data')

    logger.info(f"Evaluating expressions")
    # Выполнение дополнительных расчетов для задачи
    for expression in task['evals']:
        exec(expression, globals(), locals())
    logger.debug(f"Evaluated data {data}")

    # Запрос предсказания через REST
    model_input = calc_lags(data, model_params)
    if NaN not in model_input:
        prediction = ask_rest(model_input, model_name)
        logger.info(f'Made prediction {prediction} for {task["name"]}')

        # Формирование таблицы резульатов
        resdf = pd.DataFrame()
        df_data = {
            'unit_id': task['unit_id'],
            'sensor_name': f'{task["predict_var"]}_v{task["version"]}',
            'date_time_utc': dt,
            'period_sec': 0,
            'indication': prediction,
            'confidence': 100
        }
        resdf = resdf.append(df_data, ignore_index=True)
        logger.debug(f'Resulting DF: \n {resdf}')

        # Запись данных в Clickhouse
        # save_data(resdf)


def executeTasks(dt=datetime.datetime.utcnow()-datetime.timedelta(hours=7)):
    """Функция выполнения задач по одной
    """
    global config

    for task in config['tasks']:
        try:
            executeTask(task, dt)
        except Exception as error:
            logger.error(f"Error on writing to clickhouse: {error}")
            continue


def load_tasks(taskFile):
    """Функция загрузки конфига задачи обновления

    Args:
        taskFile (string): Имя файла json
    """
    global config
    try:
        with open(taskFile, "r") as read_file:
            config = json.load(read_file)
        logger.debug(f"Loaded tasks config file {taskFile}")
        return True
    except Exception as error:
        logger.error(f"Error in parsing tasks config file {taskFile, error}")
        return False


if __name__ == "__main__":
    start = time.time()
    logger.info("Start executing predictor script")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="configFile",
                        help="Task config file", default="tasks.json",)

    args = parser.parse_args()

    if (load_tasks(args.configFile.strip())):
        executeTasks()
    end = time.time()
    logger.info(f"End executing, execution time is {end - start} seconds")