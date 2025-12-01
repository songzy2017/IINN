import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import warnings
warnings.filterwarnings("ignore")


def get_mse(preds, trues):
    return mean_squared_error(trues, preds)


def get_mae(preds, trues):
    return mean_absolute_error(trues, preds)


def get_mape(preds, trues):
    return mean_absolute_percentage_error(trues, preds)


def get_R2(preds, trues):
    return r2_score(trues, preds)


def get_result(trues, preds, data):
    trues = data.inverse_transform_col(trues, -1)
    preds = data.inverse_transform_col(preds, -1)
    metrice = {
        'MSE': get_mse(preds, trues),
        'MAE': get_mae(preds, trues),
        'MAPE': get_mape(preds, trues),
        'R2': get_R2(preds, trues)}
    return metrice
