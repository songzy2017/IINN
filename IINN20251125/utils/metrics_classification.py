import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


def get_mse(preds, trues):
    non_zeros_pos = trues != 0
    return np.fabs((trues[non_zeros_pos] - preds[non_zeros_pos])).mean()


def get_accuracy(preds, trues):
    return accuracy_score(trues, preds)


def get_precision(preds, trues):
    return precision_score(trues, preds, average='binary')


def get_recall(preds, trues):
    return recall_score(trues, preds, average='binary')


def get_f1(preds, trues):
    return f1_score(trues, preds, average='binary')


def get_cohen_kappa(preds, trues):
    return cohen_kappa_score(trues, preds)


def get_confusion_matrix(trues, preds):
    return confusion_matrix(trues, preds)


def get_report(trues, preds):
    print(classification_report(trues, preds))
    print(get_confusion_matrix(trues, preds))


def get_result(trues, preds):
    metrice = {
        'accuracy': get_accuracy(preds, trues),
        'precision': get_precision(preds, trues),
        'recall': get_recall(preds, trues),
        'f1': get_f1(preds, trues),
        'cohen_kappa': get_cohen_kappa(preds, trues)}
    return metrice
