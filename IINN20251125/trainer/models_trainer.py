import time
import torch
import copy
import pandas as pd

from scripts import timers
from trainer import models_predictor
from utils import metrics_predication


def trainDNM(model, train_config, data):
    device = torch.device(train_config['device'])
    optimizer = getattr(torch.optim, train_config['optimizer'])(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
    criterion_config = train_config['criterion']
    criterion = eval(f'torch.nn.{criterion_config}()')
    current_loss = 0
    for i, (data_batch, label_batch) in enumerate(data):
        data_batch = data_batch.float().to(device)
        label_batch = label_batch.float().to(device)
        outputs = model(data_batch)
        loss = criterion(outputs, label_batch)
        current_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return current_loss


def train(model, model_config, train_config, data, train_data, test_data):
    model_name = model_config['train_type_fn']
    iters_num = train_config['iters_num']
    convergence_buffer = []
    for iter in range(iters_num):
        start_time = time.time()
        current_loss = eval(f'train{model_name}(model, train_config, train_data)')
        convergence_buffer.append(current_loss.item())
        o_preds, preds, trues = models_predictor.predict(model, model_config, train_config, test_data)
        metrice_buffer = metrics_predication.get_result(o_preds, trues, data)
        print(f'{iter:0>3d}/{iters_num-1:0>3d}, ', end='')
        for key, value in metrice_buffer.items():
            print(f'{key[:3]}: {value:.3%}, ', end='')
        print(f'** {timers.timeSince(start_time)}', end='\r')
    print(f'{iter:0>3d}/{iters_num-1:0>3d}, ', end='')
    for key, value in metrice_buffer.items():
        print(f'{key[:3]}: {value:.2%}, ', end='')
    print(f'**  {timers.timeSince(start_time)}', end='\n')
    print()
    convergence = pd.DataFrame([convergence_buffer], columns=range(0, len(convergence_buffer)))
    return model, convergence
