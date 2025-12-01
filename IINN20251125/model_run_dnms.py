import pandas as pd
import torch
import argparse

from torch.utils.data import DataLoader
from models import RDNMs
from utils import data_loader, metrics_predication
from trainer import models_trainer, models_predictor
from scripts import fileio


def main(args):
    dataset_configs = fileio.load_config(args.dataset_config_path)
    model_configs = fileio.load_config(args.model_config_path)
    train_config = fileio.load_config(args.train_config_path)
    dataset_config = dataset_configs[args.dataset_config]
    model_config = model_configs[args.model_config]
    run_times = train_config['run_times']
    dataset_name = args.dataset_config
    model_name = args.model_config
    datafile = fileio.filefolermaker(model_name, dataset_name)
    data = data_loader.Covid_19_Dataset(dataset_config)
    trainDataLoader = DataLoader(data.data_train, batch_size=train_config['batch_size'], shuffle=True)
    testDataLoader = DataLoader(data.data_test, batch_size=train_config['batch_size'], shuffle=False)
    device = torch.device(train_config['device'])
    input_size = len(dataset_config['features'])
    output_size = dataset_config['output_size']
    dendrite_num = dataset_config['step_num']
    num_layers = 1
    flexible_synapse = model_config['flexible_synapse']
    flexible_soma = model_config['flexible_soma']
    print(f'Model: {model_name}')
    print(f'Dataset: {dataset_name}')
    print(f'Devive: {device}')
    print(f'Input Size: {input_size}')
    print(f'Dendrite Num: {dendrite_num}')
    print(f'Layer Num: {num_layers}')
    print(f'Synapse: {flexible_synapse}')
    print(f'Soma: {flexible_soma}')
    print(f'Train Data: {len(data.data_train)}')
    print(f'Test Data: {len(data.data_test)}')
    for run_time in range(run_times):
        makername = model_config['maker']
        model = eval(f'{makername}s.{makername}(input_size, dendrite_num, num_layers, output_size, flexible_synapse, flexible_soma).to(device)')
        print(f'Dataset: {dataset_name:-^6}, Model: {model_name:-^12}, Time: {run_time:0>2d}/{run_times-1:0>2d}')
        model, convergence = models_trainer.train(model, model_config, train_config, data, trainDataLoader, testDataLoader)
        torch.save(model, f'{datafile.foldername_save_model}/{model_name}_{str(run_time).zfill(2)}.pkl')
        train_o_preds, train_preds, train_trues = models_predictor.predict(model, model_config, train_config, trainDataLoader)
        test_o_preds, test_preds, test_trues = models_predictor.predict(model, model_config, train_config, testDataLoader)
        train_metrics_buffer = metrics_predication.get_result(train_o_preds, train_trues, data)
        test_metrics_buffer = metrics_predication.get_result(test_o_preds, test_trues, data)
        train_metrics = pd.DataFrame(train_metrics_buffer, index=[0])
        test_metrics = pd.DataFrame(test_metrics_buffer, index=[0])
        train_o_preds = data.inverse_transform_col(train_o_preds, -1)
        train_preds = data.inverse_transform_col(train_preds, -1)
        train_trues = data.inverse_transform_col(train_trues, -1)
        test_o_preds = data.inverse_transform_col(test_o_preds, -1)
        test_preds = data.inverse_transform_col(test_preds, -1)
        test_trues = data.inverse_transform_col(test_trues, -1)
        train_o_preds_buffer = pd.DataFrame([train_o_preds], columns=range(0, len(train_o_preds)))
        train_preds_buffer = pd.DataFrame([train_preds], columns=range(0, len(train_preds)))
        train_trues_buffer = pd.DataFrame([train_trues], columns=range(0, len(train_trues)))
        test_o_preds_buffer = pd.DataFrame([test_o_preds], columns=range(0, len(test_o_preds)))
        test_preds_buffer = pd.DataFrame([test_preds], columns=range(0, len(test_preds)))
        test_trues_buffer = pd.DataFrame([test_trues], columns=range(0, len(test_trues)))
        datafile.save_convergence = pd.concat([datafile.save_convergence, convergence]).reset_index(drop=True)
        datafile.save_train_o_preds = pd.concat([datafile.save_train_o_preds, train_o_preds_buffer]).reset_index(drop=True)
        datafile.save_train_preds = pd.concat([datafile.save_train_preds, train_preds_buffer]).reset_index(drop=True)
        datafile.save_train_trues = pd.concat([datafile.save_train_trues, train_trues_buffer]).reset_index(drop=True)
        datafile.save_test_o_preds = pd.concat([datafile.save_test_o_preds, test_o_preds_buffer]).reset_index(drop=True)
        datafile.save_test_preds = pd.concat([datafile.save_test_preds, test_preds_buffer]).reset_index(drop=True)
        datafile.save_test_trues = pd.concat([datafile.save_test_trues, test_trues_buffer]).reset_index(drop=True)
        datafile.save_train_metrics = pd.concat([datafile.save_train_metrics, train_metrics]).reset_index(drop=True)
        datafile.save_test_metrics = pd.concat([datafile.save_test_metrics, test_metrics]).reset_index(drop=True)
        datafile.data_saver()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', type=str, default='./config/dataset_config.yaml')
    parser.add_argument('--model_config_path', type=str, default='./config/model_config.yaml')
    parser.add_argument('--train_config_path', type=str, default='./config/train_config.yaml')
    parser.add_argument('--dataset_config', type=str, default='D1_Max_6')
    parser.add_argument('--model_config', type=str, default='RFDNM')
    args = parser.parse_args()
    main(args)
