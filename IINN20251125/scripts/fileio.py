import os
import yaml
import numpy as np
import pandas as pd


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def makefolder(filename):
    if os.path.exists(filename) is not True:
        os.makedirs(filename, mode=0o777)
    else:
        print(filename + " folder exists")


def makefile(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print(filename + " file does not exist")


def saver(filename, datarecord):
    with open(filename, 'a', newline='') as f:
        np.savetxt(f, [datarecord], delimiter=",")


class filefolermaker:
    def __init__(self, model_name, dataset_name):
        super(filefolermaker, self).__init__()
        self.save_convergence = None
        self.save_train_metrics = None
        self.save_train_o_preds = None
        self.save_train_preds = None
        self.save_train_trues = None
        self.save_test_metrics = None
        self.save_test_o_preds = None
        self.save_test_preds = None
        self.save_test_trues = None

        self.foldername_save_model = f'./output/{dataset_name}/{model_name}'
        self.foldername_save_log = f'./output/{dataset_name}/{model_name}_log'
        self.foldername_save_convergence = f'./result/{dataset_name}/train/convergence'
        self.foldername_save_train_metrics = f'./result/{dataset_name}/train/metrics'
        self.foldername_save_train_o_preds = f'./result/{dataset_name}/train/o_preds'
        self.foldername_save_train_preds = f'./result/{dataset_name}/train/preds'
        self.foldername_save_train_trues = f'./result/{dataset_name}/train/trues'
        self.foldername_save_test_metrics = f'./result/{dataset_name}/test/metrics'
        self.foldername_save_test_o_preds = f'./result/{dataset_name}/test/o_preds'
        self.foldername_save_test_preds = f'./result/{dataset_name}/test/preds'
        self.foldername_save_test_trues = f'./result/{dataset_name}/test/trues'
        self.filename_save_convergence = f'./result/{dataset_name}/train/convergence/{model_name}.csv'
        self.filename_save_train_metrics = f'./result/{dataset_name}/train/metrics/{model_name}.csv'
        self.filename_save_train_o_preds = f'./result/{dataset_name}/train/o_preds/{model_name}.csv'
        self.filename_save_train_preds = f'./result/{dataset_name}/train/preds/{model_name}.csv'
        self.filename_save_train_trues = f'./result/{dataset_name}/train/trues/{model_name}.csv'
        self.filename_save_test_metrics = f'./result/{dataset_name}/test/metrics/{model_name}.csv'
        self.filename_save_test_o_preds = f'./result/{dataset_name}/test/o_preds/{model_name}.csv'
        self.filename_save_test_preds = f'./result/{dataset_name}/test/preds/{model_name}.csv'
        self.filename_save_test_trues = f'./result/{dataset_name}/test/trues/{model_name}.csv'

        self.premaker()

    def premaker(self):
        makefolder(self.foldername_save_model)
        makefolder(self.foldername_save_convergence)
        makefolder(self.foldername_save_train_metrics)
        makefolder(self.foldername_save_train_o_preds)
        makefolder(self.foldername_save_train_preds)
        makefolder(self.foldername_save_train_trues)
        makefolder(self.foldername_save_test_metrics)
        makefolder(self.foldername_save_test_o_preds)
        makefolder(self.foldername_save_test_preds)
        makefolder(self.foldername_save_test_trues)
        makefile(self.filename_save_convergence)
        makefile(self.filename_save_train_metrics)
        makefile(self.filename_save_train_o_preds)
        makefile(self.filename_save_train_preds)
        makefile(self.filename_save_train_trues)
        makefile(self.filename_save_test_metrics)
        makefile(self.filename_save_test_o_preds)
        makefile(self.filename_save_test_preds)
        makefile(self.filename_save_test_trues)

    def data_saver(self):
        self.save_convergence.to_csv(self.filename_save_convergence)
        self.save_train_metrics.to_csv(self.filename_save_train_metrics)
        self.save_train_o_preds.to_csv(self.filename_save_train_o_preds)
        self.save_train_preds.to_csv(self.filename_save_train_preds)
        self.save_train_trues.to_csv(self.filename_save_train_trues)
        self.save_test_metrics.to_csv(self.filename_save_test_metrics)
        self.save_test_o_preds.to_csv(self.filename_save_test_o_preds)
        self.save_test_preds.to_csv(self.filename_save_test_preds)
        self.save_test_trues.to_csv(self.filename_save_test_trues)


def log_saver(filepath, record):
    with open(filepath, 'a') as file:
        file.write(record)


def data_reader(filepath):
    with open(filepath, 'r') as file:
        data = pd.read_csv(file, delimiter=",", index_col=0)
    return data
