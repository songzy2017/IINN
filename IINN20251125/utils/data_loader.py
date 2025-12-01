import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


def change_grade(grade):
    if grade == "A":
        grade = 0
    elif grade == "B":
        grade = 0
    elif grade == "C":
        grade = 1
    elif grade == "D":
        grade = 1
    elif grade == "F":
        grade = 1
    else:
        raise ValueError("invalid grade")
    return grade


class ALPDataset(Dataset):
    def __init__(self, dataset_config):
        self.features = dataset_config['features']
        self.step_num = dataset_config['step_num']
        self.data_train = self.get_data(dataset_config['train_data_names'])
        self.data_test = self.get_data(dataset_config['test_data_names'])

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        data, label = self.data_train[index]
        return data, label

    def get_data(self, data_names):
        output = []
        for data_name in data_names:
            data_path = f'./dataset/alp_{data_name}.csv'
            label_path = f'./dataset/grade_{data_name}.csv'
            data_raw = pd.read_csv(data_path)
            label_raw = pd.read_csv(label_path)
            label_raw['grade'] = label_raw['grade'].apply(change_grade)
            for user in np.unique(data_raw['userid']):
                data_filted_userid = data_raw.loc[data_raw['userid'] == user]
                label_change = label_raw.loc[label_raw['userid'] == user]
                label = np.array(label_change['grade']).astype(np.float32).squeeze(-1)
                data_filted_classid = data_filted_userid.loc[data_filted_userid['classid'] <= self.step_num]
                data = np.array(data_filted_classid[self.features]).astype(np.float32)
                output.append((data, label))
        return output


class Covid_19_Dataset(Dataset):
    def __init__(self, dataset_config):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = dataset_config['features']
        self.step_num = dataset_config['step_num']
        self.data_train, self.data_test = self.get_data(dataset_config['train_data_names'])

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, index):
        data, label = self.data_train[index]
        return data, label

    def inverse_transform_col(self, data, n_col):
        data = data.copy()
        data -= self.scaler.min_[n_col]
        data /= self.scaler.scale_[n_col]
        return data

    def get_data(self, data_names):
        train_data = []
        test_data = []
        for data_name in data_names:
            data_path = f'./dataset/{data_name}.csv'
            data_raw_buffer = pd.read_csv(data_path)
            data_raw = data_raw_buffer[self.features]
            self.scaler.fit(data_raw)
            data = self.scaler.transform(data_raw)
            # data = np.array(data_raw)
            rate = 0.8
            mid_point = int(len(data) * rate)
            for i in range(len(data_raw) - self.step_num):
                start_point = i
                end_point = start_point + self.step_num
                data_x = data[start_point:end_point]
                data_y = data[end_point, data.shape[1] - 1]
                # print("x", data_x)
                # print("y", data_y)
                if i <= mid_point:
                    train_data.append((data_x, data_y))
                else:
                    test_data.append((data_x, data_y))
        return train_data, test_data
