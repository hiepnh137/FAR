import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_path_dict = {
    'BikeNYC': 'datasets/BikeNYC_short.json',
    'Crowd': 'datasets/Crowd_short.json',
    'TaxiBJin': 'datasets/TaxiBJin_short.json',
    'TaxiBJout': 'datasets/TaxiBJout_short.json',
}

def normalize_data(data):
    norm_data = []
    scaler_list = []
    for i in range(data.shape[0]):
        sample = data[i]
        scaler = MinMaxScaler(feature_range=(0, 1))
        sample = scaler.fit_transform(sample.reshape(sample.shape[0], -1)).reshape(sample.shape)
        norm_data.append(sample)
        scaler_list.append(scaler)
    return np.array(norm_data), scaler_list

def normalize_data_2(data):
    max_data = max(data.reshape(-1))
    min_data = min(data.reshape(-1))
    data = (data - min_data) / (max_data - min_data)

    return data, min_data, max_data

def read_data(path, num_samples=None):
    data = json.load(open(path, 'r'))
    test_data = data['X_test'][0]  # N x T x H x W
    video = np.array(test_data) 
    if num_samples is not None:
        video = video[:num_samples]
    scaled_video, scaler_list = normalize_data(video)
    
    print('test_data: ', video.shape)
    return video, scaled_video, scaler_list


def read_data_2(path):
    data = json.load(open(path, 'r'))
    test_data = data['X_test'][0]  # N x T x H x W
    video = np.array(test_data) 

    scaled_video, min_data, max_data = normalize_data_2(video)
    
    print('test_data: ', video.shape)
    return video, scaled_video, min_data, max_data