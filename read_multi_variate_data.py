import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataset = 'ETTh1'
seq_len = 512
pred_len = 96
L = seq_len + pred_len

data_path_dict = {
    'ETTh1': 'datasets/multivariate_benchmark/ETT-small/ETTh1.jsonl',
    'ETTh2': 'datasets/multivariate_benchmark/ETT-small/ETTh2.jsonl',
    'ETTm1': 'datasets/multivariate_benchmark/ETT-small/ETTm1.jsonl',
    'ETTm2': 'datasets/multivariate_benchmark/ETT-small/ETTm2.jsonl',
}



def normalize_data(data, border1s=None, border2s=None):
    train_data = data[border1s[0]:border2s[0]]
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler

def read_data(path):
    if dataset == 'ETTh1' or dataset == 'ETTh2':
        border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset == 'ETTm1' or dataset == 'ETTm2':
        border1s = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        border1s = None
        border2s = None
    # read jsonl file
    # path = data_path_dict[dataset]
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    data = [d["sequence"] for d in data]
    data = np.array(data)  # N x T
    N = len(data[0])
    
    if border1s is not None:
        eval_data = data[:,border1s[0]:border2s[0]]
    else:
        n_test = int(N*0.2)
        eval_data = eval_data[:-n_test]
    
    eval_scaler = normalize_data(data, border1s, border2s)

    data = eval_data
    scaler_list = []
    
    all_samples = []

    for d in data:
        # assert len(d) == n_test
        scaler = MinMaxScaler(feature_range=(0, 1))
        d = scaler.fit_transform(np.array(d).reshape(-1, 1)).reshape(d.shape)
        scaler_list.append(scaler)

        samples = []
        for i in range(len(d)):
            if i+L < len(d):
                samples.append(d[i:i+L])
        all_samples.append(np.array(samples))
    all_samples = np.concatenate(np.expand_dims(all_samples, axis=0), axis=0)  # (N*num_samples) x L x C
    all_samples = all_samples.transpose((1, 0, 2))  # N x C x L
    
    return all_samples, scaler_list, eval_scaler
