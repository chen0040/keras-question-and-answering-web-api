import os
import numpy as np


def main():
    model_dir_path = './models'
    models = ['seq2seq-qa', 'seq2seq-qa-v2', 'seq2seq-qa-glove']
    histories = list()
    for model_name in models:
        history_file_name = model_name + '-history.npy'
        history_file_path = os.path.join(model_dir_path, history_file_name)
        history = np.load(history_file_path).item()
        for index in range(0, 201, 50):
            acc_data = history['acc']
            epoch = min(len(acc_data)-1, index)
            acc = acc_data[epoch]
            print(model_name, epoch, acc)


if __name__ == '__main__':
    main()
