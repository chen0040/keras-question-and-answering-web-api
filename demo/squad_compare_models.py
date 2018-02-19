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
        print(history_file_name, history)


if __name__ == '__main__':
    main()
