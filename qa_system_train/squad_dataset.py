import json
import nltk
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/SQuAD/train-v1.1.json'
WHITE_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890,.?'
MAX_CONTEXT_SEQ_LENGTH = 300
MAX_QUESTION_SEQ_LENGTH = 60
MAX_TARGET_SEQ_LENGTH = 50
MAX_VOCAB_SIZE = 5000
MAX_DATA_COUNT = 5000


def in_white_list(_word):
    for char in _word:
        if char in WHITE_LIST:
            return True

    return False


class SquADDataSet(object):
    data = None

    def __init__(self):
        self.data = []

        with open(DATA_PATH) as file:
            json_data = json.load(file)

            for instance in json_data['data']:
                for paragraph in instance['paragraphs']:
                    context = paragraph['context']
                    context_wids = [w.lower() for w in nltk.word_tokenize(context) if in_white_list(w)]
                    if len(context_wids) > MAX_CONTEXT_SEQ_LENGTH:
                        continue
                    qas = paragraph['qas']
                    for qas_instance in qas:
                        question = qas_instance['question']
                        question_wids = [w.lower() for w in nltk.word_tokenize(question) if in_white_list(w)]
                        if len(question_wids) > MAX_QUESTION_SEQ_LENGTH:
                            continue
                        answers = qas_instance['answers']
                        for answer in answers:
                            ans = answer['text']
                            answer_wids = [w.lower() for w in nltk.word_tokenize(ans) if in_white_list(w)]
                            if len(answer_wids) > MAX_TARGET_SEQ_LENGTH:
                                continue
                            if len(self.data) < MAX_DATA_COUNT:
                                self.data.append((context, question, ans))

                    if len(self.data) >= MAX_DATA_COUNT:
                        break

                if len(self.data) >= MAX_DATA_COUNT:
                    break

    def get_data(self, index):
        return self.data[index]

    def size(self):
        return len(self.data)


class SQuADSeq2Seq(object):
    input_max_seq_length = None
    target_max_seq_length = None

    input_word2idx = None
    target_word2idx = None

    input_idx2word = None
    target_idx2word = None

    num_input_tokens = None
    num_target_tokens = None

    samples = None

    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.input_data_samples = []
        self.output_data_samples = []

        self.input_max_seq_length = 0
        self.target_max_seq_length = 0

        input_counter = Counter()
        target_counter = Counter()

        input_data_samples = []
        output_data_samples = []

        for sample in self.dataset.data:
            paragraph, question, answer = sample
            paragraph_word_list = [w.lower() for w in nltk.word_tokenize(paragraph) if in_white_list(w)]
            question_word_list = [w.lower() for w in nltk.word_tokenize(question) if in_white_list(w)]
            answer_word_list = [w.lower() for w in nltk.word_tokenize(answer) if in_white_list(w)]

            input_data = paragraph_word_list + ['Q'] + question_word_list
            output_data = ['START'] + answer_word_list + ['END']

            input_data_samples.append(input_data)
            output_data_samples.append(output_data)

            for w in input_data:
                input_counter[w] += 1
            for w in output_data:
                target_counter[w] += 1

            self.input_max_seq_length = max(self.input_max_seq_length, len(input_data))
            self.target_max_seq_length = max(self.target_max_seq_length, len(output_data))

        self.input_word2idx = dict()
        self.target_word2idx = dict()
        for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
            self.input_word2idx[word[0]] = idx + 2
        for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
            self.target_word2idx[word[0]] = idx + 1

        self.target_word2idx['UNK'] = 0
        self.input_word2idx['PAD'] = 0
        self.input_word2idx['UNK'] = 1

        self.input_idx2word = dict([(idx, word) for word, idx in self.input_word2idx.items()])
        self.target_idx2word = dict([(idx, word) for word, idx in self.target_word2idx.items()])

        self.num_input_tokens = len(self.input_idx2word)
        self.num_target_tokens = len(self.target_idx2word)

        input_encoded_data_samples = []
        target_encoded_data_samples = []

        for input_data, output_data in zip(input_data_samples, output_data_samples):
            input_encoded_data = []
            target_encoded_data = []
            for word in input_data:
                if word in self.input_word2idx:
                    input_encoded_data.append(self.input_word2idx[word])
                else:
                    input_encoded_data.append(1)
            for word in output_data:
                if word in self.target_word2idx:
                    target_encoded_data.append(self.target_word2idx[word])
                else:
                    target_encoded_data.append(0)
            input_encoded_data_samples.append(input_encoded_data)
            target_encoded_data_samples.append(target_encoded_data)

        self.samples = [input_encoded_data_samples, target_encoded_data_samples]

    def save(self, dir_path):
        np.save(dir_path + '/seq2seq-input-word2idx.npy', self.input_word2idx)
        np.save(dir_path + '/seq2seq-input-idx2word.npy', self.input_idx2word)
        np.save(dir_path + '/seq2seq-target-word2idx.npy', self.target_word2idx)
        np.save(dir_path + '/seq2seq-target-idx2word.npy', self.target_idx2word)

        config = dict()
        config['num_input_tokens'] = self.num_input_tokens
        config['num_target_tokens'] = self.num_target_tokens
        config['input_max_seq_length'] = self.input_max_seq_length
        config['target_max_seq_length'] = self.target_max_seq_length

        print(config)
        np.save(dir_path + '/seq2seq-config.npy', config)

    def size(self):
        return self.dataset.size()

    def get_samples(self):
        return self.samples

    def split(self, test_size, random_state):
        input_data, target_data = self.samples
        return train_test_split(input_data, target_data, test_size=test_size,
                                random_state=random_state)


class SQuADSeq2SeqEmb(object):
    input_max_seq_length = None
    target_max_seq_length = None

    word2emb = None

    target_word2idx = None

    target_idx2word = None

    num_target_tokens = None

    samples = None

    dataset = None

    def __init__(self, dataset, word2emb, embed_size):
        self.dataset = dataset
        self.word2emb = word2emb
        self.input_data_samples = []
        self.output_data_samples = []

        self.input_max_seq_length = 0
        self.target_max_seq_length = 0

        unknown_emb = np.zeros(shape=embed_size)

        target_counter = Counter()

        input_data_samples = []
        output_data_samples = []

        for sample in self.dataset.data:
            paragraph, question, answer = sample
            paragraph_word_list = [w.lower() for w in nltk.word_tokenize(paragraph) if in_white_list(w)]
            question_word_list = [w.lower() for w in nltk.word_tokenize(question) if in_white_list(w)]
            answer_word_list = [w.lower() for w in nltk.word_tokenize(answer) if in_white_list(w)]

            input_data = paragraph_word_list + ['question'] + question_word_list
            output_data = ['START'] + answer_word_list + ['END']

            input_data_samples.append(input_data)
            output_data_samples.append(output_data)

            for w in output_data:
                target_counter[w] += 1

            self.input_max_seq_length = max(self.input_max_seq_length, len(input_data))
            self.target_max_seq_length = max(self.target_max_seq_length, len(output_data))

        self.target_word2idx = dict()
        for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
            self.target_word2idx[word[0]] = idx + 1

        self.target_word2idx['UNK'] = 0
        self.target_idx2word = dict([(idx, word) for word, idx in self.target_word2idx.items()])

        self.num_target_tokens = len(self.target_idx2word)

        input_encoded_data_samples = []
        target_encoded_data_samples = []

        for input_data, output_data in zip(input_data_samples, output_data_samples):
            input_encoded_data = []
            target_encoded_data = []
            for word in input_data:
                if word in self.word2emb:
                    input_encoded_data.append(self.word2emb[word])
                else:
                    input_encoded_data.append(unknown_emb)
            for word in output_data:
                if word in self.target_word2idx:
                    target_encoded_data.append(self.target_word2idx[word])
                else:
                    target_encoded_data.append(0)
            input_encoded_data_samples.append(input_encoded_data)
            target_encoded_data_samples.append(target_encoded_data)

        self.samples = [input_encoded_data_samples, target_encoded_data_samples]

    def save(self, dir_path, embed_type):
        np.save(dir_path + '/seq2seq-' + embed_type + '-target-word2idx.npy', self.target_word2idx)
        np.save(dir_path + '/seq2seq-' + embed_type + '-target-idx2word.npy', self.target_idx2word)

        config = dict()
        config['num_target_tokens'] = self.num_target_tokens
        config['input_max_seq_length'] = self.input_max_seq_length
        config['target_max_seq_length'] = self.target_max_seq_length

        print(config)
        np.save(dir_path + '/seq2seq-' + embed_type + '-config.npy', config)

    def size(self):
        return self.dataset.size()

    def get_samples(self):
        return self.samples

    def split(self, test_size, random_state):
        input_data, target_data = self.samples
        return train_test_split(input_data, target_data, test_size=test_size,
                                random_state=random_state)

def main():
    ds = SquADDataSet()
    print(ds.get_data(2))


if __name__ == '__main__':
    main()
