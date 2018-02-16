from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from keras_question_and_answering_system.library.utility.text_utils import in_white_list
import nltk


class SQuADSeq2SeqEmbTripleSamples(object):
    input_paragraph_max_seq_length = None
    input_question_max_seq_length = None
    target_max_seq_length = None

    word2emb = None

    target_word2idx = None

    target_idx2word = None

    num_target_tokens = None

    samples = None

    dataset = None

    def __init__(self, dataset, word2emb, embed_size, max_target_vocab_size=None):
        if max_target_vocab_size is None:
            max_target_vocab_size = 5000

        self.dataset = dataset
        self.word2emb = word2emb
        self.input_data_samples = []
        self.output_data_samples = []

        self.input_paragraph_max_seq_length = 0
        self.input_question_max_seq_length = 0
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

            output_data = ['START'] + answer_word_list + ['END']

            input_data_samples.append([paragraph_word_list, question_word_list])
            output_data_samples.append(output_data)

            for w in output_data:
                target_counter[w] += 1

            self.input_paragraph_max_seq_length = max(self.input_paragraph_max_seq_length, len(paragraph_word_list))
            self.input_question_max_seq_length = max(self.input_question_max_seq_length, len(question_word_list))
            self.target_max_seq_length = max(self.target_max_seq_length, len(output_data))

        self.target_word2idx = dict()
        for idx, word in enumerate(target_counter.most_common(max_target_vocab_size)):
            self.target_word2idx[word[0]] = idx + 1

        self.target_word2idx['UNK'] = 0
        self.target_idx2word = dict([(idx, word) for word, idx in self.target_word2idx.items()])

        self.num_target_tokens = len(self.target_idx2word)

        input_encoded_data_samples = []
        target_encoded_data_samples = []

        for input_data, output_data in zip(input_data_samples, output_data_samples):
            input_paragraph_encoded_data = []
            input_question_encoded_data = []
            target_encoded_data = []
            input_paragraph_data, input_question_data = input_data
            for word in input_question_data:
                if word in self.word2emb:
                    input_question_encoded_data.append(self.word2emb[word])
                else:
                    input_question_encoded_data.append(unknown_emb)
            for word in input_paragraph_data:
                if word in self.word2emb:
                    input_paragraph_encoded_data.append(self.word2emb[word])
                else:
                    input_paragraph_encoded_data.append(unknown_emb)
            for word in output_data:
                if word in self.target_word2idx:
                    target_encoded_data.append(self.target_word2idx[word])
                else:
                    target_encoded_data.append(0)
            input_encoded_data_samples.append([input_paragraph_encoded_data, input_question_encoded_data])
            target_encoded_data_samples.append(target_encoded_data)

        self.samples = [input_encoded_data_samples, target_encoded_data_samples]

    def save(self, dir_path, embed_type):
        np.save(dir_path + '/seq2seq-' + embed_type + '-target-word2idx.npy', self.target_word2idx)
        np.save(dir_path + '/seq2seq-' + embed_type + '-target-idx2word.npy', self.target_idx2word)

        config = dict()
        config['num_target_tokens'] = self.num_target_tokens
        config['input_question_max_seq_length'] = self.input_question_max_seq_length
        config['input_paragraph_max_seq_length'] = self.input_paragraph_max_seq_length
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


class SQuADSeq2SeqEmbTupleSamples(object):
    input_max_seq_length = None
    target_max_seq_length = None

    word2emb = None

    target_word2idx = None

    target_idx2word = None

    num_target_tokens = None

    samples = None

    data_set = None

    def __init__(self, data_set, word2emb, embed_size, max_target_vocab_size=None):
        if max_target_vocab_size is None:
            max_target_vocab_size = 5000

        self.data_set = data_set
        self.word2emb = word2emb
        self.input_data_samples = []
        self.output_data_samples = []

        self.input_max_seq_length = 0
        self.target_max_seq_length = 0

        unknown_emb = np.zeros(shape=embed_size)

        target_counter = Counter()

        input_data_samples = []
        output_data_samples = []

        for sample in self.data_set.data:
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
        for idx, word in enumerate(target_counter.most_common(max_target_vocab_size)):
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
        return self.data_set.size()

    def get_samples(self):
        return self.samples

    def split(self, test_size, random_state):
        input_data, target_data = self.samples
        return train_test_split(input_data, target_data, test_size=test_size,
                                random_state=random_state)
