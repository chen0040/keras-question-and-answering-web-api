import json
import nltk
from collections import Counter
import numpy as np

DATA_PATH = 'data/SQuAD/train-v1.1.json'
WHITE_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890,.?'
MAX_CONTEXT_SEQ_LENGTH = 300
MAX_QUESTION_SEQ_LENGTH = 60
MAX_TARGET_SEQ_LENGTH = 50
MAX_VOCAB_SIZE = 1000
MAX_DATA_COUNT = 5000


def in_white_list(_word):
    for char in _word:
        if char in WHITE_LIST:
            return True

    return False


class SquADDataSet(object):
    data = None
    wids = None

    context_max_seq_length = None
    question_max_seq_length = None
    answer_max_seq_length = None

    context_word2idx = None
    question_word2idx = None
    answer_word2idx = None

    context_idx2word = None
    question_idx2word = None
    answer_idx2word = None

    num_context_tokens = None
    num_question_tokens = None
    num_answer_tokens = None

    def __init__(self):
        self.data = []
        self.wids = []

        self.context_max_seq_length = 0
        self.question_max_seq_length = 0
        self.answer_max_seq_length = 0

        context_counter = Counter()
        question_counter = Counter()
        answer_counter = Counter()

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
                                self.wids.append((context_wids, question_wids, answer_wids))

                            for w in context_wids:
                                context_counter[w] += 1
                            for w in question_wids:
                                question_counter[w] += 1
                            for w in answer_wids:
                                answer_counter[w] += 1

                            self.context_max_seq_length = max(self.context_max_seq_length, len(context))
                            self.question_max_seq_length = max(self.question_max_seq_length, len(question))
                            self.answer_max_seq_length = max(self.answer_max_seq_length, len(ans))

                    if len(self.data) >= MAX_DATA_COUNT:
                        break

                if len(self.data) >= MAX_DATA_COUNT:
                    break

        self.context_word2idx = dict()
        self.question_word2idx = dict()
        self.answer_word2idx = dict()
        for idx, word in enumerate(question_counter.most_common(MAX_VOCAB_SIZE)):
            self.question_word2idx[word[0]] = idx + 2
        for idx, word in enumerate(context_counter.most_common(MAX_VOCAB_SIZE)):
            self.context_word2idx[word[0]] = idx + 2
        for idx, word in enumerate(answer_counter.most_common(MAX_VOCAB_SIZE)):
            self.answer_word2idx[word[0]] = idx

        self.context_word2idx['PAD'] = 0
        self.context_word2idx['UNK'] = 1
        self.question_word2idx['PAD'] = 0
        self.question_word2idx['UNK'] = 1

        self.context_idx2word = dict([(idx, word) for word, idx in self.context_word2idx.items()])
        self.question_idx2word = dict([(idx, word) for word, idx in self.question_word2idx.items()])
        self.answer_idx2word = dict([(idx, word) for word, idx in self.answer_word2idx.items()])

        self.num_context_tokens = len(self.context_idx2word)
        self.num_question_tokens = len(self.question_idx2word)
        self.num_answer_tokens = len(self.answer_idx2word)

    def get_data(self, index):
        return self.data[index]

    def get_wids(self, index):
        return self.wids[index]

    def size(self):
        return len(self.data)

    def save(self, dir_path):
        np.save(dir_path + '/squad-context-word2idx.npy', self.context_word2idx)
        np.save(dir_path + '/squad-context-idx2word.npy', self.context_idx2word)
        np.save(dir_path + '/squad-question-word2idx.npy', self.question_word2idx)
        np.save(dir_path + '/squad-question-idx2word.npy', self.question_idx2word)
        np.save(dir_path + '/squad-ans-sent2idx.npy', self.answer_word2idx)
        np.save(dir_path + '/squad-ans-idx2sent.npy', self.answer_idx2word)

        config = dict()
        config['num_context_tokens'] = self.num_context_tokens
        config['num_question_tokens'] = self.num_question_tokens
        config['num_answer_tokens'] = self.num_answer_tokens
        config['context_max_seq_length'] = self.context_max_seq_length
        config['question_max_seq_length'] = self.question_max_seq_length
        config['answer_max_seq_length'] = self.answer_max_seq_length

        print(config)
        np.save(dir_path + '/squad-data-config.npy', config)


def main():
    dataset = SquADDataSet()
    print(dataset.get_data(2))

if __name__ == '__main__':
    main()
