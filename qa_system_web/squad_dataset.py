import json
import nltk

DATA_PATH = '../qa_system_train/data/SQuAD/train-v1.1.json'
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
                            ans_wids = [w.lower() for w in nltk.word_tokenize(ans) if in_white_list(w)]
                            ans_wids = ['START'] + ans_wids + ['END']
                            if len(ans_wids) > MAX_TARGET_SEQ_LENGTH:
                                continue
                            if len(self.data) < MAX_DATA_COUNT:
                                self.data.append((context, question, ans))
                if len(self.data) >= MAX_DATA_COUNT:
                    break

    def get_data(self, index):
        return self.data[index]


def main():
    dataset = SquADDataSet()
    print(dataset.get_data(2))

if __name__ == '__main__':
    main()
