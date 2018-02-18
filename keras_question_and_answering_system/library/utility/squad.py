import json
import nltk

from keras_question_and_answering_system.library.utility.qa_data_utils import QADataSet
from keras_question_and_answering_system.library.utility.text_utils import in_white_list


def load_squad(data, data_path, max_data_count=None,
               max_context_seq_length=None,
               max_question_seq_length=None,
               max_target_seq_length=None):
    if data_path is None:
        return

    if max_data_count is None:
        max_data_count = 10000
    if max_context_seq_length is None:
        max_context_seq_length = 300
    if max_question_seq_length is None:
        max_question_seq_length = 60
    if max_target_seq_length is None:
        max_target_seq_length = 50

    with open(data_path) as file:
        json_data = json.load(file)

        for instance in json_data['data']:
            for paragraph in instance['paragraphs']:
                context = paragraph['context']
                context_wid_list = [w.lower() for w in nltk.word_tokenize(context) if in_white_list(w)]
                if len(context_wid_list) > max_context_seq_length:
                    continue
                qas = paragraph['qas']
                for qas_instance in qas:
                    question = qas_instance['question']
                    question_wid_list = [w.lower() for w in nltk.word_tokenize(question) if in_white_list(w)]
                    if len(question_wid_list) > max_question_seq_length:
                        continue
                    answers = qas_instance['answers']
                    for answer in answers:
                        ans = answer['text']
                        answer_wid_list = [w.lower() for w in nltk.word_tokenize(ans) if in_white_list(w)]
                        if len(answer_wid_list) > max_target_seq_length:
                            continue
                        if len(data) < max_data_count:
                            data.append((context, question, ans))

                if len(data) >= max_data_count:
                    break

                break


class SquADDataSet(QADataSet):

    def __init__(self, data_path, max_data_count=None,
                 max_context_seq_length=None,
                 max_question_seq_length=None,
                 max_target_seq_length=None):
        super(SquADDataSet, self).__init__()

        load_squad(self.data, data_path=data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length)

    def load_model(self, data_path, max_data_count=None,
                   max_context_seq_length=None,
                   max_question_seq_length=None,
                   max_target_seq_length=None):
        load_squad(self.data, data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length
                   )
