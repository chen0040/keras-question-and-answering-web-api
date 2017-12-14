from keras.models import model_from_json, Model
import numpy as np
from keras.layers import LSTM, Input, Embedding, Dropout, RepeatVector, add, Dense
from keras.preprocessing.sequence import pad_sequences
import nltk
from qa_system_web.squad_dataset import SquADDataSet

MODEL_DIR = '../qa_system_train/models/SQuaD'
EMBED_HIDDEN_UNITS = 64
HIDDEN_UNITS = 256
WHITE_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890,.?'


def in_white_list(_word):
    for char in _word:
        if char in WHITE_LIST:
            return True

    return False


class SquadRnnEmbQA(object):
    model = None
    question_word2idx = None
    question_idx2word = None
    context_word2idx = None
    context_idx2word = None

    context_max_seq_length = None
    question_max_seq_length = None

    target_sent2idx = None
    target_idx2sent = None

    ans_size = None

    def __init__(self):

        self.question_word2idx = np.load(MODEL_DIR + '/rnn-emb-question-word2idx.npy').item()
        self.question_idx2word = np.load(MODEL_DIR + '/rnn-emb-question-idx2word.npy').item()

        self.context_word2idx = np.load(MODEL_DIR + '/rnn-emb-context-word2idx.npy').item()
        self.context_idx2word = np.load(MODEL_DIR + '/rnn-emb-context-idx2word.npy').item()

        self.target_sent2idx = np.load(MODEL_DIR + '/rnn-emb-ans-sent2idx.npy').item()
        self.target_idx2sent = np.load(MODEL_DIR + '/rnn-emb-ans-idx2sent.npy').item()

        self.context = np.load(MODEL_DIR + '/rnn-emb-squad-context.npy').item()
        num_context_tokens = self.context['num_context_tokens']
        self.context_max_seq_length = self.context['context_max_seq_length']
        num_question_tokens = self.context['num_question_tokens']
        self.question_max_seq_length = self.context['question_max_seq_length']
        self.ans_size = self.context['ans_size']

        context_inputs = Input(shape=(None,), name='context_inputs')
        encoded_context = Embedding(input_dim=num_context_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                    input_length=self.context_max_seq_length, name='context_embedding')(context_inputs)
        encoded_context = Dropout(0.3)(encoded_context)

        question_inputs = Input(shape=(None,), name='question_inputs')
        encoded_question = Embedding(input_dim=num_question_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                     input_length=self.question_max_seq_length, name='question_embedding')(question_inputs)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = LSTM(units=EMBED_HIDDEN_UNITS, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(self.context_max_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])
        merged = LSTM(units=HIDDEN_UNITS, name='decoder_lstm')(merged)
        merged = Dropout(0.3)(merged)
        preds = Dense(self.ans_size, activation='softmax')(merged)

        self.model = Model([context_inputs, question_inputs], preds)

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def predict(self, question_context, question):
        question_context = [w.lower() for w in nltk.word_tokenize(question_context) if in_white_list(w)]
        question = [w.lower() for w in nltk.word_tokenize(question) if in_white_list(w)]
        context_data_batch = []
        question_data_batch = []

        question_wids = []
        for w in question:
            wid = 1
            if w in self.question_word2idx:
                wid = self.question_word2idx[w]
            question_wids.append(wid)

        context_wids = []
        for w in question_context:
            wid = 1
            if w in self.context_word2idx:
                wid = self.context_word2idx[w]
            context_wids.append(wid)

        context_data_batch.append(context_wids)
        question_data_batch.append(question_wids)

        context_data_batch = pad_sequences(context_data_batch, self.context_max_seq_length)
        question_data_batch = pad_sequences(question_data_batch, self.question_max_seq_length)

        states_value = self.model.predict([context_data_batch, question_data_batch])
        predicted_index = np.argmax(states_value)
        predicted_value = self.target_idx2sent[predicted_index]
        return predicted_value

    def test_run(self):
        dataset = SquADDataSet()
        for i in range(10):
            record = dataset.get_data(i * 50)
            question_context = record[0]
            question = record[1]
            answer = record[2]
            print('in: ', question_context)
            print('question: ', question)
            print('guess: ', self.predict(question_context, question))
            print('answer: ', answer)


if __name__ == '__main__':
    app = SquadRnnEmbQA()
    app.test_run()
