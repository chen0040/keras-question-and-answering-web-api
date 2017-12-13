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


class SquadSeq2SeqQA(object):
    model = None
    question_word2idx = None
    question_idx2word = None
    context_word2idx = None
    context_idx2word = None
    context = None
    encoder_model = None
    decoder_model = None

    context_max_seq_length = None
    question_max_seq_length = None

    target_word2idx = None
    target_idx2word = None

    num_decoder_tokens = None

    def __init__(self):

        self.question_word2idx = np.load(MODEL_DIR + '/word-question-word2idx.npy').item()
        self.question_idx2word = np.load(MODEL_DIR + '/word-question-idx2word.npy').item()

        self.context_word2idx = np.load(MODEL_DIR + '/word-context-word2idx.npy').item()
        self.context_idx2word = np.load(MODEL_DIR + '/word-context-idx2word.npy').item()

        self.target_word2idx = np.load(MODEL_DIR + '/word-ans-word2idx.npy').item()
        self.target_idx2word = np.load(MODEL_DIR + '/word-ans-idx2word.npy').item()

        self.context = np.load(MODEL_DIR + '/word-squad-context.npy').item()
        num_context_tokens = self.context['num_context_tokens']
        self.context_max_seq_length = self.context['context_max_seq_length']
        num_question_tokens = self.context['num_question_tokens']
        self.question_max_seq_length = self.context['question_max_seq_length']
        self.num_decoder_tokens = self.context['num_decoder_tokens']
        self.max_decoder_seq_length = self.context['ans_max_seq_length']

        context_inputs = Input(shape=(None,), name='context_inputs')
        encoded_context = Embedding(input_dim=num_context_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                    input_length=self.context_max_seq_length, name='context_embedding')(context_inputs)
        encoded_context = Dropout(0.3)(encoded_context)

        question_inputs = Input(shape=(None,), name='question_inputs')
        encoded_question = Embedding(input_dim=num_question_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                     input_length=self.question_max_seq_length, name='question_embedding')(
            question_inputs)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = LSTM(units=EMBED_HIDDEN_UNITS, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(self.context_max_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])
        encoder_outputs, encoder_state_h, encoder_state_c = LSTM(units=HIDDEN_UNITS,
                                                                 name='encoder_lstm', return_state=True)(merged)

        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)

        model.load_weights(MODEL_DIR + '/word-weights.h5')
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.encoder_model = Model([context_inputs, question_inputs], encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

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

        states_value = self.encoder_model.predict([context_data_batch, question_data_batch])
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            # if sample_word != 'START' and sample_word != 'END':
            target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()

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
    app = SquadSeq2SeqQA()
    app.test_run()
