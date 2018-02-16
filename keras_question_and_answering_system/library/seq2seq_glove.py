from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk

from keras_question_and_answering_system.library.utility.glove_model import GloveModel
from keras_question_and_answering_system.library.utility.text_utils import in_white_list


class Seq2SeqGloveQA(object):
    name = 'seq2seq-glove'

    def __init__(self):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.target_word2idx = None
        self.target_idx2word = None
        self.max_decoder_seq_length = None
        self.max_encoder_seq_length = None
        self.num_decoder_tokens = None
        self.glove_model = None

    def load_glove_model(self, data_dir_path):
        self.glove_model = GloveModel()
        self.glove_model.load_model(data_dir_path)

    def load_model(self, model_dir_path):
        self.target_word2idx = np.load(
            model_dir_path + '/' + Seq2SeqGloveQA.name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(
            model_dir_path + '/' + Seq2SeqGloveQA.name + '-target-idx2word.npy').item()
        context = np.load(model_dir_path + '/' + Seq2SeqGloveQA.name + '-config.npy').item()
        self.max_encoder_seq_length = context['input_max_seq_length']
        self.max_decoder_seq_length = context['target_max_seq_length']
        self.num_decoder_tokens = context['num_target_tokens']

        self.create_model()
        self.model.load_weights(model_dir_path + '/' + Seq2SeqGloveQA.name + '-weights.h5')

    def create_model(self):
        hidden_units = 256

        encoder_inputs = Input(shape=(None, self.glove_model.embedding_size), name='encoder_inputs')
        encoder_lstm = LSTM(units=hidden_units, return_state=True, name="encoder_lstm")
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=hidden_units, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(hidden_units,)), Input(shape=(hidden_units,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, paragraph, question):
        input_seq = []
        input_emb = []
        input_text = paragraph.lower() + ' question ' + question.lower()
        for word in nltk.word_tokenize(input_text):
            if not in_white_list(word):
                continue
            emb = self.glove_model.encode_word(word)
            input_emb.append(emb)
        input_seq.append(input_emb)
        input_seq = pad_sequences(input_seq, self.max_encoder_seq_length)
        states_value = self.encoder_model.predict(input_seq)
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

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()

    def test_run(self, ds, index=None):
        if index is None:
            index = 0
        paragraph, question, actual_answer = ds.get_data(index)
        predicted_answer = self.reply(paragraph, question)
        print({'predict': predicted_answer, 'actual': actual_answer})



