import nltk
import numpy as np
from keras.layers import Input, LSTM, Dense, Dropout, add, RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from keras_question_and_answering_system.library.utility.glove_model import GloveModel
from qa_system_web.text_utils import in_white_list
import qa_system_web.glove_loader as glove_loader

from qa_system_web.squad_dataset import SquADDataSet
import os

class Seq2SeqGloveV2QA(object):
    model_name = 'seq2seq-qa-glove-v2'

    def __init__(self):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.target_word2idx = None
        self.target_idx2word = None
        self.max_decoder_seq_length = None
        self.max_encoder_paragraph_seq_length = None
        self.max_encoder_question_seq_length = None
        self.num_decoder_tokens = None
        self.glove_model = GloveModel()

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqGloveV2QA.model_name + '-weights.h5')

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqGloveV2QA.model_name + '-architecture.json')

    def load_glove_model(self, data_dir_path):
        self.glove_model.load_model(data_dir_path)

    def load_model(self, model_dir_path):
        self.target_word2idx = np.load(
            model_dir_path + '/' + Seq2SeqGloveV2QA.model_name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(
            model_dir_path + '/' + Seq2SeqGloveV2QA.model_name + '-target-idx2word.npy').item()
        context = np.load(model_dir_path + '/' + Seq2SeqGloveV2QA.model_name + '-config.npy').item()
        self.max_encoder_paragraph_seq_length = context['input_paragraph_max_seq_length']
        self.max_encoder_question_seq_length = context['input_question_max_seq_length']
        self.max_decoder_seq_length = context['target_max_seq_length']
        self.num_decoder_tokens = context['num_target_tokens']

        self.create_model()
        self.model.load_weights(Seq2SeqGloveV2QA.get_weight_file_path(model_dir_path))

    def create_model(self):
        hidden_units = 256

        context_inputs = Input(shape=(None, self.glove_model.embedding_size), name='context_inputs')
        encoded_context = Dropout(0.3)(context_inputs)

        question_inputs = Input(shape=(None, glove_loader.GLOVE_EMBEDDING_SIZE), name='question_inputs')
        encoded_question = Dropout(0.3)(question_inputs)
        encoded_question = LSTM(units=glove_loader.GLOVE_EMBEDDING_SIZE, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(self.max_encoder_paragraph_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])
        encoder_outputs, encoder_state_h, encoder_state_c = LSTM(units=hidden_units,
                                                                 name='encoder_lstm', return_state=True)(merged)

        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=hidden_units, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.encoder_model = Model([context_inputs, question_inputs], encoder_states)

        decoder_state_inputs = [Input(shape=(hidden_units,)), Input(shape=(hidden_units,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, paragraph, question):
        input_paragraph_seq = []
        input_question_seq = []
        input_paragraph_emb = []
        input_question_emb = []
        input_paragraph_text = paragraph.lower()
        input_question_text = question.lower()
        for word in nltk.word_tokenize(input_paragraph_text):
            if not in_white_list(word):
                continue
            emb = self.glove_model.encode_word(word)
            input_paragraph_emb.append(emb)
        for word in nltk.word_tokenize(input_question_text):
            if not in_white_list(word):
                continue
            emb = self.glove_model.encode_word(word)
            input_question_emb.append(emb)
        input_paragraph_seq.append(input_paragraph_emb)
        input_question_seq.append(input_question_emb)
        input_paragraph_seq = pad_sequences(input_paragraph_seq, self.max_encoder_paragraph_seq_length)
        input_question_seq = pad_sequences(input_question_seq, self.max_encoder_question_seq_length)
        states_value = self.encoder_model.predict([input_paragraph_seq, input_question_seq])
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

    def test_run(self, ds, index):
        paragraph, question, actual_answer = ds.get_data(index)
        predicted_answer = self.reply(paragraph, question)
        # print({'context': paragraph, 'question': question})
        print({'predict': predicted_answer, 'actual': actual_answer})

