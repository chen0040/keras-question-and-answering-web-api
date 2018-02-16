from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout, add, RepeatVector
from keras.preprocessing.sequence import pad_sequences

from keras_question_and_answering_system.library.utility import text_utils
from keras_question_and_answering_system.library.utility.qa_data_utils import Seq2SeqTripleSamples
import numpy as np
import nltk
import os


def generate_batch(ds, input_data, output_data, batch_size):
    num_batches = len(input_data) // batch_size
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            encoder_input_paragraph_data_batch = []
            encoder_input_question_data_batch = []
            for input_paragraph_data, input_question_data in input_data[start:end]:
                encoder_input_paragraph_data_batch.append(input_paragraph_data)
                encoder_input_question_data_batch.append(input_question_data)
            encoder_input_paragraph_data_batch = pad_sequences(encoder_input_paragraph_data_batch,
                                                               ds.input_paragraph_max_seq_length)
            encoder_input_question_data_batch = pad_sequences(encoder_input_question_data_batch,
                                                              ds.input_question_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(batch_size, ds.target_max_seq_length, ds.num_target_tokens))
            decoder_input_data_batch = np.zeros(shape=(batch_size, ds.target_max_seq_length, ds.num_target_tokens))
            for lineIdx, target_wid_list in enumerate(output_data[start:end]):
                for idx, wid in enumerate(target_wid_list):
                    if wid == 0:  # UNKNOWN
                        continue
                    decoder_input_data_batch[lineIdx, idx, wid] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
            yield [encoder_input_paragraph_data_batch, encoder_input_question_data_batch,
                   decoder_input_data_batch], decoder_target_data_batch


class Seq2SeqV2QA(object):
    model_name = 'seq2seq-qa-v2'

    def __init__(self):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.input_paragraph_word2idx = None
        self.input_paragraph_idx2word = None
        self.input_question_word2idx = None
        self.input_question_idx2word = None
        self.target_word2idx = None
        self.target_idx2word = None
        self.max_encoder_paragraph_seq_length = None
        self.max_encoder_question_seq_length = None
        self.max_decoder_seq_length = None
        self.num_encoder_paragraph_tokens = None
        self.num_encoder_question_tokens = None
        self.num_decoder_tokens = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqV2QA.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqV2QA.model_name + '-weights.h5')

    def load_model(self, model_dir_path):
        self.input_paragraph_word2idx = np.load(
            model_dir_path + '/' + self.model_name + '-input-paragraph-word2idx.npy').item()
        self.input_paragraph_idx2word = np.load(
            model_dir_path + '/' + self.model_name + '-input-paragraph-idx2word.npy').item()
        self.input_question_word2idx = np.load(
            model_dir_path + '/' + self.model_name + '-input-question-word2idx.npy').item()
        self.input_question_idx2word = np.load(
            model_dir_path + '/' + self.model_name + '-input-question-idx2word.npy').item()
        self.target_word2idx = np.load(model_dir_path + '/' + self.model_name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(model_dir_path + '/' + self.model_name + '-target-idx2word.npy').item()
        context = np.load(model_dir_path + '/' + self.model_name + '-config.npy').item()
        self.max_encoder_paragraph_seq_length = context['input_paragraph_max_seq_length']
        self.max_encoder_question_seq_length = context['input_question_max_seq_length']
        self.max_decoder_seq_length = context['target_max_seq_length']
        self.num_encoder_paragraph_tokens = context['num_input_paragraph_tokens']
        self.num_encoder_question_tokens = context['num_input_question_tokens']
        self.num_decoder_tokens = context['num_target_tokens']

        print(self.max_encoder_paragraph_seq_length)
        print(self.max_encoder_question_seq_length)
        print(self.max_decoder_seq_length)
        print(self.num_encoder_paragraph_tokens)
        print(self.num_encoder_question_tokens)
        print(self.num_decoder_tokens)

        self.create_model()
        weight_file_path = self.get_weight_file_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_model(self):

        hidden_units = 256
        embed_hidden_units = 100

        context_inputs = Input(shape=(None,), name='context_inputs')
        encoded_context = Embedding(input_dim=self.num_encoder_paragraph_tokens, output_dim=embed_hidden_units,
                                    input_length=self.max_encoder_paragraph_seq_length,
                                    name='context_embedding')(context_inputs)
        encoded_context = Dropout(0.3)(encoded_context)

        question_inputs = Input(shape=(None,), name='question_inputs')
        encoded_question = Embedding(input_dim=self.num_encoder_question_tokens, output_dim=embed_hidden_units,
                                     input_length=self.max_encoder_question_seq_length,
                                     name='question_embedding')(question_inputs)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = LSTM(units=embed_hidden_units, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(self.max_encoder_paragraph_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])

        encoder_lstm = LSTM(units=hidden_units, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(merged)
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
        input_paragraph_wid_list = []
        input_question_wid_list = []
        input_paragraph_text = paragraph.lower()
        input_question_text = question.lower()
        for word in nltk.word_tokenize(input_paragraph_text):
            if not text_utils.in_white_list(word):
                continue
            idx = 1  # default [UNK]
            if word in self.input_paragraph_word2idx:
                idx = self.input_paragraph_word2idx[word]
            input_paragraph_wid_list.append(idx)
        for word in nltk.word_tokenize(input_question_text):
            if not text_utils.in_white_list(word):
                continue
            idx = 1  # default [UNK]
            if word in self.input_question_word2idx:
                idx = self.input_question_word2idx[word]
            input_question_wid_list.append(idx)
        input_paragraph_seq.append(input_paragraph_wid_list)
        input_question_seq.append(input_question_wid_list)
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

    def test_run(self, ds, index=None):
        if index is None:
            index = 0
        paragraph, question, actual_answer = ds.get_data(index)
        predicted_answer = self.reply(paragraph, question)
        # print({'context': paragraph, 'question': question})
        print({'predict': predicted_answer, 'actual': actual_answer})

    def fit(self, data_set, model_dir_path, epochs=None, batch_size=None, test_size=None, random_state=None,
            save_best_only=False, max_input_vocab_size=None, max_target_vocab_size=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 100
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if max_input_vocab_size is None:
            max_input_vocab_size = 5000
        if max_target_vocab_size is None:
            max_target_vocab_size = 5000

        data_set_seq2seq = Seq2SeqTripleSamples(data_set, max_input_vocab_size=max_input_vocab_size,
                                                max_target_vocab_size=max_target_vocab_size)
        data_set_seq2seq.save(model_dir_path, 'qa-v2')

        x_train, x_test, y_train, y_test = data_set_seq2seq.split(test_size=test_size, random_state=random_state)

        print(len(x_train))
        print(len(x_test))

        self.max_encoder_question_seq_length = data_set_seq2seq.input_question_max_seq_length
        self.max_encoder_paragraph_seq_length = data_set_seq2seq.input_paragraph_max_seq_length
        self.max_decoder_seq_length = data_set_seq2seq.target_max_seq_length
        self.num_encoder_question_tokens = data_set_seq2seq.num_input_question_tokens
        self.num_encoder_paragraph_tokens = data_set_seq2seq.num_input_paragraph_tokens
        self.num_decoder_tokens = data_set_seq2seq.num_target_tokens

        weight_file_path = self.get_weight_file_path(model_dir_path)
        architecture_file_path = self.get_architecture_file_path(model_dir_path)

        self.create_model()

        with open(architecture_file_path, 'w') as f:
            f.write(self.model.to_json())

        train_gen = generate_batch(data_set_seq2seq, x_train, y_train, batch_size)
        test_gen = generate_batch(data_set_seq2seq, x_test, y_test, batch_size)

        train_num_batches = len(x_train) // batch_size
        test_num_batches = len(x_test) // batch_size

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=save_best_only)

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])

        self.model.save_weights(weight_file_path)

        np.save(os.path.join(model_dir_path, Seq2SeqV2QA.model_name + '-history.npy'), history.history)

        return history
