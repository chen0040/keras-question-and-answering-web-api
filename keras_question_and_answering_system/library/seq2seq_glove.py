from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
import os
from keras_question_and_answering_system.library.utility.glove_model import GloveModel
from keras_question_and_answering_system.library.utility.qa_embed_data_utils import SQuADSeq2SeqEmbTupleSamples
from keras_question_and_answering_system.library.utility.text_utils import in_white_list


def generate_batch(ds, input_word2em_data, output_data, batch_size):
    num_batches = len(input_word2em_data) // batch_size
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], ds.input_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(batch_size, ds.target_max_seq_length, ds.num_target_tokens))
            decoder_input_data_batch = np.zeros(shape=(batch_size, ds.target_max_seq_length, ds.num_target_tokens))
            for lineIdx, target_wid_list in enumerate(output_data[start:end]):
                for idx, wid in enumerate(target_wid_list):
                    if wid == 0:  # UNKNOWN
                        continue
                    decoder_input_data_batch[lineIdx, idx, wid] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


class Seq2SeqGloveQA(object):
    model_name = 'seq2seq-qa-glove'

    def __init__(self):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.target_word2idx = None
        self.target_idx2word = None
        self.max_decoder_seq_length = None
        self.max_encoder_seq_length = None
        self.num_decoder_tokens = None
        self.glove_model = GloveModel()

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqGloveQA.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqGloveQA.model_name + '-weights.h5')

    def load_glove_model(self, data_dir_path):
        self.glove_model.load_model(data_dir_path)

    def load_model(self, model_dir_path):
        self.target_word2idx = np.load(
            model_dir_path + '/' + Seq2SeqGloveQA.model_name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(
            model_dir_path + '/' + Seq2SeqGloveQA.model_name + '-target-idx2word.npy').item()
        context = np.load(model_dir_path + '/' + Seq2SeqGloveQA.model_name + '-config.npy').item()
        self.max_encoder_seq_length = context['input_max_seq_length']
        self.max_decoder_seq_length = context['target_max_seq_length']
        self.num_decoder_tokens = context['num_target_tokens']

        self.create_model()
        self.model.load_weights(Seq2SeqGloveQA.get_weight_file_path(model_dir_path))

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

    def fit(self, data_set, model_dir_path, epochs=None, batch_size=None, test_size=None, random_state=None,
            save_best_only=False, max_target_vocab_size=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 100
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if max_target_vocab_size is None:
            max_target_vocab_size = 5000

        data_set_seq2seq = SQuADSeq2SeqEmbTupleSamples(data_set, self.glove_model.word2em,
                                                       self.glove_model.embedding_size,
                                                       max_target_vocab_size=max_target_vocab_size)
        data_set_seq2seq.save(model_dir_path, 'qa-glove')

        x_train, x_test, y_train, y_test = data_set_seq2seq.split(test_size=test_size, random_state=random_state)

        print(len(x_train))
        print(len(x_test))

        self.max_encoder_seq_length = data_set_seq2seq.input_max_seq_length
        self.max_decoder_seq_length = data_set_seq2seq.target_max_seq_length
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

        np.save(os.path.join(model_dir_path, Seq2SeqGloveQA.model_name + '-history.npy'), history.history)

        return history

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



