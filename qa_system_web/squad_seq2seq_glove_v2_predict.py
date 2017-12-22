import nltk
import numpy as np
from keras.layers import Input, LSTM, Dense, Dropout, add, RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from qa_system_web.text_utils import in_white_list
import qa_system_web.glove_loader as glove_loader

from qa_system_web.squad_dataset import SquADDataSet

HIDDEN_UNITS = 256
DATA_SET_NAME = 'SQuAD'
MODEL_DIR_PATH = '../qa_system_train/models/' + DATA_SET_NAME


class SQuADSeq2SeqGloveV2Model(object):
    model = None
    encoder_model = None
    decoder_model = None
    target_word2idx = None
    target_idx2word = None
    max_decoder_seq_length = None
    max_encoder_paragraph_seq_length = None
    max_encoder_question_seq_length = None
    num_decoder_tokens = None
    word2em = None

    def __init__(self):
        name = 'seq2seq-glove-v2'
        self.word2em = glove_loader.load_glove()

        self.target_word2idx = np.load(
            MODEL_DIR_PATH + '/' + name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(
            MODEL_DIR_PATH + '/' + name + '-target-idx2word.npy').item()
        context = np.load(MODEL_DIR_PATH + '/' + name + '-config.npy').item()
        self.max_encoder_paragraph_seq_length = context['input_paragraph_max_seq_length']
        self.max_encoder_question_seq_length = context['input_question_max_seq_length']
        self.max_decoder_seq_length = context['target_max_seq_length']
        self.num_decoder_tokens = context['num_target_tokens']

        context_inputs = Input(shape=(None, glove_loader.GLOVE_EMBEDDING_SIZE), name='context_inputs')
        encoded_context = Dropout(0.3)(context_inputs)

        question_inputs = Input(shape=(None, glove_loader.GLOVE_EMBEDDING_SIZE), name='question_inputs')
        encoded_question = Dropout(0.3)(question_inputs)
        encoded_question = LSTM(units=glove_loader.GLOVE_EMBEDDING_SIZE, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(self.max_encoder_paragraph_seq_length)(encoded_question)

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

        self.model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)

        # model_json = open(MODEL_DIR_PATH + '/' + name + '-architecture.json', 'r').read()
        # self.model = model_from_json(model_json)
        self.model.load_weights(MODEL_DIR_PATH + '/' + name + '-weights.h5')
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.encoder_model = Model([context_inputs, question_inputs], encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
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
            emb = np.zeros(shape=glove_loader.GLOVE_EMBEDDING_SIZE)
            if word in self.word2em:
                emb = self.word2em[word]
            input_paragraph_emb.append(emb)
        for word in nltk.word_tokenize(input_question_text):
            if not in_white_list(word):
                continue
            emb = np.zeros(shape=glove_loader.GLOVE_EMBEDDING_SIZE)
            if word in self.word2em:
                emb = self.word2em[word]
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


def main():
    model = SQuADSeq2SeqGloveV2Model()
    dataset = SquADDataSet()
    for i in range(20):
        model.test_run(dataset, i * 10)


if __name__ == '__main__':
    main()
