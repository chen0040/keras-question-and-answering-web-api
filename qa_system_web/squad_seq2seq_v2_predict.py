from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense, Embedding, Dropout, add, RepeatVector
from keras.preprocessing.sequence import pad_sequences
from qa_system_web.squad_dataset import SquADDataSet
import qa_system_web.text_utils as text_utils
import numpy as np
import nltk

HIDDEN_UNITS = 256
EMBED_HIDDEN_UNITS = 100
MODEL_DIR_PATH = '../qa_system_train/models/SQuAD'


class SQuADSeq2SeqModel(object):
    model = None
    encoder_model = None
    decoder_model = None
    input_paragraph_word2idx = None
    input_paragraph_idx2word = None
    input_question_word2idx = None
    input_question_idx2word = None 
    target_word2idx = None
    target_idx2word = None
    max_encoder_paragraph_seq_length = None
    max_encoder_question_seq_length = None 
    max_decoder_seq_length = None
    num_encoder_paragraph_tokens = None
    num_encoder_question_tokens = None
    num_decoder_tokens = None

    def __init__(self):
        name = 'seq2seq-v2'
        self.input_paragraph_word2idx = np.load(MODEL_DIR_PATH + '/' + name + '-input-paragraph-word2idx.npy').item()
        self.input_paragraph_idx2word = np.load(MODEL_DIR_PATH + '/' + name + '-input-paragraph-idx2word.npy').item()
        self.input_question_word2idx = np.load(MODEL_DIR_PATH + '/' + name + '-input-question-word2idx.npy').item()
        self.input_question_idx2word = np.load(MODEL_DIR_PATH + '/' + name + '-input-question-idx2word.npy').item()
        self.target_word2idx = np.load(MODEL_DIR_PATH + '/' + name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(MODEL_DIR_PATH + '/' + name + '-target-idx2word.npy').item()
        context = np.load(MODEL_DIR_PATH + '/' + name + '-config.npy').item()
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

        context_inputs = Input(shape=(None,), name='context_inputs')
        encoded_context = Embedding(input_dim=self.num_encoder_paragraph_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                    input_length=self.max_encoder_paragraph_seq_length,
                                    name='context_embedding')(context_inputs)
        encoded_context = Dropout(0.3)(encoded_context)

        question_inputs = Input(shape=(None,), name='question_inputs')
        encoded_question = Embedding(input_dim=self.num_encoder_question_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                     input_length=self.max_encoder_question_seq_length,
                                     name='question_embedding')(question_inputs)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = LSTM(units=EMBED_HIDDEN_UNITS, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(self.max_encoder_paragraph_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])

        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(merged)
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

    def test_run(self, ds, index):
        paragraph, question, actual_answer = ds.get_data(index)
        predicted_answer = self.reply(paragraph, question)
        # print({'context': paragraph, 'question': question})
        print({'predict': predicted_answer, 'actual': actual_answer})


def main():
    model = SQuADSeq2SeqModel()
    dataset = SquADDataSet()
    for i in range(20):
        model.test_run(dataset, i * 10)

if __name__ == '__main__':
    main()
