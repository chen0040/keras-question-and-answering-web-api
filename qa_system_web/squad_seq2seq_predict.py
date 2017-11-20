from keras.models import model_from_json, Model
import numpy as np
from keras.layers import LSTM, Input, Embedding, Dropout, RepeatVector, add, Dense
from keras.preprocessing.sequence import pad_sequences
import nltk

MODEL_DIR = '../qa_system_train/models/SQuaD'
EMBED_HIDDEN_UNITS = 64
HIDDEN_UNITS = 256


class SquadSeq2SeqQA(object):
    model = None
    word2idx = None
    idx2word = None
    context = None
    encoder_model = None
    decoder_model = None

    def __init__(self):
        self.context = np.load(MODEL_DIR + '/word-context.npy').items()
        num_context_tokens = self.context['num_context_tokens']
        context_max_seq_length = self.context['context_max_seq_length']
        num_question_tokens = self.context['num_question_tokens']
        question_max_seq_length = self.context['question_max_seq_length']
        num_decoder_tokens = self.context['num_decoder_tokens']

        context_inputs = Input(shape=(None,), name='context_inputs')
        encoded_context = Embedding(input_dim=num_context_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                    input_length=context_max_seq_length, name='context_embedding')(context_inputs)
        encoded_context = Dropout(0.3)(encoded_context)

        question_inputs = Input(shape=(None,), name='question_inputs')
        encoded_question = Embedding(input_dim=num_question_tokens, output_dim=EMBED_HIDDEN_UNITS,
                                     input_length=question_max_seq_length, name='question_embedding')(question_inputs)
        encoded_question = Dropout(0.3)(encoded_question)
        encoded_question = LSTM(units=EMBED_HIDDEN_UNITS, name='question_lstm')(encoded_question)
        encoded_question = RepeatVector(context_max_seq_length)(encoded_question)

        merged = add([encoded_context, encoded_question])
        encoder_outputs, encoder_state_h, encoder_state_c = LSTM(units=HIDDEN_UNITS,
                                                                 name='encoder_lstm', return_state=True)(merged)

        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
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
        # to be implemented
        pass

    def test_run(self):
        question_context = 'i liked the Da Vinci Code a lot.'
        question = 'what do i like?'
        print(self.predict(question_context, question))

if __name__ == '__main__':
    app = SquadSeq2SeqQA()
    app.test_run()
