import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.layers import add, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from experiments.glove_loader import Glove
from experiments.squad_dataset import SquADDataSet, SQuADSeq2SeqEmbTripleSamples

np.random.seed(42)

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
DATA_SET_NAME = 'SQuAD'
MODEL_DIR_PATH = 'models/' + DATA_SET_NAME
WEIGHT_FILE_PATH = MODEL_DIR_PATH + '/seq2seq-glove-v2-weights.h5'
ARCHITECTURE_FILE_PATH = MODEL_DIR_PATH + '/seq2seq-glove-v2-architecture.json'

glove = Glove()

dataset = SquADDataSet(10000)
dataset_seq2seq = SQuADSeq2SeqEmbTripleSamples(dataset, glove.word2em, glove.GLOVE_EMBEDDING_SIZE)
dataset_seq2seq.save(MODEL_DIR_PATH, 'glove-v2')


def generate_batch(ds, input_word2em_data, output_data):
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_paragraph_data_batch = []
            encoder_input_question_data_batch = []
            for input_paragraph_data, input_question_data in input_word2em_data[start:end]:
                encoder_input_paragraph_data_batch.append(input_paragraph_data)
                encoder_input_question_data_batch.append(input_question_data)
            encoder_input_paragraph_data_batch = pad_sequences(encoder_input_paragraph_data_batch,
                                                               ds.input_paragraph_max_seq_length)
            encoder_input_question_data_batch = pad_sequences(encoder_input_question_data_batch,
                                                              ds.input_question_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, ds.target_max_seq_length, ds.num_target_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, ds.target_max_seq_length, ds.num_target_tokens))
            for lineIdx, target_wid_list in enumerate(output_data[start:end]):
                for idx, wid in enumerate(target_wid_list):
                    if wid == 0:  # UNKNOWN
                        continue
                    decoder_input_data_batch[lineIdx, idx, wid] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
            yield [encoder_input_paragraph_data_batch, encoder_input_question_data_batch, decoder_input_data_batch], \
                  decoder_target_data_batch


context_inputs = Input(shape=(None, glove.GLOVE_EMBEDDING_SIZE), name='context_inputs')
encoded_context = Dropout(0.3)(context_inputs)

question_inputs = Input(shape=(None, glove.GLOVE_EMBEDDING_SIZE), name='question_inputs')
encoded_question = Dropout(0.3)(question_inputs)
encoded_question = LSTM(units=glove.GLOVE_EMBEDDING_SIZE, name='question_lstm')(encoded_question)
encoded_question = RepeatVector(dataset_seq2seq.input_paragraph_max_seq_length)(encoded_question)

merged = add([encoded_context, encoded_question])
encoder_outputs, encoder_state_h, encoder_state_c = LSTM(units=HIDDEN_UNITS,
                                                         name='encoder_lstm', return_state=True)(merged)

encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, dataset_seq2seq.num_target_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=dataset_seq2seq.num_target_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

open(ARCHITECTURE_FILE_PATH, 'w').write(model.to_json())

Xtrain, Xtest, Ytrain, Ytest = dataset_seq2seq.split(test_size=0.2, random_state=42)

print(len(Xtrain))
print(len(Xtest))

train_gen = generate_batch(dataset_seq2seq, Xtrain, Ytrain)
test_gen = generate_batch(dataset_seq2seq, Xtest, Ytest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)
