import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from experiments.glove_loader import Glove
from experiments.squad_dataset import SquADDataSet, SQuADSeq2SeqEmbTupleSamples

np.random.seed(42)

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
DATA_SET_NAME = 'SQuAD'
MODEL_DIR_PATH = 'models/' + DATA_SET_NAME
WEIGHT_FILE_PATH = MODEL_DIR_PATH + '/seq2seq-glove-weights.h5'
ARCHITECTURE_FILE_PATH = MODEL_DIR_PATH + '/seq2seq-glove-architecture.json'

glove = Glove()

dataset = SquADDataSet(10000)
dataset_seq2seq = SQuADSeq2SeqEmbTupleSamples(dataset, glove.word2em, glove.GLOVE_EMBEDDING_SIZE)
dataset_seq2seq.save(MODEL_DIR_PATH, 'glove')


def generate_batch(ds, input_word2em_data, output_data):
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], ds.input_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, ds.target_max_seq_length, ds.num_target_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, ds.target_max_seq_length, ds.num_target_tokens))
            for lineIdx, target_wid_list in enumerate(output_data[start:end]):
                for idx, wid in enumerate(target_wid_list):
                    if wid == 0:  # UNKNOWN
                        continue
                    decoder_input_data_batch[lineIdx, idx, wid] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


encoder_inputs = Input(shape=(None, glove.GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, dataset_seq2seq.num_target_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=dataset_seq2seq.num_target_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

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
