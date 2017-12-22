from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding, Dropout, RepeatVector, add
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from qa_system_train.squad_dataset import SquADDataSet, SQuADSeq2SeqTripleSamples
import numpy as np

np.random.seed(42)

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
EMBED_HIDDEN_UNITS = 100
MODEL_DIR_PATH = 'models/SQuAD'
WEIGHT_FILE_PATH = MODEL_DIR_PATH + '/seq2seq-v2-weights.h5'
ARCHITECTURE_FILE_PATH = MODEL_DIR_PATH + '/seq2seq-v2-architecture.json'


dataset = SquADDataSet(10000)
dataset_seq2seq = SQuADSeq2SeqTripleSamples(dataset)
dataset_seq2seq.save(MODEL_DIR_PATH, 'v2')


def generate_batch(ds, input_data, target_data):
    num_batches = len(input_data) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_data[start:end], ds.input_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, ds.target_max_seq_length,
                                                        ds.num_target_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, ds.target_max_seq_length,
                                                       ds.num_target_tokens))
            for lineIdx, target_wid_list in enumerate(target_data[start:end]):
                for idx, wid in enumerate(target_wid_list):
                    if wid == 0:  # UNKNOWN
                        continue
                    decoder_input_data_batch[lineIdx, idx, wid] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


context_inputs = Input(shape=(None,), name='context_inputs')
encoded_context = Embedding(input_dim=dataset_seq2seq.num_input_paragraph_tokens, output_dim=EMBED_HIDDEN_UNITS,
                            input_length=dataset_seq2seq.input_paragraph_max_seq_length, name='context_embedding')(context_inputs)
encoded_context = Dropout(0.3)(encoded_context)

question_inputs = Input(shape=(None,), name='question_inputs')
encoded_question = Embedding(input_dim=dataset_seq2seq.num_input_question_tokens, output_dim=EMBED_HIDDEN_UNITS,
                             input_length=dataset_seq2seq.input_question_max_seq_length, name='question_embedding')(question_inputs)
encoded_question = Dropout(0.3)(encoded_question)
encoded_question = LSTM(units=EMBED_HIDDEN_UNITS, name='question_lstm')(encoded_question)
encoded_question = RepeatVector(dataset_seq2seq.input_paragraph_max_seq_length)(encoded_question)

merged = add([encoded_context, encoded_question])

encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(merged)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, dataset_seq2seq.num_target_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=dataset_seq2seq.num_target_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

json = model.to_json()
open(ARCHITECTURE_FILE_PATH, 'w').write(json)

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
