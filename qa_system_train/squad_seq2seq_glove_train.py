from keras.models import Model
from keras.layers import add, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import json
import zipfile
import urllib.request
import sys

np.random.seed(42)

BATCH_SIZE = 64
NUM_EPOCHS = 100
EMBED_HIDDEN_UNITS = 64
HIDDEN_UNITS = 256
MAX_DATA_COUNT = 5000
MAX_CONTEXT_SEQ_LENGTH = 300
MAX_QUESTION_SEQ_LENGTH = 60
MAX_TARGET_SEQ_LENGTH = 50
MAX_VOCAB_SIZE = 1000
MODEL_DIR = 'models/SQuAD'
DATA_PATH = 'data/SQuAD/train-v1.1.json'
GLOVE_EMBEDDING_SIZE = 100
GLOVE_MODEL = "very_large_data/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WEIGHT_FILE_PATH = MODEL_DIR + '/glove-weights.h5'

WHITE_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890,.?'


def in_white_list(_word):
    for char in _word:
        if char in WHITE_LIST:
            return True

    return False


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove():
    if not os.path.exists(GLOVE_MODEL):

        glove_zip = 'very_large_data/glove.6B.zip'

        if not os.path.exists('very_large_data'):
            os.makedirs('very_large_data')

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('very_large_data')
        zip_ref.close()


def load_glove():
    download_glove()
    _word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


word2em = load_glove()

ans_counter = Counter()

context_max_seq_length = 0
question_max_seq_length = 0
ans_max_seq_length = 0


data = []
with open(DATA_PATH) as file:
    json_data = json.load(file)

    for instance in json_data['data']:
        for paragraph in instance['paragraphs']:
            context = [w.lower() for w in nltk.word_tokenize(paragraph['context']) if in_white_list(w)]
            if len(context) > MAX_CONTEXT_SEQ_LENGTH:
                continue
            qas = paragraph['qas']
            for qas_instance in qas:
                question = [w.lower() for w in nltk.word_tokenize(qas_instance['question']) if in_white_list(w)]
                if len(question) > MAX_QUESTION_SEQ_LENGTH:
                    continue
                answers = qas_instance['answers']
                for answer in answers:
                    ans = [w.lower() for w in nltk.word_tokenize(answer['text']) if in_white_list(w)]
                    ans = ['start'] + ans + ['end']
                    if len(ans) > MAX_TARGET_SEQ_LENGTH:
                        continue
                    if len(data) < MAX_DATA_COUNT:
                        data.append((context, question, ans))
                        context_max_seq_length = max(context_max_seq_length, len(context))
                        question_max_seq_length = max(question_max_seq_length, len(question))
                        ans_max_seq_length = max(ans_max_seq_length, len(ans))
        if len(data) >= MAX_DATA_COUNT:
            break

ans_word2idx = dict()

ans_word2idx['UNK'] = 0

ans_idx2word = dict([(idx, word) for word, idx in ans_word2idx.items()])

num_decoder_tokens = len(ans_idx2word)

np.save(MODEL_DIR + '/glove-ans-word2idx.npy', ans_word2idx)
np.save(MODEL_DIR + '/glove-ans-idx2word.npy', ans_idx2word)

config = dict()
config['num_decoder_tokens'] = num_decoder_tokens
config['context_max_seq_length'] = context_max_seq_length
config['question_max_seq_length'] = question_max_seq_length
config['ans_max_seq_length'] = ans_max_seq_length

print(config)
np.save(MODEL_DIR + '/glove-context.npy', config)

context_unknown = np.random.rand(GLOVE_EMBEDDING_SIZE)
question_unknown = np.random.rand(GLOVE_EMBEDDING_SIZE)

def generate_batch(source):
    num_batches = len(source) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            source_batch = source[start:end]
            context_data_batch = []
            question_data_batch = []
            ans_data_batch = []
            ans_target_data_batch = []
            for (_context, _question, _ans) in source_batch:
                context_emb = []
                question_emb = []
                ans_emb = []
                ans_wids = []
                for w in _context:
                    emb = context_unknown
                    if w in word2em:
                        emb = word2em[w]
                    context_emb.append(emb)
                for w in _question:
                    emb = 1
                    if w in word2em:
                        emb = word2em[w]
                    question_emb.append(emb)
                for w in _ans:
                    wid = 0
                    if w in ans_word2idx:
                        wid = ans_word2idx[w]
                    ans_wids.append(wid)
                    if w in word2em:
                        emb = word2em[w]
                        ans_emb.append(emb)
                context_data_batch.append(context_emb)
                question_data_batch.append(question_emb)
                ans_data_batch.append(ans_emb)
                ans_target_data_batch.append(ans_wids)
            context_data_batch = pad_sequences(context_data_batch, context_max_seq_length)
            question_data_batch = pad_sequences(question_data_batch, question_max_seq_length)
            ans_data_batch = pad_sequences(ans_data_batch, ans_max_seq_length)

            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, ans_max_seq_length, num_decoder_tokens))
            for lineIdx, ans_wids in enumerate(ans_target_data_batch):
                for idx, w2idx in enumerate(ans_wids):
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [context_data_batch, question_data_batch, ans_data_batch], decoder_target_data_batch


context_inputs = Input(shape=(None,GLOVE_EMBEDDING_SIZE), name='context_inputs')
encoded_context = Dropout(0.3)(context_inputs)

question_inputs = Input(shape=(None,GLOVE_EMBEDDING_SIZE), name='question_inputs')
encoded_question = Dropout(0.3)(question_inputs)
encoded_question = LSTM(units=EMBED_HIDDEN_UNITS, name='question_lstm')(encoded_question)
encoded_question = RepeatVector(context_max_seq_length)(encoded_question)

merged = add([encoded_context, encoded_question])
encoder_outputs, encoder_state_h, encoder_state_c = LSTM(units=HIDDEN_UNITS,
                                                         name='encoder_lstm', return_state=True)(merged)

encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([context_inputs, question_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

json = model.to_json()
open(MODEL_DIR + '/glove-architecture.json', 'w').write(json)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, data, test_size=0.2, random_state=42)

print(len(Xtrain))
print(len(Xtest))

train_gen = generate_batch(Xtrain)
test_gen = generate_batch(Xtest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])


model.save_weights(WEIGHT_FILE_PATH)
