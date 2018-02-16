import urllib.request
import os
import sys
import zipfile
import numpy as np


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


def download_glove(data_dir_path, embedding_size=None):
    if embedding_size is None:
        embedding_size = 100
    file_path = data_dir_path + "/glove.6B." + str(embedding_size) + "d.txt"
    if not os.path.exists(file_path):

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


def load_glove(data_dir_path, embedding_size=None):
    if embedding_size is None:
        embedding_size = 100
    file_path = data_dir_path + "/glove.6B." + str(embedding_size) + "d.txt"
    download_glove(data_dir_path, embedding_size)
    _word2em = {}
    file = open(file_path, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


class GloveModel(object):

    def __init__(self):
        self.word2em = None
        self.embedding_size = 100

    def load_model(self, data_dir_path, embedding_size=None):
        if embedding_size is None:
            embedding_size = 100
        self.embedding_size = embedding_size
        self.word2em = load_glove(data_dir_path, embedding_size)

    def encode_word(self, word):
        if word in self.word2em:
            return self.word2em[word]
        else:
            return np.zeros(shape=self.embedding_size)
