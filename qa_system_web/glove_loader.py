import os
import sys
import zipfile
import urllib.request
import numpy as np

GLOVE_EMBEDDING_SIZE = 100
GLOVE_MODEL = "../qa_system_train/very_large_data/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"


def download_glove():
    if not os.path.exists(GLOVE_MODEL):

        glove_zip = '../qa_system_train/very_large_data/glove.6B.zip'

        if not os.path.exists('qa_system_train/very_large_data'):
            os.makedirs('../qa_system_train/very_large_data')

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('../qa_system_train/very_large_data')
        zip_ref.close()


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


def load_glove():
    download_glove()
    word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        word2em[word] = embeds
    file.close()
    return word2em
