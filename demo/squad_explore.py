from keras_question_and_answering_system.library.utility.squad import SquADDataSet


def main():
    data_set = SquADDataSet(data_path='./data/SQuAD/train-v1.1.json')
    print('size: ', data_set.size())


if __name__ == '__main__':
    main()
