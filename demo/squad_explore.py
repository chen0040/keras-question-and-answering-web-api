from keras_question_and_answering_system.library.utility.squad import SquADDataSet


def main():
    data_set = SquADDataSet(data_path='./data/SQuAD/train-v1.1.json')
    print('size: ', data_set.size())
    for index in range(20):
        context, question, answer = data_set.get_data(index)
        print('#' + str(index) + ' context:', context)
        print('#' + str(index) + ' question:', question)
        print('#' + str(index) + ' answer:', answer)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
    main()
