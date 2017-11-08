import json

DATA_PATH = 'data/SQuAD/train-v1.1.json'

with open(DATA_PATH) as file:
    json_data = json.load(file)

    for instance in json_data['data']:
        for paragraph in instance['paragraphs']:
            context = paragraph['context']
            print('context: ', context)
            qas = paragraph['qas']
            for qas_instance in qas:
                question = qas_instance['question']
                print('question: ', question)
                answers = qas_instance['answers']
                for answer in answers:
                    print('answer: ', answer['text'])