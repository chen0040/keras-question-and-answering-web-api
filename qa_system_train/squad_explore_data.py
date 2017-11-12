import json
import nltk

DATA_PATH = 'data/SQuAD/train-v1.1.json'
data = []

max_context_length = 0
max_question_length = 0
max_answer_length = 0

with open(DATA_PATH) as file:
    json_data = json.load(file)

    for instance in json_data['data']:
        for paragraph in instance['paragraphs']:
            context = paragraph['context']
            max_context_length = max(max_context_length, len([w for w in nltk.word_tokenize(context)]))
            qas = paragraph['qas']
            for qas_instance in qas:
                question = qas_instance['question']
                answers = qas_instance['answers']
                max_question_length = max(max_question_length, len([w for w in nltk.word_tokenize(question)]))
                for answer in answers:
                    max_answer_length = max(max_answer_length, len([w for w in nltk.word_tokenize(answer['text'])]))
                    data.append((context, question, answer['text']))

print('max context length: ', max_context_length)
print('max question length: ', max_question_length)
print('max answer length: ', max_answer_length)

print('data count: ', len(data))
