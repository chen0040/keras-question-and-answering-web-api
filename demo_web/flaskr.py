from flask import Flask, request, redirect, render_template, flash, abort, jsonify
from keras_question_and_answering_system.library.seq2seq import Seq2SeqQA
from keras_question_and_answering_system.library.utility.squad import SquADDataSet

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

seq2seq = Seq2SeqQA()
data_set = SquADDataSet(data_path=None)
qa_list = list()


@app.route('/')
def home():
    return render_template('home.html', qa_list=qa_list)


@app.route('/about')
def about():
    return 'About Us'


@app.route('/qa', methods=['POST', 'GET'])
def qa():
    if request.method == 'POST':
        if 'context' not in request.form and 'question' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['context'] == '' or request.form['question'] == '':
            flash('No context or question')
            redirect(request.url)
        else:
            question_context = request.form['context']
            question = request.form['question']
            ans = seq2seq.reply(question_context, question)
            return render_template('qa.html', question_context=question_context,
                                   question=question, answer=ans)
    elif request.method == 'GET':
        context_index = int(request.args.get('context_index'))
        question_index = request.args.get('question_index')
        qa_item = qa_list[context_index]
        qa_pair = qa_item[question_index]
        question = qa_pair[0]
        context = qa_item[0]
    return render_template('qa.html', question_context=context,
                           question=question, answer='')


@app.route('/qa_api', methods=['POST', 'GET'])
def qa_api():
    if request.method == 'POST':
        if not request.json or 'context' not in request.json or 'question' not in request.json or 'agent' not in request.json:
            abort(400)
        context = request.json['context']
        question = request.json['question']
        agent = request.json['agent']
    else:
        context = request.args.get('context')
        question = request.args.get('question')
        agent = request.args.get('agent')

    target_text = context
    if agent == 'seq2seq':
        target_text = seq2seq.reply(paragraph=context, question=question)
    elif agent == 'seq2seq_v2':
        target_text = seq2seq.reply(paragraph=context, question=question)
    elif agent == 'seq2seq_glove':
        target_text = seq2seq.reply(paragraph=context, question=question)
    elif agent == 'seq2seq_v2_glove':
        target_text = seq2seq.reply(paragraph=context, question=question)
    return jsonify({
        'context': context,
        'question': question,
        'agent': agent,
        'answer': target_text
    })


def main():
    data_set.load_model(data_path='../demo/data/SQuAD/train-v1.1.json')
    qa_list = data_set.to_tree()

    seq2seq.load_model(model_dir_path='../demo/models')
    seq2seq.test_run(data_set)
    app.run(debug=True)


if __name__ == '__main__':
    main()
