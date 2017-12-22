from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for
from qa_system_web.squad_seq2seq_predict import SQuADSeq2SeqModel

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

squad_s2s_qa = SQuADSeq2SeqModel()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/squad_qa', methods=['POST', 'GET'])
def squad_qa():
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
            ans = squad_s2s_qa.reply(question_context, question)
            return render_template('squad_qa.html', question_context=question_context,
                                   question=question, answer=ans)
    return render_template('squad_qa.html')


def main():
    squad_s2s_qa.test_run()
    app.run(debug=True)


if __name__ == '__main__':
    main()
