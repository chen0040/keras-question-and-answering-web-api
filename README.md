# keras-question-and-answering-web-api

Question answering system developed using seq2seq and memory network model in Keras

This project explores 

* Two different ways to merge the paragraph context and the question as the input of the encoder
* Two different encoding for the input of the encoder (one hot encoding and GloVe word2vec encoding)

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

The question answering models are trained using [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) data set and are available in the 
"qa_system_train/models" directory. During runtime, the flask app will load these trained models to perform the 
qa-reply

## Training 

While the trained models are already included in the "qa_system_train/models" folder in the project, the training is
not completed due to my limited time (the models are only from the check point model by running only a few epochs). Therefore, if you like to tune the parameters of the seq2seq and complete the training of the models, you can use the 
following command to run the training:

```bash
cd qa_system_train
python squad_seq2seq_train.py
```

The above commands will train one-hot encoding seq2seq model using SQuAD and store the trained model
in "qa_system_train/models/SQuAD/seq2seq-**"

If you like to train other models, you can use the same command above on another train python scripts:

* squad_seq2seq_v2_train.py: one hot encoding seq2seq but different from squad_seq2seq_train.py in that the paragraph context and the question are added after the LSTM + RepeatVector layer
* squad_seq2seq_glove_train.py: train on SQuAD on word-level (GloVe word2vec encoding) with input = paragraph_context + ' Q ' + question
* squad_seq2seq_glove_v2_train.py: train on SQuAD on word-level (GloVe word2vec encoding) but different from squad_seq2seq_glove_train.py in that the paragraph context and the question are added after the LSTM + RepeatVector layer

## Predict Answers

Once the trained models are generated. the predictors in the qa_system_web can be used to load these model and predict answer based on the paragraph context and the question:

* squad_seq2seq_predict.py: one-hot encoding input that is paragraph_context + ' Q ' + question
* squad_seq2seq_v2_predict.py: one-hot encoding input that is add(paragraph_context, RepeatVector(LSTM(question))) 
* squad_seq2seq_glove_predict.py: GloVe encoding input that is paragraph_context + ' Q ' + question
* squad_seq2seq_glove_v2_predict.py: GloVe encoding input that is add(paragraph_context, RepeatVector(LSTM(question)))

## Running Web Api Server (WIP)

Goto qa_system_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained seq2seq models:

* Word-level seq2seq models (One Hot Encoding)
* Word-level seq2seq models (GloVe Encoding)
