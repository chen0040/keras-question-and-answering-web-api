# keras-question-and-answering-web-api

Question answering system developed using seq2seq and memory network model in Keras

This project explores 

* Two different ways to merge the paragraph context and the question as the input of the encoder
* Two different encoding for the input of the encoder (one hot encoding and GloVe word2vec encoding)

The implementation of these models can be found in
 [keras_question_and_answering_system/library](keras_question_and_answering_system/library)
 
The implemented models include:

* [seq2seq.py](keras_question_and_answering_system/library/seq2seq.py): one-hot encoding input that is paragraph_context + ' Q ' + question
* [seq2seq_v2.py](keras_question_and_answering_system/library/seq2seq_v2.py): one-hot encoding input that is add(paragraph_context, RepeatVector(LSTM(question))) 
* [seq2seq_glove.py](keras_question_and_answering_system/library/seq2seq_glove.py)): GloVe encoding input that is paragraph_context + ' Q ' + question
* [seq2seq_v2_glove.py](keras_question_and_answering_system/library/seq2seq_v2_glove.py): GloVe encoding input that is add(paragraph_context, RepeatVector(LSTM(question)))

 
The demo codes of using these models with [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) data set
can be found in [demo](demo)

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

The question answering models are trained using [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) data set and are available in the 
"qa_system_train/models" directory. During runtime, the flask app will load these trained models to perform the 
qa-reply

### Training 

While the trained models are already included in the [demo/models](demo/models) folder in the project, the training is
not completed due to my limited time (the models are only from the check point model by running only a few epochs). Therefore, if you like to tune the parameters of the seq2seq and complete the training of the models, you can use the 
following command to run the training:

```bash
cd demo
python squad_seq2seq_train.py
```

The above commands will train one-hot encoding seq2seq model using SQuAD and store the trained model
in [demo/models/SQuAD/seq2seq-**"](demo/models)

If you like to train other models, you can use the same command above on another train python scripts:

* [squad_seq2seq_v2_train.py](demo/squad_seq2seq_v2_train.py): one hot encoding seq2seq but different from squad_seq2seq_train.py in that the paragraph context and the question are added after the LSTM + RepeatVector layer
* [squad_seq2seq_glove_train.py](demo/squad_seq2seq_glove_train.py): train on SQuAD on word-level (GloVe word2vec encoding) with input = paragraph_context + ' Q ' + question
* [squad_seq2seq_v2_glove_train.py](demo/squad_seq2seq_v2_glove_train.py): train on SQuAD on word-level (GloVe word2vec encoding) but different from squad_seq2seq_glove_train.py in that the paragraph context and the question are added after the LSTM + RepeatVector layer

### Predict Answers

Once the trained models are generated. the predictors in the [demo/models](demo/models) can be used to load these model and predict answer based on the paragraph context and the question:

* [squad_seq2seq_predict.py](demo/squad_seq2seq_predict.py)
* [squad_seq2seq_v2_predict.py](demo/squad_seq2seq_v2_predict.py)
* [squad_seq2seq_glove_predict.py](demo/squad_seq2seq_glove_predict.py)
* [squad_seq2seq_v2_glove_predict.py](demo/squad_seq2seq_v2_glove_predict.py)

### Running Web Api Server (WIP)

Goto [demo_web](demo_web) directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained seq2seq models:

* Word-level seq2seq models (One Hot Encoding)
* Word-level seq2seq models (GloVe Encoding)

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 

