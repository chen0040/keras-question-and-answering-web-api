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
* [seq2seq_glove.py](keras_question_and_answering_system/library/seq2seq_glove.py): GloVe encoding input that is paragraph_context + ' Q ' + question
* [seq2seq_v2_glove.py](keras_question_and_answering_system/library/seq2seq_v2_glove.py): GloVe encoding input that is add(paragraph_context, RepeatVector(LSTM(question)))

 
The demo codes of using these models with [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) data set
can be found in [demo](demo)

# Usage

The question answering models are trained using [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) data set and are available in the 
[demo/models](demo/models) directory. 

### Project Dependencies

Run the following command to install the keras, tensorflow, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```



### Training 

The trained models are included in the [demo/models](demo/models) folder in the project.

The training was done on with 200 epochs and batch size of 64 on tensorflow-gpu.
 
If you like to tune the parameters of the seq2seq and complete the training of the models, you can use the 
following command to run the training:

```bash
cd demo
python squad_seq2seq_train.py
```

The above commands will train one-hot encoding seq2seq model using SQuAD and store the trained model
in [demo/models/seq2seq-**"](demo/models)

Below is the source codes of [squad_seq2seq_train.py](demo/squad_seq2seq_train.py) which trains Seq2SeqQA on the SQuAD:

```python
from keras_question_and_answering_system.library.seq2seq import Seq2SeqQA
from keras_question_and_answering_system.library.utility.squad import SquADDataSet
import numpy as np


def main():
    random_state = 42
    output_dir_path = './models'

    np.random.seed(random_state)
    data_set = SquADDataSet(data_path='./data/SQuAD/train-v1.1.json')

    qa = Seq2SeqQA()
    batch_size = 64
    epochs = 200
    history = qa.fit(data_set, model_dir_path=output_dir_path,
                     batch_size=batch_size, epochs=epochs,
                     random_state=random_state)


if __name__ == '__main__':
    main()
```

Note that [SquADDataSet](keras_question_and_answering_system/library/utility/squad.py) is a sub class
of [QADataSet](keras_question_and_answering_system/library/utility/qa_data_utils.py). Therefore it is 
possible to train Seq2SeqQA on any data set that implements [QADataSet](keras_question_and_answering_system/library/utility/qa_data_utils.py)

If you like to train other models, you can use the same command above on another train python scripts:

* [squad_seq2seq_v2_train.py](demo/squad_seq2seq_v2_train.py): one hot encoding seq2seq but different from squad_seq2seq_train.py in that the paragraph context and the question are added after the LSTM + RepeatVector layer
* [squad_seq2seq_glove_train.py](demo/squad_seq2seq_glove_train.py): train on SQuAD on word-level (GloVe word2vec encoding) with input = paragraph_context + ' Q ' + question
* [squad_seq2seq_v2_glove_train.py](demo/squad_seq2seq_v2_glove_train.py): train on SQuAD on word-level (GloVe word2vec encoding) but different from squad_seq2seq_glove_train.py in that the paragraph context and the question are added after the LSTM + RepeatVector layer

### Predict Answers

Once the trained models are generated. the predictors in the [demo/models](demo/models) can be used to load these model and predict answer based on the paragraph context and the question.

For example, to test the trained model of [Seq2SeqQA](keras_question_and_answering_system/library/seq2seq.py) on
 SQuAD data set, run the following command:

```bash
cd demo
python squad_seq2seq_predict.py
```

Below is the sample code of [squad_seq2seq_predict.py](demo/squad_seq2seq_predict.py) which tries to
predict the answer based on context provided and question being asked:

```python
from keras_question_and_answering_system.library.seq2seq import Seq2SeqQA
from keras_question_and_answering_system.library.utility.squad import SquADDataSet


def main():
    qa = Seq2SeqQA()
    qa.load_model(model_dir_path='./models')
    data_set = SquADDataSet(data_path='./data/SQuAD/train-v1.1.json')
    for i in range(20):
        index = i * 10
        paragraph, question, actual_answer = data_set.get_data(index)
        predicted_answer = qa.reply(paragraph, question)
        print('context: ', paragraph)
        print('question: ', question)
        print({'guessed_answer': predicted_answer, 'actual_answer': actual_answer})


if __name__ == '__main__':
    main()

```

Below show the console output from [squad_seq2seq_predict.py](demo/squad_seq2seq_predict.py):

```
context:  Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
question:  To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
{'guessed_answer': 'the 10,000', 'actual_answer': 'Saint Bernadette Soubirous'}
context:  The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.
question:  Where is the headquarters of the Congregation of the Holy Cross?
{'guessed_answer': 'moreau seminary', 'actual_answer': 'Rome'}
context:  All of Notre Dame's undergraduate students are a part of one of the five undergraduate colleges at the school or are in the First Year of Studies program. The First Year of Studies program was established in 1962 to guide incoming freshmen in their first year at the school before they have declared a major. Each student is given an academic advisor from the program who helps them to choose classes that give them exposure to any major in which they are interested. The program also includes a Learning Resource Center which provides time management, collaborative learning, and subject tutoring. This program has been recognized previously, by U.S. News & World Report, as outstanding.
question:  What entity provides help with the management of time for new students at Notre Dame?
{'guessed_answer': 'international scientific', 'actual_answer': 'Learning Resource Center'}
context:  The Joan B. Kroc Institute for International Peace Studies at the University of Notre Dame is dedicated to research, education and outreach on the causes of violent conflict and the conditions for sustainable peace. It offers PhD, Master's, and undergraduate degrees in peace studies. It was founded in 1986 through the donations of Joan B. Kroc, the widow of McDonald's owner Ray Kroc. The institute was inspired by the vision of the Rev. Theodore M. Hesburgh CSC, President Emeritus of the University of Notre Dame. The institute has contributed to international policy discussions about peace building practices.
question:  What is the title of Notre Dame's Theodore Hesburgh?
{'guessed_answer': 'president emeritus of the university of notre dame', 'actual_answer': 'President Emeritus of the University of Notre Dame'}
context:  Notre Dame is known for its competitive admissions, with the incoming class enrolling in fall 2015 admitting 3,577 from a pool of 18,156 (19.7%). The academic profile of the enrolled class continues to rate among the top 10 to 15 in the nation for national research universities. The university practices a non-restrictive early action policy that allows admitted students to consider admission to Notre Dame as well as any other colleges to which they were accepted. 1,400 of the 3,577 (39.1%) were admitted under the early action plan. Admitted students came from 1,311 high schools and the average student traveled more than 750 miles to Notre Dame, making it arguably the most representative university in the United States. While all entering students begin in the College of the First Year of Studies, 25% have indicated they plan to study in the liberal arts or social sciences, 24% in engineering, 24% in business, 24% in science, and 3% in architecture.
question:  What percentage of students were admitted to Notre Dame in fall 2015?
{'guessed_answer': '19.7', 'actual_answer': '19.7%'}
context:  Father Joseph Carrier, C.S.C. was Director of the Science Museum and the Library and Professor of Chemistry and Physics until 1874. Carrier taught that scientific research and its promise for progress were not antagonistic to the ideals of intellectual and moral culture endorsed by the Church. One of Carrier's students was Father John Augustine Zahm (1851–1921) who was made Professor and Co-Director of the Science Department at age 23 and by 1900 was a nationally prominent scientist and naturalist. Zahm was active in the Catholic Summer School movement, which introduced Catholic laity to contemporary intellectual issues. His book Evolution and Dogma (1896) defended certain aspects of evolutionary theory as true, and argued, moreover, that even the great Church teachers Thomas Aquinas and Augustine taught something like it. The intervention of Irish American Catholics in Rome prevented Zahm's censure by the Vatican. In 1913, Zahm and former President Theodore Roosevelt embarked on a major expedition through the Amazon.
question:  What was the lifespan of John Augustine Zahm?
{'guessed_answer': 'evolution and science', 'actual_answer': '1851–1921'}
context:  The Lobund Institute grew out of pioneering research in germ-free-life which began in 1928. This area of research originated in a question posed by Pasteur as to whether animal life was possible without bacteria. Though others had taken up this idea, their research was short lived and inconclusive. Lobund was the first research organization to answer definitively, that such life is possible and that it can be prolonged through generations. But the objective was not merely to answer Pasteur's question but also to produce the germ free animal as a new tool for biological and medical research. This objective was reached and for years Lobund was a unique center for the study and production of germ free animals and for their use in biological and medical investigations. Today the work has spread to other universities. In the beginning it was under the Department of Biology and a program leading to the master's degree accompanied the research program. In the 1940s Lobund achieved independent status as a purely research organization and in 1950 was raised to the status of an Institute. In 1958 it was brought back into the Department of Biology as integral part of that department, but with its own program leading to the degree of PhD in Gnotobiotics.
question:  Around what time did Lobund of Notre Dame become independent?
{'guessed_answer': 'the 1940s', 'actual_answer': 'the 1940s'}
context:  As of 2012[update] research continued in many fields. The university president, John Jenkins, described his hope that Notre Dame would become "one of the pre–eminent research institutions in the world" in his inaugural address. The university has many multi-disciplinary institutes devoted to research in varying fields, including the Medieval Institute, the Kellogg Institute for International Studies, the Kroc Institute for International Peace studies, and the Center for Social Concerns. Recent research includes work on family conflict and child development, genome mapping, the increasing trade deficit of the United States with China, studies in fluid mechanics, computational science and engineering, and marketing trends on the Internet. As of 2013, the university is home to the Notre Dame Global Adaptation Index which ranks countries annually based on how vulnerable they are to climate change and how prepared they are to adapt.
question:  What does the Kroc Institute at Notre Dame focus on?
{'guessed_answer': 'international peace studies', 'actual_answer': 'International Peace studies'}
context:  About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no traditional social fraternities or sororities at the university, but a majority of students live in the same residence hall for all four years. Some intramural sports are based on residence hall teams, where the university offers the only non-military academy program of full-contact intramural American football. At the end of the intramural season, the championship game is played on the field in Notre Dame Stadium.
question:  How many dorms for males are on the Notre Dame campus?
{'guessed_answer': '2,000', 'actual_answer': '15'}
context:  This Main Building, and the library collection, was entirely destroyed by a fire in April 1879, and the school closed immediately and students were sent home. The university founder, Fr. Sorin and the president at the time, the Rev. William Corby, immediately planned for the rebuilding of the structure that had housed virtually the entire University. Construction was started on the 17th of May and by the incredible zeal of administrator and workers the building was completed before the fall semester of 1879. The library collection was also rebuilt and stayed housed in the new Main Building for years afterwards. Around the time of the fire, a music hall was opened. Eventually becoming known as Washington Hall, it hosted plays and musical acts put on by the school. By 1880, a science program was established at the university, and a Science Hall (today LaFortune Student Center) was built in 1883. The hall housed multiple classrooms and science labs needed for early research at the university.
question:  Who was the president of Notre Dame in 1879?
{'guessed_answer': 'rev . william corby', 'actual_answer': 'Rev. William Corby'}
context:  One of the main driving forces in the growth of the University was its football team, the Notre Dame Fighting Irish. Knute Rockne became head coach in 1918. Under Rockne, the Irish would post a record of 105 wins, 12 losses, and five ties. During his 13 years the Irish won three national championships, had five undefeated seasons, won the Rose Bowl in 1925, and produced players such as George Gipp and the "Four Horsemen". Knute Rockne has the highest winning percentage (.881) in NCAA Division I/FBS football history. Rockne's offenses employed the Notre Dame Box and his defenses ran a 7–2–2 scheme. The last game Rockne coached was on December 14, 1930 when he led a group of Notre Dame all-stars against the New York Giants in New York City.
question:  In what year did the team lead by Knute Rockne win the Rose Bowl?
{'guessed_answer': '1925', 'actual_answer': '1925'}
context:  Holy Cross Father John Francis O'Hara was elected vice-president in 1933 and president of Notre Dame in 1934. During his tenure at Notre Dame, he brought numerous refugee intellectuals to campus; he selected Frank H. Spearman, Jeremiah D. M. Ford, Irvin Abell, and Josephine Brownson for the Laetare Medal, instituted in 1883. O'Hara strongly believed that the Fighting Irish football team could be an effective means to "acquaint the public with the ideals that dominate" Notre Dame. He wrote, "Notre Dame football is a spiritual service because it is played for the honor and glory of God and of his Blessed Mother. When St. Paul said: 'Whether you eat or drink, or whatsoever else you do, do all for the glory of God,' he included football."
question:  Irvin Abell was given what award by Notre Dame?
{'guessed_answer': 'laetare medal', 'actual_answer': 'Laetare Medal'}
context:  The Rev. Theodore Hesburgh, C.S.C., (1917–2015) served as president for 35 years (1952–87) of dramatic transformations. In that time the annual operating budget rose by a factor of 18 from $9.7 million to $176.6 million, and the endowment by a factor of 40 from $9 million to $350 million, and research funding by a factor of 20 from $735,000 to $15 million. Enrollment nearly doubled from 4,979 to 9,600, faculty more than doubled 389 to 950, and degrees awarded annually doubled from 1,212 to 2,500.
question:  What was the size of the Notre Dame endowment when Theodore Hesburgh became president?
{'guessed_answer': '9 million', 'actual_answer': '$9 million'}
context:  In the 18 years under the presidency of Edward Malloy, C.S.C., (1987–2005), there was a rapid growth in the school's reputation, faculty, and resources. He increased the faculty by more than 500 professors; the academic quality of the student body has improved dramatically, with the average SAT score rising from 1240 to 1360; the number of minority students more than doubled; the endowment grew from $350 million to more than $3 billion; the annual operating budget rose from $177 million to more than $650 million; and annual research funding improved from $15 million to more than $70 million. Notre Dame's most recent[when?] capital campaign raised $1.1 billion, far exceeding its goal of $767 million, and is the largest in the history of Catholic higher education.
question:  When Malloy became president of Notre Dame what was the size of the endowment?
{'guessed_answer': '350 million', 'actual_answer': '$350 million'}
context:  Because of its Catholic identity, a number of religious buildings stand on campus. The Old College building has become one of two seminaries on campus run by the Congregation of Holy Cross. The current Basilica of the Sacred Heart is located on the spot of Fr. Sorin's original church, which became too small for the growing college. It is built in French Revival style and it is decorated by stained glass windows imported directly from France. The interior was painted by Luigi Gregori, an Italian painter invited by Fr. Sorin to be artist in residence. The Basilica also features a bell tower with a carillon. Inside the church there are also sculptures by Ivan Mestrovic. The Grotto of Our Lady of Lourdes, which was built in 1896, is a replica of the original in Lourdes, France. It is very popular among students and alumni as a place of prayer and meditation, and it is considered one of the most beloved spots on campus.
question:  In which architectural style is the Basilica of the Sacred Heart at Notre Dame made?
{'guessed_answer': 'french revival', 'actual_answer': 'French Revival'}
context:  The University of Notre Dame has made being a sustainability leader an integral part of its mission, creating the Office of Sustainability in 2008 to achieve a number of goals in the areas of power generation, design and construction, waste reduction, procurement, food services, transportation, and water.As of 2012[update] four building construction projects were pursuing LEED-Certified status and three were pursuing LEED Silver. Notre Dame's dining services sources 40% of its food locally and offers sustainably caught seafood as well as many organic, fair-trade, and vegan options. On the Sustainable Endowments Institute's College Sustainability Report Card 2010, University of Notre Dame received a "B" grade. The university also houses the Kroc Institute for International Peace Studies. Father Gustavo Gutierrez, the founder of Liberation Theology is a current faculty member.
question:  Notre Dame got a "B" for its sustainability practices from which entity?
{'guessed_answer': 'sustainable endowments institute', 'actual_answer': 'Sustainable Endowments Institute'}
context:  The College of Arts and Letters was established as the university's first college in 1842 with the first degrees given in 1849. The university's first academic curriculum was modeled after the Jesuit Ratio Studiorum from Saint Louis University. Today the college, housed in O'Shaughnessy Hall, includes 20 departments in the areas of fine arts, humanities, and social sciences, and awards Bachelor of Arts (B.A.) degrees in 33 majors, making it the largest of the university's colleges. There are around 2,500 undergraduates and 750 graduates enrolled in the college.
question:  How many BA majors does the College of Arts and Letters at Notre Dame offer?
{'guessed_answer': '33', 'actual_answer': '33'}
context:  The School of Architecture was established in 1899, although degrees in architecture were first awarded by the university in 1898. Today the school, housed in Bond Hall, offers a five-year undergraduate program leading to the Bachelor of Architecture degree. All undergraduate students study the third year of the program in Rome. The university is globally recognized for its Notre Dame School of Architecture, a faculty that teaches (pre-modernist) traditional and classical architecture and urban planning (e.g. following the principles of New Urbanism and New Classical Architecture). It also awards the renowned annual Driehaus Architecture Prize.
question:  Which prestigious prize does the School of Architecture at Notre Dame give out?
{'guessed_answer': 'driehaus architecture prize', 'actual_answer': 'Driehaus Architecture Prize'}
context:  The University of Notre Dame du Lac (or simply Notre Dame /ˌnoʊtərˈdeɪm/ NOH-tər-DAYM) is a Catholic research university located adjacent to South Bend, Indiana, in the United States. In French, Notre Dame du Lac means "Our Lady of the Lake" and refers to the university's patron saint, the Virgin Mary. The main campus covers 1,250 acres in a suburban setting and it contains a number of recognizable landmarks, such as the Golden Dome, the "Word of Life" mural (commonly known as Touchdown Jesus), and the Basilica.
question:  The school known as Notre Dame is known by a more lengthy name, what is it?
{'guessed_answer': 'university of notre dame du', 'actual_answer': 'University of Notre Dame du'}
context:  Besides its prominence in sports, Notre Dame is also a large, four-year, highly residential research University, and is consistently ranked among the top twenty universities in the United States  and as a major global university. The undergraduate component of the university is organized into four colleges (Arts and Letters, Science, Engineering, Business) and the Architecture School. The latter is known for teaching New Classical Architecture and for awarding the globally renowned annual Driehaus Architecture Prize. Notre Dame's graduate program has more than 50 master's, doctoral and professional degree programs offered by the five schools, with the addition of the Notre Dame Law School and a MD-PhD program offered in combination with IU medical School. It maintains a system of libraries, cultural venues, artistic and scientific museums, including Hesburgh Library and the Snite Museum of Art. Over 80% of the university's 8,000 undergraduates live on campus in one of 29 single-sex residence halls, each with its own traditions, legacies, events and intramural sports teams. The university counts approximately 120,000 alumni, considered among the strongest alumni networks among U.S. colleges.
question:  Where among US universities does Notre Dame rank?
{'guessed_answer': 'among the top twenty', 'actual_answer': 'among the top twenty'}
```

Other available scripts for testing the various deep learning models are:

* [squad_seq2seq_v2_predict.py](demo/squad_seq2seq_v2_predict.py)
* [squad_seq2seq_glove_predict.py](demo/squad_seq2seq_glove_predict.py)
* [squad_seq2seq_v2_glove_predict.py](demo/squad_seq2seq_v2_glove_predict.py)

### Running Web Api Server (WIP)

Goto [demo_web](demo_web) directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000

To use the question and answering system as web api, you can send the following curl command to the 
web api server running at http://localhost:5000

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"agent":"seq2seq", "context":"...", "question":"..."}' http://localhost:5000/qa_api
```

And the following will be the json response:

```json
{
    "agent": "seq2seq",
    "context": "...",
    "question": "...",
    "answer": "..."
}
```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 

