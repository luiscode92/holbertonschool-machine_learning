# Repository for Holberton School Machine Learning projects.
Computers can learn similarly as we do. Machine learning is a subfield of artificial intelligence (AI). Conventionally, algorithms are instructions to computers for solving a particular problem. The paradigm of Machine learning algorithms is to train computers with big sets of data inputs and by statistical analysis output (predict) values that are in a specific range.

This repository is entirely written in Python (*.py) and Jupyter Notebook (\*.ipynb), and has 4 categories of projects that are the base of Machine Learning.

The first category is "Math", which includes Linear Algebra, Plotting, Calculus, Probability, Convolutions and Pooling, Advanced Linear Algebra, Multivariate Probability, and ty.

The second category is "Supervised Learning", in which input data is labeled so that after training, the machine can predict labels for unseen data.

Here is an example: [chatbot](https://github.com/felipeserna/holbertonschool-machine_learning/tree/master/supervised_learning/0x13-qa_bot)

A chatbot is trained with a set of documents and then it can answer questions.
```
$ cat 4-main.py
#!/usr/bin/env python3

question_answer = __import__('4-qa').question_answer

question_answer('ZendeskArticles')
$ ./4-main.py
Q: When are PLDs?
A: on - site days from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
$
```

The third category is "Unsupervised Learning", where training data is unlabeled and the machine learns to classify data with similar properties.

The fourth category is "Reinforcement Learning", where an agent is trained based on positive and negative rewards.

## Installing most important libraries
```
$ sudo apt-get install python3-pip
```
Installing pip 19.1
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
```
Installing numpy 1.15, scipy 1.3, and pycodestyle 2.5
```
$ pip install --user numpy==1.15
$ pip install --user scipy==1.3
$ pip install --user pycodestyle==2.5
```
To check that all have been successfully downloaded, use `pip list`.

The most difficult challenges encountered with these projects have been related to translate mathematics into Python code. Some other challenges have been related to hardware resources such as memory and processor. The latter have been solved using Colab from Google, which allows us to use a remote GPU for expensive computations.

## Author:
Felipe Serna <1509@holbertonschool.com>

Chemical Engineer with skills in the design of industrial processes for the elaboration of new products with added value, taking into account economic, environmental and safety restrictions. Knowledge in water treatment, integrated management systems HSEQ and GLP. High interest in fats and oils, particularly for the manufacture of biodiesel. In his Chemical Engineering thesis he designed and built the first microreactor in Colombia for manufacturing biodiesel.

[LinkedIn profile](https://www.linkedin.com/in/felipesernabarbosa/)

[Twitter](https://twitter.com/felipesernabar1)

[Portfolio Project repository](https://github.com/felipeserna/Portfolio_Foundations)
