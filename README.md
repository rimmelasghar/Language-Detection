# Language-Detection
Language detection is a natural language processing task where we need to identify the language of a text or document. Using machine learning for language identification was a difficult task a few years ago because there was not a lot of data on languages, but with the availability of data with ease, several powerful machine learning models are already available for language identification.
As a human, you can easily detect the languages you know. For example, I can easily identify Urdu and English, but being an Pakistani, it is also not possible for me to identify all languages. This is where the language identification task can be used. Google Translate is one of the most popular language translators in the world which is used by so many people around the world. It also includes a machine learning model to detect languages that you can use if you don’t know which language you want to translate.

## Note:
The most important part of training a language detection model is data. The more data you have about every language, the more accurate your model will perform in real-time. The dataset that I am using is collected from Kaggle, which contains data about 22 popular languages and contains 1000 sentences in each of the languages, so it will be an appropriate dataset for training a language detection model with machine learning.

## Project
At the start I've imported the libraries required for this task. After loading the dataset. I've found out that this dataset contain 22000 rows of data and 2 columns Text,language respectively.
transform the columns to an numpy array and declared a CountVectorizer() the functionality of CountVectorizer() is explained in the bottom.Scale the training data using fit_transform() method. 
split the dataset into two parts (train,test).As this is a problem of multiclass classification, so I will be using the Multinomial Naïve Bayes algorithm to train the language detection model as this algorithm always performs very well on the problems based on multiclass classification.
after training the model now it is time to test it. I have taken a user input , transforms the input to array and predicted the input. The prediction is 99% accurate and The model Performs Well. 
## Important Note:
 This model can only detect the languages mentioned in the dataset.

## Prerequisites

-> Pandas
-> Numpy
-> sklearn

Functions used in this rep:

.CountVectorizer()
Convert a collection of text documents to a matrix of token counts.
.fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data
.transform() Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data. 

## Multinomial Naive Bayes Algorithm
Multinomial Naive Bayes is one of the variations of the Naive Bayes algorithm in machine learning which is very useful to use on a dataset that is distributed multinomially. When there are multiple classes to classify, this algorithm can be used because to predict the label of the text it calculates the probability of each label for the input text and then generates the label with the highest probability as output.

Some of the advantages of using this algorithm for multinomial classification are:

It is easy to use on continuous and discrete data
It can handle large data sets
It can classify data with multiple labels
Best to use for training natural language processing models


code by rimmel asghar with ❤
