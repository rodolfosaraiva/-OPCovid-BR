from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import os
import re
import nltk
import csv
import unicodedata
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

nltk.download('punkt')

base_path = 'reli'
ReLiTrain = []
TweetSentBRTrain = []
covidOptionsBRTest = []

files = [os.path.join(base_path, f) for f in os.listdir(base_path)]

for file in files:
    t = 1 if '_Positivos' in file else -1
    with open(file, 'r', encoding = "ISO-8859-1") as content_file:
        content = content_file.read()
        all = re.findall('\[.*?\]', content)
        for w in all:
            ReLiTrain.append((w[1:-1], t))


def clean_tweet(tweet):
    string = str(unicodedata.normalize('NFKD', tweet).encode('ascii','ignore'))[2:]
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", string).split())
    
# Train by Henrico DATASET
filePos = open('henrico/trainTT.pos', 'r')
fileNeg = open('henrico/trainTT.neg', 'r')
for line in filePos:
  TweetSentBRTrain.append((clean_tweet(line[19:]), 1))
for line in fileNeg:
  TweetSentBRTrain.append((clean_tweet(line[19:]), -1))


# Make test Using CovidOptions.BR
# ----------------------

file = csv.reader(open('test/SentCovid-BR.csv'), delimiter=',')
for line in file:
  tweet = line[1]
  sentiment = int(line[2])
  covidOptionsBRTest.append((clean_tweet(tweet), sentiment))

with open('resultado.txt', 'a') as f:
      
  def Experiment(trainData, testData):
    dtc = DecisionTreeClassifier(trainData)
    cl = NaiveBayesClassifier(trainData)

    # Make y_test and y_pred
    y_test = []
    y_pred_dtc = []
    y_pred_cl = []

    for key, value in testData:
      y_test.append(value)
      y_pred_dtc.append(dtc.classify(key))
      y_pred_cl.append(cl.classify(key))

    # Precision
    print("Precision", file=f)
    print('NaiveBayesClassifier:', precision_score(y_test, y_pred_cl), file=f)
    print('DecisionTreeClassifier:', precision_score(y_test, y_pred_dtc), file=f)

    #Accuracy
    print("\nAccuracy", file=f)
    print('NaiveBayesClassifier:', cl.accuracy(testData), file=f)
    print('DecisionTreeClassifier:', dtc.accuracy(testData), file=f)

    # Score
    print("\nF1 Score", file=f)
    print('NaiveBayesClassifier:', f1_score(y_test, y_pred_cl), file=f)
    print('DecisionTreeClassifier:', f1_score(y_test, y_pred_dtc), file=f)

    # Recall
    print("\nRecall", file=f)
    print('NaiveBayesClassifier:', recall_score(y_test, y_pred_cl), file=f)
    print('DecisionTreeClassifier:', recall_score(y_test, y_pred_dtc), file=f)


    # # DATASET Train PLOT 
    # # -------------
    # print("\n Divisão do dataset de treino")
    # df = pd.DataFrame(train, columns=["text", "sentiment"]) 
    # df["sentiment"].value_counts(sort=False).plot(kind='barh')
    # plt.ion()
    # plt.show()

    # # DATASET Test PLOT 
    # # -------------
    # print("\n Divisão do dataset de teste")
    # df2 = pd.DataFrame(testData, columns=["text", "sentiment"]) 
    # df2["sentiment"].value_counts(sort=False).plot(kind='barh')
    # plt.show()

    #Classification Report
    print("\nClassification Report of Naive Bayes", file=f)
    print(classification_report(y_test, y_pred_cl), file=f)

    print("\nClassification Report of Decision Tree", file=f)
    print(classification_report(y_test, y_pred_dtc), file=f)


  # Experiments

  Experiment - Train With ReLI 
  print("1. Experimento - Treinamento apenas com o ReLi", file=f)
  Experiment(ReLiTrain, covidOptionsBRTest)

  # Experiment - Train With TweetSentBR 
  print("2. Experimento - Treinamento com TweetSentBR", file=f)
  Experiment(TweetSentBRTrain, covidOptionsBRTest)


  # Experiment - Train With ReLI + TweetSentBR 
  print("3. Experimento - Treinamento com ReLI + TweetSentBR", file=f)
  Experiment(ReLiTrain + TweetSentBRTrain, covidOptionsBRTest)

  # Experiment - Train With ReLI + TweetSentBR + CovidOptions.BR
  print("4. Experimento - Treinamento com ReLI + TweetSentBR + CovidOptions.BR (.25 separado para teste)", file=f)
  train, test = train_test_split(covidOptionsBRTest, test_size=0.25)
  Experiment(ReLiTrain + TweetSentBRTrain + train, test)


  # Experiment - Train With ReLI + TweetSentBR + CovidOptions.BR
  print("5. Experimento - CovidOptions.BR (.25 separado para teste)", file=f)
  train, test = train_test_split(covidOptionsBRTest, test_size=0.25)
  Experiment(train, test)
