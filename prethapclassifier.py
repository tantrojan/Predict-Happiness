import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle

df=pd.read_csv("train.csv",sep=',')

df=df[['User_ID','Description','Is_Response']]

all_words = []
tokenizer=RegexpTokenizer(r'\w+')
for sent in df['Description']:
    tokens=tokenizer.tokenize(sent)
    for w in tokens:
        all_words.append(w.lower())

stop_words = set(stopwords.words("English"))
filtered_words = []
for w in all_words:
    if w not in stop_words:
        filtered_words.append(w)
#
filtered_words = nltk.FreqDist(filtered_words)
word_features = list(filtered_words.keys())[:3000]

def find_features(document):
    words = set(document)
    #dictionary ---> {word,True or False}
    features = {}
    for w in word_features:
        features[w.lower()] = (w.lower() in words)
    return features

featuresets = []

for (sent,response) in zip(df['Description'],df['Is_Response']):
    featuresets.append((find_features(tokenizer.tokenize(sent)),response))

#TRAINING
classifier = nltk.NaiveBayesClassifier.train(featuresets)

#saving classifier using pickle
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

save_list = open("top3000words.pickle","wb")
pickle.dump(word_features,save_list)
save_list.close()
#TESTING
# df2=pd.read_csv("test.csv",sep=',')
# df2=df2[['User_ID','Description','Is_Response']]
#
# testing_set = []
#
# for (sent,response) in zip(df2['Description'],df2['Is_Response']):
#     testing_set.append((find_features(tokenizer.tokenize(sent)),response))
#
# print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

#Prediction
# results = []
# df2=pd.read_csv("test.csv",sep=',')
# df2=df2[['User_ID','Description']]
#
# for sent in df2['Description']:
#     results.append(classifier.classify(find_features(tokenizer.tokenize(sent))))
#
# print(results)
