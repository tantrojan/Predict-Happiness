import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle

df = pd.read_csv("test.csv",sep=',')
# print(df.head())

df = df[['User_ID','Description']]
users = df.User_ID.tolist()
stop_words = set(stopwords.words("English"))

# loading classifier and word_list
classifierfile = open("naivebayes.pickle","rb")
classifier = pickle.load(classifierfile)
classifierfile.close()

list_loader = open("top3000words.pickle","rb")
top_words = pickle.load(list_loader)
list_loader.close()

#Creating featureset
def find_features(document):
    words = set(document)
    #dictionary ---> {word,True or False}
    features = {}
    for w in top_words:
        features[w.lower()] = (w.lower() in words)
    return features

results = []

tokenizer=RegexpTokenizer(r'\w+')
for sent in df['Description']:
    tokens=tokenizer.tokenize(sent)
    filtered_words = []
    for w in tokens:
        if w not in stop_words:
            filtered_words.append(w)
    results.append(classifier.classify(find_features(filtered_words)))

final_results = []

for w in results:
    if w == 'not happy':
        final_results.append('not_happy')
    else:
        final_results.append('happy')

output = {'User_ID' : users,'Is_Response': final_results}
outputdata = pd.DataFrame(output)
outputdata.set_index('User_ID', inplace=True)
outputdata.to_csv("prehapoutput.csv", sep=',')
