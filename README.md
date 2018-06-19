# Predict-Happiness
A Naive Approach to predict whether a Customer is Happy from the room service or not. HACKEREARTH PROBLEM

## Dataset Provided
|Variable |	Description |
| --------| ---------- |
| User_ID | unique ID of the customer |
| Description | description of the review posted |
| Browser_Used |	browser used to post the review |
| Device_Used | device used to post the review |
| Is_Response | 	target Variable |

## My Approach to Solving the Problem
- Reading the Dataset 
```sh
df=pd.read_csv("train.csv",sep=',')
```
- Tokenizing the descriptions into Words and making a list 
```sh
all_words = []
tokenizer=RegexpTokenizer(r'\w+')
for sent in df['Description']:
    tokens=tokenizer.tokenize(sent)
    for w in tokens:
all_words.append(w.lower())
```
- Filtering the Words ( Removing the STOPWORDS )
```sh
stop_words = set(stopwords.words("English"))
filtered_words = []
for w in all_words:
    if w not in stop_words:
filtered_words.append(w)
```
- Creating a Dictionary *| word : it's frequency |*
- Top 3000 words are taken into consideration
```sh
filtered_words = nltk.FreqDist(filtered_words)
word_features = list(filtered_words.keys())[:3000]
```
## Creating Featureset
- Creating a Matrix of shape **||no. of Description |x| 3000 words ||**
- Where each cell **(i,j)** consists of **TRUE** or **FALSE** 
- **True** : signifies that particular word is present in the Description
- **False** : signifies that particular word is not present in the Description

```sh
def find_features(document):
    words = set(document)
    #dictionary ---> {word,True or False}
    features = {}
    for w in word_features:
        features[w.lower()] = (w.lower() in words)
return features
```
## Classification
#### Classifier used : *Naive Bayes Classifier*
- The classification Model is trained using the Featureset
```sh
#TRAINING
classifier = nltk.NaiveBayesClassifier.train(featuresets)
```
#### Saving the Classifier using PICKLE
```sh
#saving classifier using pickle
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
```
