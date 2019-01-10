import random
import nltk
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# shuffle the documents
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print('Most common words: {}'.format(all_words.most_common(15)))
print('The word happy: {}'.format(all_words["happy"]))

# print(len(all_words))
word_features = list(all_words.keys())[:4000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Lets use an example from a negative review
features = find_features(movie_reviews.words('neg/cv000_29416.txt'))


featuresets = [(find_features(rev), category) for (rev, category) in documents]

from sklearn import model_selection

# define a seed for reproducibility
seed = 1

training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))