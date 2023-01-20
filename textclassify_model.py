
# imports used in this file
from logging import getLoggerClass
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
import numpy as np
import sys
from collections import Counter
import math

"""
Nick Bagley

This file contains classes that train text classifiers. These 
classes are intended to train classifiers given example reviews
with positive or negative labels, score new sentences with the 
trained classifier, and classify new sentences into one of the two
classes.

"""

"""
Cite your sources here:

. https://stackoverflow.com/questions/45574832/initailizing-nested-dict-with-fromkeys
. Jurafsky, Daniel. Martin, James. Speech and Language Processing 3rd edition. (2020)\
. nltk list of stopwords from ntlk.corpus package https://www.nltk.org/api/nltk.corpus.html
. nltk lemmatizer from nltk.stem.wordnet package https://www.nltk.org/_modules/nltk/stem/wordnet.html 
"""

def generate_tuples_from_file(training_file_path):
  """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
  f = open(training_file_path, "r", encoding="utf8")
  listOfExamples = []
  for review in f:
    if len(review.strip()) == 0:
      continue
    dataInReview = review.split("\t")
    for i in range(len(dataInReview)):
      # remove any extraneous whitespace
      dataInReview[i] = dataInReview[i].strip()
    t = tuple(dataInReview)
    listOfExamples.append(t)
  f.close()
  return listOfExamples

def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """

  # initializing counts for total and true positives at 0
  pos = 0
  true_pos = 0
  # iterating through actuals and predicted labels to count true and total positives
  for g, p in zip(gold_labels, predicted_labels):
    if p == '1':
      pos += 1
      if g == '1':
        true_pos += 1
  
  # calculating precision 
  if pos == 0:
    prec = 0
  else:
    prec = true_pos / pos

  return prec


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """

  # initializing counts for true positives and false negatives at 0
  true_pos = 0
  false_neg = 0
  # iterating through actuals and predicted labels to count true positives and false negatives
  for g, p in zip(gold_labels, predicted_labels):
    if g == '1':
      if p == '1':
        true_pos += 1
      else:
        false_neg += 1

  # calculating recall 
  if true_pos + false_neg == 0:
    rec = 0
  else:
    rec = true_pos / (true_pos + false_neg)

  return rec


def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
  
  # finding precision and recall using above functions
  prec = precision(gold_labels, predicted_labels)
  rec = recall(gold_labels, predicted_labels)

  # calculating f1 score
  if prec + rec == 0:
    f1_score = 0
  else:
    f1_score = 2 * (prec * rec) / (prec + rec)

  return f1_score


def remove_punc(sentence):
  """
  Removes punctuation from the given sentence
  Parameters:
    sentence (string): a string to remove punctuation from
  Returns: string new_sentence (sentence with removed punctuation)
  """

  # setting a punctuation string
  punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

  # looping through sentence and replacing punctuation with ""
  new_sentence = sentence
  for c in new_sentence:
    if c in punc:
      new_sentence = new_sentence.replace(c, "")

  return new_sentence


def normalize_text(sentence):
  """
  Normalizes a sentence by removing punctuation, converting to lowercase,
  and lemmatizing
  Parameters:
    sentence (string): a string to apply normalization to
  Returns: list sentence_list (normalized version of given sentence as a list)
  """

  # removing punctuation
  new_sentence = remove_punc(sentence)
  # converting sentence to lowercase
  new_sentence = new_sentence.lower()
  # applying lemmatization
  lemmatizer = WordNetLemmatizer()
  sentence_list = new_sentence.split()
  for w in range(len(sentence_list)):
    sentence_list[w] = lemmatizer.lemmatize(sentence_list[w])
  
  return sentence_list 


def remove_stopwords(text):
  """
  Removes the stop words found in nltk.corpus stopwords
  Parameters:
    text (list): list of strings to remove stopwords from
  Returns: list filtered_text (text with stopwords removed)
  """

  # pulling stop words from nltk.corpus stopwords
  stop_words = set(stopwords.words('english'))
  # removing stopwords and creating a filtered list
  filtered_text = []
  for w in text:
    if w not in stop_words:
      filtered_text.append(w)

  return filtered_text


class TextClassify:
  """
  Class designed to train a Naive Bayes text classifier.
  This class is intended to train a classifier given example reviews with 
  corresponding positive or negative labels, score given sentences using
  the trained model, and classify given sentences as positive or negative.
  """


  def __init__(self):
    
    # set of vocabulary
    self.vocabulary = None
    # nested dictionary representing counts of each word under each label
    self.word_counts = None
    # contains the log priors for each class
    self.priors = None
    # stores a list of all tokens found in all documents, seperated by classes
    self.bigdoc = None
    # contains the log likelihoods for each word and for each class
    self.likelihoods = None

  def set_vocab(self, examples):
    """
    Sets the unique vocabulary found across all examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # creating a list of all tokens in the given examples
    tokens = []
    for e in examples:
      text = e[1].split()
      tokens.extend(text)
    
    # creating a set to find the unique vocabulary
    tokens = set(tokens)

    # setting self.vocabulary 
    self.vocabulary = tokens
 
  def count_words(self, examples):
    """
    Counts the amount of times each word occurs for each class across all examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # initializing a nested dictionary representing each word with their count for each class (initialized at 0)
    labels = ['0', '1']
    words = {k: dict.fromkeys(labels, 0) for k in self.vocabulary}
    # updating the nested dictionary based on the given examples
    for e in examples:
      label = e[2]
      tokens = e[1].split()
      if label == '0':
        for t in tokens:
          words[t]['0'] += 1
      elif label == '1':
        for t in tokens:
          words[t]['1'] += 1

    # setting self.word_counts
    self.word_counts = words

  def get_prior(self, examples):
    """
    Calculates the log priors for the two classes
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    
    # initializing the counts of each class to 0
    labels = {'0': 0, '1': 0}
    # iterating over all examples to count number of documents in each class
    for e in examples:
      if e[2] == '0':
        labels['0'] += 1
      elif e[2] == '1':
        labels['1'] += 1
    
    # calculating the log priors for each class
    prior0 = np.log(labels['0'] / len(examples))
    prior1 = np.log(labels['1'] / len(examples))
    priors = {'0': prior0, '1': prior1}

    # setting self.priors
    self.priors = priors

  def get_bigdoc(self, examples):
    """
    Generates a list of all tokens for each respective class
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # initializing the list of tokens for each class
    bigdoc0 = []
    bigdoc1 = []
    # iterating over the examples to add tokens to the respective classes
    for e in examples:
      tokens = e[1].split()
      if e[2] == '0':
        bigdoc0.extend(tokens)
      elif e[2] == '1':
        bigdoc1.extend(tokens)
    
    # storing the token lists in a dictionary for each class
    bigdoc = {'0': bigdoc0, '1': bigdoc1}

    # setting self.bigdoc
    self.bigdoc = bigdoc

  def get_likelihoods(self):
    """
    Finds the log likelihoods for each word and each class
    Parameters:
      None
    Return: None
    """

    # initializing a nested dictionary representing each word and their log likelihood for each class
    labels = ['0', '1']
    likelihoods = {k: dict.fromkeys(labels, 0) for k in self.vocabulary}

    # getting variables needed for the log likelihood calculation
    vocab_length = len(self.vocabulary)
    class0_length = len(self.bigdoc['0'])
    class1_length = len(self.bigdoc['1'])

    # iterating through each word in the vocabulary to get log likelihoods for each class
    for w in self.vocabulary:
      # finding the counts of the word in each class
      count0 = self.word_counts[w]['0']
      count1 = self.word_counts[w]['1']
      # calculating the log likelihood for the word for each class
      likelihood0 = np.log((count0 + 1) / (class0_length + vocab_length))
      likelihood1 = np.log((count1 + 1) / (class1_length + vocab_length))
      # storing the likelihood in the nested dictionary
      likelihoods[w]['0'] = likelihood0
      likelihoods[w]['1'] = likelihood1

    # setting self.likelihoods
    self.likelihoods = likelihoods

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # calling function to assign self.vocabulary
    self.set_vocab(examples)
    # calling function to assign self.word_counts
    self.count_words(examples)
    # calling function to assign self.priors 
    self.get_prior(examples)
    # calling function to assign self.bigdoc
    self.get_bigdoc(examples)
    # calling function to assign self.likelihoods
    self.get_likelihoods()

  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    
    # storing the string as a list of words
    tokens = data.split()

    # looping through the words to sum the log likelihoods
    sum0 = 0
    sum1 = 0
    for w in tokens:
      if w in self.vocabulary:
        sum0 += self.likelihoods[w]['0']
        sum1 += self.likelihoods[w]['1']

    # adding the priors to the sums
    sum0 += self.priors['0']
    sum1 += self.priors['1']
    # computing the final score by taking the exponent
    score0 = math.exp(sum0)
    score1 = math.exp(sum1)

    return {'0': score0, '1': score1}

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """

    # getting scores for each class
    scores = self.score(data)

    # choosing the max score to return
    label = max(scores, key=scores.get)

    return label

  def featurize(self, data):
    """
    we use this format to make implementation of your TextClassifyImproved model more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """

    # creating a tuple (word, True) for existing words and appending them to a list
    features = []
    for w in data.split():
      if w in self.vocabulary:
        t = (w, True)
        features.append(t)

    return features  

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:
  """
  Class designed to train a Naive Bayes text classifier.
  This class is intended to train a classifier given example reviews with 
  corresponding positive or negative labels, score given sentences using
  the trained model, and classify given sentences as positive or negative.

  This class is very similar to TextClassify with several improvements:
    - text is normalized by removing punctuation, normalizing word case, and applying a lemmatizer
    - applies a list of stop words from nltk which are removed when training
  """

  def __init__(self):
    
    # set of vocabulary
    self.vocabulary = None
    # nested dictionary representing counts of each word under each label
    self.word_counts = None
    # contains the log priors for each class
    self.priors = None
    # stores a list of all tokens found in all documents, seperated by classes
    self.bigdoc = None
    # contains the log likelihoods for each word and for each class
    self.likelihoods = None


  def set_vocab(self, examples):
    """
    Sets the unique vocabulary found across all examples
    Applies text normalization and removes stop words
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # creating a list of all tokens in the given examples
    tokens = []
    for e in examples:
      # normalizing the text
      text = normalize_text(e[1])
      tokens.extend(text)

    # creating a set to find the unique vocabulary
    tokens = set(tokens)

    # removing stop words
    tokens = remove_stopwords(tokens)

    # setting self.vocabulary 
    self.vocabulary = tokens

  def count_words(self, examples):
    """
    Counts the amount of times each word occurs for each class across all examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # initializing a nested dictionary representing each word with their count for each class (initialized at 0)
    labels = ['0', '1']
    words = {k: dict.fromkeys(labels, 0) for k in self.vocabulary}
    # updating the nested dictionary based on the given examples
    for e in examples:
      label = e[2]
      # normalizing text
      tokens = normalize_text(e[1])
      # removing stopwords
      tokens = remove_stopwords(tokens)

      if label == '0':
        for t in tokens:
          words[t]['0'] += 1
      elif label == '1':
        for t in tokens:
          words[t]['1'] += 1

    # setting self.word_counts
    self.word_counts = words

  def get_prior(self, examples):
    """
    Calculates the log priors for the two classes
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    
    # initializing the counts of each class to 0
    labels = {'0': 0, '1': 0}
    # iterating over all examples to count number of documents in each class
    for e in examples:
      if e[2] == '0':
        labels['0'] += 1
      elif e[2] == '1':
        labels['1'] += 1
    
    # calculating the log priors for each class
    prior0 = np.log(labels['0'] / len(examples))
    prior1 = np.log(labels['1'] / len(examples))
    priors = {'0': prior0, '1': prior1}

    # setting self.priors
    self.priors = priors

  def get_bigdoc(self, examples):
    """
    Generates a list of all tokens for each respective class
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    # initializing the list of tokens for each class
    bigdoc0 = []
    bigdoc1 = []
    # iterating over the examples to add tokens to the respective classes
    for e in examples:
      # normalizing text
      tokens = normalize_text(e[1])
      # removing stopwords 
      tokens = remove_stopwords(tokens)
      if e[2] == '0':
        bigdoc0.extend(tokens)
      elif e[2] == '1':
        bigdoc1.extend(tokens)
    
    # storing the token lists in a dictionary for each class
    bigdoc = {'0': bigdoc0, '1': bigdoc1}

    # setting self.bigdoc
    self.bigdoc = bigdoc

  def get_likelihoods(self):
    """
    Finds the log likelihoods for each word and each class
    Parameters:
      None
    Return: None
    """

    # initializing a nested dictionary representing each word and their log likelihood for each class
    labels = ['0', '1']
    likelihoods = {k: dict.fromkeys(labels, 0) for k in self.vocabulary}

    # getting variables needed for the log likelihood calculation
    vocab_length = len(self.vocabulary)
    class0_length = len(self.bigdoc['0'])
    class1_length = len(self.bigdoc['1'])

    # iterating through each word in the vocabulary to get log likelihoods for each class
    for w in self.vocabulary:
      # finding the counts of the word in each class
      count0 = self.word_counts[w]['0']
      count1 = self.word_counts[w]['1']
      # calculating the log likelihood for the word for each class
      likelihood0 = np.log((count0 + 1) / (class0_length + vocab_length))
      likelihood1 = np.log((count1 + 1) / (class1_length + vocab_length))
      # storing the likelihood in the nested dictionary
      likelihoods[w]['0'] = likelihood0
      likelihoods[w]['1'] = likelihood1

    # setting self.likelihoods
    self.likelihoods = likelihoods

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    
    # calling function to assign self.vocabulary
    self.set_vocab(examples)
    # calling function to assign self.word_counts
    self.count_words(examples)
    # calling function to assign self.priors 
    self.get_prior(examples)
    # calling function to assign self.bigdoc
    self.get_bigdoc(examples)
    # calling function to assign self.likelihoods
    self.get_likelihoods()

  def score(self, data):
    """
    Score a given piece of text
    you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """

    # normalizing the input data
    tokens = normalize_text(data)
    # removing stopwords
    tokens = remove_stopwords(tokens)

    # looping through the words to sum the log likelihoods
    sum0 = 0
    sum1 = 0
    for w in tokens:
      if w in self.vocabulary:
        sum0 += self.likelihoods[w]['0']
        sum1 += self.likelihoods[w]['1']

    # adding the priors to the sums
    sum0 += self.priors['0']
    sum1 += self.priors['1']
    # computing the final score by taking the exponent
    score0 = math.exp(sum0)
    score1 = math.exp(sum1)

    return {'0': score0, '1': score1}

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    
    # getting scores for each class
    scores = self.score(data)

    # choosing the max score to return
    label = max(scores, key=scores.get)

    return label

  def featurize(self, data):
    """
    we use this format to make implementation of this class more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    
    # creating a tuple (word, True) for existing words and appending them to a list
    features = []
    for w in data.split():
      if w in self.vocabulary:
        t = (w, True)
        features.append(t)

    return features  

  def __str__(self):
    return "Improved naive bayes - normalization and stop words"



def main():

  training = sys.argv[1]
  testing = sys.argv[2]

  classifier = TextClassify()
  print('\n')
  print(classifier)

  # turning the file paths into structured examples
  trainingex = generate_tuples_from_file(training)
  testingex = generate_tuples_from_file(testing)
  # training the classifier on the training data
  classifier.train(trainingex)

  # classifying each testing example and recording predicted and actual labels
  predicted = []
  actuals = []
  for ex in testingex:
    actuals.append(ex[2])
    pred = classifier.classify(ex[1])
    predicted.append(pred)
  
  # report precision, recall, f1
  prec_baseline = precision(actuals, predicted)
  rec_baseline = recall(actuals, predicted)
  f1_baseline = f1(actuals, predicted)
  print("--------------------------")
  print("BASELINE MODEL PERFORMANCE")
  print("Precision: {}".format(prec_baseline))
  print("Recall: {}".format(rec_baseline))
  print("F1 Score: {}".format(f1_baseline))
  

  improved = TextClassifyImproved()
  print('\n')
  print(improved)
  
  # training the improved classifier on the training data
  improved.train(trainingex)

  # classifying each testing example and recording predicted labels
  improved_pred = []
  for ex in testingex:
    pred = improved.classify(ex[1])
    improved_pred.append(pred)

  # report final precision, recall, f1 (for your best model)
  prec_improved = precision(actuals, improved_pred)
  rec_improved = recall(actuals, improved_pred)
  f1_improved = f1(actuals, improved_pred)
  print("--------------------------")
  print("IMPROVED MODEL PERFORMANCE")
  print("Precision: {}".format(prec_improved))
  print("Recall: {}".format(rec_improved))
  print("F1 Score: {}".format(f1_improved))




if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
 








