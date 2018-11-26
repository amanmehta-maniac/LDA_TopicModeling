import pandas as pd
import gensim
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import STOPWORDS
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
# from nltk.stem.porter import *
import numpy as np
from nltk.stem.porter import *
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from auxilaryFunctions import *
# nltk.download('wordnet')
from gensim import corpora, models
from pprint import pprint


data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text
print len(documents)


print(WordNetLemmatizer().lemmatize('went', pos='v'))

# stemmer = SnowballStemmer('english')
# original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
#            'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
#            'traditional', 'reference', 'colonizer','plotted']
# singles = [stemmer.stem(plural) for plural in original_words]
# pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})

np.random.seed(2018)
doc_sample = documents[documents['index'] == 4310].values[0][0]

# print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
# print(words)
# print('\n\n tokenized and lemmatized document: ')
# print(preprocess(doc_sample))


processed_docs = documents['headline_text'][:20].map(preprocess)
print processed_docs


### dictonary of all words of the headlines ###

dictionary = gensim.corpora.Dictionary(processed_docs)
# count = 0
# for k, v in dictionary.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break

# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print bow_corpus[10]


bow_doc_10 = bow_corpus[10]

for i in range(len(bow_doc_10)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_10[i][0], 
                                                     dictionary[bow_doc_10[i][0]], 
                                                     bow_doc_10[i][1]))

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

for doc in corpus_tfidf:
    pprint(doc)
    break

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)