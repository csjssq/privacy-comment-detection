## Reusing the code for  Quora Question-pairs 
# need load the pre-trained word vectors and prepare the word-to-index mapping
# GoogleNews-vectors-negative300.bin

import sys
import numpy as np
import pandas as pd
import re, nltk, gensim

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import keras
from keras_preprocessing import sequence
from keras.models import Model
from keras.layers import LSTM, Embedding, Input, Lambda
from keras.optimizers import Adadelta
import keras.backend as K


## Helper functions
# Pre-process and convert text to a list of words
def text_clean(corpus, keep_list):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs_list = []
        for word in row.split():
            word = word.lower()
            word = re.sub(r"[^a-zA-Z0-9^.']"," ",word)
            word = re.sub(r"what's", "what is ", word)
            word = re.sub(r"\'ve", " have ", word)
            word = re.sub(r"can't", "cannot ", word)
            word = re.sub(r"n't", " not ", word)
            word = re.sub(r"i'm", "i am ", word)
            word = re.sub(r"\'re", " are ", word)
            word = re.sub(r"\'d", " would ", word)
            word = re.sub(r"\'ll", " will ", word)
            # If the word contains numbers with decimals, this will preserve it
            if bool(re.search(r'\d', word) and re.search(r'\.', word)) and word not in keep_list:
                keep_list.append(word)
            # Preserves certain frequently occuring dot words
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                qs_list.append(p1)
            else : qs_list.append(word)
        
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs_list)))
    
    return cleaned_corpus


def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = True, remove_stopwords = True):
    
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    ''' All stopwords except the 'wh-' words are removed '''
    if remove_stopwords == True:
        wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        stop = set(stopwords.words('english'))
        for word in wh_words:
            stop.remove(word)
        corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        lem = WordNetLemmatizer()
        corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    
    if stemming == True:
        if stem_type == 'snowball':
            stemmer = SnowballStemmer(language = 'english')
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
        else :
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    
        
    return corpus


def exponent_neg_manhattan_distance(left, right):
    ''' 
    Purpose : Helper function for the similarity estimate of the LSTMs outputs
    Inputs : Two n-dimensional vectors
    Output : Manhattan distance between the input vectors
    
    '''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))



# Based on the training set, a keep list of common dot words was prepared
common_dot_words = ['u.s.', 'b.tech', 'm.tech', 'st.', 'e.g.', 'rs.', 'vs.', 'mr.',
                    'dr.', 'u.s', 'i.e.', 'node.js']


'''
for our own data
'''

train = pd.read_csv('<file_path>',encoding_errors='ignore')
q1 = pd.Series(train.text_x.tolist()).astype(str)
q2 = pd.Series(train.text_y.tolist()).astype(str)
all_corpus = q1.append(q2)
all_corpus = preprocess(all_corpus, keep_list = common_dot_words, remove_stopwords = False)

import pickle
# Dump processed list to file for future re-use
with open('outfile', 'wb') as fp:
    pickle.dump(all_corpus, fp)

# Read back processed list from file at a later time
with open ('outfile', 'rb') as fp:
    all_corpus = pickle.load(fp)

# Separating processed questions
q1 = all_corpus[0:q1.shape[0]]
q2 = all_corpus[q2.shape[0]::]

#print("\n Text pre-processing done")

# Loading pre-trained word vectors
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary = True)
w2v = dict(zip(word2vec_model.wv.index2word, word2vec_model.wv.syn0))
 
#print("\n Pre-trained word vectors loaded")


# Prepare word-to-index mapping
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

qs = pd.DataFrame({'q1': q1, 'q2': q2})
questions_cols = ['q1', 'q2']
from tqdm import tqdm
for index, row in tqdm(qs.iterrows()):
    
    for question in questions_cols:
        
        q2n = []  # q2n -> numerical vector representation of each question
        for word in row[question]:
            # Check for stopwords who do not have a word2vec mapping and ignore them
            if word in set(stopwords.words('english')) and word not in word2vec_model.vocab:
                continue

            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                q2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                q2n.append(vocabulary[word])

        # Replace questions with equivalent numerical vector/ word-indices
        qs.at[index, question] = q2n

# Prepare embedding layer
embedding_dim = 300
embeddings = np.random.randn(len(vocabulary)+1, embedding_dim) # Embedding matrix
embeddings[0] = 0 # This is to ignore the zero padding at the beginning of the sequence

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec_model.vocab:
        embeddings[index] = w2v[word]

del word2vec_model, w2v

#print("\n Embedding matrix prepared")
X = qs[questions_cols]
# Feature-space of the two questions
X_test = {'left': X.q1, 'right': X.q2
          }

## Truncating and padding sequences to a length of 50
max_seq_length = 128
X_test['left'] = sequence.pad_sequences(X_test['left'], maxlen = max_seq_length)
X_test['right'] = sequence.pad_sequences(X_test['right'], maxlen = max_seq_length)

# Checking shapes and sizes to ensure no errors occur
assert X_test['left'].shape == X_test['right'].shape

#print("\n Begin model building")
## Define model architecture
# Model variables
n_hidden = 30
batch_size = 64
n_epoch = 1

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32', name = 'input_1')
right_input = Input(shape=(max_seq_length,), dtype='int32', name = 'input_2')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, 
                            trainable=False, name = 'embed_new')

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden, activation = 'relu', name = 'lstm_1_2')

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]), 
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Combine all of the above in a Model
model = Model([left_input, right_input], [malstm_distance])

#print("\nModel built")
## Loading weights from a pre-trained model
model.load_weights("model30_relu_epoch_3.h5", by_name = True)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#print("\n Weights loaded and compiled")

#print("\n Making prediction")
## Predict using pre-trained model
pred = model.predict([X_test['left'], X_test['right']])

print("\n")
res = pd.DataFrame(pred)
res.columns = ["prediction"]
res.to_csv("<similarity_file_path>.csv",index=None)

    