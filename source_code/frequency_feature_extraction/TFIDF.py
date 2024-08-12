import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import pickle
from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA  
import numpy as np
import scipy


df = pd.read_csv('<file_path>',encoding_errors='ignore')
df = df.fillna('') # avoid washing none

tfidf_vect = TfidfVectorizer(min_df=5, max_df=0.7)
text_x_vectorize = tfidf_vect.fit_transform(df['clean_text_x']+df['clean_text_y'])

X = scipy.sparse.hstack((text_x_vectorize))

# Save vectorizer.vocabulary_
pickle.dump(tfidf_vect.vocabulary_,open("<pkl_file_path>","wb"))

df_res = pd.DataFrame(scipy.sparse.csr_matrix.todense(X)) 

# PCA
pca = PCA(n_components=500)  
df_res = pca.fit_transform(df_res)  
df_res = pd.DataFrame(df_res) 

df_res.to_csv('<frequency_feature_path >.csv',index=False,header=None)

'''
#Load it later
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))
'''