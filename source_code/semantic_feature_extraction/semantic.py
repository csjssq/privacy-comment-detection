import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
# Load pre-trained model
model = SentenceTransformer('sentence-transformers')

# Load data
data = pd.read_csv('<file_path>',encoding_errors='ignore')

# # Concatenate post and comment as input
input_sentences = data['text_x'].astype(str).str.cat(data['text_y'].astype(str), sep=' ')
# post = 'Gonna give my son a thank you card for not making me a grandfather before I turned 40'
# comment = 'You should. I was a grandfather at 35'

# Concatenate post and comment as input
# input_sentences = post+' '+comment

# Get embeddings for input sentences
embeddings = model.encode(input_sentences)

# Print result
print("Embeddings shape:", embeddings.shape)
np.savetxt("<semantic_features_file_path>",embeddings, delimiter=",")