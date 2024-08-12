from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np

df = pd.read_csv('<file_path>',encoding_errors='ignore')
df_id = df['conversation_id']
# Define the entity types and their indices
# thus i ignor the 'O' 
entity_types = {"B-PER": 0, "I-PER": 1, "B-ORG": 2, "I-ORG": 3, "B-LOC": 4, "I-LOC": 5, "B-MISC": 6, "I-MISC": 7}

# Create empty DataFrames to store the feature vectors
feature_vectors_A = pd.DataFrame(columns=list(entity_types.keys()))
feature_vectors_B = pd.DataFrame(columns=list(entity_types.keys()))

# Define the fixed-length feature vector
tokenizer = AutoTokenizer.from_pretrained("dslim")
model = AutoModelForTokenClassification.from_pretrained("dslim")
# ner_model = pipeline("ner", model="bert-base-cased", tokenizer="bert-base-cased")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
# Loop through each row in the DataFrame
# Loop through each row in the DataFrame
num = 0
for i, row in df.iterrows():
    # Get the input sentences
    text_A = row["text_x"]
    text_B = row["text_y"]
    try:
    # Run NER model on input sentences
        ner_results_A = ner_pipeline(text_A)
        ner_results_B = ner_pipeline(text_B)
    # Accumulate scores for the entities in sentence A
        feature_vector_A = pd.Series([0] * len(entity_types), index=list(entity_types.keys()))
        for result in ner_results_A:
            entity_label = result["entity"]
            entity_score = result["score"]
            if entity_label in entity_types:
                entity_index = entity_types[entity_label]
                entity_key = list(entity_types.keys())[entity_index]
                feature_vector_A[entity_key] += entity_score
        
        # Append feature vector for sentence A to DataFrame
        feature_vectors_A = feature_vectors_A.append(feature_vector_A, ignore_index=True)
        
        # Accumulate scores for the entities in sentence B
        feature_vector_B = pd.Series([0] * len(entity_types), index=list(entity_types.keys()))
        for result in ner_results_B:
            entity_label = result["entity"]
            entity_score = result["score"]
            if entity_label in entity_types:
                entity_index = entity_types[entity_label]
                entity_key = list(entity_types.keys())[entity_index]
                feature_vector_B[entity_key] += entity_score
        
        # Append feature vector for sentence B to DataFrame
        feature_vectors_B = feature_vectors_B.append(feature_vector_B, ignore_index=True)
    except:
        feature_vector_A = pd.Series([0] * len(entity_types), index=list(entity_types.keys()))
        feature_vector_B = pd.Series([0] * len(entity_types), index=list(entity_types.keys()))
        feature_vectors_A = feature_vectors_A.append(feature_vector_A, ignore_index=True)
        feature_vectors_B = feature_vectors_B.append(feature_vector_B, ignore_index=True)
    num += 1
    print("success:",num)
# Combine feature vectors with original DataFrame
result = pd.concat([df_id, feature_vectors_A, feature_vectors_B], axis=1)
# Print the entity scores
print(result.sample(2))
result.to_csv('<entity_feature_file_path>',index=None)