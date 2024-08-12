from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os
import csv

analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv('<file_path>',encoding_errors='ignore')
sentences = []
for index,row in df.iterrows():
    text_A = row['text_x']
    text_Reply = row['text_y']
    ID = row['conversation_id']
    
    try:
        vs_A = analyzer.polarity_scores(text_A)
        vs_A_neg,vs_A_neu,vs_A_pos = vs_A['neg'],vs_A['neu'],vs_A['pos']
        vs_reply = analyzer.polarity_scores(text_Reply)
        vs_reply_neg,vs_reply_neu,vs_reply_pos = vs_reply['neg'],vs_reply['neu'],vs_reply['pos']

    except:
        if text_A == '':
            vs_A_neg,vs_A_neu,vs_A_pos=0,0,0
        elif text_Reply =='':
            vs_reply_neg,vs_reply_neu,vs_reply_po=0,0,0 
    sentences.append([ID,vs_A_neg,vs_A_neu,vs_A_pos,vs_reply_neg,vs_reply_neu,vs_reply_pos])
df1 = pd.DataFrame(data=sentences,
                      columns=['id','vs_A_neg','vs_A_neu','vs_A_pos','vs_reply_neg','vs_reply_neu','vs_reply_pos'])

df1.to_csv('<sentiment_file_path>',index=False)  # 1803