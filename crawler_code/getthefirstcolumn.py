import pandas as pd
import os

# filter en and match

'''
df = pd.read_csv('<sample>.csv')
df_clear = df[df['lang'].str.contains('en')]
df_clear = df_clear[df_clear['id']!=df_clear['conversation_id']]

# == to match; != to not match
# df_clear = df_clear[df_clear['id']==df_clear['conversation_id']] 

df_clear.to_csv('<sample_enoncereply>.csv',index=0)
'''


# filter conversation_id

'''
data = pd.read_csv('<sample_enoncereply>.csv')
data.drop_duplicates(subset=['conversation_id'], keep='first', inplace=True)
data = data['conversation_id']

data.to_csv('<sample_enoncereplyid>.csv', index=0, header=None)
'''