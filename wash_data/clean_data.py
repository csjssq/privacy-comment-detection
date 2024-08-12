import re
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words('english'))

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet

def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
if __name__=="__main__":
    df = pd.read_csv(r'<file_path>',encoding_errors='ignore') # change file

    df['clean_text_x'] = df['text_x'].apply(lambda x: remove_url(str(x)))
    df['clean_text_x'] = df['clean_text_x'].apply(lambda x: remove_emoji(str(x)))
    df['clean_text_x'] = df['clean_text_x'].apply(lambda x: remove_html(str(x)))
    df['clean_text_x'] = df['clean_text_x'].apply(lambda x: remove_punct(str(x)))
    df['clean_text_x'] = df['clean_text_x'].apply(word_tokenize)
    df['clean_text_x'] = df['clean_text_x'].apply(lambda x: [word.lower() for word in x])
    df['clean_text_x'] = df['clean_text_x'].apply(lambda x: [word for word in x if word not in stop])
    df['clean_text_x'] = df['clean_text_x'].apply(nltk.tag.pos_tag)
    df['clean_text_x'] = df['clean_text_x'].apply(
        lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()
    df['clean_text_x'] = df['clean_text_x'].apply(
        lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    df['clean_text_x'] = df['clean_text_x'].apply(
        lambda x: [word for word in x if word not in stop])
    df['clean_text_x'] = [' '.join(map(str, l)) for l in df['clean_text_x']]

    df['clean_text_y'] = df['text_y'].apply(lambda x: remove_usernames_links(str(x)))
    df['clean_text_y'] = df['clean_text_y'].apply(lambda x: remove_url(str(x)))
    df['clean_text_y'] = df['clean_text_y'].apply(lambda x: remove_emoji(str(x)))
    df['clean_text_y'] = df['clean_text_y'].apply(lambda x: remove_html(str(x)))
    df['clean_text_y'] = df['clean_text_y'].apply(lambda x: remove_punct(str(x)))
    df['clean_text_y'] = df['clean_text_y'].apply(word_tokenize)
    df['clean_text_y'] = df['clean_text_y'].apply(lambda x: [word.lower() for word in x])
    df['clean_text_y'] = df['clean_text_y'].apply(lambda x: [word for word in x if word not in stop])
    df['clean_text_y'] = df['clean_text_y'].apply(nltk.tag.pos_tag)
    df['clean_text_y'] = df['clean_text_y'].apply(
        lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()
    df['clean_text_y'] = df['clean_text_y'].apply(
        lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    df['clean_text_y'] = df['clean_text_y'].apply(
        lambda x: [word for word in x if word not in stop])
    df['clean_text_y'] = [' '.join(map(str, l)) for l in df['clean_text_y']]
    print(df.sample(2))
    df.to_csv('<new_file_path>',index=False)