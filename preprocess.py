import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def preprocess_text(text):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in tokens if word.isalnum()]
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Lemmatization using WordNetLemmatizer
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)

    return text
  

def clean_df(text, rare_threshold):
    text = text[["topic","title"]]
    text["title"] = text["title"].astype(str)
    text = text[text["title"] != "nan"].reset_index(drop=True)
    most_common = freq_rare_words(text)
    RAREWORDS = [w for (w, word_count) in most_common if word_count < rare_threshold]

    for i in range(len(text)):
        text.loc[i,"title"] = remove_rare_words(text.loc[i,"title"], RAREWORDS)
        text.loc[i,"title"] = preprocess_text(text.loc[i,"title"])

    target = text["topic"]
    text = text["title"]
    return (text,target)

def freq_rare_words(text):
    full_text = ' '.join(text)
    split_text = full_text.split()
    count = Counter(split_text)
    most_common = count.most_common()
    
    return most_common

def remove_rare_words(text, RAREWORDS):
    split_text = text.split()
    filtered_words = [ word for word in split_text if word not in RAREWORDS ]

    filtered_text = ' '.join(filtered_words)
    return filtered_text