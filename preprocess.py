import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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
  

def clean_df(data):
    data["Poem"] = data["Poem"].astype(str)
    data = data[data["Poem"] != "nan"].reset_index(drop=True)
    for i in range(len(data)):
        data.loc[i,"Poem"] = preprocess_text(data.loc[i,"Poem"])
    return data