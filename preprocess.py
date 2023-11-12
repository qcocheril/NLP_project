import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter


# This function cleans the initial dataset to keep only the columns we care about
def clean_df(data):

    filtered_data = data[["topic","title"]] # topic is our target, title is our training sample set
    filtered_data.loc[:,"title"] = filtered_data.loc[:,"title"].astype(str) 
    filtered_data = filtered_data[filtered_data["title"] != "nan"].reset_index(drop=True) # Removes potential missing data

    target = filtered_data["topic"] # set target
    X = filtered_data["title"] # set X
    return (X,target)

# This function defines the words to remove from the text. 
def define_vocab_to_remove(text, low_threshold, high_treshold):
    
    translation_table = str.maketrans(dict.fromkeys(string.punctuation))  # List of punctuation to remove
    stop_words = set(stopwords.words('english')) # List of stop words to remove

    full_text = ' '.join(text).lower() # Join the text together
    full_text_wo_punct = full_text.translate(translation_table) # removes punctuations

    split_text = full_text_wo_punct.split() # split all the text into list of words
    split_text_wo_sw = [word for word in split_text if (word not in stop_words) and word.isalnum()]

    vocab_count = Counter(split_text_wo_sw) # Define a Counter object

    RAREWORDS = [word for word in split_text_wo_sw if vocab_count[word]<=low_threshold] # Set the rare words to remove based on a specified threshold
    FREQWORDS = [word for word in split_text_wo_sw if vocab_count[word]>high_treshold] # Set the freq words to remove 

    words_to_remove = stop_words | set(RAREWORDS) | set(FREQWORDS) # unique list of all the words to remove
    
    return words_to_remove

# This is the preprocessor function that is called in the model pipeline. It takes the words to remove list as an extra parameter
def preprocessor(text, words_to_remove):
    
    text = text.lower() # Lowercase the text
    translation_table = str.maketrans(dict.fromkeys(string.punctuation)) 
    text = text.translate(translation_table) # Removes the punctuation
 
    tokens = word_tokenize(text) # Tokenizes the text
    filtered_tokens = [word for word in tokens if (word not in words_to_remove) and word.isalnum()]

    stemmer = PorterStemmer() # Instanciate a PorterStemmer object
    
    filtered_words = [ stemmer.stem(word) for word in filtered_tokens] # filter and stems each words
    filtered_text = ' '.join(filtered_words) # Join the filtered words back together
    
    return filtered_text 