import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def setup_nltk():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nltk_data_dir = os.path.join(base_dir, 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print(f"Downloading NLTK data to {nltk_data_dir}...")
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('punkt_tab', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def process_nlp(df):
    setup_nltk()
    stop_words = set(stopwords.words('english'))
    
    df['clean_description'] = df['job_description'].apply(clean_text)
    
    def tokenize_and_remove_stopwords(text):
        tokens = word_tokenize(text)
        return [t for t in tokens if t not in stop_words]
    
    df['tokens'] = df['clean_description'].apply(tokenize_and_remove_stopwords)
    return df
