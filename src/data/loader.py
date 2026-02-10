import pandas as pd
import numpy as np
import os
from ..config import DATA_RAW

def load_data():
    file_path = os.path.join(DATA_RAW, 'job_postings.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    titles = ['Data Scientist'] * 50 + ['Data Analyst'] * 50 + ['Machine Learning Engineer'] * 50
    descriptions = [
        "Experience with Python, SQL, and machine learning models.",
        "Proficiency in Tableau, Excel, and data visualization.",
        "Strong background in TensorFlow, NLP, and deep learning.",
        "Knowledge of AWS, Spark, and big data technologies.",
        "Skills in Scikit-learn, Pandas, and data cleaning."
    ] * 30
    
    np.random.seed(42)
    data = {
        'job_title': titles,
        'job_description': np.random.choice(descriptions, 150),
        'location': np.random.choice(['New York', 'San Francisco', 'Remote', 'London'], 150),
        'date_posted': pd.date_range(start='2023-01-01', periods=150).tolist()
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df
