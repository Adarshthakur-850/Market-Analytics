import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ..config import TARGET_SKILLS

def extract_skills(df):
    def find_skills(tokens):
        return [skill for skill in TARGET_SKILLS if skill in tokens]
    
    df['extracted_skills'] = df['tokens'].apply(find_skills)
    df['skill_count'] = df['extracted_skills'].apply(len)
    
    for skill in TARGET_SKILLS:
        df[f'has_{skill}'] = df['extracted_skills'].apply(lambda x: 1 if skill in x else 0)
        
    return df

def calculate_tfidf(df):
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df['clean_description'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return pd.concat([df, tfidf_df], axis=1)
