import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ..config import RANDOM_STATE, BASE_DIR

def train_model(df):
    features = [col for col in df.columns if col.startswith('has_') or col in ['skill_count']] 
    X = df[features]
    y = df['job_title']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(BASE_DIR, 'models', 'role_classifier.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    return clf
