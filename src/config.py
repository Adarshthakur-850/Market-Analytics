import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_SKILLS = ['python', 'sql', 'aws', 'spark', 'tableau', 'excel', 'tensorflow', 'scikit-learn', 'nlp', 'pandas']
