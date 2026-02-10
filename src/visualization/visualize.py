import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from ..config import PLOTS_DIR, TARGET_SKILLS

def plot_top_skills(df):
    skill_counts = {}
    for skill in TARGET_SKILLS:
        count = df[f'has_{skill}'].sum()
        skill_counts[skill] = count
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(skill_counts.keys()), y=list(skill_counts.values()))
    plt.title('Top Requested Skills')
    plt.xlabel('Skill')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_skills.png'))
    plt.close()

def plot_role_heatmap(df):
    heatmap_data = pd.DataFrame()
    for skill in TARGET_SKILLS:
        heatmap_data[skill] = df.groupby('job_title')[f'has_{skill}'].mean()
        
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
    plt.title('Skill Demand by Job Role')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'role_heatmap.png'))
    plt.close()

def plot_skill_trends(df):
    df['month'] = pd.to_datetime(df['date_posted']).dt.to_period('M')
    monthly_skills = df.groupby('month')[[f'has_{skill}' for skill in TARGET_SKILLS]].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_skills.plot(kind='line')
    plt.title('Skill Trends Over Time')
    plt.xlabel('Month')
    plt.ylabel('Demand Count')
    plt.legend(title='Skills', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'skill_trends.png'))
    plt.close()
