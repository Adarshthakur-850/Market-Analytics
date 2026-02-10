from src.data.loader import load_data
from src.data.preprocessing import process_nlp
from src.features.build_features import extract_skills, calculate_tfidf
from src.visualization.visualize import plot_top_skills, plot_role_heatmap, plot_skill_trends
from src.models.train_model import train_model

def main():
    df = load_data()
    df = process_nlp(df)
    df = extract_skills(df)
    df = calculate_tfidf(df)
    
    plot_top_skills(df)
    plot_role_heatmap(df)
    plot_skill_trends(df)
    
    train_model(df)

if __name__ == "__main__":
    main()
