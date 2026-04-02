import pandas as pd
import re

def simple_sent_tokenize(text):
    # A very simplified sentence tokenizer just for demonstration
    # Splits on . ! ? followed by space and capital letter, or at the end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def explore_data():
    # Use test.csv since it's smaller and faster to load than train.csv
    dataset_path = "./dataset/cnn_dailymail/test.csv"
    # print(f"Loading data from {dataset_path}...")
    
    # Load just the first few rows to save memory and time
    df = pd.read_csv(dataset_path, nrows=5)
    
    print("\n--- Dataset Columns ---")
    print(df.columns.tolist())
    
    #standard CNN/DailyMail column names
    text_col = 'article' if 'article' in df.columns else df.columns[0]
    summary_col = 'highlights' if 'highlights' in df.columns else df.columns[1]

    print(f"\n--- First '{text_col}' Snippet ---")
    first_article = str(df[text_col].iloc[0])
    print(first_article[:500] + "...\n")
    
    print(f"--- '{summary_col}' for First Article ---")
    first_highlights = str(df[summary_col].iloc[0])
    print(first_highlights + "\n")
    
    # Demonstrate sentence tokenization
    print("--- Breaking the Article into Sentences ---")
    sentences = simple_sent_tokenize(first_article)
    print(f"Total sentences extracted: {len(sentences)}")
    print("First 3 sentences:")
    for i, sentence in enumerate(sentences[:3]):
        print(f"  {i+1}. {sentence}")

if __name__ == "__main__":
    explore_data()
