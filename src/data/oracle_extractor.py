import pandas as pd
import re
import json
from rouge_score import rouge_scorer
from tqdm import tqdm

def simple_sent_tokenize(text):
    """
    A robust regex-based sentence tokenizer to avoid NLTK download hangs.
    Splits on period, exclamation, or question mark followed by a space and a capital letter.
    """
    if not isinstance(text, str):
        return []
        
    # Split using punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    # Clean up and remove empty strings
    return [s.strip() for s in sentences if s.strip()]

def get_oracle_labels(article_sentences, highlight_sentences, scorer):
    """
    Greedily selects one article sentence for each summary sentence 
    based on the highest ROUGE-L score.
    Returns a binary list assigning 1 to selected sentences, 0 otherwise.
    """
    labels = [0] * len(article_sentences)
    
    if not article_sentences or not highlight_sentences:
        return labels

    selected_indices = set()
    
    for sum_sent in highlight_sentences:
        max_score = -1.0
        best_idx = -1
        
        for i, art_sent in enumerate(article_sentences):
            # Skip if already selected representing a different summary sentence
            if i in selected_indices:
                continue
                
            score = scorer.score(sum_sent, art_sent)['rougeL'].fmeasure
            if score > max_score:
                max_score = score
                best_idx = i
                
        if best_idx != -1:
            selected_indices.add(best_idx)
            labels[best_idx] = 1
            
    return labels

def create_oracle_dataset(input_csv, output_csv, num_samples=None):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv, nrows=num_samples)
    
    text_col = 'article' if 'article' in df.columns else df.columns[0]
    summary_col = 'highlights' if 'highlights' in df.columns else df.columns[1]
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    all_labels = []
    
    print("Generating Oracle Labels...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        article_text = row[text_col]
        summary_text = row[summary_col]
        
        art_sents = simple_sent_tokenize(article_text)
        sum_sents = simple_sent_tokenize(summary_text)
        
        labels = get_oracle_labels(art_sents, sum_sents, scorer)
        all_labels.append(json.dumps(labels))
        
    df['extractive_labels'] = all_labels
    
    print(f"Saving with extracted labels to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    # We will test on just 10 rows to verify it works quickly
    create_oracle_dataset(
        input_csv="dataset/cnn_dailymail/test.csv", 
        output_csv="src/data/test_oracle_subset.csv",
        num_samples=10
    )
