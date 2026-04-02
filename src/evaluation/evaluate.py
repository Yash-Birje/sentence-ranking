import pandas as pd
import json
import sys
import os
from rouge_score import rouge_scorer
from tqdm import tqdm

# Add the parent directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.textrank import TextRank
from models.semantic_hybrid_textrank import SemanticHybridTextRank
from data.oracle_extractor import simple_sent_tokenize

def evaluate_models(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    text_col = 'article' if 'article' in df.columns else df.columns[0]
    summary_col = 'highlights' if 'highlights' in df.columns else df.columns[1]
    
    model_baseline = TextRank(damping_factor=0.85)
    model_hybrid = SemanticHybridTextRank(positional_decay=0.5)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores_baseline = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    scores_hybrid = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    print("Evaluating models against reference highlights...")
    valid_samples = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        article_text = row[text_col]
        reference_summary = row[summary_col]
        
        if 'extractive_labels' in df.columns:
            labels = json.loads(row['extractive_labels'])
            k = sum(labels)
        else:
            k = len(simple_sent_tokenize(reference_summary))
            
        if k == 0: k = 3
        
        art_sents = simple_sent_tokenize(article_text)
        if not art_sents: continue
        
        # --- Baseline TextRank ---
        tr_scores = model_baseline.rank_sentences(art_sents)
        top_k_indices_base = model_baseline.get_top_k_indices(tr_scores, k=k)
        top_k_indices_base.sort()
        summary_base = " ".join([art_sents[i] for i in top_k_indices_base])
        r_base = scorer.score(reference_summary, summary_base)
        
        scores_baseline['rouge1'] += r_base['rouge1'].fmeasure
        scores_baseline['rouge2'] += r_base['rouge2'].fmeasure
        scores_baseline['rougeL'] += r_base['rougeL'].fmeasure
        
        # --- Semantic Hybrid TextRank ---
        sys_scores = model_hybrid.rank_sentences(art_sents)
        top_k_indices_hybrid = model_hybrid.get_top_k_indices(sys_scores, k=k)
        top_k_indices_hybrid.sort()
        summary_hybrid = " ".join([art_sents[i] for i in top_k_indices_hybrid])
        r_hybrid = scorer.score(reference_summary, summary_hybrid)
        
        scores_hybrid['rouge1'] += r_hybrid['rouge1'].fmeasure
        scores_hybrid['rouge2'] += r_hybrid['rouge2'].fmeasure
        scores_hybrid['rougeL'] += r_hybrid['rougeL'].fmeasure
        
        valid_samples += 1
        
    if valid_samples > 0:
        print("\n--- Final Average ROUGE Scores (F-Measure) ---")
        print(f"{'Metric':<10} | {'Baseline TextRank':<20} | {'Semantic Hybrid TextRank':<30}")
        print("-" * 65)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            base_score = scores_baseline[metric] / valid_samples
            hybrid_score = scores_hybrid[metric] / valid_samples
            print(f"{metric.upper():<10} | {base_score:<20.4f} | {hybrid_score:<30.4f}")
    else:
        print("No valid samples found to evaluate.")

if __name__ == "__main__":
    evaluate_models("src/data/test_oracle_subset.csv")
