import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SemanticHybridTextRank:
    def __init__(self, model_name='all-MiniLM-L6-v2', damping_factor=0.85, positional_decay=0.5):
        """
        model_name: The Sentence-Transformer model to use. MiniLM is fast and lightweight.
        damping_factor: Standard PageRank damping factor.
        positional_decay: Controls how gently we penalize later sentences. Given a sentence at 
                          index i (0-indexed), its initial weight is 1 / ((i + 1) ** positional_decay).
                          A decay of 0.5 means a square root decay, keeping the bias gentle.
        """
        self.damping_factor = damping_factor
        self.positional_decay = positional_decay
        # Load the sentence transformer model
        self.encoder = SentenceTransformer(model_name)
        
    def _build_similarity_matrix(self, sentences):
        # 1. Convert sentences to semantic embeddings
        embeddings = self.encoder.encode(sentences)
        
        # 2. Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        
        # 3. Prevent negative similarities (if any occur from cosine) and self-loops
        # PageRank expects non-negative weights
        similarity_matrix = np.clip(similarity_matrix, 0, None)
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
        
    def _get_personalized_weights(self, num_sentences):
        """
        Creates a bias vector that gently favors sentences appearing earlier in the document.
        """
        weights = {}
        total_weight = 0.0
        
        for i in range(num_sentences):
            # Gentle positional rank weight e.g., 1/sqrt(position)
            weight = 1.0 / ((i + 1) ** self.positional_decay)
            weights[i] = weight
            total_weight += weight
            
        # Normalize weights so they sum to 1
        for i in range(num_sentences):
            weights[i] /= total_weight
            
        return weights

    def rank_sentences(self, sentences):
        if not sentences:
            return []
            
        num_sents = len(sentences)
        if num_sents == 1:
            return [1.0]

        # 1. Compute Semantic Similarity
        sim_matrix = self._build_similarity_matrix(sentences)
        
        # 2. Build graph from similarity matrix
        nx_graph = nx.from_numpy_array(sim_matrix)
        
        # 3. Calculate Personalization vector (Positional Bias)
        personalization = self._get_personalized_weights(num_sents)
        
        # 4. Apply Personalized PageRank
        # 'personalization' acts as the probability distribution for the random walker teleporting
        try:
            scores = nx.pagerank(
                nx_graph, 
                alpha=self.damping_factor, 
                weight='weight', 
                personalization=personalization
            )
            score_list = [scores.get(i, 0.0) for i in range(num_sents)]
            return score_list
        except nx.PowerIterationFailedConvergence:
            # Fallback
            return [personalization[i] for i in range(num_sents)]
            
    def get_top_k_indices(self, scores, k):
        if not scores:
            return []
            
        ranked_indices = np.argsort(scores)[::-1]
        k = min(k, len(scores))
        return ranked_indices[:k].tolist()

if __name__ == "__main__":
    sample_text = [
        "The quick brown fox jumps over the lazy dog.",
        "Foxes are known for their speed and agility.",
        "Dogs often chase foxes in the wild.",
        "The sun sets in the west.", # Unrelated sentence
        "Many brown foxes live in the forest."
    ]
    
    print("Testing Semantic Hybrid TextRank...")
    # Suppress warnings for downloading models if needed
    model = SemanticHybridTextRank(positional_decay=0.5) 
    scores = model.rank_sentences(sample_text)
    
    for i, (sent, score) in enumerate(zip(sample_text, scores)):
        print(f"Sentence {i} Score: {score:.4f} -> {sent}")
        
    top_2 = model.get_top_k_indices(scores, k=2)
    print(f"\nTop 2 sentences indices: {top_2}")
    for idx in top_2:
        print(f" - {sample_text[idx]}")
