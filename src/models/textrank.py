import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextRank:
    def __init__(self, damping_factor=0.85):
        self.damping_factor = damping_factor
        
    def _build_similarity_matrix(self, sentences):
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # We need at least one word to vectorize
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError: # e.g. if all sentences only contain stop words
            return np.zeros((len(sentences), len(sentences)))
            
        # Compute cosine similarity between all sentence pairs
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Zero out the diagonal to avoid self-loops skewing PageRank
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
        
    def rank_sentences(self, sentences):
        """
        Takes a list of sentences and returns their PageRank scores.
        """
        if not sentences:
            return []
            
        if len(sentences) == 1:
            return [1.0]

        # 1. Compute similarity matrix
        sim_matrix = self._build_similarity_matrix(sentences)
        
        # 2. Build graph from similarity matrix
        # Create a graph where nodes are sentences and edges are similarities
        nx_graph = nx.from_numpy_array(sim_matrix)
        
        # 3. Apply PageRank algorithm
        try:
            scores = nx.pagerank(nx_graph, alpha=self.damping_factor, weight='weight')
            # scores is a dictionary {node_index: score}
            # Convert to list ordered by original sentence order
            score_list = [scores.get(i, 0.0) for i in range(len(sentences))]
            return score_list
        except nx.PowerIterationFailedConvergence:
            # Fallback if PageRank fails to converge
            return [1.0 / len(sentences)] * len(sentences)
            
    def get_top_k_indices(self, scores, k):
        """
        Returns the original indices of the top K highest scoring sentences.
        """
        if not scores:
            return []
            
        # Get indices sorted by score descending
        ranked_indices = np.argsort(scores)[::-1]
        
        # Ensure k is not larger than available sentences
        k = min(k, len(scores))
        
        return ranked_indices[:k].tolist()

if __name__ == "__main__":
    # A quick test of the TextRank class
    sample_text = [
        "The quick brown fox jumps over the lazy dog.",
        "Foxes are known for their speed and agility.",
        "Dogs often chase foxes in the wild.",
        "The sun sets in the west.", # Unrelated sentence
        "Many brown foxes live in the forest."
    ]
    
    print("Testing TextRank model...")
    model = TextRank()
    scores = model.rank_sentences(sample_text)
    
    for i, (sent, score) in enumerate(zip(sample_text, scores)):
        print(f"Sentence {i} Score: {score:.4f} -> {sent}")
        
    top_2 = model.get_top_k_indices(scores, k=2)
    print(f"\nTop 2 sentences indices: {top_2}")
    for idx in top_2:
        print(f" - {sample_text[idx]}")
