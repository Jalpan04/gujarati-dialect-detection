from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd

class GujaratiTextFeaturizer:
    def __init__(self, use_lexicon=True):
        self.ngram_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            max_features=5000, # Limit for memory/speed in MVP
            min_df=2
        )
        self.use_lexicon = use_lexicon
        # MVP Dialect Markers (Example: Kathiawadi/Saurashtra vs Standard)
        self.lexicon = {
            'sa': 'kathiawadi_marker', # 'Chhe' vs 'Sa'
            'hata': 'standard',
            'huta': 'kathiawadi',
            'mare': 'standard',
            'mahu': 'surti', # Example
        }
        self.lexicon_terms = list(self.lexicon.keys())

    def fit(self, texts):
        """Fit the N-gram vectorizer."""
        self.ngram_vectorizer.fit(texts)

    def transform(self, texts):
        """
        Returns:
            processed_features: Sparse matrix or aggregated features
        """
        # 1. N-grams (Sparse)
        ngram_feats = self.ngram_vectorizer.transform(texts)
        
        # 2. Lexicon Features (Dense)
        if self.use_lexicon:
            lex_feats = self._extract_lexicon_features(texts)
            # In a real pipeline, we might concat these. 
            # For MVP SVM, we might just use N-grams or concat using scipy.sparse.hstack
            from scipy.sparse import hstack
            combined = hstack([ngram_feats, lex_feats])
            return combined
        
        return ngram_feats

    def _extract_lexicon_features(self, texts):
        """
        Binary/Count features for lexicon presence.
        Shape: (n_samples, n_lexicon_terms)
        """
        features = np.zeros((len(texts), len(self.lexicon_terms)))
        
        for i, text in enumerate(texts):
            words = set(text.lower().split()) # Simple split
            for j, term in enumerate(self.lexicon_terms):
                if term in words: # Exact match
                    features[i, j] = 1
                    
        return features

if __name__ == "__main__":
    # Test
    corpus = [
        "તમે ક્યાં જાવ છો", # Standard
        "તમે ક્યાં જાવ સો", # Kathiawadi (so)
    ]
    
    featurizer = GujaratiTextFeaturizer()
    featurizer.fit(corpus)
    feats = featurizer.transform(corpus)
    
    print(f"Feature shape: {feats.shape}")
    print("N-gram vocabulary size:", len(featurizer.ngram_vectorizer.vocabulary_))
