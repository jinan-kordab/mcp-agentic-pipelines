"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from typing import List, Set
from nltk.stem import PorterStemmer


# ── Lazy-loaded NLTK stopwords, minus content-bearing words ────
# Words like "other", "more", "same" are stopwords in general NLP
# but ARE content in document section titles and technical text.
_CONTENT_WORDS_TO_KEEP: Set[str] = {
    "other", "more", "most", "some", "such", "only", "own", "same",
    "very", "just", "both", "few", "each", "every", "any", "all",
    "no", "not", "nor",  # negation is semantically important
}

def _load_nltk_stopwords() -> Set[str]:
    """Return NLTK stopwords minus content-bearing words."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english")) - _CONTENT_WORDS_TO_KEEP
    except (ImportError, LookupError, OSError):
        pass
    # Minimal fallback
    return {"i", "me", "my", "we", "our", "you", "your", "he", "him",
            "his", "she", "her", "it", "its", "they", "them", "their",
            "this", "that", "these", "those", "am", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "a", "an", "the", "and", "but", "if", "or",
            "because", "as", "of", "at", "by", "for", "with", "about",
            "between", "into", "through", "during", "before", "after",
            "to", "from", "in", "on", "off", "over", "under",
            "can", "will", "should", "now", "don", "doesn", "didn",
            "won", "wouldn", "couldn", "shouldn", "isn", "aren"}


class PrecisStemmer:
    """Combines Porter stemming with domain-specific rules. Acronyms (KYC, AML, ESG) are preserved as-is."""

    _KNOWN_ACRONYMS: Set[str] = {"kyc", "aml", "esg", "gaap", "ifrs", "sec", "fdic", "finra",
                                  "soc", "iso", "hipaa", "gdpr", "ccpa", "sox", "cfpb", "finra"}

    # ── Precis-specific additions (query-structure words) ─────
    _PRECIS_STOPWORDS: Set[str] = {
        "summarize", "summary", "summarise", "explain", "describe",
        "list", "identify", "compare", "contrast", "discuss", "analyze",
        "key", "finding", "findings", "detail", "details", "overview",
        "section", "chapter", "paragraph", "figure", "table", "page",
        "get", "make", "made", "see", "show", "shown", "find", "found",
    }

    # ── Merged set: NLTK standard + Precis custom ─────────────
    _STOPWORDS: Set[str] = _load_nltk_stopwords() | _PRECIS_STOPWORDS

    def __init__(self) -> None:
        self._stemmer = PorterStemmer()

    def stem(self, word: str) -> str:
        """Stem a single word. Preserves known acronyms. Filters stopwords to empty string."""
        word_lower = word.strip().lower()
        if not word_lower:
            return ""
        if word_lower in self._KNOWN_ACRONYMS:
            return word_lower
        if word_lower in self._STOPWORDS:
            return ""
        return self._stemmer.stem(word_lower)

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem a list of tokens, filtering out stopwords and empty results."""
        return [s for token in tokens if (s := self.stem(token))]

    def add_acronym(self, acronym: str) -> None:
        """Register a domain-specific acronym to preserve during stemming."""
        self._KNOWN_ACRONYMS.add(acronym.strip().lower())
