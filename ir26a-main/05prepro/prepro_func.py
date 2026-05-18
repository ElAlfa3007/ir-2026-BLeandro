import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_resources() -> None:
    """Download required NLTK data if it is not already available."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package)


def remove_special_characters(text: str) -> str:
    """Remove non-alphanumeric characters except accented Spanish letters and whitespace."""
    return re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s]', '', text)


def tokenize(text: str, language: str = 'spanish', remove_stopwords: bool = True) -> list[str]:
    """Tokenize text using NLTK and optionally remove stopwords."""
    language = language.lower()
    if language not in {'spanish', 'english'}:
        raise ValueError("Language must be 'spanish' or 'english'")

    tokens = word_tokenize(text, language=language)
    tokens = [token.lower() for token in tokens if token.isalpha()]

    if remove_stopwords:
        stop_words = set(stopwords.words(language))
        tokens = [token for token in tokens if token not in stop_words]

    return tokens


def stemming_tokens(tokens: list[str], language: str = 'spanish') -> list[str]:
    """Apply Snowball stemmer to a list of tokens."""
    language = language.lower()
    if language not in {'spanish', 'english'}:
        raise ValueError("Language must be 'spanish' or 'english'")

    stemmer = SnowballStemmer(language)
    return [stemmer.stem(token) for token in tokens]


def lemmatization_tokens(tokens: list[str], language: str = 'spanish') -> list[str]:
    """Lemmatization alias using the Snowball stemmer for Spanish/English."""
    return stemming_tokens(tokens, language=language)


def build_tfidf_matrix(documents: list[str]) -> tuple[TfidfVectorizer, 'scipy.sparse.csr_matrix']:
    """Build a TF-IDF matrix from a list of documents."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def score_queries_tfidf(vectorizer: TfidfVectorizer, tfidf_matrix, queries: list[str]) -> list[list[float]]:
    """Score a list of queries against the TF-IDF matrix using cosine similarity."""
    query_matrix = vectorizer.transform(queries)
    return cosine_similarity(tfidf_matrix, query_matrix).tolist()
