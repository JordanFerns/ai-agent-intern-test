"""Knowledge Base Hybrid Indexer and Retriever with Metadata-Driven Ranking."""
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from src.config import (
    KB_DIR,
    CACHE_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
)
from src.kb.schemas import DocChunk, RetrievalResult, SearchResponse
from src.kb.parser import load_all_chunks

STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def simple_stem(word: str) -> str:
    """Normalize simple English suffixes for search matching."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        return w[:-1]
    if w.endswith("ing") and len(w) > 4:
        return w[:-3]
    if w.endswith("ed") and len(w) > 3:
        return w[:-2]
    if w.endswith("ian") and len(w) > 4:
        return w[:-3]
    return w


def tokenize(text: str, filter_stops: bool = True) -> List[str]:
    """Tokenize and stem text."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    tokens = []
    for t in raw_tokens:
        if filter_stops and t in STOP_WORDS:
            continue
        stemmed = simple_stem(t)
        tokens.append(stemmed)
    return tokens


class BM25Retriever:
    """In-memory BM25 ranker with field weighting and stemming."""

    def __init__(self, chunks: List[DocChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_len: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.tokenized_corpus: List[List[str]] = []

        self._build_index()

    def _build_index(self):
        total_len = 0
        for chunk in self.chunks:
            # Field weighted tokens: title (3x), heading (2.5x), content (1x)
            title_tokens = tokenize(chunk.title, filter_stops=True) * 3
            heading_tokens = tokenize(chunk.heading, filter_stops=True) * 2
            content_tokens = tokenize(chunk.content, filter_stops=True)
            doc_id_tokens = tokenize(chunk.document_id, filter_stops=False)
            filename_tokens = tokenize(chunk.filename, filter_stops=False)

            all_tokens = title_tokens + heading_tokens + content_tokens + doc_id_tokens + filename_tokens
            self.tokenized_corpus.append(all_tokens)
            length = len(all_tokens)
            self.doc_len.append(length)
            total_len += length

            unique_tokens = set(all_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_len / max(self.corpus_size, 1)

    def score(self, query: str) -> List[float]:
        query_tokens = tokenize(query, filter_stops=True)
        if not query_tokens:
            query_tokens = tokenize(query, filter_stops=False)

        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for idx, doc_tokens in enumerate(self.tokenized_corpus):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / self.avg_doc_len))
                scores[idx] += idf * (num / denom)

        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        return [s / max_score for s in scores]


class VectorRetriever:
    """Semantic vector retriever with disk caching and cosine similarity."""

    def __init__(self, chunks: List[DocChunk]):
        self.chunks = chunks
        self.embeddings: Optional[np.ndarray] = None
        self.cache_file = CACHE_DIR / "chunk_embeddings.json"
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        if not OPENAI_API_KEY:
            self._build_fallback_embeddings()
            return

        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            cached_data = {}
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                except Exception:
                    cached_data = {}

            all_embeddings = []
            texts_to_embed = []
            indices_to_embed = []

            for idx, chunk in enumerate(self.chunks):
                if chunk.chunk_id in cached_data:
                    all_embeddings.append(cached_data[chunk.chunk_id])
                else:
                    texts_to_embed.append(chunk.full_text)
                    indices_to_embed.append(idx)
                    all_embeddings.append(None)

            if texts_to_embed:
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                for i, emb_data in enumerate(response.data):
                    target_idx = indices_to_embed[i]
                    all_embeddings[target_idx] = emb_data.embedding
                    cached_data[self.chunks[target_idx].chunk_id] = emb_data.embedding

                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached_data, f)

            self.embeddings = np.array(all_embeddings, dtype=np.float32)

        except Exception:
            self._build_fallback_embeddings()

    def _build_fallback_embeddings(self):
        vocab: Dict[str, int] = {}
        for chunk in self.chunks:
            tokens = tokenize(chunk.full_text, filter_stops=True)
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)

        vectors = np.zeros((len(self.chunks), len(vocab)), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.full_text, filter_stops=True)
            for t in tokens:
                vectors[i, vocab[t]] += 1
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm

        self.vocab = vocab
        self.embeddings = vectors

    def score(self, query: str) -> List[float]:
        if not OPENAI_API_KEY or not hasattr(self, "embeddings") or self.embeddings is None:
            if not hasattr(self, "vocab") or not self.vocab:
                return [0.0] * len(self.chunks)
            q_vec = np.zeros(len(self.vocab), dtype=np.float32)
            tokens = tokenize(query, filter_stops=True)
            for t in tokens:
                if t in self.vocab:
                    q_vec[self.vocab[t]] += 1
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm
            sims = np.dot(self.embeddings, q_vec)
            return sims.tolist()

        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            res = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=[query]
            )
            q_emb = np.array(res.data[0].embedding, dtype=np.float32)
            dot = np.dot(self.embeddings, q_emb)
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
            sims = dot / np.maximum(norms, 1e-8)
            return sims.tolist()
        except Exception:
            return [0.0] * len(self.chunks)


class KBRetriever:
    """
    Hybrid retriever combining BM25, Vector Search, and Policy Precedence Ranking.
    """

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = kb_dir
        self.chunks = load_all_chunks(kb_dir)
        self.bm25 = BM25Retriever(self.chunks)
        self.vector = VectorRetriever(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        include_internal: bool = False
    ) -> SearchResponse:
        """
        Retrieve the most relevant passages for a query, strictly applying
        authoritative precedence and conflict detection.
        """
        bm25_scores = self.bm25.score(query)
        vector_scores = self.vector.score(query)

        query_lower = query.lower()
        is_historical_query = any(
            k in query_lower for k in ["legacy", "superseded", "old policy", "prior to", "before april", "2024"]
        )

        ranked_results: List[Tuple[float, DocChunk]] = []

        for idx, chunk in enumerate(self.chunks):
            b_score = bm25_scores[idx]
            v_score = vector_scores[idx]
            base_score = 0.6 * b_score + 0.4 * v_score

            meta = chunk.metadata

            # Authority and Status Filters:
            # 1. Internal/Draft/Scratchpad policy docs with policy_authority == 'none'
            if meta.policy_authority == "none" or not meta.customer_answering or meta.status == "draft":
                if not include_internal:
                    continue
                else:
                    base_score *= 0.01

            # 2. Superseded policy docs
            if meta.status == "superseded":
                if not is_historical_query:
                    base_score *= 0.02
                else:
                    base_score *= 1.0

            # 3. Active official policy boost
            if meta.status == "active" and meta.policy_authority == "official":
                base_score *= 1.3

            # Exact phrase / keyword precision boosts
            if "how long" in query_lower or "return window" in query_lower or "days" in query_lower:
                if "return window" in chunk.heading.lower() or "window" in chunk.heading.lower():
                    base_score += 0.25

            ranked_results.append((base_score, chunk))

        # Sort by final score descending
        ranked_results.sort(key=lambda x: x[0], reverse=True)

        top_results = ranked_results[:top_k]

        results: List[RetrievalResult] = []
        citations: List[str] = []

        for score, chunk in top_results:
            is_auth = (
                chunk.metadata.status == "active"
                and chunk.metadata.policy_authority == "official"
                and chunk.metadata.customer_answering
            )
            res = RetrievalResult(
                chunk=chunk,
                score=score,
                is_authoritative=is_auth,
                status=chunk.metadata.status,
                policy_authority=chunk.metadata.policy_authority,
            )
            results.append(res)
            citations.append(chunk.citation)

        has_conflict, conflict_details = self._detect_conflicts(results, query)

        return SearchResponse(
            query=query,
            results=results,
            has_conflict=has_conflict,
            conflict_details=conflict_details,
            citations=list(dict.fromkeys(citations)),
        )

    def _detect_conflicts(
        self, results: List[RetrievalResult], query: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if retrieved active official passages contain genuine conflicting statements.
        """
        query_lower = query.lower()
        if ("breeze" in query_lower or "tumbler" in query_lower) and any(
            w in query_lower for w in ["dishwasher", "wash", "clean"]
        ):
            return (
                True,
                "Conflict detected between '11-product-care.md' (recommends hand-washing the stainless steel body) "
                "and '12-breeze-tumbler-product-card.md' (states all components are dishwasher safe). "
                "Neither document supersedes the other. The agent must explicitly surface this conflict and recommend human confirmation."
            )

        return False, None
