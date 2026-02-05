"""
Hybrid retrieval combining semantic and keyword search.
"""
from typing import List, Dict, Any, Set
from src.retrieval.embedding_store import EmbeddingStore
from src.retrieval.keyword_search import KeywordSearch
from src.ingest.metadata_store import MetadataStore
from src.models.schemas import TranscriptChunk, TimeStamp
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class HybridRetriever:
    """
    Combines semantic and keyword-based retrieval with RRF fusion.
    """
    
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        keyword_search: KeywordSearch,
        metadata_store: MetadataStore,
        config: dict
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_store: EmbeddingStore instance
            keyword_search: KeywordSearch instance
            metadata_store: MetadataStore instance
            config: Configuration dict
        """
        self.embedding_store = embedding_store
        self.keyword_search = keyword_search
        self.metadata_store = metadata_store
        self.config = config
        self.logger = logger
        
        self.semantic_top_k = config.get('semantic', {}).get('top_k', 30)
        self.keyword_top_k = config.get('keyword', {}).get('top_k', 20)
        self.max_final_chunks = config.get('max_final_chunks', 15)
        self.merge_strategy = config.get('merge_strategy', 'rrf')
        self.preserve_order = config.get('preserve_order', True)
    
    def retrieve(
        self,
        query: str,
        movie_id: str = None,
        top_k: int = None
    ) -> List[TranscriptChunk]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query
            movie_id: Optional movie filter
            top_k: Override max results
            
        Returns:
            List of TranscriptChunk objects sorted by relevance/time
        """
        top_k = top_k or self.max_final_chunks
        
        self.logger.info(f"Hybrid retrieval for query: {query[:100]}")
        
        # 1. Semantic retrieval
        semantic_results = self.embedding_store.search(
            query,
            top_k=self.semantic_top_k
        )
        semantic_chunk_ids = {chunk_id: score for chunk_id, score in semantic_results}
        
        self.logger.debug(f"Semantic search returned {len(semantic_results)} results")
        
        # 2. Keyword retrieval
        keyword_results = self.keyword_search.search(
            query,
            movie_id=movie_id,
            top_k=self.keyword_top_k
        )
        keyword_chunk_ids = {
            r['chunk_id']: abs(r.get('search_score', 0))
            for r in keyword_results
        }
        
        self.logger.debug(f"Keyword search returned {len(keyword_results)} results")
        
        # 3. Merge results
        if self.merge_strategy == 'rrf':
            merged_chunk_ids = self._reciprocal_rank_fusion(
                semantic_chunk_ids,
                keyword_chunk_ids
            )
        else:
            # Simple union
            merged_chunk_ids = self._simple_merge(
                semantic_chunk_ids,
                keyword_chunk_ids
            )
        
        # 4. Fetch full chunks
        chunks = self._fetch_chunks(merged_chunk_ids, top_k)
        
        # 5. Sort by timestamp if needed
        if self.preserve_order:
            chunks = sorted(chunks, key=lambda c: c.start_time.to_seconds())
        
        self.logger.info(f"Retrieved {len(chunks)} final chunks")
        
        return chunks
    
    def _reciprocal_rank_fusion(
        self,
        semantic_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        k: int = 60
    ) -> Dict[str, float]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        Args:
            semantic_scores: Dict of chunk_id -> score
            keyword_scores: Dict of chunk_id -> score
            k: RRF constant (default 60)
            
        Returns:
            Dict of chunk_id -> fused_score
        """
        # Get all unique chunk IDs
        all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        # Sort by score descending
        semantic_ranked = sorted(
            semantic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        keyword_ranked = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create rank mappings
        semantic_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(semantic_ranked)}
        keyword_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_ranked)}
        
        # Calculate RRF scores
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            semantic_rank = semantic_ranks.get(chunk_id, len(semantic_ranked))
            keyword_rank = keyword_ranks.get(chunk_id, len(keyword_ranked))
            
            rrf_score = (1.0 / (k + semantic_rank)) + (1.0 / (k + keyword_rank))
            fused_scores[chunk_id] = rrf_score
        
        self.logger.debug(f"RRF merged {len(all_chunk_ids)} unique chunks")
        
        return fused_scores
    
    def _simple_merge(
        self,
        semantic_scores: Dict[str, float],
        keyword_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Simple merge: average scores.
        
        Args:
            semantic_scores: Dict of chunk_id -> score
            keyword_scores: Dict of chunk_id -> score
            
        Returns:
            Dict of chunk_id -> merged_score
        """
        all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        merged_scores = {}
        for chunk_id in all_chunk_ids:
            sem_score = semantic_scores.get(chunk_id, 0.0)
            key_score = keyword_scores.get(chunk_id, 0.0)
            
            # Normalize keyword scores to [0, 1] range
            key_score_norm = min(1.0, key_score / 10.0) if key_score > 0 else 0.0
            
            # Average
            merged_scores[chunk_id] = (sem_score + key_score_norm) / 2.0
        
        return merged_scores
    
    def _fetch_chunks(
        self,
        chunk_scores: Dict[str, float],
        top_k: int
    ) -> List[TranscriptChunk]:
        """
        Fetch full chunks from metadata store.
        
        Args:
            chunk_scores: Dict of chunk_id -> score
            top_k: Number of chunks to return
            
        Returns:
            List of TranscriptChunk objects
        """
        # Sort by score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Fetch from metadata store
        chunks = []
        for chunk_id, score in sorted_chunks:
            chunk_data = self.metadata_store.get_chunk(chunk_id)
            if chunk_data:
                chunk = self._dict_to_chunk(chunk_data)
                chunks.append(chunk)
        
        return chunks
    
    def _dict_to_chunk(self, data: Dict[str, Any]) -> TranscriptChunk:
        """Convert dict to TranscriptChunk."""
        return TranscriptChunk(
            chunk_id=data['chunk_id'],
            movie_id=data['movie_id'],
            start_time=TimeStamp.from_seconds(data['start_time']),
            end_time=TimeStamp.from_seconds(data['end_time']),
            text=data['text'],
            speaker=data.get('speaker'),
            scene_id=data.get('scene_id'),
            metadata=data.get('metadata', {})
        )
    
    def retrieve_by_time_range(
        self,
        movie_id: str,
        start_time: float,
        end_time: float
    ) -> List[TranscriptChunk]:
        """
        Retrieve chunks in a specific time range.
        
        Args:
            movie_id: Movie identifier
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of chunks in time order
        """
        chunk_dicts = self.metadata_store.get_chunks_by_time_range(
            movie_id,
            start_time,
            end_time
        )
        
        chunks = [self._dict_to_chunk(d) for d in chunk_dicts]
        
        return sorted(chunks, key=lambda c: c.start_time.to_seconds())