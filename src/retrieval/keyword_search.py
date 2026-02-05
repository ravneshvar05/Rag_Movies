"""
Keyword and entity-based search.
"""
from typing import List, Dict, Any, Optional
from src.ingest.metadata_store import MetadataStore
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class KeywordSearch:
    """
    Keyword-based search using SQLite full-text search.
    """
    
    def __init__(self, metadata_store: MetadataStore, config: dict):
        """
        Initialize keyword search.
        
        Args:
            metadata_store: MetadataStore instance
            config: Configuration dict
        """
        self.metadata_store = metadata_store
        self.config = config
        self.logger = logger
        
        self.top_k = config.get('top_k', 20)
        self.min_score = config.get('min_match_score', 0.5)
        self.enable_fuzzy = config.get('enable_fuzzy', True)
    
    def search(
        self,
        query: str,
        movie_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks matching keywords.
        
        Args:
            query: Search query
            movie_id: Optional movie filter
            top_k: Number of results (overrides config)
            
        Returns:
            List of chunk dicts with scores
        """
        top_k = top_k or self.top_k
        
        # Prepare FTS5 query
        fts_query = self._prepare_fts_query(query)
        
        self.logger.debug(f"Keyword search query: {fts_query}")
        
        # Search using metadata store
        results = self.metadata_store.keyword_search(
            fts_query,
            movie_id=movie_id,
            limit=top_k
        )
        
        # Filter by minimum score
        filtered_results = [
            r for r in results
            if abs(r.get('search_score', 0)) >= self.min_score
        ]
        
        self.logger.info(
            f"Keyword search returned {len(filtered_results)} results "
            f"(filtered from {len(results)})"
        )
        
        return filtered_results
    
    def _prepare_fts_query(self, query: str) -> str:
        """
        Prepare query for FTS5 full-text search.
        
        Args:
            query: Raw query string
            
        Returns:
            FTS5-formatted query
        """
        # Clean and tokenize
        tokens = query.lower().split()
        
        # Remove common stop words (basic list)
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with'
        }
        
        significant_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        if not significant_tokens:
            # Fallback to original tokens
            significant_tokens = tokens
        
        # Build FTS5 query
        # Use OR for flexibility, but you can adjust
        if self.enable_fuzzy:
            # FTS5 doesn't have built-in fuzzy, but we can use prefix matching
            query_parts = [f'"{token}"*' for token in significant_tokens]
        else:
            query_parts = [f'"{token}"' for token in significant_tokens]
        
        fts_query = ' OR '.join(query_parts)
        
        return fts_query
    
    def search_entities(
        self,
        entities: List[str],
        movie_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks containing specific entities (character names, etc).
        
        Args:
            entities: List of entity strings to search
            movie_id: Optional movie filter
            top_k: Number of results
            
        Returns:
            List of chunk dicts
        """
        if not entities:
            return []
        
        # Combine entities into query
        query = ' OR '.join(f'"{entity}"' for entity in entities)
        
        return self.search(query, movie_id=movie_id, top_k=top_k)
    
    def search_exact_quote(
        self,
        quote: str,
        movie_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for exact quote match.
        
        Args:
            quote: Exact quote to search
            movie_id: Optional movie filter
            
        Returns:
            List of matching chunks
        """
        # Use exact phrase matching
        fts_query = f'"{quote}"'
        
        results = self.metadata_store.keyword_search(
            fts_query,
            movie_id=movie_id,
            limit=10  # Exact matches should be few
        )
        
        return results