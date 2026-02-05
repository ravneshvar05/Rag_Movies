"""
FAISS-based vector store for semantic search.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models.schemas import TranscriptChunk, TimeStamp
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class EmbeddingStore:
    """
    FAISS-based vector store for semantic embeddings.
    """
    
    def __init__(self, config: dict):
        """
        Initialize embedding store.
        
        Args:
            config: Configuration dict
        """
        self.config = config
        self.logger = logger  # Initialize logger first!
        
        # Auto-detect environment and choose appropriate model
        import os
        is_huggingface_space = os.getenv("SPACE_ID") is not None
        
        if is_huggingface_space:
            # Use lightweight model for HuggingFace Spaces (limited RAM)
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.dimension = 384  # Smaller dimension for lightweight model
            self.logger.info("🌐 Running on HuggingFace Spaces - using lightweight model")
        else:
            # Use full-size model for local deployment
            self.model_name = config.get('model_name', 'BAAI/bge-base-en-v1.5')
            self.dimension = config.get('dimension', 768)
            self.logger.info("💻 Running locally - using full-size model")
        
        self.normalize = config.get('normalize', True)
        self.batch_size = config.get('batch_size', 32)
        
        self.store_path = Path(config.get('store_path', 'data/processed/vector_db'))
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {self.model_name}")
        # Force CPU to avoid issues with metadata tensors on HF Spaces free tier
        self.model = SentenceTransformer(self.model_name, device="cpu")
        
        # Initialize FAISS index
        self.index = None
        self.chunk_ids = []  # Maintain mapping from index position to chunk_id
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        import faiss
        
        index_path = self.store_path / 'faiss.index'
        metadata_path = self.store_path / 'metadata.pkl'
        
        if index_path.exists() and metadata_path.exists():
            self.logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                self.chunk_ids = pickle.load(f)
            self.logger.info(f"Loaded index with {len(self.chunk_ids)} vectors")
        else:
            self.logger.info("Creating new FAISS index")
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            if self.normalize:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.chunk_ids = []
    
    def add_chunks(self, chunks: List[TranscriptChunk]) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of TranscriptChunk objects
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        self.logger.info(f"Encoding {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunk IDs
        for chunk in chunks:
            self.chunk_ids.append(chunk.chunk_id)
        
        self.logger.info(f"Added {len(chunks)} vectors to index")
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 30,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Semantic search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Index is empty")
            return []
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        # Search
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunk_ids):
                # FAISS returns distances, convert to similarity
                similarity = float(distance)
                
                if similarity >= threshold:
                    chunk_id = self.chunk_ids[idx]
                    results.append((chunk_id, similarity))
        
        self.logger.debug(f"Found {len(results)} results above threshold {threshold}")
        return results
    
    def save(self):
        """Save index and metadata to disk."""
        import faiss
        
        index_path = self.store_path / 'faiss.index'
        metadata_path = self.store_path / 'metadata.pkl'
        
        faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_ids, f)
        
        self.logger.info(f"Saved index with {len(self.chunk_ids)} vectors")
    
    def clear(self):
        """Clear the index."""
        import faiss
        
        if self.normalize:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.chunk_ids = []
        self.logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            'num_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'model': self.model_name,
            'normalize': self.normalize
        }