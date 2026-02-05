import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock config fixture
@pytest.fixture
def test_config(temp_dirs):
    return {
        'paths': {
            'metadata_db': 'tests/temp_data/processed/metadata.db',  # Use file-based DB to persist schema
            'vector_db': 'tests/temp_vector_db',
            'raw_srt': 'tests/temp_data/raw',
            'processed': 'tests/temp_data/processed'
        },
        'ingestion': {
            'chunking_strategy': 'scene_aware',
            'max_chunk_seconds': 120,
            'min_chunk_seconds': 10,
            'overlap_seconds': 5,
            'scene_break_threshold': 3.0
        },
        'embedding': {
            'model_name': 'mock-model',
            'dimension': 384,
            'batch_size': 32,
            'normalize': True,
            'store_path': 'tests/temp_vector_db'
        },
        'retrieval': {
            'semantic': {'top_k': 10, 'similarity_threshold': 0.5},
            'keyword': {'top_k': 10, 'min_match_score': 0.1, 'enable_fuzzy': False},
            'hybrid': {'merge_strategy': 'rrf', 'max_final_chunks': 5, 'preserve_order': True}
        },
        'llm': {
            'router': {'model': 'mock-router'},
            'judge': {'model': 'mock-judge'},
            'distiller': {'model': 'mock-distiller'},
            'answerer': {'primary_model': 'mock-answerer', 'fallback_model': 'mock-fallback'}
        }
    }

# Cleanup fixture
@pytest.fixture
def temp_dirs():
    """Create and cleanup temporary directories."""
    dirs = ['tests/temp_vector_db', 'tests/temp_data/raw', 'tests/temp_data/processed']
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        Path(d).mkdir(parents=True, exist_ok=True)
    yield
    for d in dirs:
        if Path(d).exists():
            shutil.rmtree(d)

# Mock Embedding Model
@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer to avoid downloading models."""
    with patch('src.retrieval.embedding_store.SentenceTransformer') as mock_class:
        mock_model = MagicMock()
        # Mock encoding: return list of lists based on input length
        def side_effect(texts, **kwargs):
            import numpy as np
            # Return random vectors of correct dimension
            return np.random.rand(len(texts), 384).astype('float32')
            
        mock_model.encode.side_effect = side_effect
        mock_class.return_value = mock_model
        yield mock_model

# Sample SRT content
@pytest.fixture
def sample_srt_content():
    return """1
00:00:01,000 --> 00:00:04,000
Start of the movie.

2
00:00:05,000 --> 00:00:08,000
[HERO] This is a dialogue.

3
00:00:20,000 --> 00:00:25,000
(Significant gap before this)
Start of new scene.
"""
