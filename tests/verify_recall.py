import time
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import initialize_system
from src.utils.logger import MovieRAGLogger

# Setup logger
logger = MovieRAGLogger.get_logger(__name__)

def verify_recall():
    """Verify that the RAG pipeline can retrieve small details."""
    
    # Initialize system
    config = {
        'paths': {
            'metadata_db': 'data/processed/metadata.db',
            'vector_db': 'data/processed/vector_db'
        },
        'embedding': {
            'model_name': 'BAAI/bge-base-en-v1.5',
            'top_k_retrieval': 25
        },
        'retrieval': {
            'keyword': {'top_k': 10},
            'hybrid': {'alpha': 0.5}
        },
        'llm': {
            'router': {'model': 'llama-3.1-8b-instant'},
            'judge': {'model': 'llama-3.1-8b-instant'},
            'distiller': {'model': 'llama-3.1-8b-instant'},
            'answerer': {'model': 'llama-3.3-70b-versatile'}
        }
    }
    
    pipeline, _, _ = initialize_system(config)
    
    # Test queries focusing on small details
    test_queries = [
        "what brand of cigarettes does the character smoke?", # Detail question
        "what is the license plate number of the car?", # Detail question
        "who is rockey" # General question
    ]
    
    print("\n" + "="*50)
    print("VERIFICATION: RAG Recall Optimization")
    print("="*50 + "\n")
    
    for query in test_queries:
        print(f"\nTesting Query: {query}")
        start_time = time.time()
        
        try:
            result = pipeline.process(query)
            
            print(f"Time: {time.time() - start_time:.2f}s")
            print(f"Retrieved Chunks: {result.retrieved_chunks}")
            print(f"Relevant Chunks: {result.relevant_chunks}")
            print(f"Context Length: {result.metadata.get('distilled_context_length', 0)}")
            print(f"Answer Confidence: {result.answer.confidence}")
            print("-" * 30)
            print(f"Answer: {result.answer.answer}")
            print("-" * 30)
            
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    verify_recall()
