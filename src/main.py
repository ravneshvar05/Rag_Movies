"""
Main entry point for the Movie Transcript RAG system.
"""
import os
print("[INFO] Setting offline mode programmatically...")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import yaml
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Add src to path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Detect HuggingFace Spaces using multiple methods
IS_HF_SPACE = (
    os.getenv("SPACE_ID") is not None or 
    os.getenv("SYSTEM") == "spaces" or
    os.getenv("SPACE_AUTHOR_NAME") is not None
)
DATA_DIR = Path("/data") if IS_HF_SPACE else Path("data/processed")


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_system(config: dict):
    """
    Initialize all system components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (pipeline, metadata_store, embedding_store)
    """
    # Lazy imports to avoid slow startup
    from src.utils.logger import MovieRAGLogger
    from src.ingest.metadata_store import MetadataStore
    from src.retrieval.embedding_store import EmbeddingStore
    from src.retrieval.keyword_search import KeywordSearch
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.llm.router import QuestionRouter
    from src.llm.relevance_judge import RelevanceJudge
    from src.llm.distiller import ContextDistiller
    from src.llm.answerer import Answerer
    from src.pipeline.rag_pipeline import RAGPipeline

    # Setup logging
    MovieRAGLogger.setup_logging(config)
    logger = MovieRAGLogger.get_logger(__name__)
    
    logger.info("Initializing Movie RAG system...")
    
    # Override paths for HF Spaces persistence
    if IS_HF_SPACE:
        logger.info("🌐 Running on HuggingFace Spaces - Using persistent storage at /data")
        base_data_path = Path("/data")
        
        # Override ALL data paths to use persistent storage
        config['paths']['raw_srt'] = str(base_data_path / "raw_srt")
        config['paths']['processed'] = str(base_data_path / "processed")
        config['paths']['metadata_db'] = str(base_data_path / "processed" / "metadata.db")
        config['paths']['vector_db'] = str(base_data_path / "processed" / "vector_db")
        
        # Ensure directories exist
        base_data_path.mkdir(parents=True, exist_ok=True)
        Path(config['paths']['raw_srt']).mkdir(parents=True, exist_ok=True)
        Path(config['paths']['processed']).mkdir(parents=True, exist_ok=True)
    else:
        logger.info("💻 Running locally - Using ./data directory")
    
    # Initialize stores
    metadata_store = MetadataStore(config['paths']['metadata_db'])
    
    embedding_config = config['embedding'].copy()
    embedding_config['store_path'] = config['paths']['vector_db']
    embedding_store = EmbeddingStore(embedding_config)
    
    # Initialize retrieval
    keyword_search = KeywordSearch(metadata_store, config['retrieval']['keyword'])
    
    hybrid_retriever = HybridRetriever(
        embedding_store,
        keyword_search,
        metadata_store,
        config['retrieval']['hybrid']
    )
    
    # Initialize LLM components
    router = QuestionRouter(config['llm']['router'])
    judge = RelevanceJudge(config['llm']['judge'])
    distiller = ContextDistiller(config['llm']['distiller'])
    answerer = Answerer(config['llm']['answerer'])
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        hybrid_retriever,
        router,
        judge,
        distiller,
        answerer
    )
    
    logger.info("System initialized successfully")
    
    return pipeline, metadata_store, embedding_store


def ingest_srt(srt_path: str, movie_id: str, config: dict):
    """
    Ingest an SRT file into the system.
    
    Args:
        srt_path: Path to SRT file
        movie_id: Unique movie identifier
        config: Configuration dictionary
    """
    from src.utils.logger import MovieRAGLogger
    from src.ingest.srt_parser import SRTParser
    from src.ingest.chunker import TranscriptChunker
    
    logger = MovieRAGLogger.get_logger(__name__)
    
    logger.info(f"Ingesting SRT file: {srt_path}")
    logger.info(f"Movie ID: {movie_id}")
    
    # Initialize components
    _, metadata_store, embedding_store = initialize_system(config)
    
    # ENFORCE SINGLE MOVIE POLICY: Clear existing data
    logger.info("Enforcing Single Movie Policy: Clearing existing data...")
    metadata_store.clear_all()
    embedding_store.clear()
    logger.info("Data cleared.")
    
    # Parse SRT
    parser = SRTParser()
    entries = parser.parse_file(srt_path)
    
    # Chunk entries
    chunker = TranscriptChunker(config['ingestion'])
    chunks = chunker.chunk_entries(entries, movie_id)
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Store metadata
    metadata_store.insert_chunks(chunks)
    logger.info("Metadata stored")
    
    # Generate and store embeddings
    embedding_store.add_chunks(chunks)
    embedding_store.save()
    logger.info("Embeddings stored")
    
    logger.info("Ingestion complete!")


def interactive_mode(config: dict):
    """
    Run interactive question-answering mode.
    
    Args:
        config: Configuration dictionary
    """
    from src.utils.logger import MovieRAGLogger
    logger = MovieRAGLogger.get_logger(__name__)
    
    pipeline, _, _ = initialize_system(config)
    
    print("\n" + "="*80)
    print("Movie Transcript RAG - Interactive Mode")
    print("="*80)
    print("\nType your questions below. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            # Process question
            result = pipeline.process(question)
            
            # Display result
            print("\n" + "-"*80)
            print(f"Category: {result.route.category}")
            print(f"Chunks retrieved: {result.retrieved_chunks}")
            print(f"Chunks relevant: {result.relevant_chunks}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Model used: {result.answer.model_used}")
            print(f"Confidence: {result.answer.confidence}")
            print("-"*80)
            print(f"\nAnswer:\n{result.answer.answer}")
            
            if result.answer.supporting_timestamps:
                print(f"\nTimestamps: {', '.join(result.answer.supporting_timestamps)}")
            
            print("-"*80)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Movie Transcript RAG System")
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest SRT file')
    ingest_parser.add_argument('srt_file', help='Path to SRT file')
    ingest_parser.add_argument('movie_id', help='Unique movie identifier')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Run interactive QA mode')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask a single question')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--movie-id', help='Optional movie filter')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Execute command
    if args.command == 'ingest':
        ingest_srt(args.srt_file, args.movie_id, config)
    
    elif args.command == 'interactive':
        interactive_mode(config)
    
    elif args.command == 'query':
        pipeline, _, _ = initialize_system(config)
        result = pipeline.process(args.question, args.movie_id)
        
        print(f"\nAnswer: {result.answer.answer}")
        if result.answer.supporting_timestamps:
            print(f"Timestamps: {', '.join(result.answer.supporting_timestamps)}")
        print(f"\nProcessing time: {result.processing_time:.2f}s")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()