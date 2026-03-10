"""
Main entry point for the Movie Transcript RAG system.
"""
import os
# [MODIFIED] Removed forced offline mode to allow model download on new envs (e.g. HF Spaces)
# Local execution will use cached models if available.
# print("[INFO] Setting offline mode programmatically...")
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
        
        # Configure HuggingFace Cache to use persistent storage
        # This prevents re-downloading models on every restart and avoids ephemeral disk limits
        os.environ['HF_HOME'] = str(base_data_path / ".huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(base_data_path / ".huggingface")
        
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
        data_dir = DATA_DIR # Use the globally defined DATA_DIR for local execution
        logger.info(f"Running locally - Using {data_dir} directory")
    
    # Initialize stores
    metadata_store = MetadataStore(config['paths']['metadata_db'])
    
    embedding_config = config['embedding'].copy()
    embedding_config['store_path'] = config['paths']['vector_db']
    embedding_store = EmbeddingStore(embedding_config)
    
    # Initialize retrieval
    keyword_search = KeywordSearch(metadata_store, config['retrieval']['keyword'])
    
    # [FIX] Backfill chunk_movie_map for legacy chunks if missing
    if len(embedding_store.chunk_movie_map) < len(embedding_store.chunk_ids):
        logger.warning(f"⚠️ Chunk map mismatch (Map: {len(embedding_store.chunk_movie_map)}, IDs: {len(embedding_store.chunk_ids)}). Backfilling from MetadataStore...")
        full_map = metadata_store.get_chunk_movie_map()
        embedding_store.chunk_movie_map = full_map
        embedding_store.save()
        logger.info(f"✅ Backfilled chunk map with {len(full_map)} entries and saved to disk.")
    
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
        answerer,
        config
    )
    
    logger.info("System initialized successfully")
    
    return pipeline, metadata_store, embedding_store


def ingest_file(file_path: str, movie_id: str, config: dict):
    """
    Ingest a file (SRT, TXT, PDF) into the system.
    
    Args:
        file_path: Path to the file
        movie_id: Unique movie identifier
        config: Configuration dictionary
    """
    from src.utils.logger import MovieRAGLogger
    from src.ingest.srt_parser import SRTParser
    from src.ingest.text_parser import TextParser
    from src.ingest.pdf_parser import PDFParser
    from src.ingest.chunker import TranscriptChunker
    
    logger = MovieRAGLogger.get_logger(__name__)
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = file_path_obj.suffix.lower()
    logger.info(f"Ingesting {ext.upper()} file: {file_path}")
    logger.info(f"Movie ID: {movie_id}")
    
    # Initialize components
    _, metadata_store, embedding_store = initialize_system(config)
    
    # Prepare ingestion
    logger.info(f"Preparing ingestion for {movie_id}...")
    metadata_store.clear_movie(movie_id)
    
    # Select parser
    if ext == '.srt':
        parser = SRTParser()
    elif ext == '.txt':
        parser = TextParser()
    elif ext == '.pdf':
        parser = PDFParser()
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .srt, .txt, .pdf")
    
    try:
        entries = parser.parse_file(str(file_path_obj))
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return
    
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
    ingest_parser = subparsers.add_parser('ingest', help='Ingest file (SRT, TXT, PDF)')
    ingest_parser.add_argument('file', help='Path to file')
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
        ingest_file(args.file, args.movie_id, config)
    
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