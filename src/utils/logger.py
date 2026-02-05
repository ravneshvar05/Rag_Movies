"""
Logging configuration for the movie RAG system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class MovieRAGLogger:
    """Centralized logger for the system."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[dict] = None) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (usually __name__)
            config: Optional configuration dict with 'level', 'file', 'console'
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        
        # Default configuration
        if config is None:
            config = {
                'level': 'INFO',
                'file': 'logs/movie_rag.log',
                'console': True,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        
        # Set level
        log_level = getattr(logging, config.get('level', 'INFO').upper())
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(config.get('format'))
        
        # Console handler
        if config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        log_file = config.get('file')
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def setup_logging(cls, config: dict) -> None:
        """
        Setup logging configuration for the entire system.
        
        Args:
            config: Configuration dict from YAML
        """
        log_config = config.get('logging', {})
        
        # Ensure log directory exists
        log_file = log_config.get('file', 'logs/movie_rag.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config.get('level', 'INFO').upper()))
        
        # Store config for future loggers
        cls._default_config = log_config