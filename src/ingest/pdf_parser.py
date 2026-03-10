"""
PDF file parser for movie transcripts.
"""
from pathlib import Path
from typing import List
from src.models.schemas import SRTEntry
from src.utils.logger import MovieRAGLogger
from src.ingest.text_parser import TextParser

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

logger = MovieRAGLogger.get_logger(__name__)


class PDFParser:
    """
    Parse PDF files by extracting text and using TextParser logic.
    """
    
    def __init__(self):
        self.logger = logger
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is required for PDF parsing. Install it with: pip install pypdf")
        self.text_parser = TextParser()
    
    def parse_file(self, file_path: str) -> List[SRTEntry]:
        """
        Parse a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of SRTEntry objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Parsing PDF file: {file_path}")
        
        try:
            reader = PdfReader(file_path)
            full_text = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text.append(text)
            
            extracted_content = "\n".join(full_text)
            
            # Delegate basic parsing to TextParser
            entries = self.text_parser._parse_content(extracted_content)
            
            if not entries:
                raise ValueError(f"No valid timestamps found in PDF {file_path}. File must contain timestamps.")
            
            self.logger.info(f"Parsed {len(entries)} entries from PDF")
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to parse PDF: {e}")
            raise ValueError(f"PDF parsing error: {e}")
