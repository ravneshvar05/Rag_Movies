
import os
import sys
from pathlib import Path
import unittest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.text_parser import TextParser

class TestParsers(unittest.TestCase):
    
    def setUp(self):
        self.valid_text = """
[00:00:10] This is a valid line.
[00:00:20] This is another valid line with {brackets}.
(00:00:30) Different bracket style.
00:00:40 - 00:00:45 Range style.
"""
        self.invalid_text = """
This is just some text.
It has no timestamps at all.
No numbers here either.
        """
        
        self.valid_file = "test_valid.txt"
        self.invalid_file = "test_invalid.txt"
        
        with open(self.valid_file, "w") as f:
            f.write(self.valid_text)
            
        with open(self.invalid_file, "w") as f:
            f.write(self.invalid_text)

    def tearDown(self):
        if os.path.exists(self.valid_file):
            os.remove(self.valid_file)
        if os.path.exists(self.invalid_file):
            os.remove(self.invalid_file)

    def test_text_parser_valid(self):
        parser = TextParser()
        entries = parser.parse_file(self.valid_file)
        print(f"Parsed {len(entries)} entries from valid file.")
        self.assertTrue(len(entries) >= 4)
        self.assertEqual(entries[0].text, "This is a valid line.")
        self.assertEqual(entries[0].start_time.seconds, 10)

    def test_text_parser_invalid(self):
        parser = TextParser()
        with self.assertRaises(ValueError):
            parser.parse_file(self.invalid_file)
        print("Successfully caught invalid file without timestamps.")

    def test_pdf_parser_mock(self):
        from unittest.mock import MagicMock, patch
        from src.ingest.pdf_parser import PDFParser
        
        # Mock PdfReader
        with patch('src.ingest.pdf_parser.PdfReader') as MockReader:
            # Setup mock to return pages with text
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "[00:00:05] Hello from PDF."
            
            mock_instance = MockReader.return_value
            mock_instance.pages = [mock_page]
            
            # Create a dummy file just to pass path existence check
            with open("dummy.pdf", "w") as f:
                f.write("dummy")
            
            try:
                parser = PDFParser()
                entries = parser.parse_file("dummy.pdf")
                
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0].text, "Hello from PDF.")
                self.assertEqual(entries[0].start_time.seconds, 5)
                print("Successfully tested PDFParser with mock.")
                
            finally:
                if os.path.exists("dummy.pdf"):
                    os.remove("dummy.pdf")

if __name__ == '__main__':
    unittest.main()
