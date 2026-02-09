"""
SQLite-based metadata and keyword search store.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.models.schemas import TranscriptChunk
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class MetadataStore:
    """
    SQLite store for chunk metadata and keyword search.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize metadata store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Main chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                movie_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT,
                scene_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Full-text search virtual table
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                content='chunks',
                content_rowid='rowid'
            )
        ''')
        
        # Triggers to keep FTS in sync
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, chunk_id, text)
                VALUES (new.rowid, new.chunk_id, new.text);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text)
                VALUES('delete', old.rowid, old.chunk_id, old.text);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text)
                VALUES('delete', old.rowid, old.chunk_id, old.text);
                INSERT INTO chunks_fts(rowid, chunk_id, text)
                VALUES (new.rowid, new.chunk_id, new.text);
            END
        ''')
        
        # Indexes for efficient queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_movie_id ON chunks(movie_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_time_range ON chunks(start_time, end_time)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_scene_id ON chunks(scene_id)
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Metadata store initialized at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def insert_chunks(self, chunks: List[TranscriptChunk]) -> int:
        """
        Insert multiple chunks into the store.
        
        Args:
            chunks: List of TranscriptChunk objects
            
        Returns:
            Number of chunks inserted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        inserted = 0
        for chunk in chunks:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, movie_id, start_time, end_time, text, speaker, scene_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk.chunk_id,
                    chunk.movie_id,
                    chunk.start_time.to_seconds(),
                    chunk.end_time.to_seconds(),
                    chunk.text,
                    chunk.speaker,
                    chunk.scene_id,
                    json.dumps(chunk.metadata)
                ))
                inserted += 1
            except Exception as e:
                self.logger.error(f"Failed to insert chunk {chunk.chunk_id}: {e}")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Inserted {inserted} chunks into metadata store")
        return inserted
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data as dict or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, movie_id, start_time, end_time, text, speaker, scene_id, metadata
            FROM chunks
            WHERE chunk_id = ?
        ''', (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_chunks_by_movie(self, movie_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a movie.
        
        Args:
            movie_id: Movie identifier
            
        Returns:
            List of chunk dicts
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, movie_id, start_time, end_time, text, speaker, scene_id, metadata
            FROM chunks
            WHERE movie_id = ?
            ORDER BY start_time
        ''', (movie_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def keyword_search(
        self,
        query: str,
        movie_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text keyword search.
        
        Args:
            query: Search query
            movie_id: Optional movie filter
            limit: Maximum results
            
        Returns:
            List of matching chunks with scores
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if movie_id:
            cursor.execute('''
                SELECT c.chunk_id, c.movie_id, c.start_time, c.end_time, 
                       c.text, c.speaker, c.scene_id, c.metadata,
                       bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                WHERE chunks_fts MATCH ? AND c.movie_id = ?
                ORDER BY score
                LIMIT ?
            ''', (query, movie_id, limit))
        else:
            cursor.execute('''
                SELECT c.chunk_id, c.movie_id, c.start_time, c.end_time,
                       c.text, c.speaker, c.scene_id, c.metadata,
                       bm25(chunks_fts) as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            ''', (query, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            chunk_dict = self._row_to_dict(row[:-1])  # Exclude score
            chunk_dict['search_score'] = row[-1]
            results.append(chunk_dict)
        
        return results
    
    def get_chunks_by_time_range(
        self,
        movie_id: str,
        start_time: float,
        end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Get chunks within a time range.
        
        Args:
            movie_id: Movie identifier
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of chunk dicts
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, movie_id, start_time, end_time, text, speaker, scene_id, metadata
            FROM chunks
            WHERE movie_id = ?
              AND NOT (end_time < ? OR start_time > ?)
            ORDER BY start_time
        ''', (movie_id, start_time, end_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        return {
            'chunk_id': row[0],
            'movie_id': row[1],
            'start_time': row[2],
            'end_time': row[3],
            'text': row[4],
            'speaker': row[5],
            'scene_id': row[6],
            'metadata': json.loads(row[7]) if row[7] else {}
        }
    
    def clear_movie(self, movie_id: str) -> int:
        """
        Delete all chunks for a movie.
        
        Args:
            movie_id: Movie identifier
            
        Returns:
            Number of chunks deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM chunks WHERE movie_id = ?', (movie_id,))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Deleted {deleted} chunks for movie {movie_id}")
        return deleted

    def clear_all(self) -> int:
        """
        Delete all chunks from the store.
        
        Returns:
            Number of chunks deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM chunks')
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Cleared {deleted} chunks from metadata store")
        return deleted

    def list_movies(self) -> List[str]:
        """
        Get list of all ingested movie IDs.
        
        Returns:
            List of movie IDs
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT movie_id FROM chunks ORDER BY movie_id')
        rows = cursor.fetchall()
        
        conn.close()
        return [row[0] for row in rows]