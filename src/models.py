"""Data models for the exam papers scraper."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

class DownloadStatus(Enum):
    """Status of a download operation."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ProcessingStatus(Enum):
    """Status of processing operation."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExamPaper:
    """Represents an exam paper document."""
    
    # Basic information
    url: str
    filename: str
    school_code: str
    year: int
    
    # File information
    file_path: Optional[Path] = None
    file_size: Optional[int] = None
    
    # Download information
    download_status: DownloadStatus = DownloadStatus.PENDING
    download_attempts: int = 0
    download_error: Optional[str] = None
    downloaded_at: Optional[datetime] = None
    
    # Processing information
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    
    # Content information
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    
    # Embedding information
    embedding_id: Optional[str] = None
    chunks: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def school_full_name(self) -> str:
        """Get full school name."""
        from .config import Config
        return Config.get_school_full_names().get(self.school_code, self.school_code.upper())
    
    @property
    def is_downloaded(self) -> bool:
        """Check if the paper has been downloaded."""
        return self.download_status == DownloadStatus.COMPLETED
    
    @property
    def is_processed(self) -> bool:
        """Check if the paper has been processed."""
        return self.processing_status == ProcessingStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "filename": self.filename,
            "school_code": self.school_code,
            "year": self.year,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_size": self.file_size,
            "download_status": self.download_status.value,
            "download_attempts": self.download_attempts,
            "download_error": self.download_error,
            "downloaded_at": self.downloaded_at.isoformat() if self.downloaded_at else None,
            "processing_status": self.processing_status.value,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "processing_error": self.processing_error,
            "page_count": self.page_count,
            "embedding_id": self.embedding_id,
            "chunks_count": len(self.chunks),
            "metadata": self.metadata
        }

@dataclass
class ScrapingSession:
    """Represents a scraping session with summary statistics."""
    
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Statistics
    total_papers_found: int = 0
    total_papers_downloaded: int = 0
    total_papers_failed: int = 0
    total_papers_skipped: int = 0
    
    # Breakdown by school
    school_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Breakdown by year
    year_stats: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_papers_found == 0:
            return 0.0
        return self.total_papers_downloaded / self.total_papers_found * 100
    
    def add_paper_result(self, paper: ExamPaper) -> None:
        """Add a paper result to the session statistics."""
        
        # Update totals
        self.total_papers_found += 1
        
        if paper.download_status == DownloadStatus.COMPLETED:
            self.total_papers_downloaded += 1
        elif paper.download_status == DownloadStatus.FAILED:
            self.total_papers_failed += 1
        elif paper.download_status == DownloadStatus.SKIPPED:
            self.total_papers_skipped += 1
        
        # Update school stats
        if paper.school_code not in self.school_stats:
            self.school_stats[paper.school_code] = {
                "found": 0, "downloaded": 0, "failed": 0, "skipped": 0
            }
        
        self.school_stats[paper.school_code]["found"] += 1
        
        if paper.download_status == DownloadStatus.COMPLETED:
            self.school_stats[paper.school_code]["downloaded"] += 1
        elif paper.download_status == DownloadStatus.FAILED:
            self.school_stats[paper.school_code]["failed"] += 1
        elif paper.download_status == DownloadStatus.SKIPPED:
            self.school_stats[paper.school_code]["skipped"] += 1
        
        # Update year stats
        if paper.year not in self.year_stats:
            self.year_stats[paper.year] = {
                "found": 0, "downloaded": 0, "failed": 0, "skipped": 0
            }
        
        self.year_stats[paper.year]["found"] += 1
        
        if paper.download_status == DownloadStatus.COMPLETED:
            self.year_stats[paper.year]["downloaded"] += 1
        elif paper.download_status == DownloadStatus.FAILED:
            self.year_stats[paper.year]["failed"] += 1
        elif paper.download_status == DownloadStatus.SKIPPED:
            self.year_stats[paper.year]["skipped"] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "total_papers_found": self.total_papers_found,
            "total_papers_downloaded": self.total_papers_downloaded,
            "total_papers_failed": self.total_papers_failed,
            "total_papers_skipped": self.total_papers_skipped,
            "success_rate": self.success_rate,
            "school_stats": self.school_stats,
            "year_stats": self.year_stats,
            "errors": self.errors
        }