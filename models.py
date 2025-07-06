"""Data models for the exam papers system."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
import json

class Status(Enum):
    """Generic status enum."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ExamPaper:
    """Represents an exam paper document."""
    
    # Identifiers
    url: str
    filename: str
    school_code: str
    year: int
    
    # File information
    local_path: Optional[Path] = None
    file_size: Optional[int] = None
    
    # Processing status
    download_status: Status = Status.PENDING
    processing_status: Status = Status.PENDING
    embedding_status: Status = Status.PENDING
    
    # Timestamps
    downloaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    embedded_at: Optional[datetime] = None
    
    # Content
    extracted_text: Optional[str] = None
    chunks: List[str] = field(default_factory=list)
    page_count: Optional[int] = None
    
    # Error tracking
    last_error: Optional[str] = None
    retry_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def school_name(self) -> str:
        """Get full school name."""
        from config import Config
        return Config.SCHOOL_CODES.get(self.school_code, self.school_code.upper())
    
    @property 
    def is_downloaded(self) -> bool:
        """Check if paper is downloaded."""
        return self.download_status == Status.COMPLETED
    
    @property
    def is_processed(self) -> bool:
        """Check if paper is processed."""
        return self.processing_status == Status.COMPLETED
    
    @property
    def is_embedded(self) -> bool:
        """Check if paper is embedded."""
        return self.embedding_status == Status.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        
        if data["local_path"]:
            data["local_path"] = str(data["local_path"])
        
        for field_name in ["downloaded_at", "processed_at", "embedded_at"]:
            if data[field_name]:
                data[field_name] = data[field_name].isoformat()
        
        data["download_status"] = data["download_status"].value
        data["processing_status"] = data["processing_status"].value  
        data["embedding_status"] = data["embedding_status"].value
        
        data.pop("extracted_text", None)
        data["chunk_count"] = len(data["chunks"])
        data.pop("chunks", None)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExamPaper":
        """Create instance from dictionary."""
        if data.get("local_path"):
            data["local_path"] = Path(data["local_path"])
        
        for field_name in ["downloaded_at", "processed_at", "embedded_at"]:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        data["download_status"] = Status(data["download_status"])
        data["processing_status"] = Status(data["processing_status"])
        data["embedding_status"] = Status(data["embedding_status"])
        
        allowed_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        return cls(**filtered_data)

@dataclass
class ScrapingSession:
    """Track scraping session statistics."""
    
    session_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    total_found: int = 0
    total_downloaded: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_found == 0:
            return 0.0
        return (self.total_downloaded / self.total_found) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def add_result(self, paper: ExamPaper) -> None:
        """Add paper result to session stats."""
        self.total_found += 1
        
        if paper.download_status == Status.COMPLETED:
            self.total_downloaded += 1
        elif paper.download_status == Status.FAILED:
            self.total_failed += 1 
        elif paper.download_status == Status.SKIPPED:
            self.total_skipped += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_found": self.total_found,
            "total_downloaded": self.total_downloaded,
            "total_failed": self.total_failed,
            "total_skipped": self.total_skipped,
            "success_rate": self.success_rate,
            "errors": self.errors
        }

def save_papers_data(papers: List[ExamPaper], filepath: Path) -> None:
    """Save papers data to JSON file."""
    data = [paper.to_dict() for paper in papers]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_papers_data(filepath: Path) -> List[ExamPaper]:
    """Load papers data from JSON file."""
    if not filepath.exists():
        return []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [ExamPaper.from_dict(item) for item in data]