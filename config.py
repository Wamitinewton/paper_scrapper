"""Configuration management for the exam papers scraper."""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    BASE_URL: str = "https://exampapers.must.ac.ke"
    DOWNLOAD_DIR: Path = Path("downloads")
    LOGS_DIR: Path = Path("logs")
    DATA_DIR: Path = Path("data")
    
    SCHOOL_CODES: Dict[str, str] = {
        "safs": "School of Agriculture and Food Sciences",
        "sbe": "School of Business and Economics", 
        "sci": "School of Science",
        "sea": "School of Engineering and Applied Sciences",
        "sed": "School of Education",
        "shs": "School of Health Sciences",
        "son": "School of Nursing",
        "spas": "School of Public Administration and Management",
        "tvet": "Technical and Vocational Education and Training"
    }
    
    YEARS: List[int] = list(range(2014, 2025))
    
    MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    DELAY_BETWEEN_REQUESTS: float = float(os.getenv("DELAY_BETWEEN_REQUESTS", "1.0"))
    
    SSL_VERIFY: bool = os.getenv("SSL_VERIFY", "false").lower() == "true"
    SSL_CERT_PATH: str = os.getenv("SSL_CERT_PATH", "")
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "exam_papers")
    
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories."""
        cls.DOWNLOAD_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        
        for school_code in cls.SCHOOL_CODES.keys():
            (cls.DOWNLOAD_DIR / school_code).mkdir(exist_ok=True)
    
    @classmethod
    def validate_embedding_config(cls) -> None:
        """Validate configuration for embedding operations."""
        missing_fields = []
        
        if not cls.OPENAI_API_KEY:
            missing_fields.append("OPENAI_API_KEY")
        
        if not cls.QDRANT_URL:
            missing_fields.append("QDRANT_URL")
            
        if not cls.QDRANT_API_KEY:
            missing_fields.append("QDRANT_API_KEY")
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")