

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the scraper."""
    
    # Base configuration
    BASE_URL: str = os.getenv("BASE_URL", "https://exampapers.must.ac.ke")
    DOWNLOAD_DIR: Path = Path("downloads")
    LOGS_DIR: Path = Path("logs")
    
    # School codes and years
    SCHOOL_CODES: List[str] = [
        "safs", "sbe", "sci", "sea", "sed", "shs", "son", "spas", "tvet"
    ]
    
    YEARS: List[int] = list(range(2014, 2025))  # 2014 to 2024
    
    # Scraping configuration
    MAX_CONCURRENT_DOWNLOADS: int = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))
    DOWNLOAD_TIMEOUT: int = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))
    RETRY_ATTEMPTS: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    
    # SSL Configuration
    VERIFY_SSL: bool = os.getenv("VERIFY_SSL", "true").lower() in ("true", "1", "yes", "on")
    SSL_CERT_PATH: str = os.getenv("SSL_CERT_PATH", "")  # Custom certificate path if needed
    
    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    
    # Qdrant configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "exam_papers")
    
    # Processing configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    
    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = Path(os.getenv("LOG_FILE", "logs/scraper.log"))
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return missing required fields."""
        errors = {}
        warnings = {}
        
        if not cls.OPENAI_API_KEY:
            errors["OPENAI_API_KEY"] = "OpenAI API key is required"
        
        if not cls.QDRANT_URL:
            errors["QDRANT_URL"] = "Qdrant URL is required"
        
        if not cls.QDRANT_API_KEY:
            errors["QDRANT_API_KEY"] = "Qdrant API key is required"
        
        # SSL warnings
        if not cls.VERIFY_SSL:
            warnings["VERIFY_SSL"] = "SSL verification is disabled - not recommended for production"
        
        return {"errors": errors, "warnings": warnings}
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories."""
        cls.DOWNLOAD_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        
        for school in cls.SCHOOL_CODES:
            (cls.DOWNLOAD_DIR / school).mkdir(exist_ok=True)
    
    @classmethod
    def get_school_full_names(cls) -> Dict[str, str]:
        """Get full names for school codes."""
        return {
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