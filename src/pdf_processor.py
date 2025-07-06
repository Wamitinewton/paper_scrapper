
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
import re
from datetime import datetime

import PyPDF2
import pdfplumber
from tqdm.asyncio import tqdm

from .config import Config
from .models import ExamPaper, ProcessingStatus
from .logger import get_logger

logger = get_logger(__name__)

class PDFProcessor:
    """Process PDF files to extract text content."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """Initialize the PDF processor."""
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _extract_text_pypdf2(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                page_count = len(reader.pages)
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(self._clean_text(text))
                
                return ' '.join(text_parts), page_count
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path}: {e}")
            return "", 0
    
    def _extract_text_pdfplumber(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using pdfplumber (more accurate but slower)."""
        try:
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(self._clean_text(text))
                
                return ' '.join(text_parts), page_count
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path}: {e}")
            return "", 0
    
    def extract_text(self, file_path: Path) -> Tuple[str, int]:
        """Extract text from PDF file using fallback methods."""
        # Try pdfplumber first (more accurate)
        text, page_count = self._extract_text_pdfplumber(file_path)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text and page_count == 0:
            logger.info(f"Falling back to PyPDF2 for {file_path}")
            text, page_count = self._extract_text_pypdf2(file_path)
        
        return text, page_count
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(words):
                break
        
        return chunks
    
    async def process_paper(self, paper: ExamPaper) -> ExamPaper:
        """Process a single paper to extract text and create chunks."""
        if not paper.file_path or not paper.file_path.exists():
            paper.processing_status = ProcessingStatus.FAILED
            paper.processing_error = "File not found"
            return paper
        
        paper.processing_status = ProcessingStatus.PROCESSING
        
        try:
            # Extract text in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            text, page_count = await loop.run_in_executor(
                None, self.extract_text, paper.file_path
            )
            
            if not text:
                paper.processing_status = ProcessingStatus.FAILED
                paper.processing_error = "No text extracted from PDF"
                return paper
            
            # Create chunks
            chunks = self.create_chunks(text)
            
            # Update paper with extracted information
            paper.extracted_text = text
            paper.page_count = page_count
            paper.chunks = chunks
            paper.processing_status = ProcessingStatus.COMPLETED
            paper.processed_at = datetime.now()
            
            # Add metadata
            paper.metadata.update({
                'text_length': len(text),
                'word_count': len(text.split()),
                'chunk_count': len(chunks),
                'extraction_method': 'pdfplumber_pypdf2_fallback'
            })
            
            logger.info(f"Processed {paper.filename}: {len(chunks)} chunks, {page_count} pages")
            
        except Exception as e:
            paper.processing_status = ProcessingStatus.FAILED
            paper.processing_error = str(e)
            logger.error(f"Error processing {paper.filename}: {e}")
        
        return paper
    
    async def process_papers(self, papers: List[ExamPaper]) -> List[ExamPaper]:
        """Process multiple papers concurrently."""
        if not papers:
            return []
        
        # Filter papers that need processing
        papers_to_process = [
            p for p in papers 
            if p.is_downloaded and p.processing_status != ProcessingStatus.COMPLETED
        ]
        
        if not papers_to_process:
            logger.info("No papers need processing")
            return papers
        
        logger.info(f"Processing {len(papers_to_process)} papers")
        
        # Process papers with progress bar
        tasks = [self.process_paper(paper) for paper in papers_to_process]
        results = await tqdm.gather(*tasks, desc="Processing PDFs")
        
        # Update papers list
        paper_dict = {p.url: p for p in papers}
        for processed_paper in results:
            paper_dict[processed_paper.url] = processed_paper
        
        # Count results
        completed = sum(1 for r in results if r.processing_status == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.processing_status == ProcessingStatus.FAILED)
        
        logger.info(f"Processing completed: {completed} successful, {failed} failed")
        
        return list(paper_dict.values())
    
    async def extract_metadata(self, paper: ExamPaper) -> dict:
        """Extract additional metadata from the paper."""
        metadata = paper.metadata.copy()
        
        if not paper.extracted_text:
            return metadata
        
        text = paper.extracted_text.lower()
        
        # Extract academic information
        if 'semester' in text:
            metadata['has_semester_info'] = True
        
        if any(term in text for term in ['exam', 'test', 'quiz', 'assignment']):
            metadata['assessment_type'] = 'exam'
        
        course_codes = re.findall(r'\b[A-Z]{2,4}\s*\d{3,4}\b', paper.extracted_text)
        if course_codes:
            metadata['course_codes'] = list(set(course_codes))
        
        # Extract years mentioned in text
        years = re.findall(r'\b(19|20)\d{2}\b', paper.extracted_text)
        if years:
            metadata['mentioned_years'] = list(set(years))
        
        # Extract question patterns
        question_patterns = len(re.findall(r'\bquestion\s+\d+', text))
        if question_patterns > 0:
            metadata['question_count'] = question_patterns
        
        return metadata