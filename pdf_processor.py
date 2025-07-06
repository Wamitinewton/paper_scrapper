"""PDF text extraction and processing."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import PyPDF2
import pdfplumber

from config import Config
from models import ExamPaper, Status

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF files to extract text content."""
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
    
    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[str, int]:
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
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using pdfplumber (more accurate)."""
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
        """
        Extract text from PDF with fallback methods.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        text, page_count = self._extract_with_pdfplumber(file_path)
        
        if not text and page_count == 0:
            logger.info(f"Falling back to PyPDF2 for {file_path}")
            text, page_count = self._extract_with_pypdf2(file_path)
        
        return text, page_count
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
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
            
            start = end - self.chunk_overlap
            if start >= len(words):
                break
        
        return chunks
    
    def extract_metadata(self, text: str, paper: ExamPaper) -> dict:
        """Extract metadata from text content."""
        metadata = {}
        
        if not text:
            return metadata
        
        text_lower = text.lower()
        
        metadata['text_length'] = len(text)
        metadata['word_count'] = len(text.split())
        
        if any(term in text_lower for term in ['semester', 'trimester', 'quarter']):
            metadata['has_semester_info'] = True
        
        if any(term in text_lower for term in ['exam', 'test', 'quiz', 'assignment']):
            metadata['assessment_type'] = 'exam'
        
        course_codes = re.findall(r'\b[A-Z]{2,4}[-\s]*\d{3,4}\b', text)
        if course_codes:
            metadata['course_codes'] = list(set(course_codes))
        
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            metadata['mentioned_years'] = list(set(years))
        
        question_patterns = len(re.findall(r'\bquestion\s+\d+', text_lower))
        if question_patterns > 0:
            metadata['question_count'] = question_patterns
        
        time_patterns = re.findall(r'\b\d+\s*(hour|hr|minute|min)s?\b', text_lower)
        if time_patterns:
            metadata['has_time_info'] = True
        
        return metadata
    
    async def process_paper(self, paper: ExamPaper) -> ExamPaper:
        """
        Process a single paper to extract text and create chunks.
        
        Args:
            paper: ExamPaper to process
            
        Returns:
            Updated ExamPaper with extracted content
        """
        if not paper.local_path or not paper.local_path.exists():
            paper.processing_status = Status.FAILED
            paper.last_error = "File not found"
            return paper
        
        if not paper.is_downloaded:
            paper.processing_status = Status.FAILED
            paper.last_error = "Paper not downloaded"
            return paper
        
        paper.processing_status = Status.IN_PROGRESS
        
        try:
            loop = asyncio.get_event_loop()
            text, page_count = await loop.run_in_executor(
                None, self.extract_text, paper.local_path
            )
            
            if not text:
                paper.processing_status = Status.FAILED
                paper.last_error = "No text extracted from PDF"
                return paper
            
            chunks = self.create_chunks(text)
            
            metadata = self.extract_metadata(text, paper)
            
            paper.extracted_text = text
            paper.chunks = chunks
            paper.page_count = page_count
            paper.metadata.update(metadata)
            paper.processing_status = Status.COMPLETED
            paper.processed_at = datetime.now()
            
            logger.info(f"Processed {paper.filename}: {len(chunks)} chunks, {page_count} pages")
            
        except Exception as e:
            paper.processing_status = Status.FAILED
            paper.last_error = str(e)
            logger.error(f"Error processing {paper.filename}: {e}")
        
        return paper
    
    async def process_papers(self, papers: List[ExamPaper], max_concurrent: int = 5) -> List[ExamPaper]:
        """
        Process multiple papers concurrently.
        
        Args:
            papers: List of papers to process
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processed papers
        """
        papers_to_process = [
            p for p in papers 
            if p.is_downloaded and p.processing_status != Status.COMPLETED
        ]
        
        if not papers_to_process:
            logger.info("No papers need processing")
            return papers
        
        logger.info(f"Processing {len(papers_to_process)} papers")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(paper):
            async with semaphore:
                return await self.process_paper(paper)
        
        tasks = [process_with_semaphore(paper) for paper in papers_to_process]
        
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processing progress: {i + 1}/{len(papers_to_process)} completed")
        
        paper_dict = {p.url: p for p in papers}
        for processed_paper in results:
            paper_dict[processed_paper.url] = processed_paper
        
        completed = sum(1 for r in results if r.processing_status == Status.COMPLETED)
        failed = sum(1 for r in results if r.processing_status == Status.FAILED)
        
        logger.info(f"Processing completed: {completed} successful, {failed} failed")
        
        return list(paper_dict.values())