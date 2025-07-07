"""Enhanced PDF text extraction and processing for clean embeddings."""

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
    """Process PDF files to extract clean text content for embeddings."""
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Size of text chunks in words
            chunk_overlap: Overlap between chunks in words
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers from exam papers."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Common header/footer patterns for MUST papers
        skip_patterns = [
            r'MERU UNIVERSITY OF SCIENCE AND TECHNOLOGY',
            r'P\.O\. Box \d+-\d+ – Meru-Kenya',
            r'Tel: \+254.*',
            r'Website: .*',
            r'Email: .*',
            r'University Examinations \d{4}/\d{4}',
            r'Meru University of Science & Technology is ISO.*',
            r'Foundation of Innovations',
            r'Page \d+',
            r'^\d+$',  # Page numbers alone
            r'^\s*Page \d+ of \d+\s*$',
            r'^\s*\d+\s*/\s*\d+\s*$',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip if line matches any header/footer pattern
            skip_line = False
            for pattern in skip_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    skip_line = True
                    break
            
            if not skip_line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove headers and footers first
        text = self._remove_headers_footers(text)
        
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[.]{4,}', '...', text)  # Excessive dots
        text = re.sub(r'[-]{4,}', '---', text)  # Excessive dashes
        text = re.sub(r'_{4,}', '___', text)    # Excessive underscores
        
        # Remove standalone numbers that are likely page numbers
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Clean up exam metadata lines but preserve important info
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Keep important exam information but clean it up
            if any(keyword in line.upper() for keyword in 
                   ['QUESTION', 'ANSWER', 'MARKS', 'TIME', 'INSTRUCTIONS', 'EXAMINATION FOR']):
                cleaned_lines.append(line)
            elif re.match(r'^[A-Z\s]{10,}$', line):  # Skip all-caps headers
                continue
            elif len(line) > 20:  # Keep substantial content lines
                cleaned_lines.append(line)
            elif re.match(r'^[a-z]\)', line) or re.match(r'^\d+\.', line):  # Keep list items
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using pdfplumber with better accuracy."""
        try:
            text_parts = []
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text with custom settings
                        page_text = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=True,
                            x_density=7.25,
                            y_density=13
                        )
                        
                        if page_text:
                            cleaned_text = self._clean_text(page_text)
                            if cleaned_text and len(cleaned_text.strip()) > 50:  # Only add substantial content
                                text_parts.append(cleaned_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1} from {file_path}: {e}")
                        continue
                
            full_text = '\n\n'.join(text_parts)
            return full_text, page_count
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path}: {e}")
            return "", 0
    
    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using PyPDF2 as fallback."""
        try:
            text_parts = []
            page_count = 0
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self._clean_text(page_text)
                            if cleaned_text and len(cleaned_text.strip()) > 50:
                                text_parts.append(cleaned_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1} from {file_path}: {e}")
                        continue
                
            full_text = '\n\n'.join(text_parts)
            return full_text, page_count
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path}: {e}")
            return "", 0
    
    def extract_text(self, file_path: Path) -> Tuple[str, int]:
        """
        Extract text from PDF with fallback methods.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        # Try pdfplumber first (more accurate)
        text, page_count = self._extract_with_pdfplumber(file_path)
        
        # Fallback to PyPDF2 if needed
        if not text or len(text.strip()) < 100:
            logger.info(f"Falling back to PyPDF2 for {file_path.name}")
            text, page_count = self._extract_with_pypdf2(file_path)
        
        # Final validation
        if text and len(text.strip()) < 50:
            logger.warning(f"Extracted text too short for {file_path.name}: {len(text)} chars")
            return "", page_count
            
        return text, page_count
    
    def _create_question_aware_chunks(self, text: str) -> List[str]:
        """Create chunks that respect question boundaries for better context."""
        if not text:
            return []
        
        # First, try to split by questions
        question_pattern = r'(?=QUESTION\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|\d+))'
        question_sections = re.split(question_pattern, text, flags=re.IGNORECASE)
        
        chunks = []
        
        for section in question_sections:
            section = section.strip()
            if not section or len(section) < 50:
                continue
                
            words = section.split()
            
            if len(words) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(section)
            else:
                # Section needs to be split further
                # Try to split at sub-questions or parts
                subquestion_pattern = r'(?=^[a-z]\)|\n\d+\.|\n[a-z]\))'
                subsections = re.split(subquestion_pattern, section, flags=re.MULTILINE)
                
                current_chunk = ""
                
                for subsection in subsections:
                    subsection = subsection.strip()
                    if not subsection:
                        continue
                    
                    # Check if adding this subsection would exceed chunk size
                    test_chunk = f"{current_chunk}\n{subsection}".strip()
                    test_words = test_chunk.split()
                    
                    if len(test_words) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = subsection
                        
                        # If even the subsection is too large, split it
                        if len(subsection.split()) > self.chunk_size:
                            sub_chunks = self._split_large_text(subsection)
                            chunks.extend(sub_chunks[:-1])  # Add all but last
                            current_chunk = sub_chunks[-1] if sub_chunks else ""
                
                # Add final chunk
                if current_chunk:
                    chunks.append(current_chunk)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_large_text(self, text: str) -> List[str]:
        """Split large text into overlapping chunks."""
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
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into context-aware chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Use question-aware chunking for better context preservation
        chunks = self._create_question_aware_chunks(text)
        
        # Filter out very short chunks
        valid_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 30 and len(chunk.split()) > 10:
                valid_chunks.append(chunk.strip())
        
        return valid_chunks
    
    def extract_metadata(self, text: str, paper: ExamPaper) -> dict:
        """Extract relevant metadata from text content."""
        metadata = {}
        
        if not text:
            return metadata
        
        text_lower = text.lower()
        
        # Basic text statistics
        metadata['text_length'] = len(text)
        metadata['word_count'] = len(text.split())
        
        # Extract course codes
        course_codes = re.findall(r'\b[A-Z]{2,4}[-\s]*\d{3,4}\b', text)
        if course_codes:
            metadata['course_codes'] = list(set(course_codes))
        
        # Extract time information
        time_matches = re.findall(r'TIME:\s*(\d+)\s*HOUR', text, re.IGNORECASE)
        if time_matches:
            metadata['exam_duration_hours'] = int(time_matches[0])
        
        # Count questions
        question_count = len(re.findall(r'QUESTION\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|\d+)', text, re.IGNORECASE))
        if question_count > 0:
            metadata['question_count'] = question_count
        
        # Extract degree information
        degree_match = re.search(r'DEGREE OF\s+([A-Z\s]+)', text, re.IGNORECASE)
        if degree_match:
            metadata['degree_program'] = degree_match.group(1).strip()
        
        # Extract semester information
        semester_match = re.search(r'(FIRST|SECOND|THIRD)\s+SEMESTER', text, re.IGNORECASE)
        if semester_match:
            metadata['semester'] = semester_match.group(1).lower()
        
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
            # Extract text in executor to avoid blocking
            loop = asyncio.get_event_loop()
            text, page_count = await loop.run_in_executor(
                None, self.extract_text, paper.local_path
            )
            
            if not text or len(text.strip()) < 100:
                paper.processing_status = Status.FAILED
                paper.last_error = f"Insufficient text extracted: {len(text) if text else 0} chars"
                return paper
            
            # Create chunks
            chunks = self.create_chunks(text)
            
            if not chunks:
                paper.processing_status = Status.FAILED
                paper.last_error = "No valid chunks created"
                return paper
            
            # Extract metadata
            metadata = self.extract_metadata(text, paper)
            
            # Update paper
            paper.extracted_text = text
            paper.chunks = chunks
            paper.page_count = page_count
            paper.metadata.update(metadata)
            paper.processing_status = Status.COMPLETED
            paper.processed_at = datetime.now()
            
            logger.info(f"Processed {paper.filename}: {len(chunks)} chunks from {len(text)} chars")
            
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
        completed_count = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed_count += 1
            
            if completed_count % 100 == 0:
                logger.info(f"Progress: {completed_count}/{len(papers_to_process)} completed")
        
        # Update original papers
        paper_dict = {p.url: p for p in papers}
        for processed_paper in results:
            paper_dict[processed_paper.url] = processed_paper
        
        successful = sum(1 for r in results if r.processing_status == Status.COMPLETED)
        failed = sum(1 for r in results if r.processing_status == Status.FAILED)
        
        logger.info(f"Processing completed: {successful} successful, {failed} failed")
        
        return list(paper_dict.values())