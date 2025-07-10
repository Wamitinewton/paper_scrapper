"""Enhanced PDF text extraction and processing for clean embeddings."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

import PyPDF2
import pdfplumber

from config import Config
from models import ExamPaper, Status

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF files to extract clean text content optimized for embeddings."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def _remove_university_headers(self, text: str) -> str:
        """Remove MUST university specific headers and footers."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # University-specific patterns to remove
        skip_patterns = [
            r'MERU UNIVERSITY OF SCIENCE AND TECHNOLOGY',
            r'Meru University of Science & Technology',
            r'P\.O\. Box \d+-\d+ – Meru-Kenya',
            r'Tel: \+254.*',
            r'Website: .*must\.ac\.ke.*',
            r'Email: .*must\.ac\.ke.*',
            r'University Examinations \d{4}/\d{4}',
            r'.*ISO 9001:2015.*ISO/IEC 27001:2013.*',
            r'Foundation of Innovations',
            r'Page \d+',
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^\s*Page \d+ of \d+\s*$',
            r'.*Certified\s*$',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip university header/footer lines
            should_skip = any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns)
            
            if not should_skip:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_question_structure(self, text: str) -> str:
        """Normalize question formatting for better chunking."""
        if not text:
            return ""
        
        # Standardize question headers
        text = re.sub(r'QUESTION\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)', 
                     r'QUESTION \1', text, flags=re.IGNORECASE)
        
        # Standardize numbered questions
        text = re.sub(r'QUESTION\s+(\d+)', r'QUESTION \1', text, flags=re.IGNORECASE)
        
        # Clean up sub-question formatting
        text = re.sub(r'^\s*([a-z])\)\s*', r'\1) ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+)\.\s*', r'\1. ', text, flags=re.MULTILINE)
        
        # Normalize mark allocations
        text = re.sub(r'\(\s*(\d+)\s+[Mm]arks?\s*\)', r'(\1 Marks)', text)
        text = re.sub(r'\(\s*(\d+)\s+[Mm]ark\s*\)', r'(\1 Mark)', text)
        
        return text
    
    def _extract_course_metadata(self, text: str) -> Dict[str, Any]:
        """Extract course-specific metadata from exam paper."""
        metadata = {}
        
        if not text:
            return metadata
        
        # Extract course code (e.g., BCJ 3152, CSC 2101)
        course_match = re.search(r'\b([A-Z]{2,4})\s*(\d{4})\b', text)
        if course_match:
            metadata['course_prefix'] = course_match.group(1)
            metadata['course_number'] = course_match.group(2)
            metadata['course_code'] = f"{course_match.group(1)} {course_match.group(2)}"
        
        # Extract course name
        course_name_match = re.search(r'[A-Z]{2,4}\s*\d{4}:\s*([A-Z\s]+)', text)
        if course_name_match:
            metadata['course_name'] = course_name_match.group(1).strip()
        
        # Extract exam duration
        time_match = re.search(r'TIME:\s*(\d+(?:\.\d+)?)\s*HOURS?', text, re.IGNORECASE)
        if time_match:
            metadata['exam_duration_hours'] = float(time_match.group(1))
        
        # Extract degree program
        degree_match = re.search(r'DEGREE OF\s+([A-Z\s]+?)(?:\n|\s+[A-Z]{2,4})', text)
        if degree_match:
            metadata['degree_program'] = degree_match.group(1).strip()
        
        # Extract semester information
        semester_match = re.search(r'(FIRST|SECOND|THIRD)\s+(?:YEAR\s+)?(FIRST|SECOND|THIRD)\s+SEMESTER', text, re.IGNORECASE)
        if semester_match:
            metadata['academic_year'] = semester_match.group(1).lower()
            metadata['semester'] = semester_match.group(2).lower()
        
        # Extract year information
        year_match = re.search(r'(FIRST|SECOND|THIRD|FOURTH)\s+YEAR', text, re.IGNORECASE)
        if year_match:
            metadata['study_year'] = year_match.group(1).lower()
        
        # Count total questions
        question_count = len(re.findall(r'QUESTION\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+)', text, re.IGNORECASE))
        if question_count > 0:
            metadata['total_questions'] = question_count
        
        # Extract instructions
        instructions_match = re.search(r'INSTRUCTIONS?:\s*([^\n]+)', text, re.IGNORECASE)
        if instructions_match:
            metadata['exam_instructions'] = instructions_match.group(1).strip()
        
        return metadata
    
    def _clean_content_text(self, text: str) -> str:
        """Clean the main content while preserving structure."""
        if not text:
            return ""
        
        # Remove university headers first
        text = self._remove_university_headers(text)
        
        # Normalize question structure
        text = self._normalize_question_structure(text)
        
        # Clean up excessive whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Clean PDF artifacts
        text = re.sub(r'[.]{4,}', '...', text)
        text = re.sub(r'[-]{4,}', '---', text)
        text = re.sub(r'_{4,}', '___', text)
        
        # Remove standalone page numbers
        text = re.sub(r'\n\d+\n', '\n', text)
        
        return text.strip()
    
    def _create_semantic_chunks(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Create chunks that preserve semantic meaning and context."""
        if not text:
            return []
        
        chunks = []
        
        # Split by major questions first
        question_pattern = r'(?=QUESTION\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+))'
        question_sections = re.split(question_pattern, text, flags=re.IGNORECASE)
        
        # Add course context to each chunk
        course_context = ""
        if 'course_code' in metadata and 'course_name' in metadata:
            course_context = f"Course: {metadata['course_code']} - {metadata['course_name']}\n"
        
        for section in question_sections:
            section = section.strip()
            if len(section) < 30:  # Skip very short sections
                continue
            
            # Add course context to maintain context across chunks
            section_with_context = course_context + section
            
            words = section_with_context.split()
            
            if len(words) <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(section_with_context)
            else:
                # Split large sections while preserving sub-questions
                sub_chunks = self._split_preserving_subquestions(section_with_context)
                chunks.extend(sub_chunks)
        
        # Filter and validate chunks
        valid_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.strip()
            word_count = len(cleaned_chunk.split())
            
            # Only keep substantial chunks with meaningful content
            if (word_count >= 15 and 
                len(cleaned_chunk) >= 50 and
                not re.match(r'^[\s\d\.\-_]+$', cleaned_chunk)):
                valid_chunks.append(cleaned_chunk)
        
        return valid_chunks
    
    def _split_preserving_subquestions(self, text: str) -> List[str]:
        """Split text while preserving sub-question boundaries."""
        # Try to split at sub-question boundaries
        subquestion_pattern = r'(?=^[a-z]\)|\n\d+\.|\n[a-z]\))'
        parts = re.split(subquestion_pattern, text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            test_chunk = f"{current_chunk}\n{part}".strip()
            test_words = test_chunk.split()
            
            if len(test_words) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If part is still too large, split it further
                if len(part.split()) > self.chunk_size:
                    chunks.extend(self._split_by_sentences(part))
                    current_chunk = ""
                else:
                    current_chunk = part
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split large text by sentences with overlap."""
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            test_chunk = f"{current_chunk} {sentence}".strip()
            
            if len(test_chunk.split()) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Tuple[str, int]:
        """Primary extraction method using pdfplumber."""
        try:
            text_parts = []
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text(
                            x_tolerance=2,
                            y_tolerance=2,
                            layout=True,
                            x_density=7.25,
                            y_density=13
                        )
                        
                        if page_text and len(page_text.strip()) > 20:
                            text_parts.append(page_text)
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page.page_number} from {file_path.name}: {e}")
                        continue
            
            return '\n\n'.join(text_parts), page_count
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path.name}: {e}")
            return "", 0
    
    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[str, int]:
        """Fallback extraction using PyPDF2."""
        try:
            text_parts = []
            page_count = 0
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 20:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"PyPDF2 page extraction failed: {e}")
                        continue
            
            return '\n\n'.join(text_parts), page_count
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path.name}: {e}")
            return "", 0
    
    def extract_text(self, file_path: Path) -> Tuple[str, int]:
        """Extract text with fallback strategy."""
        # Try pdfplumber first
        text, page_count = self._extract_with_pdfplumber(file_path)
        
        # Fallback to PyPDF2 if extraction insufficient
        if not text or len(text.strip()) < 100:
            logger.debug(f"Using PyPDF2 fallback for {file_path.name}")
            text, page_count = self._extract_with_pypdf2(file_path)
        
        return text, page_count
    
    async def process_paper(self, paper: ExamPaper) -> ExamPaper:
        """Process a single paper for optimized embeddings."""
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
            raw_text, page_count = await loop.run_in_executor(None, self.extract_text, paper.local_path)
            
            if not raw_text or len(raw_text.strip()) < 50:
                paper.processing_status = Status.FAILED
                paper.last_error = f"Insufficient text extracted: {len(raw_text) if raw_text else 0} chars"
                return paper
            
            # Extract metadata before cleaning
            metadata = self._extract_course_metadata(raw_text)
            
            # Clean and normalize text
            cleaned_text = self._clean_content_text(raw_text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 30:
                paper.processing_status = Status.FAILED
                paper.last_error = "No meaningful content after cleaning"
                return paper
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(cleaned_text, metadata)
            
            if not chunks:
                paper.processing_status = Status.FAILED
                paper.last_error = "No valid chunks created"
                return paper
            
            # Update paper with results
            paper.extracted_text = cleaned_text
            paper.chunks = chunks
            paper.page_count = page_count
            paper.metadata.update(metadata)
            paper.processing_status = Status.COMPLETED
            paper.processed_at = datetime.now()
            
            logger.info(f"Processed {paper.filename}: {len(chunks)} chunks, {len(cleaned_text)} chars")
            
        except Exception as e:
            paper.processing_status = Status.FAILED
            paper.last_error = str(e)
            logger.error(f"Processing failed for {paper.filename}: {e}")
        
        return paper
    
    async def process_papers(self, papers: List[ExamPaper], max_concurrent: int = 5) -> List[ExamPaper]:
        """Process multiple papers with controlled concurrency."""
        papers_to_process = [
            p for p in papers 
            if p.is_downloaded and p.processing_status != Status.COMPLETED
        ]
        
        if not papers_to_process:
            return papers
        
        logger.info(f"Processing {len(papers_to_process)} papers")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(paper):
            async with semaphore:
                return await self.process_paper(paper)
        
        tasks = [process_with_semaphore(paper) for paper in papers_to_process]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        processed_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Processing task failed: {result}")
            else:
                processed_papers.append(result)
        
        # Update original papers list
        paper_dict = {p.url: p for p in papers}
        for processed_paper in processed_papers:
            paper_dict[processed_paper.url] = processed_paper
        
        successful = sum(1 for p in processed_papers if p.processing_status == Status.COMPLETED)
        failed = sum(1 for p in processed_papers if p.processing_status == Status.FAILED)
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        return list(paper_dict.values())