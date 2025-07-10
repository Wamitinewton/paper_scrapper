"""Parallel PDF processing and embedding generation pipeline."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from collections import deque

from config import Config
from models import ExamPaper, Status, save_papers_data, load_papers_data
from pdf_processor import PDFProcessor
from embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)

class ParallelPipeline:
    """Parallel processing pipeline for PDF processing and embedding generation."""
    
    def __init__(self, 
                 max_processing: int = 5,
                 max_embedding: int = 3,
                 batch_size: int = 50):
        """
        Initialize parallel pipeline.
        
        Args:
            max_processing: Max concurrent PDF processing
            max_embedding: Max concurrent embedding generation
            batch_size: Batch size for processing updates
        """
        self.max_processing = max_processing
        self.max_embedding = max_embedding
        self.batch_size = batch_size
        
        self.processor = PDFProcessor()
        self.embeddings_manager = None
        
        # Processing queues
        self.processing_queue = asyncio.Queue()
        self.embedding_queue = asyncio.Queue()
        self.completed_papers = deque()
        
        # Stats
        self.stats = {
            'total_papers': 0,
            'processed': 0,
            'embedded': 0,
            'failed_processing': 0,
            'failed_embedding': 0,
            'start_time': None
        }
    
    async def _initialize_embeddings(self):
        """Initialize embeddings manager with validation."""
        try:
            Config.validate_embedding_config()
            self.embeddings_manager = EmbeddingsManager()
            logger.info("Embeddings manager initialized")
        except ValueError as e:
            logger.error(f"Embeddings config error: {e}")
            raise
    
    async def _pdf_processor_worker(self, worker_id: int):
        """Worker for PDF processing."""
        while True:
            try:
                paper = await self.processing_queue.get()
                if paper is None:  # Shutdown signal
                    break
                
                processed_paper = await self.processor.process_paper(paper)
                
                if processed_paper.is_processed and processed_paper.chunks:
                    # Queue for embedding
                    await self.embedding_queue.put(processed_paper)
                    self.stats['processed'] += 1
                else:
                    self.stats['failed_processing'] += 1
                
                self.completed_papers.append(processed_paper)
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}")
                self.processing_queue.task_done()
    
    async def _embedding_worker(self, worker_id: int):
        """Worker for embedding generation."""
        while True:
            try:
                paper = await self.embedding_queue.get()
                if paper is None:  # Shutdown signal
                    break
                
                embedded_paper = await self.embeddings_manager.embed_paper(paper)
                
                if embedded_paper.is_embedded:
                    self.stats['embedded'] += 1
                else:
                    self.stats['failed_embedding'] += 1
                
                # Update the paper in completed queue
                for i, completed_paper in enumerate(self.completed_papers):
                    if completed_paper.url == embedded_paper.url:
                        self.completed_papers[i] = embedded_paper
                        break
                
                self.embedding_queue.task_done()
                
            except Exception as e:
                logger.error(f"Embedding worker {worker_id} error: {e}")
                self.embedding_queue.task_done()
    
    async def _progress_monitor(self):
        """Monitor and report progress."""
        last_processed = 0
        last_embedded = 0
        
        while True:
            await asyncio.sleep(30)  # Report every 30 seconds
            
            processed = self.stats['processed']
            embedded = self.stats['embedded']
            
            if processed > last_processed or embedded > last_embedded:
                elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                rate_processing = processed / elapsed * 60 if elapsed > 0 else 0
                rate_embedding = embedded / elapsed * 60 if elapsed > 0 else 0
                
                logger.info(
                    f"Progress: {processed}/{self.stats['total_papers']} processed "
                    f"({rate_processing:.1f}/min), {embedded} embedded ({rate_embedding:.1f}/min)"
                )
                
                last_processed = processed
                last_embedded = embedded
    
    async def _periodic_save(self, output_file: Path):
        """Periodically save progress."""
        while True:
            await asyncio.sleep(300)  # Save every 5 minutes
            
            if self.completed_papers:
                papers_list = list(self.completed_papers)
                save_papers_data(papers_list, output_file)
                logger.info(f"Progress saved: {len(papers_list)} papers")
    
    async def process_papers(self, 
                           papers: List[ExamPaper],
                           output_file: Optional[Path] = None) -> List[ExamPaper]:
        """Process papers with parallel PDF processing and embedding."""
        
        # Filter papers that need processing
        papers_to_process = [
            p for p in papers 
            if p.is_downloaded and (
                p.processing_status != Status.COMPLETED or 
                not p.chunks or 
                len(p.chunks) == 0
            )
        ]
        
        if not papers_to_process:
            logger.info("No papers need processing")
            return papers
        
        # Initialize embeddings
        await self._initialize_embeddings()
        
        # Setup output file
        if not output_file:
            session_id = str(uuid.uuid4())[:8]
            output_file = Config.DATA_DIR / f"processed_papers_{session_id}.json"
        
        # Initialize stats
        self.stats['total_papers'] = len(papers_to_process)
        self.stats['start_time'] = datetime.now()
        
        logger.info(f"Starting parallel processing of {len(papers_to_process)} papers")
        
        # Start workers
        processing_workers = [
            asyncio.create_task(self._pdf_processor_worker(i))
            for i in range(self.max_processing)
        ]
        
        embedding_workers = [
            asyncio.create_task(self._embedding_worker(i))
            for i in range(self.max_embedding)
        ]
        
        # Start monitoring tasks
        monitor_task = asyncio.create_task(self._progress_monitor())
        save_task = asyncio.create_task(self._periodic_save(output_file))
        
        try:
            # Queue papers for processing
            for paper in papers_to_process:
                await self.processing_queue.put(paper)
            
            # Wait for processing to complete
            await self.processing_queue.join()
            
            # Signal processing workers to stop
            for _ in processing_workers:
                await self.processing_queue.put(None)
            await asyncio.gather(*processing_workers)
            
            # Wait for embedding to complete
            await self.embedding_queue.join()
            
            # Signal embedding workers to stop
            for _ in embedding_workers:
                await self.embedding_queue.put(None)
            await asyncio.gather(*embedding_workers)
            
        finally:
            # Cancel monitoring tasks
            monitor_task.cancel()
            save_task.cancel()
            
            try:
                await monitor_task
                await save_task
            except asyncio.CancelledError:
                pass
        
        # Collect results
        processed_papers = list(self.completed_papers)
        
        # Update original papers
        paper_dict = {p.url: p for p in papers}
        for processed_paper in processed_papers:
            paper_dict[processed_paper.url] = processed_paper
        
        final_papers = list(paper_dict.values())
        
        # Final save
        save_papers_data(final_papers, output_file)
        
        # Final stats
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info(
            f"Pipeline completed in {elapsed:.1f}s: "
            f"{self.stats['processed']} processed, "
            f"{self.stats['embedded']} embedded, "
            f"{self.stats['failed_processing']} processing failures, "
            f"{self.stats['failed_embedding']} embedding failures"
        )
        
        return final_papers

# Standalone function for easy usage
async def run_parallel_processing(input_file: Optional[Path] = None,
                                output_file: Optional[Path] = None,
                                schools: Optional[List[str]] = None) -> List[ExamPaper]:
    """Run the complete parallel processing pipeline."""
    
    # Load papers
    if not input_file:
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            input_file = latest_file
        else:
            papers_files = list(Config.DATA_DIR.glob("*papers_*.json"))
            if not papers_files:
                raise FileNotFoundError("No papers data found")
            input_file = max(papers_files, key=lambda p: p.stat().st_mtime)
    
    papers = load_papers_data(input_file)
    
    if not papers:
        raise ValueError("No papers loaded")
    
    # Filter by schools if specified
    if schools:
        papers = [p for p in papers if p.school_code in schools]
        logger.info(f"Filtered to {len(papers)} papers for schools: {schools}")
    
    # Run pipeline
    pipeline = ParallelPipeline(
        max_processing=5,
        max_embedding=3,
        batch_size=50
    )
    
    return await pipeline.process_papers(papers, output_file)