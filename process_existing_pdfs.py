"""
Usage:
    python process_existing_pdfs.py
    python process_existing_pdfs.py --schools sci sea --concurrent-processing 8 --concurrent-embedding 4
    python process_existing_pdfs.py --reprocess-failed --test-search
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import uuid
from typing import List, Optional, AsyncGenerator
import time

from config import Config
from models import ExamPaper, Status, save_papers_data, load_papers_data
from pdf_processor import PDFProcessor
from embeddings import EmbeddingsManager

Config.create_directories()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AsyncProcessingPipeline:
    """Manages concurrent PDF processing and embedding generation."""
    
    def __init__(self, max_concurrent_processing: int = 6, max_concurrent_embedding: int = 3):
        self.max_concurrent_processing = max_concurrent_processing
        self.max_concurrent_embedding = max_concurrent_embedding
        self.pdf_processor = PDFProcessor()
        self.embeddings_manager = None
        
        # Processing queues
        self.processing_queue = asyncio.Queue()
        self.embedding_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            "processing_started": 0,
            "processing_completed": 0,
            "processing_failed": 0,
            "embedding_started": 0,
            "embedding_completed": 0,
            "embedding_failed": 0,
            "start_time": None
        }
    
    async def initialize_embeddings(self) -> bool:
        """Initialize embeddings manager with validation."""
        try:
            Config.validate_embedding_config()
            self.embeddings_manager = EmbeddingsManager()
            return True
        except ValueError as e:
            logger.error(f"Embeddings config error: {e}")
            return False
    
    async def process_paper_worker(self, worker_id: int) -> None:
        """Worker for processing PDFs."""
        logger.info(f"PDF processing worker {worker_id} started")
        
        while True:
            try:
                paper = await self.processing_queue.get()
                if paper is None:  # Shutdown signal
                    break
                
                self.stats["processing_started"] += 1
                
                # Process the paper
                processed_paper = await self.pdf_processor.process_paper(paper)
                
                if processed_paper.processing_status == Status.COMPLETED:
                    self.stats["processing_completed"] += 1
                    
                    # Add to embedding queue if embeddings are enabled
                    if self.embeddings_manager:
                        await self.embedding_queue.put(processed_paper)
                else:
                    self.stats["processing_failed"] += 1
                    logger.warning(f"Processing failed for {paper.filename}: {processed_paper.last_error}")
                
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}")
                self.processing_queue.task_done()
    
    async def embedding_worker(self, worker_id: int) -> None:
        """Worker for generating embeddings."""
        if not self.embeddings_manager:
            return
        
        logger.info(f"Embedding worker {worker_id} started")
        
        while True:
            try:
                paper = await self.embedding_queue.get()
                if paper is None:  # Shutdown signal
                    break
                
                self.stats["embedding_started"] += 1
                
                # Generate embeddings
                embedded_paper = await self.embeddings_manager.embed_paper(paper)
                
                if embedded_paper.embedding_status == Status.COMPLETED:
                    self.stats["embedding_completed"] += 1
                else:
                    self.stats["embedding_failed"] += 1
                    logger.warning(f"Embedding failed for {paper.filename}: {embedded_paper.last_error}")
                
                self.embedding_queue.task_done()
                
            except Exception as e:
                logger.error(f"Embedding worker {worker_id} error: {e}")
                self.embedding_queue.task_done()
    
    async def progress_monitor(self, total_papers: int, save_interval: int = 300) -> None:
        """Monitor progress and periodically save results."""
        last_save = time.time()
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Calculate progress
            processed = self.stats["processing_completed"] + self.stats["processing_failed"]
            embedded = self.stats["embedding_completed"] + self.stats["embedding_failed"]
            
            # Log progress
            if processed > 0:
                processing_rate = processed / (time.time() - self.stats["start_time"]) * 60
                logger.info(f"Progress: {processed}/{total_papers} processed ({processing_rate:.1f}/min), "
                          f"{embedded} embedded")
            
            # Periodic save (every 5 minutes by default)
            if time.time() - last_save > save_interval:
                logger.info("Performing periodic save...")
                # This would be implemented to save current state
                last_save = time.time()
            
            # Check if work is complete
            if (processed >= total_papers and 
                self.embedding_queue.empty() and 
                self.processing_queue.empty()):
                break
    
    async def process_papers_pipeline(self, papers: List[ExamPaper], 
                                    enable_embeddings: bool = True,
                                    reprocess_failed: bool = False) -> List[ExamPaper]:
        """Main processing pipeline with concurrent workers."""
        
        # Filter papers that need processing
        if reprocess_failed:
            papers_to_process = [
                p for p in papers 
                if p.is_downloaded and (
                    p.processing_status == Status.FAILED or
                    p.embedding_status == Status.FAILED
                )
            ]
            # Reset failed papers
            for paper in papers_to_process:
                if paper.processing_status == Status.FAILED:
                    paper.processing_status = Status.PENDING
                if paper.embedding_status == Status.FAILED:
                    paper.embedding_status = Status.PENDING
        else:
            papers_to_process = [
                p for p in papers 
                if p.is_downloaded and p.processing_status != Status.COMPLETED
            ]
        
        if not papers_to_process:
            logger.info("No papers need processing")
            return papers
        
        # Initialize embeddings if enabled
        if enable_embeddings:
            if not await self.initialize_embeddings():
                logger.warning("Embeddings disabled due to configuration issues")
                enable_embeddings = False
        
        logger.info(f"Starting pipeline for {len(papers_to_process)} papers")
        logger.info(f"PDF processing workers: {self.max_concurrent_processing}")
        if enable_embeddings:
            logger.info(f"Embedding workers: {self.max_concurrent_embedding}")
        
        self.stats["start_time"] = time.time()
        
        # Start workers
        processing_workers = [
            asyncio.create_task(self.process_paper_worker(i))
            for i in range(self.max_concurrent_processing)
        ]
        
        embedding_workers = []
        if enable_embeddings:
            embedding_workers = [
                asyncio.create_task(self.embedding_worker(i))
                for i in range(self.max_concurrent_embedding)
            ]
        
        # Start progress monitor
        monitor_task = asyncio.create_task(
            self.progress_monitor(len(papers_to_process))
        )
        
        # Add papers to processing queue
        for paper in papers_to_process:
            await self.processing_queue.put(paper)
        
        # Wait for processing to complete
        await self.processing_queue.join()
        
        # Signal processing workers to stop
        for _ in processing_workers:
            await self.processing_queue.put(None)
        await asyncio.gather(*processing_workers)
        
        # Wait for embedding queue to empty if embeddings enabled
        if enable_embeddings:
            await self.embedding_queue.join()
            
            # Signal embedding workers to stop
            for _ in embedding_workers:
                await self.embedding_queue.put(None)
            await asyncio.gather(*embedding_workers)
        
        # Stop monitor
        monitor_task.cancel()
        
        # Calculate final statistics
        total_time = time.time() - self.stats["start_time"]
        
        logger.info("Pipeline completed")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Processing: {self.stats['processing_completed']} successful, "
                   f"{self.stats['processing_failed']} failed")
        
        if enable_embeddings:
            logger.info(f"Embeddings: {self.stats['embedding_completed']} successful, "
                       f"{self.stats['embedding_failed']} failed")
        
        return papers

def load_papers_data_safely() -> Optional[List[ExamPaper]]:
    """Load papers data with error handling."""
    try:
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            logger.info(f"Loading from latest: {latest_file}")
            return load_papers_data(latest_file)
        
        # Find most recent file
        papers_files = list(Config.DATA_DIR.glob("*papers_*.json"))
        if papers_files:
            latest_file = max(papers_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading from: {latest_file}")
            return load_papers_data(latest_file)
        
        logger.error("No papers data found - run scraping first")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load papers data: {e}")
        return None

async def test_search_functionality(embeddings_manager: EmbeddingsManager):
    """Test search with sample queries."""
    test_queries = [
        "public relations management functions",
        "database design normalization",
        "calculus integration methods",
        "business communication strategies",
        "software engineering principles"
    ]
    
    logger.info("Testing search functionality...")
    
    for query in test_queries:
        try:
            results = await embeddings_manager.search_similar(query, limit=3, score_threshold=0.75)
            
            if results:
                logger.info(f"Query '{query}': {len(results)} results")
                for i, result in enumerate(results, 1):
                    metadata = result["metadata"]
                    logger.info(f"  {i}. {metadata.get('course_code', 'Unknown')} - "
                               f"{metadata.get('paper_filename', 'Unknown')} (Score: {result['score']:.3f})")
            else:
                logger.info(f"Query '{query}': No results")
        except Exception as e:
            logger.error(f"Search test failed for '{query}': {e}")

def print_final_summary(papers: List[ExamPaper]):
    """Print comprehensive processing summary."""
    total = len(papers)
    downloaded = sum(1 for p in papers if p.is_downloaded)
    processed = sum(1 for p in papers if p.is_processed)
    embedded = sum(1 for p in papers if p.is_embedded)
    total_chunks = sum(len(p.chunks) for p in papers if p.chunks)
    
    print("\n" + "="*70)
    print("FINAL PROCESSING SUMMARY")
    print("="*70)
    print(f"Total papers: {total}")
    print(f"Downloaded: {downloaded}")
    print(f"Successfully processed: {processed}")
    print(f"Successfully embedded: {embedded}")
    print(f"Total chunks generated: {total_chunks}")
    
    if downloaded > 0:
        print(f"Processing success rate: {(processed/downloaded*100):.1f}%")
    if processed > 0:
        print(f"Embedding success rate: {(embedded/processed*100):.1f}%")
    
    print("\nSchool breakdown:")
    school_stats = {}
    for paper in papers:
        if paper.school_code not in school_stats:
            school_stats[paper.school_code] = {"processed": 0, "embedded": 0}
        if paper.is_processed:
            school_stats[paper.school_code]["processed"] += 1
        if paper.is_embedded:
            school_stats[paper.school_code]["embedded"] += 1
    
    for school_code, stats in sorted(school_stats.items()):
        school_name = Config.SCHOOL_CODES.get(school_code, school_code)
        print(f"  {school_name}: {stats['processed']} processed, {stats['embedded']} embedded")
    
    print("="*70)

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced PDF processing and embedding pipeline")
    
    parser.add_argument("--schools", nargs="+", choices=list(Config.SCHOOL_CODES.keys()),
                       help="Process specific schools only")
    parser.add_argument("--concurrent-processing", type=int, default=6,
                       help="Max concurrent PDF processing (default: 6)")
    parser.add_argument("--concurrent-embedding", type=int, default=3,
                       help="Max concurrent embedding generation (default: 3)")
    parser.add_argument("--skip-embeddings", action="store_true",
                       help="Skip embedding generation")
    parser.add_argument("--reprocess-failed", action="store_true",
                       help="Reprocess failed papers only")
    parser.add_argument("--test-search", action="store_true",
                       help="Test search functionality after completion")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ENHANCED PDF PROCESSING & EMBEDDING PIPELINE")
    print("="*70)
    print(f"PDF processing workers: {args.concurrent_processing}")
    print(f"Embedding workers: {args.concurrent_embedding}")
    print(f"Skip embeddings: {args.skip_embeddings}")
    print("="*70)
    
    # Load papers data
    papers = load_papers_data_safely()
    if not papers:
        return
    
    # Filter by schools if specified
    if args.schools:
        papers = [p for p in papers if p.school_code in args.schools]
        logger.info(f"Filtered to {len(papers)} papers for schools: {args.schools}")
    
    try:
        # Initialize pipeline
        pipeline = AsyncProcessingPipeline(
            max_concurrent_processing=args.concurrent_processing,
            max_concurrent_embedding=args.concurrent_embedding
        )
        
        # Run processing pipeline
        papers = await pipeline.process_papers_pipeline(
            papers,
            enable_embeddings=not args.skip_embeddings,
            reprocess_failed=args.reprocess_failed
        )
        
        # Save results
        session_id = str(uuid.uuid4())
        output_file = Config.DATA_DIR / f"enhanced_papers_{session_id}.json"
        save_papers_data(papers, output_file)
        
        # Update latest symlink
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(output_file.name)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Test search if requested
        if args.test_search and not args.skip_embeddings:
            try:
                Config.validate_embedding_config()
                embeddings_manager = EmbeddingsManager()
                await test_search_functionality(embeddings_manager)
            except Exception as e:
                logger.error(f"Search test failed: {e}")
        
        print_final_summary(papers)
        
        embedded_count = sum(1 for p in papers if p.is_embedded)
        if embedded_count > 0:
            print(f"\n✅ Successfully processed and embedded {embedded_count} papers!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

