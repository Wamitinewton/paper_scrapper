
import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .config import Config
from .logger import setup_logging, get_logger
from .scraper import ExamPaperScraper
from .pdf_processor import PDFProcessor
from .embeddings import EmbeddingsManager
from .models import ExamPaper

# Setup logging
setup_logging()
logger = get_logger(__name__)

class ExamPaperPipeline:
    """Complete pipeline for scraping, processing, and embedding exam papers."""
    
    def __init__(self, verify_ssl: bool = True):
        """Initialize the pipeline."""
        self.verify_ssl = verify_ssl
        self.scraper = None
        self.processor = PDFProcessor()
        self.embeddings_manager = EmbeddingsManager()
        
        config_validation = Config.validate()
        
        if config_validation["errors"]:
            logger.error("Configuration validation failed:")
            for field, error in config_validation["errors"].items():
                logger.error(f"  {field}: {error}")
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        if config_validation["warnings"]:
            logger.warning("Configuration warnings:")
            for field, warning in config_validation["warnings"].items():
                logger.warning(f"  {field}: {warning}")
    
    async def scrape_papers(self, 
                           schools: Optional[List[str]] = None,
                           years: Optional[List[int]] = None,
                           retry_failed: bool = True) -> List[ExamPaper]:
        """Scrape exam papers from the website."""
        
        logger.info("Starting paper scraping phase")
        logger.info(f"SSL verification: {'enabled' if self.verify_ssl else 'disabled'}")
        
        async with ExamPaperScraper(verify_ssl=self.verify_ssl) as scraper:
            self.scraper = scraper
            
            papers, session = await scraper.scrape_all(schools, years, verify_ssl=self.verify_ssl)
            
            if retry_failed and session.total_papers_failed > 0:
                logger.info(f"Retrying {session.total_papers_failed} failed downloads")
                papers = await scraper.retry_failed_downloads(papers)
            
            self._save_session_summary(session, "scraping_session.json")
            
            return papers
    
    async def process_papers(self, papers: List[ExamPaper]) -> List[ExamPaper]:
        """Process PDFs to extract text and create chunks."""
        
        logger.info("Starting PDF processing phase")
        
        processed_papers = await self.processor.process_papers(papers)
        
        self._save_processing_summary(processed_papers, "processing_summary.json")
        
        return processed_papers
    
    async def generate_embeddings(self, papers: List[ExamPaper]) -> List[ExamPaper]:
        """Generate embeddings and store in vector database."""
        
        logger.info("Starting embeddings generation phase")
        
        embedded_papers = await self.embeddings_manager.process_papers_embeddings(papers)
        
        self._save_embeddings_summary(embedded_papers, "embeddings_summary.json")
        
        return embedded_papers
    
    async def run_complete_pipeline(self, 
                                  schools: Optional[List[str]] = None,
                                  years: Optional[List[int]] = None,
                                  skip_scraping: bool = False,
                                  skip_processing: bool = False,
                                  skip_embeddings: bool = False) -> List[ExamPaper]:
        """Run the complete pipeline."""
        
        logger.info("Starting complete exam papers processing pipeline")
        start_time = datetime.now()
        
        papers = []
        
        try:
            if not skip_scraping:
                papers = await self.scrape_papers(schools, years)
                logger.info(f"Scraping completed: {len(papers)} papers")
            else:
                papers = self._load_existing_papers()
                logger.info(f"Loaded existing papers: {len(papers)}")
            
            if not skip_processing:
                papers = await self.process_papers(papers)
                processed_count = sum(1 for p in papers if p.is_processed)
                logger.info(f"Processing completed: {processed_count} papers processed")
            
            if not skip_embeddings:
                papers = await self.generate_embeddings(papers)
                embedded_count = sum(1 for p in papers if p.embedding_id)
                logger.info(f"Embeddings completed: {embedded_count} papers embedded")
            
            # Save final results
            self._save_papers_data(papers, "final_papers.json")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            
            # Print final summary
            self._print_final_summary(papers, duration)
            
            return papers
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _save_session_summary(self, session, filename: str) -> None:
        """Save scraping session summary."""
        output_file = Config.LOGS_DIR / filename
        with open(output_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
        logger.info(f"Session summary saved to {output_file}")
    
    def _save_processing_summary(self, papers: List[ExamPaper], filename: str) -> None:
        """Save processing summary."""
        summary = {
            "total_papers": len(papers),
            "processed_papers": sum(1 for p in papers if p.is_processed),
            "failed_processing": sum(1 for p in papers if p.processing_status.name == "FAILED"),
            "total_chunks": sum(len(p.chunks) for p in papers),
            "total_pages": sum(p.page_count or 0 for p in papers),
            "processing_time": datetime.now().isoformat()
        }
        
        output_file = Config.LOGS_DIR / filename
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Processing summary saved to {output_file}")
    
    def _save_embeddings_summary(self, papers: List[ExamPaper], filename: str) -> None:
        """Save embeddings summary."""
        summary = {
            "total_papers": len(papers),
            "embedded_papers": sum(1 for p in papers if p.embedding_id),
            "total_embeddings": sum(len(p.chunks) for p in papers if p.embedding_id),
            "collection_stats": self.embeddings_manager.get_collection_stats(),
            "embedding_time": datetime.now().isoformat()
        }
        
        output_file = Config.LOGS_DIR / filename
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Embeddings summary saved to {output_file}")
    
    def _save_papers_data(self, papers: List[ExamPaper], filename: str) -> None:
        """Save papers data."""
        papers_data = [paper.to_dict() for paper in papers]
        
        output_file = Config.LOGS_DIR / filename
        with open(output_file, 'w') as f:
            json.dump(papers_data, f, indent=2)
        logger.info(f"Papers data saved to {output_file}")
    
    def _load_existing_papers(self) -> List[ExamPaper]:
        """Load existing papers from saved data."""
        return []
    
    def _print_final_summary(self, papers: List[ExamPaper], duration: float) -> None:
        """Print final pipeline summary."""
        downloaded = sum(1 for p in papers if p.is_downloaded)
        processed = sum(1 for p in papers if p.is_processed)
        embedded = sum(1 for p in papers if p.embedding_id)
        total_chunks = sum(len(p.chunks) for p in papers)
        
        print("\n" + "="*60)
        print("EXAM PAPERS PIPELINE SUMMARY")
        print("="*60)
        print(f"Total papers found: {len(papers)}")
        print(f"Papers downloaded: {downloaded}")
        print(f"Papers processed: {processed}")
        print(f"Papers embedded: {embedded}")
        print(f"Total text chunks: {total_chunks}")
        print(f"Pipeline duration: {duration:.2f} seconds")
        print(f"SSL verification: {'enabled' if self.verify_ssl else 'disabled'}")
        print("="*60)
        
        # School breakdown
        school_stats = {}
        for paper in papers:
            if paper.school_code not in school_stats:
                school_stats[paper.school_code] = {"total": 0, "downloaded": 0, "processed": 0, "embedded": 0}
            school_stats[paper.school_code]["total"] += 1
            if paper.is_downloaded:
                school_stats[paper.school_code]["downloaded"] += 1
            if paper.is_processed:
                school_stats[paper.school_code]["processed"] += 1
            if paper.embedding_id:
                school_stats[paper.school_code]["embedded"] += 1
        
        print("\nBREAKDOWN BY SCHOOL:")
        for school, stats in sorted(school_stats.items()):
            school_name = Config.get_school_full_names().get(school, school.upper())
            print(f"  {school_name} ({school}):")
            print(f"    Total: {stats['total']}, Downloaded: {stats['downloaded']}, "
                  f"Processed: {stats['processed']}, Embedded: {stats['embedded']}")
        print("="*60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Exam Papers Scraper and Embeddings Generator")
    
    # Pipeline options
    parser.add_argument("--schools", nargs="+", choices=Config.SCHOOL_CODES, 
                       help="School codes to process")
    parser.add_argument("--years", nargs="+", type=int, 
                       help="Years to process")
    parser.add_argument("--skip-scraping", action="store_true",
                       help="Skip the scraping phase")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip the PDF processing phase")
    parser.add_argument("--skip-embeddings", action="store_true",
                       help="Skip the embeddings generation phase")
    
    # Individual operations
    parser.add_argument("--scrape-only", action="store_true",
                       help="Only run the scraping phase")
    parser.add_argument("--process-only", action="store_true",
                       help="Only run the processing phase")
    parser.add_argument("--embed-only", action="store_true",
                       help="Only run the embeddings phase")
    
    # SSL options
    parser.add_argument("--no-ssl-verify", action="store_true",
                       help="Disable SSL certificate verification (not recommended for production)")
    parser.add_argument("--ignore-ssl-errors", action="store_true",
                       help="Continue even if SSL errors occur")
    
    # Configuration overrides
    parser.add_argument("--max-concurrent", type=int,
                       help="Maximum concurrent downloads")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size for embeddings")
    parser.add_argument("--timeout", type=int,
                       help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.max_concurrent:
        Config.MAX_CONCURRENT_DOWNLOADS = args.max_concurrent
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.timeout:
        Config.DOWNLOAD_TIMEOUT = args.timeout
    
    # SSL configuration
    verify_ssl = not args.no_ssl_verify and Config.VERIFY_SSL
    
    if args.no_ssl_verify:
        logger.warning("SSL verification disabled via command line argument")
    
    # Create pipeline
    pipeline = ExamPaperPipeline(verify_ssl=verify_ssl)
    
    try:
        if args.scrape_only:
            papers = await pipeline.scrape_papers(args.schools, args.years)
            print(f"Scraping completed: {len(papers)} papers")
            
        elif args.process_only:
            papers = pipeline._load_existing_papers()
            papers = await pipeline.process_papers(papers)
            processed_count = sum(1 for p in papers if p.is_processed)
            print(f"Processing completed: {processed_count} papers processed")
            
        elif args.embed_only:
            papers = pipeline._load_existing_papers()
            papers = await pipeline.generate_embeddings(papers)
            embedded_count = sum(1 for p in papers if p.embedding_id)
            print(f"Embeddings completed: {embedded_count} papers embedded")
            
        else:
            # Run complete pipeline
            papers = await pipeline.run_complete_pipeline(
                schools=args.schools,
                years=args.years,
                skip_scraping=args.skip_scraping,
                skip_processing=args.skip_processing,
                skip_embeddings=args.skip_embeddings
            )
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if not args.ignore_ssl_errors:
            raise


if __name__ == "__main__":
    asyncio.run(main())