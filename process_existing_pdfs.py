"""

This script processes already downloaded PDFs and generates high-quality embeddings
with improved text extraction and cleaning.

Usage:
    python process_existing_pdfs.py
    python process_existing_pdfs.py --schools sci sea
    python process_existing_pdfs.py --reprocess-all
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import uuid
from typing import List, Optional

from config import Config
from models import ExamPaper, Status, save_papers_data, load_papers_data
from pdf_processor import PDFProcessor
from embeddings import EmbeddingsManager

Config.create_directories()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f'process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("MUST Exam Papers - Enhanced PDF Processing & Embeddings")
    print("=" * 70)
    print(f"Download directory: {Config.DOWNLOAD_DIR}")
    print(f"Embedding model: {Config.OPENAI_MODEL}")
    print(f"Collection: {Config.QDRANT_COLLECTION_NAME}")
    print(f"Chunk size: {Config.CHUNK_SIZE} words")
    print("=" * 70)

def print_summary(papers: List[ExamPaper]):
    """Print processing and embedding summary."""
    total_papers = len(papers)
    downloaded = sum(1 for p in papers if p.is_downloaded)
    processed = sum(1 for p in papers if p.is_processed)
    with_chunks = sum(1 for p in papers if p.is_processed and p.chunks and len(p.chunks) > 0)
    embedded = sum(1 for p in papers if p.is_embedded)
    
    total_chunks = sum(len(p.chunks) for p in papers if p.chunks)
    
    print("\n" + "=" * 70)
    print("PROCESSING & EMBEDDING SUMMARY")
    print("=" * 70)
    print(f"Total papers: {total_papers}")
    print(f"Downloaded: {downloaded}")
    print(f"Successfully processed: {processed}")
    print(f"Papers with chunks: {with_chunks}")
    print(f"Successfully embedded: {embedded}")
    print(f"Total text chunks: {total_chunks}")
    
    if downloaded > 0:
        print(f"Processing success rate: {(processed/downloaded*100):.1f}%")
    if with_chunks > 0:
        print(f"Embedding success rate: {(embedded/with_chunks*100):.1f}%")
    
    school_stats = {}
    for paper in papers:
        if paper.school_code not in school_stats:
            school_stats[paper.school_code] = {
                "total": 0, "processed": 0, "embedded": 0, "chunks": 0
            }
        school_stats[paper.school_code]["total"] += 1
        if paper.is_processed:
            school_stats[paper.school_code]["processed"] += 1
            if paper.chunks:
                school_stats[paper.school_code]["chunks"] += len(paper.chunks)
        if paper.is_embedded:
            school_stats[paper.school_code]["embedded"] += 1
    
    print("\nBreakdown by school:")
    for school_code, stats in sorted(school_stats.items()):
        school_name = Config.SCHOOL_CODES.get(school_code, school_code)
        print(f"  {school_name} ({school_code}):")
        print(f"    Processed: {stats['processed']}/{stats['total']}")
        print(f"    Embedded: {stats['embedded']}/{stats['processed'] if stats['processed'] > 0 else 0}")
        print(f"    Chunks: {stats['chunks']}")
    
    print("=" * 70)

def load_existing_data() -> Optional[List[ExamPaper]]:
    """Load existing papers data."""
    try:
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            logger.info(f"Loading data from: {latest_file}")
            return load_papers_data(latest_file)
        
        papers_files = list(Config.DATA_DIR.glob("*papers_*.json"))
        if papers_files:
            latest_file = max(papers_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading data from: {latest_file}")
            return load_papers_data(latest_file)
        
        logger.warning("No existing papers data found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

async def process_papers(papers: List[ExamPaper], 
                        reprocess_all: bool = False,
                        max_concurrent: int = 5) -> List[ExamPaper]:
    """Process PDFs with enhanced text extraction."""
    
    if reprocess_all:
        papers_to_process = [p for p in papers if p.is_downloaded]
        logger.info(f"Reprocessing all {len(papers_to_process)} downloaded papers")
        for paper in papers_to_process:
            paper.processing_status = Status.PENDING
            paper.chunks = []
            paper.extracted_text = None
    else:
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
    
    logger.info(f"Processing {len(papers_to_process)} papers")
    
    processor = PDFProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    papers = await processor.process_papers(papers, max_concurrent=max_concurrent)
    
    successful = sum(1 for p in papers_to_process 
                    if p.processing_status == Status.COMPLETED and p.chunks)
    failed = sum(1 for p in papers_to_process 
                if p.processing_status == Status.FAILED)
    
    logger.info(f"Processing completed: {successful} successful, {failed} failed")
    
    return papers

async def generate_embeddings(papers: List[ExamPaper], 
                            max_concurrent: int = 3) -> List[ExamPaper]:
    """Generate embeddings for processed papers."""
    
    try:
        Config.validate_embedding_config()
    except ValueError as e:
        logger.error(f"Embedding configuration error: {e}")
        print("\nPlease set up your .env file with:")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("QDRANT_URL=your_qdrant_url")
        print("QDRANT_API_KEY=your_qdrant_api_key")
        return papers
    
    papers_to_embed = [
        p for p in papers 
        if p.is_processed and p.chunks and len(p.chunks) > 0 and not p.is_embedded
    ]
    
    if not papers_to_embed:
        logger.info("No papers need embedding")
        return papers
    
    logger.info(f"Generating embeddings for {len(papers_to_embed)} papers")
    
    embeddings_manager = EmbeddingsManager()
    papers = await embeddings_manager.embed_papers(papers, max_concurrent=max_concurrent)
    
    successful = sum(1 for p in papers_to_embed if p.is_embedded)
    failed = sum(1 for p in papers_to_embed 
                if p.embedding_status == Status.FAILED)
    
    logger.info(f"Embedding completed: {successful} successful, {failed} failed")
    
    return papers

async def test_search(embeddings_manager: EmbeddingsManager):
    """Test search functionality."""
    test_queries = [
        "database design and management",
        "calculus and mathematical analysis", 
        "programming and software development",
        "business management principles",
        "chemistry laboratory procedures"
    ]
    
    print("\n" + "=" * 50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("=" * 50)
    
    for query in test_queries:
        try:
            results = await embeddings_manager.search_similar(
                query=query,
                limit=3,
                score_threshold=0.7
            )
            
            print(f"\nQuery: '{query}'")
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    metadata = result["metadata"]
                    print(f"  {i}. {metadata.get('paper_filename', 'Unknown')} "
                          f"({metadata.get('school_name', 'Unknown')} {metadata.get('year', 'Unknown')}) "
                          f"- Score: {result['score']:.3f}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("=" * 50)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced PDF processing and embedding generation"
    )
    
    parser.add_argument(
        "--schools",
        nargs="+",
        choices=list(Config.SCHOOL_CODES.keys()),
        help="Process only specific schools"
    )
    
    parser.add_argument(
        "--reprocess-all",
        action="store_true",
        help="Reprocess all PDFs, even if already processed"
    )
    
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip PDF processing (only generate embeddings)"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (only process PDFs)"
    )
    
    parser.add_argument(
        "--max-concurrent-processing",
        type=int,
        default=5,
        help="Max concurrent PDF processing (default: 5)"
    )
    
    parser.add_argument(
        "--max-concurrent-embedding",
        type=int,
        default=3,
        help="Max concurrent embedding generation (default: 3)"
    )
    
    parser.add_argument(
        "--test-search",
        action="store_true",
        help="Test search functionality after embedding"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    papers = load_existing_data()
    if not papers:
        logger.error("No papers data found. Run scraping first.")
        return
    
    if args.schools:
        papers = [p for p in papers if p.school_code in args.schools]
        logger.info(f"Filtered to {len(papers)} papers for schools: {args.schools}")
    
    logger.info(f"Loaded {len(papers)} papers")
    
    try:
        if not args.skip_processing:
            logger.info("Starting enhanced PDF processing...")
            papers = await process_papers(
                papers,
                reprocess_all=args.reprocess_all,
                max_concurrent=args.max_concurrent_processing
            )
        
        if not args.skip_embeddings:
            logger.info("Starting embedding generation...")
            papers = await generate_embeddings(
                papers,
                max_concurrent=args.max_concurrent_embedding
            )
        
        session_id = str(uuid.uuid4())
        output_file = Config.DATA_DIR / f"enhanced_papers_{session_id}.json"
        save_papers_data(papers, output_file)
        
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(output_file.name)
        
        logger.info(f"Results saved to: {output_file}")
        
        print_summary(papers)
        
        if args.test_search and not args.skip_embeddings:
            try:
                Config.validate_embedding_config()
                embeddings_manager = EmbeddingsManager()
                await test_search(embeddings_manager)
            except Exception as e:
                logger.error(f"Search test failed: {e}")
        
        embedded_count = sum(1 for p in papers if p.is_embedded)
        if embedded_count > 0:
            print(f"\n✅ Successfully processed and embedded {embedded_count} papers!")
            print("\nNext steps:")
            print("1. Test search: python generate_embeddings.py --test-search 'your query'")
            print("2. View collection stats: python generate_embeddings.py --stats")
            print("3. Build your search application!")
        else:
            print("\n⚠️  No papers were successfully embedded. Check the logs for issues.")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())