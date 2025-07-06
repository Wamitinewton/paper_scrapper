"""
Process Existing PDFs and Generate Embeddings Script

This script processes already downloaded PDFs and generates embeddings
without going through the scraping process again.

Usage:
    python process_existing_pdfs.py
    python process_existing_pdfs.py --schools sci sea
    python process_existing_pdfs.py --skip-embeddings
    python process_existing_pdfs.py --embeddings-only
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import re
from typing import List, Optional
import uuid

from config import Config
from models import ExamPaper, Status, save_papers_data, load_papers_data
from pdf_processor import PDFProcessor
from embeddings import EmbeddingsManager

# Setup logging
Config.create_directories()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f'process_existing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("MUST Exam Papers - Process Existing PDFs & Generate Embeddings")
    print("=" * 70)
    print(f"Download directory: {Config.DOWNLOAD_DIR}")
    print(f"Model: {Config.OPENAI_MODEL}")
    print(f"Collection: {Config.QDRANT_COLLECTION_NAME}")
    print(f"Chunk size: {Config.CHUNK_SIZE} words")
    print("=" * 70)

def discover_existing_pdfs(target_schools: Optional[List[str]] = None) -> List[ExamPaper]:
    """
    Discover all existing PDF files in the download directory.
    
    Args:
        target_schools: List of school codes to process (None for all)
        
    Returns:
        List of ExamPaper objects for discovered PDFs
    """
    logger.info("Discovering existing PDF files...")
    
    papers = []
    total_files = 0
    
    school_dirs = [d for d in Config.DOWNLOAD_DIR.iterdir() if d.is_dir()]
    
    for school_dir in school_dirs:
        school_code = school_dir.name
        
        if target_schools and school_code not in target_schools:
            continue
            
        if school_code not in Config.SCHOOL_CODES:
            logger.warning(f"Unknown school code: {school_code}, skipping...")
            continue
        
        logger.info(f"Processing school: {Config.SCHOOL_CODES[school_code]} ({school_code})")
        
        year_dirs = [d for d in school_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        
        for year_dir in year_dirs:
            year = int(year_dir.name)
            
            pdf_files = list(year_dir.glob("*.pdf"))
            total_files += len(pdf_files)
            
            for pdf_file in pdf_files:
                paper = ExamPaper(
                    url=f"file://{pdf_file.absolute()}", 
                    filename=pdf_file.name,
                    school_code=school_code,
                    year=year,
                    local_path=pdf_file,
                    file_size=pdf_file.stat().st_size,
                    download_status=Status.COMPLETED,
                    downloaded_at=datetime.fromtimestamp(pdf_file.stat().st_mtime)
                )
                papers.append(paper)
        
        school_count = len([p for p in papers if p.school_code == school_code])
        logger.info(f"  Found {school_count} PDFs for {school_code}")
    
    logger.info(f"Discovered {len(papers)} PDF files across {len(set(p.school_code for p in papers))} schools")
    return papers

def print_discovery_summary(papers: List[ExamPaper]):
    """Print summary of discovered PDFs."""
    print("\n" + "=" * 70)
    print("PDF DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Total PDFs discovered: {len(papers)}")
    
    school_stats = {}
    year_stats = {}
    total_size = 0
    
    for paper in papers:
        if paper.school_code not in school_stats:
            school_stats[paper.school_code] = 0
        school_stats[paper.school_code] += 1
        
        if paper.year not in year_stats:
            year_stats[paper.year] = 0
        year_stats[paper.year] += 1
        
        if paper.file_size:
            total_size += paper.file_size
    
    print(f"Total file size: {total_size / (1024*1024*1024):.2f} GB")
    
    print("\nBreakdown by school:")
    for school_code, count in sorted(school_stats.items()):
        school_name = Config.SCHOOL_CODES.get(school_code, school_code)
        print(f"  {school_name} ({school_code}): {count} PDFs")
    
    print("\nBreakdown by year:")
    for year, count in sorted(year_stats.items()):
        print(f"  {year}: {count} PDFs")
    
    print("=" * 70)

def print_processing_summary(papers: List[ExamPaper]):
    """Print processing summary."""
    total_papers = len(papers)
    processed_papers = sum(1 for p in papers if p.is_processed)
    embedded_papers = sum(1 for p in papers if p.is_embedded)
    failed_processing = sum(1 for p in papers if p.processing_status == Status.FAILED)
    failed_embedding = sum(1 for p in papers if p.embedding_status == Status.FAILED)
    
    total_chunks = sum(len(p.chunks) for p in papers if p.is_processed)
    
    print("\n" + "=" * 70)
    print("PROCESSING & EMBEDDING SUMMARY")
    print("=" * 70)
    print(f"Total papers: {total_papers}")
    print(f"Successfully processed: {processed_papers}")
    print(f"Processing failures: {failed_processing}")
    print(f"Successfully embedded: {embedded_papers}")
    print(f"Embedding failures: {failed_embedding}")
    print(f"Total text chunks generated: {total_chunks}")
    
    if processed_papers > 0:
        print(f"Processing success rate: {(processed_papers/total_papers*100):.1f}%")
    if processed_papers > 0:
        print(f"Embedding success rate: {(embedded_papers/processed_papers*100):.1f}%")
    
    school_stats = {}
    for paper in papers:
        if paper.school_code not in school_stats:
            school_stats[paper.school_code] = {"total": 0, "processed": 0, "embedded": 0}
        school_stats[paper.school_code]["total"] += 1
        if paper.is_processed:
            school_stats[paper.school_code]["processed"] += 1
        if paper.is_embedded:
            school_stats[paper.school_code]["embedded"] += 1
    
    print("\nBreakdown by school:")
    for school_code, stats in sorted(school_stats.items()):
        school_name = Config.SCHOOL_CODES.get(school_code, school_code)
        if stats["total"] > 0:
            proc_rate = (stats["processed"] / stats["total"]) * 100
            emb_rate = (stats["embedded"] / stats["processed"]) * 100 if stats["processed"] > 0 else 0
            print(f"  {school_name} ({school_code}):")
            print(f"    Processed: {stats['processed']}/{stats['total']} ({proc_rate:.1f}%)")
            print(f"    Embedded: {stats['embedded']}/{stats['processed']} ({emb_rate:.1f}%)")
    
    print("=" * 70)

def load_existing_data() -> Optional[List[ExamPaper]]:
    """Load existing papers data if available."""
    try:
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            logger.info(f"Loading existing data from: {latest_file}")
            return load_papers_data(latest_file)
        
        papers_files = list(Config.DATA_DIR.glob("scraped_papers_*.json"))
        if papers_files:
            latest_file = max(papers_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading existing data from: {latest_file}")
            return load_papers_data(latest_file)
        
        logger.info("No existing papers data found, will create new dataset")
        return None
        
    except Exception as e:
        logger.error(f"Error loading existing data: {e}")
        return None

async def process_and_embed_pdfs(
    target_schools: Optional[List[str]] = None,
    skip_processing: bool = False,
    skip_embeddings: bool = False,
    max_concurrent_processing: int = 5,
    max_concurrent_embedding: int = 3
) -> List[ExamPaper]:
    """
    Main function to process PDFs and generate embeddings.
    
    Args:
        target_schools: School codes to process
        skip_processing: Skip PDF text extraction
        skip_embeddings: Skip embedding generation
        max_concurrent_processing: Max concurrent PDF processing
        max_concurrent_embedding: Max concurrent embedding generation
        
    Returns:
        List of processed papers
    """
    existing_papers = load_existing_data()
    
    if existing_papers:
        logger.info(f"Loaded {len(existing_papers)} papers from existing data")
        papers = existing_papers
        
        if target_schools:
            papers = [p for p in papers if p.school_code in target_schools]
            logger.info(f"Filtered to {len(papers)} papers for target schools: {target_schools}")
    else:
        papers = discover_existing_pdfs(target_schools)
        print_discovery_summary(papers)
    
    if not papers:
        logger.error("No PDFs found to process")
        return []
    
    if not skip_processing:
        logger.info("Starting PDF processing...")
        processor = PDFProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        papers = await processor.process_papers(
            papers, 
            max_concurrent=max_concurrent_processing
        )
        
        session_id = str(uuid.uuid4())
        output_file = Config.DATA_DIR / f"processed_papers_{session_id}.json"
        save_papers_data(papers, output_file)
        logger.info(f"Processing results saved to: {output_file}")
        
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(output_file.name)
    else:
        logger.info("Skipping PDF processing (--skip-processing)")
    
    if not skip_embeddings:
        try:
            Config.validate_embedding_config()
        except ValueError as e:
            logger.error(f"Embedding configuration error: {e}")
            print("\nPlease set up your .env file with:")
            print("OPENAI_API_KEY=your_openai_api_key")
            print("QDRANT_URL=your_qdrant_url")
            print("QDRANT_API_KEY=your_qdrant_api_key")
            return papers
        
        logger.info("Starting embedding generation...")
        embeddings_manager = EmbeddingsManager()
        
        papers = await embeddings_manager.embed_papers(
            papers,
            max_concurrent=max_concurrent_embedding
        )
        
        session_id = str(uuid.uuid4())
        output_file = Config.DATA_DIR / f"embedded_papers_{session_id}.json"
        save_papers_data(papers, output_file)
        logger.info(f"Final results saved to: {output_file}")
        
        latest_file = Config.DATA_DIR / "latest_papers.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(output_file.name)
        
        try:
            stats = embeddings_manager.get_collection_stats()
            logger.info("Qdrant Collection Statistics:")
            logger.info(f"  Total vectors: {stats.get('total_vectors', 0)}")
            logger.info(f"  Collection status: {stats.get('collection_status', 'unknown')}")
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")
    else:
        logger.info("Skipping embedding generation (--skip-embeddings)")
    
    return papers

async def test_search_functionality():
    """Test the search functionality after embedding generation."""
    try:
        Config.validate_embedding_config()
        embeddings_manager = EmbeddingsManager()
        
        test_queries = [
            "database design",
            "calculus",
            "programming",
            "statistics",
            "chemistry"
        ]
        
        print("\n" + "=" * 70)
        print("TESTING SEARCH FUNCTIONALITY")
        print("=" * 70)
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = await embeddings_manager.search_similar(
                query=query,
                limit=3,
                score_threshold=0.7
            )
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    metadata = result["metadata"]
                    print(f"  {i}. {metadata.get('paper_filename', 'Unknown')} "
                          f"({metadata.get('school_name', 'Unknown')} {metadata.get('year', 'Unknown')}) "
                          f"- Score: {result['score']:.3f}")
            else:
                print("  No results found")
        
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Search test failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process existing PDFs and generate embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_existing_pdfs.py                    # Process all PDFs and generate embeddings
  python process_existing_pdfs.py --schools sci sea  # Process only specific schools
  python process_existing_pdfs.py --skip-embeddings # Only extract text, skip embeddings
  python process_existing_pdfs.py --embeddings-only  # Only generate embeddings (assume text extracted)
  python process_existing_pdfs.py --test-search      # Test search after processing
        """
    )
    
    parser.add_argument(
        "--schools",
        nargs="+",
        choices=list(Config.SCHOOL_CODES.keys()),
        help="School codes to process (default: all discovered)"
    )
    
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip PDF text extraction (assume already done)"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (only extract text)"
    )
    
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only generate embeddings (assume text already extracted)"
    )
    
    parser.add_argument(
        "--max-concurrent-processing",
        type=int,
        default=5,
        help="Maximum concurrent PDF processing operations (default: 5)"
    )
    
    parser.add_argument(
        "--max-concurrent-embedding",
        type=int,
        default=3,
        help="Maximum concurrent embedding operations (default: 3)"
    )
    
    parser.add_argument(
        "--test-search",
        action="store_true",
        help="Test search functionality after processing"
    )
    
    args = parser.parse_args()
    
    if args.embeddings_only:
        args.skip_processing = True
    
    print_banner()
    
    if args.schools:
        print(f"Target schools: {args.schools}")
    else:
        print("Target schools: all discovered")
    
    print(f"Skip processing: {args.skip_processing}")
    print(f"Skip embeddings: {args.skip_embeddings}")
    print(f"Max concurrent processing: {args.max_concurrent_processing}")
    print(f"Max concurrent embedding: {args.max_concurrent_embedding}")
    print()
    
    try:
        papers = asyncio.run(process_and_embed_pdfs(
            target_schools=args.schools,
            skip_processing=args.skip_processing,
            skip_embeddings=args.skip_embeddings,
            max_concurrent_processing=args.max_concurrent_processing,
            max_concurrent_embedding=args.max_concurrent_embedding
        ))
        
        if papers:
            print_processing_summary(papers)
            
            if args.test_search and not args.skip_embeddings:
                asyncio.run(test_search_functionality())
            
            print("\n Processing complete!")
            
            if not args.skip_embeddings:
                embedded_count = sum(1 for p in papers if p.is_embedded)
                if embedded_count > 0:
                    print(f"\n✅ {embedded_count} papers successfully embedded and searchable!")
                    print("\nYou can now:")
                    print("1. Test search: python generate_embeddings.py --test-search 'your query'")
                    print("2. View stats: python generate_embeddings.py --stats")
                    print("3. Build your search application!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()