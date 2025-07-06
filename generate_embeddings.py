"""
Embeddings Generation Script

This script generates vector embeddings from processed exam papers.
Run this after scraping and processing PDFs.

Usage:
    python generate_embeddings.py
    python generate_embeddings.py --input data/scraped_papers_abc123.json
    python generate_embeddings.py --test-search "database design"
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

from config import Config
from models import load_papers_data, save_papers_data
from embeddings import EmbeddingsManager

# Setup logging
Config.create_directories()  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f'embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("MUST Exam Papers - Embeddings Generator")
    print("=" * 60)
    print(f"Model: {Config.OPENAI_MODEL}")
    print(f"Collection: {Config.QDRANT_COLLECTION_NAME}")
    print(f"Chunk size: {Config.CHUNK_SIZE} words")
    print(f"Batch size: {Config.EMBEDDING_BATCH_SIZE}")
    print("=" * 60)

def print_embedding_summary(papers):
    """Print embedding generation summary."""
    total_papers = len(papers)
    processed_papers = sum(1 for p in papers if p.is_processed)
    embedded_papers = sum(1 for p in papers if p.is_embedded)
    total_chunks = sum(len(p.chunks) for p in papers if p.is_processed)
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total papers: {total_papers}")
    print(f"Processed papers: {processed_papers}")
    print(f"Successfully embedded: {embedded_papers}")
    print(f"Total text chunks: {total_chunks}")
    print(f"Embedding success rate: {(embedded_papers/processed_papers*100) if processed_papers > 0 else 0:.1f}%")
    
    # 
    school_stats = {}
    for paper in papers:
        if paper.school_code not in school_stats:
            school_stats[paper.school_code] = {"total": 0, "embedded": 0}
        if paper.is_processed:
            school_stats[paper.school_code]["total"] += 1
            if paper.is_embedded:
                school_stats[paper.school_code]["embedded"] += 1
    
    print("\nBreakdown by school:")
    for school_code, stats in sorted(school_stats.items()):
        school_name = Config.SCHOOL_CODES.get(school_code, school_code)
        if stats["total"] > 0:
            rate = (stats["embedded"] / stats["total"]) * 100
            print(f"  {school_name} ({school_code}): {stats['embedded']}/{stats['total']} ({rate:.1f}%)")
    
    print("=" * 60)

def find_latest_papers_file() -> Path:
    """Find the most recent papers data file."""
    latest_file = Config.DATA_DIR / "latest_papers.json"
    if latest_file.exists():
        return latest_file
    
    papers_files = list(Config.DATA_DIR.glob("scraped_papers_*.json"))
    if not papers_files:
        raise FileNotFoundError("No papers data files found. Run scrape_papers.py first.")
    
    return max(papers_files, key=lambda p: p.stat().st_mtime)

async def generate_embeddings(input_file: Path, max_concurrent: int = 3):
    """Generate embeddings for papers."""
    try:
        Config.validate_embedding_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nPlease set up your .env file with:")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("QDRANT_URL=your_qdrant_url")  
        print("QDRANT_API_KEY=your_qdrant_api_key")
        return None
    
    logger.info(f"Loading papers from: {input_file}")
    papers = load_papers_data(input_file)
    
    if not papers:
        logger.error("No papers data found")
        return None
    
    processed_papers = [p for p in papers if p.is_processed and p.chunks]
    if not processed_papers:
        logger.error("No processed papers found. Run PDF processing first.")
        print("\nTo process PDFs, run:")
        print("python scrape_papers.py --process-pdfs")
        return None
    
    logger.info(f"Found {len(processed_papers)} processed papers ready for embedding")
    
    embeddings_manager = EmbeddingsManager()
    
    logger.info("Starting embedding generation...")
    papers = await embeddings_manager.embed_papers(papers, max_concurrent=max_concurrent)
    
    save_papers_data(papers, input_file)
    
    stats = embeddings_manager.get_collection_stats()
    logger.info("Collection statistics:")
    logger.info(f"  Total vectors: {stats.get('total_vectors', 0)}")
    logger.info(f"  Collection status: {stats.get('collection_status', 'unknown')}")
    
    return papers

async def test_search(query: str, school_filter: str = None, year_filter: int = None):
    """Test search functionality."""
    try:
        Config.validate_embedding_config()
        embeddings_manager = EmbeddingsManager()
        
        print(f"\nSearching for: '{query}'")
        if school_filter:
            print(f"School filter: {school_filter}")
        if year_filter:
            print(f"Year filter: {year_filter}")
        
        results = await embeddings_manager.search_similar(
            query=query,
            limit=5,
            school_filter=school_filter,
            year_filter=year_filter,
            score_threshold=0.7
        )
        
        print(f"\nFound {len(results)} results:")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Paper: {metadata.get('paper_filename', 'Unknown')}")
            print(f"   School: {metadata.get('school_name', 'Unknown')} ({metadata.get('year', 'Unknown')})")
            print(f"   Preview: {metadata.get('chunk_text_preview', '')[:100]}...")
            print()
        
    except Exception as e:
        logger.error(f"Search test failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate vector embeddings for exam papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_embeddings.py
  python generate_embeddings.py --input data/scraped_papers_abc123.json
  python generate_embeddings.py --test-search "database design"
  python generate_embeddings.py --test-search "calculus" --school sci --year 2023
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        help="Input papers JSON file (default: find latest)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent embedding operations (default: 3)"
    )
    
    parser.add_argument(
        "--test-search",
        type=str,
        help="Test search functionality with query"
    )
    parser.add_argument(
        "--school",
        choices=list(Config.SCHOOL_CODES.keys()),
        help="Filter test search by school"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Filter test search by year"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.test_search:
        print("Testing search functionality...")
        asyncio.run(test_search(args.test_search, args.school, args.year))
        return
    
    if args.stats:
        try:
            Config.validate_embedding_config()
            embeddings_manager = EmbeddingsManager()
            stats = embeddings_manager.get_collection_stats()
            
            print("Collection Statistics:")
            print(f"  Total vectors: {stats.get('total_vectors', 0)}")
            print(f"  Status: {stats.get('collection_status', 'unknown')}")
            
            if stats.get('school_distribution'):
                print("\nSchool distribution:")
                for school, count in stats['school_distribution'].items():
                    school_name = Config.SCHOOL_CODES.get(school, school)
                    print(f"  {school_name} ({school}): {count}")
            
            if stats.get('year_distribution'):
                print("\nYear distribution:")
                for year, count in sorted(stats['year_distribution'].items()):
                    print(f"  {year}: {count}")
                    
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        return
    
    try:
        if args.input:
            input_file = args.input
            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                return
        else:
            input_file = find_latest_papers_file()
            
        print(f"Input file: {input_file}")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    try:
        papers = asyncio.run(generate_embeddings(input_file, args.max_concurrent))
        
        if papers:
            print_embedding_summary(papers)
            
            embedded_count = sum(1 for p in papers if p.is_embedded)
            if embedded_count > 0:
                print("\nEmbeddings generated successfully!")
                print("\nYou can now:")
                print("1. Test search: python generate_embeddings.py --test-search 'your query'")
                print("2. View stats: python generate_embeddings.py --stats")
                print("3. Build your own search application using the embeddings")
        
    except KeyboardInterrupt:
        logger.info("Embedding generation interrupted by user")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

if __name__ == "__main__":
    main()