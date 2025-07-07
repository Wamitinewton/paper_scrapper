"""

This script discovers and downloads exam papers from MUST website.
Run this first before generating embeddings.

Usage:
    python scrape_papers.py --schools sci sea --years 2022 2023 2024
    python scrape_papers.py --all
    python scrape_papers.py --retry-failed
"""

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

from config import Config
from models import load_papers_data, save_papers_data
from scraper import PDFScraper
from pdf_processor import PDFProcessor

Config.create_directories() 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("MUST Exam Papers Scraper")
    print("=" * 60)
    print(f"Target: {Config.BASE_URL}")
    print(f"Schools: {list(Config.SCHOOL_CODES.keys())}")
    print(f"Years: {Config.YEARS[0]} - {Config.YEARS[-1]}")
    print("=" * 60)

def print_summary(session, papers):
    """Print scraping summary."""
    print("\n" + "=" * 60)
    print("SCRAPING SUMMARY")
    print("=" * 60)
    print(f"Session ID: {session.session_id}")
    print(f"Duration: {session.duration_seconds:.2f} seconds")
    print(f"Total papers found: {session.total_found}")
    print(f"Successfully downloaded: {session.total_downloaded}")
    print(f"Failed downloads: {session.total_failed}")
    print(f"Skipped (already exist): {session.total_skipped}")
    print(f"Success rate: {session.success_rate:.1f}%")
    
    if session.errors:
        print(f"\nErrors encountered: {len(session.errors)}")
        for error in session.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # School breakdown
    school_stats = {}
    for paper in papers:
        if paper.school_code not in school_stats:
            school_stats[paper.school_code] = {"total": 0, "downloaded": 0}
        school_stats[paper.school_code]["total"] += 1
        if paper.is_downloaded:
            school_stats[paper.school_code]["downloaded"] += 1
    
    print("\nBreakdown by school:")
    for school_code, stats in sorted(school_stats.items()):
        school_name = Config.SCHOOL_CODES.get(school_code, school_code)
        print(f"  {school_name} ({school_code}): {stats['downloaded']}/{stats['total']}")
    
    print("=" * 60)

async def scrape_and_process(args):
    """Run scraping and optional processing."""
    Config.create_directories()
    
    # Initialize scraper
    scraper = PDFScraper(
        max_concurrent=args.max_concurrent,
        request_delay=args.delay,
        verify_ssl=args.verify_ssl
    )
    
    # Run scraping
    logger.info("Starting scraping process...")
    session = await scraper.run_scraping(
        schools=args.schools,
        years=args.years,
        save_results=True
    )
    
    # Load the scraped papers
    papers_file = Config.DATA_DIR / f"scraped_papers_{session.session_id}.json"
    papers = load_papers_data(papers_file)
    
    # Retry failed downloads if requested
    if args.retry_failed and session.total_failed > 0:
        logger.info(f"Retrying {session.total_failed} failed downloads...")
        papers = await scraper.retry_failed_downloads(papers, max_retries=3)
        save_papers_data(papers, papers_file)
    
    # Process PDFs if requested
    if args.process_pdfs:
        logger.info("Starting PDF processing...")
        processor = PDFProcessor()
        papers = await processor.process_papers(papers, max_concurrent=args.max_concurrent)
        save_papers_data(papers, papers_file)
    
    # Save session summary
    session_file = Config.LOGS_DIR / f"session_{session.session_id}.json"
    with open(session_file, 'w') as f:
        json.dump(session.to_dict(), f, indent=2)
    
    return session, papers

async def retry_failed_papers():
    """Retry failed downloads from previous sessions."""
    # Find the most recent papers file
    papers_files = list(Config.DATA_DIR.glob("scraped_papers_*.json"))
    if not papers_files:
        logger.error("No previous scraping data found")
        return
    
    latest_file = max(papers_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading papers from: {latest_file}")
    
    papers = load_papers_data(latest_file)
    failed_papers = [p for p in papers if not p.is_downloaded]
    
    if not failed_papers:
        logger.info("No failed papers to retry")
        return
    
    logger.info(f"Retrying {len(failed_papers)} failed papers")
    
    scraper = PDFScraper(verify_ssl=False)  # Usually SSL issues cause failures
    papers = await scraper.retry_failed_downloads(papers, max_retries=3)
    
    # Save updated results
    save_papers_data(papers, latest_file)
    logger.info("Retry completed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape exam papers from MUST website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_papers.py --all
  python scrape_papers.py --schools sci sea --years 2022 2023 2024
  python scrape_papers.py --schools safs --process-pdfs
  python scrape_papers.py --retry-failed
        """
    )
    
    # Target selection
    parser.add_argument(
        "--schools", 
        nargs="+", 
        choices=list(Config.SCHOOL_CODES.keys()),
        help="School codes to scrape (default: all)"
    )
    parser.add_argument(
        "--years", 
        nargs="+", 
        type=int,
        help="Years to scrape (default: all available)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all schools and years"
    )
    
    # Processing options
    parser.add_argument(
        "--process-pdfs",
        action="store_true", 
        help="Also extract text from downloaded PDFs"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed downloads from previous session"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=Config.MAX_CONCURRENT_DOWNLOADS,
        help=f"Maximum concurrent downloads (default: {Config.MAX_CONCURRENT_DOWNLOADS})"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=Config.DELAY_BETWEEN_REQUESTS,
        help=f"Delay between requests in seconds (default: {Config.DELAY_BETWEEN_REQUESTS})"
    )
    
    # SSL options
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable SSL certificate verification (default: disabled for institutional sites)"
    )
    
    
    parser.add_argument(
        "--retry-only",
        action="store_true",
        help="Only retry failed downloads from previous session"
    )
    
    args = parser.parse_args()
    
    if args.all:
        args.schools = None
        args.years = None
    
    if args.retry_only:
        print_banner()
        asyncio.run(retry_failed_papers())
        return
    
    if not args.schools and not args.all:
        args.schools = list(Config.SCHOOL_CODES.keys())
    
    if not args.years and not args.all:
        args.years = Config.YEARS
    
    print_banner()
    print(f"Target schools: {args.schools or 'all'}")
    print(f"Target years: {args.years or 'all'}")
    print(f"SSL verification: {'enabled' if args.verify_ssl else 'disabled'}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Request delay: {args.delay}s")
    print(f"Process PDFs: {'yes' if args.process_pdfs else 'no'}")
    print()
    
    try:
        session, papers = asyncio.run(scrape_and_process(args))
        
        print_summary(session, papers)
        
        final_file = Config.DATA_DIR / "latest_papers.json"
        papers_file = Config.DATA_DIR / f"scraped_papers_{session.session_id}.json"
        
        if papers_file.exists():
            if final_file.exists():
                final_file.unlink()
            final_file.symlink_to(papers_file.name)
            print(f"\nData saved to: {papers_file}")
            print(f"Latest data link: {final_file}")
        
        if args.process_pdfs:
            processed_count = sum(1 for p in papers if p.is_processed)
            print(f"\nPDFs processed: {processed_count}")
            if processed_count > 0:
                print("Ready for embedding generation!")
                print("Run: python generate_embeddings.py")
        else:
            downloaded_count = sum(1 for p in papers if p.is_downloaded)
            if downloaded_count > 0:
                print("\nNext steps:")
                print("1. Process PDFs: python scrape_papers.py --process-pdfs")
                print("2. Generate embeddings: python generate_embeddings.py")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise

if __name__ == "__main__":
    main()