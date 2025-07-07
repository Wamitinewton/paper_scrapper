"""PDF scraper for MUST exam papers."""

import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
import logging

from bs4 import BeautifulSoup

from config import Config
from models import ExamPaper, Status, ScrapingSession, save_papers_data
from http_client import HTTPClient

logger = logging.getLogger(__name__)

class PDFScraper:
    """Scraper for exam papers PDFs."""
    
    def __init__(self, 
                 max_concurrent: int = 3,
                 request_delay: float = 1.0,
                 verify_ssl: bool = False):
        """
        Initialize scraper.
        
        Args:
            max_concurrent: Maximum concurrent downloads
            request_delay: Delay between requests
            verify_ssl: Whether to verify SSL certificates
        """
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.verify_ssl = verify_ssl
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        Config.create_directories()
    
    def _build_year_url(self, year: int, school_code: str) -> str:
        """Build URL for a specific year and school."""
        # Handle special cases based on your observation
        if school_code in ["safs", "shs"] and year == 2024:
            return f"{Config.BASE_URL}/{year}-{school_code}-exam-papers-2/"
        return f"{Config.BASE_URL}/{year}-{school_code}-exam-papers/"
    
    def _extract_pdf_links(self, html: str, base_url: str) -> List[str]:
        """Extract PDF links from HTML content."""
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        pdf_links = []
        
        # Look for direct PDF links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                absolute_url = urljoin(base_url, href)
                pdf_links.append(absolute_url)
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/wp-content/uploads/' in href and href.lower().endswith('.pdf'):
                absolute_url = urljoin(base_url, href)
                if absolute_url not in pdf_links:
                    pdf_links.append(absolute_url)
        
        return list(set(pdf_links))  # Remove duplicates
    
    def _clean_filename(self, url: str) -> str:
        """Extract and clean filename from URL."""
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        
        # Clean filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Ensure .pdf extension
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        return filename
    
    def _get_local_path(self, school_code: str, year: int, filename: str) -> Path:
        """Get local file path for downloaded PDF."""
        return Config.DOWNLOAD_DIR / school_code / str(year) / filename
    
    async def discover_papers(self, 
                            schools: Optional[List[str]] = None,
                            years: Optional[List[int]] = None) -> List[ExamPaper]:
        """
        Discover all available papers.
        
        Args:
            schools: List of school codes to scrape (None for all)
            years: List of years to scrape (None for all)
            
        Returns:
            List of discovered exam papers
        """
        schools = schools or list(Config.SCHOOL_CODES.keys())
        years = years or Config.YEARS
        
        logger.info(f"Discovering papers for {len(schools)} schools across {len(years)} years")
        
        all_papers = []
        
        async with HTTPClient(verify_ssl=self.verify_ssl) as client:
            for school_code in schools:
                for year in years:
                    papers = await self._discover_year_papers(client, school_code, year)
                    all_papers.extend(papers)
                    
                    await asyncio.sleep(self.request_delay)
        
        logger.info(f"Discovered {len(all_papers)} papers total")
        return all_papers
    
    async def _discover_year_papers(self, 
                                  client: HTTPClient, 
                                  school_code: str, 
                                  year: int) -> List[ExamPaper]:
        """Discover papers for a specific school and year."""
        url = self._build_year_url(year, school_code)
        logger.info(f"Discovering papers for {school_code} {year}: {url}")
        
        html = await client.fetch_text(url)
        if not html:
            logger.warning(f"No content found for {school_code} {year}")
            return []
        
        pdf_links = self._extract_pdf_links(html, url)
        papers = []
        
        for pdf_url in pdf_links:
            filename = self._clean_filename(pdf_url)
            local_path = self._get_local_path(school_code, year, filename)
            
            paper = ExamPaper(
                url=pdf_url,
                filename=filename,
                school_code=school_code,
                year=year,
                local_path=local_path
            )
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers for {school_code} {year}")
        return papers
    
    async def download_paper(self, client: HTTPClient, paper: ExamPaper) -> ExamPaper:
        """Download a single paper."""
        async with self.semaphore:
            if paper.local_path and paper.local_path.exists():
                logger.info(f"File already exists: {paper.filename}")
                paper.download_status = Status.SKIPPED
                paper.file_size = paper.local_path.stat().st_size
                return paper
            
            paper.download_status = Status.IN_PROGRESS
            
            try:
                success = await client.download_file(paper.url, paper.local_path)
                
                if success:
                    paper.download_status = Status.COMPLETED
                    paper.downloaded_at = datetime.now()
                    paper.file_size = paper.local_path.stat().st_size
                    logger.info(f"Downloaded: {paper.filename} ({paper.file_size} bytes)")
                else:
                    paper.download_status = Status.FAILED
                    paper.last_error = "Download failed"
                    paper.retry_count += 1
                    
            except Exception as e:
                paper.download_status = Status.FAILED
                paper.last_error = str(e)
                paper.retry_count += 1
                logger.error(f"Error downloading {paper.filename}: {e}")
            
            # Add delay between downloads
            await asyncio.sleep(self.request_delay)
            
            return paper
    
    async def download_papers(self, papers: List[ExamPaper]) -> List[ExamPaper]:
        """Download multiple papers concurrently."""
        if not papers:
            return []
        
        logger.info(f"Starting download of {len(papers)} papers")
        
        async with HTTPClient(verify_ssl=self.verify_ssl) as client:
            # Create download tasks
            tasks = [self.download_paper(client, paper) for paper in papers]
            
            results = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)
                
                # Log progress every 10 downloads
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(papers)} downloads completed")
        
        # Count results
        completed = sum(1 for p in results if p.download_status == Status.COMPLETED)
        failed = sum(1 for p in results if p.download_status == Status.FAILED)
        skipped = sum(1 for p in results if p.download_status == Status.SKIPPED)
        
        logger.info(f"Download summary: {completed} completed, {failed} failed, {skipped} skipped")
        
        return results
    
    async def run_scraping(self, 
                          schools: Optional[List[str]] = None,
                          years: Optional[List[int]] = None,
                          save_results: bool = True) -> ScrapingSession:
        """
        Run complete scraping process.
        
        Args:
            schools: School codes to scrape
            years: Years to scrape  
            save_results: Whether to save results to file
            
        Returns:
            Scraping session with results
        """
        session = ScrapingSession(
            session_id=str(uuid.uuid4()),
            started_at=datetime.now()
        )
        
        try:
            # Discover papers
            logger.info("Starting paper discovery...")
            papers = await self.discover_papers(schools, years)
            
            if not papers:
                logger.warning("No papers discovered")
                session.completed_at = datetime.now()
                return session
            
            # Download papers
            logger.info("Starting paper downloads...")
            results = await self.download_papers(papers)
            
            # Update session statistics
            for paper in results:
                session.add_result(paper)
            
            session.completed_at = datetime.now()
            
            if save_results:
                output_file = Config.DATA_DIR / f"scraped_papers_{session.session_id}.json"
                save_papers_data(results, output_file)
                logger.info(f"Results saved to {output_file}")
            
            logger.info(f"Scraping completed: {session.success_rate:.1f}% success rate")
            
            return session
            
        except Exception as e:
            session.errors.append(str(e))
            session.completed_at = datetime.now()
            logger.error(f"Scraping failed: {e}")
            raise
    
    async def retry_failed_downloads(self, papers: List[ExamPaper], max_retries: int = 3) -> List[ExamPaper]:
        """Retry failed downloads."""
        failed_papers = [
            p for p in papers 
            if p.download_status == Status.FAILED and p.retry_count < max_retries
        ]
        
        if not failed_papers:
            logger.info("No failed papers to retry")
            return papers
        
        logger.info(f"Retrying {len(failed_papers)} failed downloads")
        
        for paper in failed_papers:
            paper.download_status = Status.PENDING
            paper.last_error = None
        
        # Retry downloads
        retried_papers = await self.download_papers(failed_papers)
        
        # Update original papers list
        paper_dict = {p.url: p for p in papers}
        for retried_paper in retried_papers:
            paper_dict[retried_paper.url] = retried_paper
        
        return list(paper_dict.values())