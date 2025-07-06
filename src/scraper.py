import asyncio
import aiohttp
import aiofiles
import ssl
import certifi
from pathlib import Path
from typing import List, Set, Optional, Tuple
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re
import uuid
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup

from .config import Config
from .models import ExamPaper, DownloadStatus, ScrapingSession
from .logger import get_logger

logger = get_logger(__name__)

class ExamPaperScraper:
    
    def __init__(self, max_concurrent: int = None, timeout: int = None, verify_ssl: bool = True):
        self.max_concurrent = max_concurrent or Config.MAX_CONCURRENT_DOWNLOADS
        self.timeout = timeout or Config.DOWNLOAD_TIMEOUT
        self.verify_ssl = verify_ssl
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        Config.create_directories()
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create a custom SSL context with proper certificate handling."""
        if not self.verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL verification disabled - use only for development/testing")
            return context
        
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            logger.info("Using certifi certificates for SSL verification")
            return context
        except Exception as e:
            logger.warning(f"Failed to create SSL context with certifi: {e}")
            return ssl.create_default_context()
    
    def _create_connector(self) -> aiohttp.TCPConnector:
        """Create a TCP connector with proper SSL configuration."""
        ssl_context = self._create_ssl_context()
        
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        return connector
    
    async def __aenter__(self):
        """Async context manager entry with SSL-aware session creation."""
        connector = self._create_connector()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=10,
            sock_read=self.timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers,
            trust_env=True
        )
        
        logger.info(f"HTTP session created with SSL verification: {self.verify_ssl}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _build_url(self, year: int, school_code: str) -> str:
        if school_code in ["safs", "shs"] and year == 2024:
            return f"{Config.BASE_URL}/{year}-{school_code}-exam-papers-2/"
        return f"{Config.BASE_URL}/{year}-{school_code}-exam-papers/"
    
    def _clean_filename(self, filename: str) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        return filename
    
    def _get_file_path(self, school_code: str, year: int, filename: str) -> Path:
        return Config.DOWNLOAD_DIR / school_code / str(year) / filename
    
    async def _fetch_page(self, url: str, retry_count: int = 3) -> Optional[str]:
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{retry_count})")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {url}")
                        return content
                    elif response.status == 403:
                        logger.warning(f"Access forbidden for {url}: HTTP {response.status}")
                        return None
                    elif response.status == 404:
                        logger.warning(f"Page not found {url}: HTTP {response.status}")
                        return None
                    else:
                        logger.warning(f"HTTP error for {url}: {response.status}")
                        if attempt < retry_count - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                        
            except aiohttp.ClientSSLError as e:
                logger.error(f"SSL error fetching {url}: {e}")
                if attempt < retry_count - 1:
                    logger.info(f"Retrying with relaxed SSL settings...")
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
                
            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection error fetching {url}: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching {url}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
        
        logger.error(f"Failed to fetch {url} after {retry_count} attempts")
        return None
    
    async def _extract_pdf_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract PDF links from HTML content."""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        pdf_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                absolute_url = urljoin(base_url, href)
                pdf_links.append(absolute_url)
        
        return pdf_links
    
    async def _discover_papers_for_year(self, year: int, school_code: str) -> List[ExamPaper]:
        """Discover all papers for a specific year and school."""
        url = self._build_url(year, school_code)
        logger.info(f"Discovering papers for {school_code} {year}: {url}")
        
        html_content = await self._fetch_page(url)
        if not html_content:
            logger.warning(f"No content retrieved for {school_code} {year}")
            return []
        
        pdf_links = await self._extract_pdf_links(html_content, url)
        papers = []
        
        for pdf_url in pdf_links:
            filename = self._clean_filename(Path(urlparse(pdf_url).path).name)
            paper = ExamPaper(
                url=pdf_url,
                filename=filename,
                school_code=school_code,
                year=year,
                file_path=self._get_file_path(school_code, year, filename)
            )
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers for {school_code} {year}")
        return papers
    
    async def _download_paper(self, paper: ExamPaper) -> ExamPaper:
        """Download a single paper with enhanced error handling."""
        async with self.semaphore:
            # Check if file already exists
            if paper.file_path and paper.file_path.exists():
                logger.info(f"File already exists: {paper.filename}")
                paper.download_status = DownloadStatus.SKIPPED
                paper.file_size = paper.file_path.stat().st_size
                return paper
            
            # Ensure directory exists
            paper.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            paper.download_status = DownloadStatus.DOWNLOADING
            paper.download_attempts += 1
            
            try:
                logger.info(f"Downloading {paper.url}")
                async with self.session.get(paper.url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Save file
                        async with aiofiles.open(paper.file_path, 'wb') as f:
                            await f.write(content)
                        
                        paper.file_size = len(content)
                        paper.download_status = DownloadStatus.COMPLETED
                        paper.downloaded_at = datetime.now()
                        
                        logger.info(f"Downloaded: {paper.filename} ({paper.file_size} bytes)")
                        
                    else:
                        paper.download_status = DownloadStatus.FAILED
                        paper.download_error = f"HTTP {response.status}"
                        logger.error(f"Failed to download {paper.url}: HTTP {response.status}")
                        
            except aiohttp.ClientSSLError as e:
                paper.download_status = DownloadStatus.FAILED
                paper.download_error = f"SSL Error: {str(e)}"
                logger.error(f"SSL error downloading {paper.url}: {e}")
                
            except Exception as e:
                paper.download_status = DownloadStatus.FAILED
                paper.download_error = str(e)
                logger.error(f"Error downloading {paper.url}: {e}")
                
                # Clean up partial file
                if paper.file_path and paper.file_path.exists():
                    paper.file_path.unlink()
            
            return paper
    
    async def discover_all_papers(self, 
                                schools: Optional[List[str]] = None,
                                years: Optional[List[int]] = None) -> List[ExamPaper]:
        """Discover all papers across schools and years."""
        
        schools = schools or Config.SCHOOL_CODES
        years = years or Config.YEARS
        
        logger.info(f"Discovering papers for {len(schools)} schools across {len(years)} years")
        
        tasks = []
        for school_code in schools:
            for year in years:
                task = self._discover_papers_for_year(year, school_code)
                tasks.append(task)
        
        results = []
        batch_size = 5 
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
            
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)
        
        all_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Discovery task failed: {result}")
            else:
                all_papers.extend(result)
        
        logger.info(f"Total papers discovered: {len(all_papers)}")
        return all_papers
    
    async def download_papers(self, papers: List[ExamPaper]) -> List[ExamPaper]:
        """Download multiple papers concurrently."""
        if not papers:
            return []
        
        logger.info(f"Starting download of {len(papers)} papers")
        
        tasks = [self._download_paper(paper) for paper in papers]
        
        results = await tqdm.gather(*tasks, desc="Downloading papers")
        
        completed = sum(1 for r in results if r.download_status == DownloadStatus.COMPLETED)
        failed = sum(1 for r in results if r.download_status == DownloadStatus.FAILED)
        skipped = sum(1 for r in results if r.download_status == DownloadStatus.SKIPPED)
        
        logger.info(f"Download completed: {completed} successful, {failed} failed, {skipped} skipped")
        
        return results
    
    async def scrape_all(self, 
                        schools: Optional[List[str]] = None,
                        years: Optional[List[int]] = None,
                        verify_ssl: bool = True) -> Tuple[List[ExamPaper], ScrapingSession]:
        """Scrape all papers from the website."""
        
        self.verify_ssl = verify_ssl
        
        session = ScrapingSession(
            session_id=str(uuid.uuid4()),
            started_at=datetime.now()
        )
        
        try:
            # Discover papers
            papers = await self.discover_all_papers(schools, years)
            
            # Download papers
            results = await self.download_papers(papers)
            
            # Update session statistics
            for paper in results:
                session.add_paper_result(paper)
            
            session.completed_at = datetime.now()
            
            logger.info(f"Scraping session completed: {session.success_rate:.1f}% success rate")
            
            return results, session
            
        except Exception as e:
            session.errors.append(str(e))
            session.completed_at = datetime.now()
            logger.error(f"Scraping session failed: {e}")
            raise
    
    async def retry_failed_downloads(self, papers: List[ExamPaper], max_retries: int = 3) -> List[ExamPaper]:
        """Retry failed downloads."""
        failed_papers = [p for p in papers 
                        if p.download_status == DownloadStatus.FAILED 
                        and p.download_attempts < max_retries]
        
        if not failed_papers:
            logger.info("No failed papers to retry")
            return papers
        
        logger.info(f"Retrying {len(failed_papers)} failed downloads")
        
        for paper in failed_papers:
            paper.download_status = DownloadStatus.PENDING
            paper.download_error = None
        
        retried_papers = await self.download_papers(failed_papers)
        
        paper_dict = {p.url: p for p in papers}
        for retried_paper in retried_papers:
            paper_dict[retried_paper.url] = retried_paper
        
        return list(paper_dict.values())