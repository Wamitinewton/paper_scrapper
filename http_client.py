
import ssl
import aiohttp
import certifi
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HTTPClient:
    
    def __init__(self, 
                 timeout: int = 30,
                 verify_ssl: bool = False,
                 custom_cert_path: Optional[str] = None):
        """
        Initialize HTTP client.
        
        Args:
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            custom_cert_path: Path to custom certificate bundle
        """
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.custom_cert_path = custom_cert_path
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context with proper certificate handling."""
        if not self.verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL verification disabled - not recommended for production")
            return context
        
        try:
            if self.custom_cert_path and Path(self.custom_cert_path).exists():
                context = ssl.create_default_context(cafile=self.custom_cert_path)
                logger.info(f"Using custom certificate bundle: {self.custom_cert_path}")
            else:
                context = ssl.create_default_context(cafile=certifi.where())
                logger.info("Using certifi certificate bundle")
            
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            return ssl.create_default_context()
    
    def _create_connector(self) -> aiohttp.TCPConnector:
        """Create TCP connector with appropriate SSL settings."""
        ssl_context = self._create_ssl_context()
        
        return aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=True,
            timeout_ceil_threshold=5
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = self._create_connector()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
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
            trust_env=True,
            raise_for_status=False  # Handle status codes manually
        )
        
        logger.info(f"HTTP client initialized (SSL verify: {self.verify_ssl})")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            # Give the session time to close properly
            await asyncio.sleep(0.1)
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make GET request with error handling."""
        if not self.session:
            raise RuntimeError("HTTP client not initialized. Use as async context manager.")
        
        return await self.session.get(url, **kwargs)
    
    async def fetch_text(self, url: str, retry_count: int = 3) -> Optional[str]:
        """
        Fetch text content from URL with retry logic.
        
        Args:
            url: URL to fetch
            retry_count: Number of retry attempts
            
        Returns:
            Text content or None if failed
        """
        last_error = None
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{retry_count})")
                
                async with self.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {url}")
                        return content
                    elif response.status == 404:
                        logger.warning(f"Page not found: {url}")
                        return None
                    elif response.status == 403:
                        logger.warning(f"Access forbidden: {url}")
                        return None
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        last_error = f"HTTP {response.status}"
                        
            except aiohttp.ClientSSLError as e:
                last_error = f"SSL error: {e}"
                logger.error(f"SSL error for {url}: {e}")
                
            except aiohttp.ClientConnectorError as e:
                last_error = f"Connection error: {e}"
                logger.error(f"Connection error for {url}: {e}")
                
            except asyncio.TimeoutError:
                last_error = "Timeout"
                logger.error(f"Timeout for {url}")
                
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.error(f"Unexpected error for {url}: {e}")
            
            if attempt < retry_count - 1:
                delay = 2 ** attempt
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        logger.error(f"Failed to fetch {url} after {retry_count} attempts. Last error: {last_error}")
        return None
    
    async def download_file(self, url: str, filepath: Path, retry_count: int = 3) -> bool:
        """
        Download file from URL to local path.
        
        Args:
            url: URL to download
            filepath: Local path to save file
            retry_count: Number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Downloading {url} to {filepath} (attempt {attempt + 1}/{retry_count})")
                
                async with self.get(url) as response:
                    if response.status == 200:
                        # Stream download for large files
                        with open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        
                        file_size = filepath.stat().st_size
                        logger.info(f"Downloaded {url} ({file_size} bytes)")
                        return True
                    else:
                        last_error = f"HTTP {response.status}"
                        logger.error(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                last_error = str(e)
                logger.error(f"Download error for {url}: {e}")
                
                # Clean up partial file
                if filepath.exists():
                    filepath.unlink()
            
            if attempt < retry_count - 1:
                delay = 2 ** attempt
                await asyncio.sleep(delay)
        
        logger.error(f"Failed to download {url} after {retry_count} attempts. Last error: {last_error}")
        return False