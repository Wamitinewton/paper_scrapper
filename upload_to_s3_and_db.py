"""
Upload PDFs to S3 and store URLs in Neon database.

Usage:
    python upload_to_s3_and_db.py
    python upload_to_s3_and_db.py --dry-run
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import os
from datetime import datetime

import asyncpg
import aioboto3
from dotenv import load_dotenv

from config import Config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class S3DatabaseUploader:
    """Upload PDFs to S3 and store metadata in Neon database."""
    
    def __init__(self):
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        self.s3_prefix = os.getenv('S3_PREFIX', 'exam-papers')
        self.database_url = os.getenv('DATABASE_URL')
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required environment variables."""
        required_vars = [
            'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 
            'AWS_REGION', 'S3_BUCKET_NAME', 'DATABASE_URL'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    
    async def _init_database(self):
        """Initialize database connection and create table if needed."""
        self.db_pool = await asyncpg.create_pool(self.database_url)
        
        # Create table if it doesn't exist
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS exam_papers (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    school_code VARCHAR(10) NOT NULL,
                    school_name VARCHAR(255) NOT NULL,
                    year INTEGER NOT NULL,
                    s3_url VARCHAR(500) NOT NULL UNIQUE,
                    file_size BIGINT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, school_code, year)
                )
            """)
            
            # Create index for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exam_papers_school_year 
                ON exam_papers(school_code, year)
            """)
    
    async def _close_database(self):
        """Close database connection pool."""
        if hasattr(self, 'db_pool'):
            await self.db_pool.close()
    
    def _generate_s3_key(self, school_code: str, year: int, filename: str) -> str:
        """Generate S3 object key."""
        return f"{self.s3_prefix}/{school_code}/{year}/{filename}"
    
    def _generate_public_url(self, s3_key: str) -> str:
        """Generate public S3 URL."""
        return f"https://{self.s3_bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
    
    async def _upload_to_s3(self, session, file_path: Path, s3_key: str) -> bool:
        """Upload single file to S3."""
        try:
            async with session.client('s3') as s3:
                with open(file_path, 'rb') as file:
                    await s3.upload_fileobj(
                        file, 
                        self.s3_bucket, 
                        s3_key,
                        ExtraArgs={
                            'ContentType': 'application/pdf',
                            'ACL': 'public-read'  # Make files publicly accessible
                        }
                    )
                return True
        except Exception as e:
            logger.error(f"S3 upload failed for {file_path.name}: {e}")
            return False
    
    async def _store_in_database(self, paper_data: Dict[str, Any]) -> bool:
        """Store paper metadata in database."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO exam_papers 
                    (filename, school_code, school_name, year, s3_url, file_size)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (filename, school_code, year) 
                    DO UPDATE SET 
                        s3_url = EXCLUDED.s3_url,
                        file_size = EXCLUDED.file_size,
                        uploaded_at = CURRENT_TIMESTAMP
                """, 
                    paper_data['filename'],
                    paper_data['school_code'],
                    paper_data['school_name'],
                    paper_data['year'],
                    paper_data['s3_url'],
                    paper_data['file_size']
                )
                return True
        except Exception as e:
            logger.error(f"Database insert failed for {paper_data['filename']}: {e}")
            return False
    
    def _discover_pdfs(self) -> List[Dict[str, Any]]:
        """Discover all PDFs in downloads directory."""
        pdfs = []
        downloads_dir = Config.DOWNLOAD_DIR
        
        if not downloads_dir.exists():
            logger.error(f"Downloads directory not found: {downloads_dir}")
            return pdfs
        
        for school_dir in downloads_dir.iterdir():
            if not school_dir.is_dir() or school_dir.name not in Config.SCHOOL_CODES:
                continue
                
            school_code = school_dir.name
            school_name = Config.SCHOOL_CODES[school_code]
            
            for year_dir in school_dir.iterdir():
                if not year_dir.is_dir() or not year_dir.name.isdigit():
                    continue
                    
                year = int(year_dir.name)
                
                for pdf_file in year_dir.glob('*.pdf'):
                    pdfs.append({
                        'file_path': pdf_file,
                        'filename': pdf_file.name,
                        'school_code': school_code,
                        'school_name': school_name,
                        'year': year,
                        'file_size': pdf_file.stat().st_size
                    })
        
        return pdfs
    
    async def _process_pdf(self, session, pdf_data: Dict[str, Any], semaphore) -> bool:
        """Process single PDF: upload to S3 and store in database."""
        async with semaphore:
            s3_key = self._generate_s3_key(
                pdf_data['school_code'], 
                pdf_data['year'], 
                pdf_data['filename']
            )
            
            # Upload to S3
            success = await self._upload_to_s3(session, pdf_data['file_path'], s3_key)
            if not success:
                return False
            
            # Generate public URL and store in database
            pdf_data['s3_url'] = self._generate_public_url(s3_key)
            success = await self._store_in_database(pdf_data)
            
            if success:
                logger.info(f"Processed: {pdf_data['school_code']}/{pdf_data['year']}/{pdf_data['filename']}")
            
            return success
    
    async def upload_all_pdfs(self, max_concurrent: int = 10, dry_run: bool = False) -> Dict[str, int]:
        """Upload all PDFs to S3 and store URLs in database."""
        pdfs = self._discover_pdfs()
        
        if not pdfs:
            logger.warning("No PDFs found to upload")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        logger.info(f"Found {len(pdfs)} PDFs to process")
        
        if dry_run:
            logger.info("DRY RUN - No actual uploads will be performed")
            for pdf in pdfs[:5]:  # Show sample
                s3_key = self._generate_s3_key(pdf['school_code'], pdf['year'], pdf['filename'])
                logger.info(f"Would upload: {pdf['file_path']} -> s3://{self.s3_bucket}/{s3_key}")
            return {'total': len(pdfs), 'success': 0, 'failed': 0}
        
        # Initialize database
        await self._init_database()
        
        # Initialize AWS session
        session = aioboto3.Session()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process all PDFs
        tasks = [
            self._process_pdf(session, pdf_data, semaphore) 
            for pdf_data in pdfs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        success_count = sum(1 for result in results if result is True)
        failed_count = len(results) - success_count
        
        await self._close_database()
        
        return {
            'total': len(pdfs),
            'success': success_count,
            'failed': failed_count
        }
    
    async def get_upload_stats(self) -> Dict[str, Any]:
        """Get upload statistics from database."""
        await self._init_database()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Overall stats
                total_count = await conn.fetchval("SELECT COUNT(*) FROM exam_papers")
                total_size = await conn.fetchval("SELECT SUM(file_size) FROM exam_papers") or 0
                
                # School breakdown
                school_stats = await conn.fetch("""
                    SELECT school_code, school_name, COUNT(*) as count,
                           SUM(file_size) as total_size
                    FROM exam_papers 
                    GROUP BY school_code, school_name
                    ORDER BY count DESC
                """)
                
                # Year breakdown
                year_stats = await conn.fetch("""
                    SELECT year, COUNT(*) as count
                    FROM exam_papers 
                    GROUP BY year
                    ORDER BY year DESC
                """)
                
                return {
                    'total_papers': total_count,
                    'total_size_mb': round(total_size / (1024 * 1024), 2),
                    'schools': [dict(row) for row in school_stats],
                    'years': [dict(row) for row in year_stats]
                }
        finally:
            await self._close_database()

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Upload PDFs to S3 and store URLs in database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent uploads")
    parser.add_argument("--stats", action="store_true", help="Show upload statistics")
    
    args = parser.parse_args()
    
    uploader = S3DatabaseUploader()
    
    if args.stats:
        stats = await uploader.get_upload_stats()
        print(f"Total papers: {stats['total_papers']}")
        print(f"Total size: {stats['total_size_mb']} MB")
        print(f"Schools: {len(stats['schools'])}")
        print(f"Years: {len(stats['years'])}")
        return
    
    try:
        start_time = datetime.now()
        results = await uploader.upload_all_pdfs(
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Upload completed in {duration:.2f}s")
        logger.info(f"Results: {results['success']}/{results['total']} successful")
        
        if results['failed'] > 0:
            logger.warning(f"{results['failed']} uploads failed")
        
    except Exception as e:
        logger.error(f"Upload process failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())