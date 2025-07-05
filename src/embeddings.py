"""OpenAI embeddings generation and Qdrant vector store management."""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.asyncio import tqdm

from .config import Config
from .models import ExamPaper
from .logger import get_logger

logger = get_logger(__name__)

class EmbeddingsManager:
    """Manage embeddings generation and vector store operations."""
    
    def __init__(self, openai_api_key: str = None, qdrant_url: str = None, qdrant_api_key: str = None):
        """Initialize the embeddings manager."""
        
        # OpenAI configuration
        self.openai_client = openai.AsyncOpenAI(
            api_key=openai_api_key or Config.OPENAI_API_KEY
        )
        self.embedding_model = Config.OPENAI_MODEL
        
        # Qdrant configuration
        self.qdrant_client = QdrantClient(
            url=qdrant_url or Config.QDRANT_URL,
            api_key=qdrant_api_key or Config.QDRANT_API_KEY,
        )
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        
        # Batch processing
        self.batch_size = Config.BATCH_SIZE
        
        # Initialize collection
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                
                # Create collection with appropriate vector configuration
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def _generate_chunk_id(self, paper: ExamPaper, chunk_index: int) -> str:
        """Generate a unique ID for a text chunk."""
        # Create a deterministic ID based on paper URL and chunk index
        content = f"{paper.url}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def _create_point_metadata(self, paper: ExamPaper, chunk_index: int, chunk_text: str) -> Dict[str, Any]:
        """Create metadata for a vector point."""
        return {
            # Paper information
            "paper_url": paper.url,
            "paper_filename": paper.filename,
            "school_code": paper.school_code,
            "school_name": paper.school_full_name,
            "year": paper.year,
            
            # Chunk information
            "chunk_index": chunk_index,
            "chunk_text": chunk_text[:500],  # Store first 500 chars for preview
            "chunk_length": len(chunk_text),
            
            # File information
            "file_size": paper.file_size,
            "page_count": paper.page_count,
            
            # Processing information
            "processed_at": paper.processed_at.isoformat() if paper.processed_at else None,
            "downloaded_at": paper.downloaded_at.isoformat() if paper.downloaded_at else None,
            
            # Additional metadata
            **paper.metadata
        }
    
    async def process_paper_embeddings(self, paper: ExamPaper) -> ExamPaper:
        """Generate embeddings for a single paper's chunks."""
        if not paper.chunks:
            logger.warning(f"No chunks found for paper: {paper.filename}")
            return paper
        
        logger.info(f"Generating embeddings for {paper.filename} ({len(paper.chunks)} chunks)")
        
        try:
            # Process chunks in batches
            points = []
            
            for i in range(0, len(paper.chunks), self.batch_size):
                batch_chunks = paper.chunks[i:i + self.batch_size]
                batch_indices = list(range(i, min(i + self.batch_size, len(paper.chunks))))
                
                # Generate embeddings for batch
                embeddings = await self._generate_embeddings_batch(batch_chunks)
                
                # Create points for batch
                for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = batch_indices[j]
                    point_id = self._generate_chunk_id(paper, chunk_index)
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=self._create_point_metadata(paper, chunk_index, chunk_text)
                    )
                    points.append(point)
            
            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Update paper with embedding information
            paper.embedding_id = self._generate_chunk_id(paper, 0)  # Use first chunk ID as paper ID
            
            logger.info(f"Successfully uploaded {len(points)} embeddings for {paper.filename}")
            
        except Exception as e:
            logger.error(f"Error processing embeddings for {paper.filename}: {e}")
            raise
        
        return paper
    
    async def process_papers_embeddings(self, papers: List[ExamPaper]) -> List[ExamPaper]:
        """Generate embeddings for multiple papers."""
        if not papers:
            return []
        
        papers_to_embed = [
            p for p in papers 
            if p.is_processed and p.chunks and not p.embedding_id
        ]
        
        if not papers_to_embed:
            logger.info("No papers need embedding generation")
            return papers
        
        logger.info(f"Generating embeddings for {len(papers_to_embed)} papers")
        
        semaphore = asyncio.Semaphore(3)  # Limit concurrent OpenAI requests
        
        async def process_with_semaphore(paper):
            async with semaphore:
                return await self.process_paper_embeddings(paper)
        
        # Process papers with progress bar
        tasks = [process_with_semaphore(paper) for paper in papers_to_embed]
        results = await tqdm.gather(*tasks, desc="Generating embeddings")
        
        # Update papers list
        paper_dict = {p.url: p for p in papers}
        for embedded_paper in results:
            paper_dict[embedded_paper.url] = embedded_paper
        
        logger.info(f"Embedding generation completed for {len(results)} papers")
        
        return list(paper_dict.values())
    
    def search_similar(self, query: str, limit: int = 10, school_filter: str = None, year_filter: int = None) -> List[Dict[str, Any]]:
        """Search for similar content in the vector store."""
        try:
            # Generate embedding for query
            loop = asyncio.get_event_loop()
            if loop.is_running():
                query_embedding = None 
            else:
                query_embedding = asyncio.run(self._generate_embedding(query))
            
            # Build filter conditions
            filter_conditions = {}
            if school_filter:
                filter_conditions["school_code"] = school_filter
            if year_filter:
                filter_conditions["year"] = year_filter
            
            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_conditions if filter_conditions else None
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "metadata": hit.payload,
                    "id": hit.id
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Get count by school
            school_counts = {}
            year_counts = {}

            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000 
            )
            
            for point in scroll_result[0]:
                school = point.payload.get("school_code", "unknown")
                year = point.payload.get("year", "unknown")
                
                school_counts[school] = school_counts.get(school, 0) + 1
                year_counts[year] = year_counts.get(year, 0) + 1
            
            return {
                "total_vectors": collection_info.vectors_count,
                "status": collection_info.status,
                "school_distribution": school_counts,
                "year_distribution": year_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_paper_embeddings(self, paper: ExamPaper) -> bool:
        """Delete all embeddings for a specific paper."""
        try:
            # Delete points with matching paper URL
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "paper_url", "match": {"value": paper.url}}
                        ]
                    }
                }
            )
            logger.info(f"Deleted embeddings for paper: {paper.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for {paper.filename}: {e}")
            return False