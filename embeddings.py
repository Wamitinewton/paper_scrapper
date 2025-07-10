"""Enhanced embeddings generation optimized for concurrent processing."""

import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from config import Config
from models import ExamPaper, Status

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Optimized embeddings manager for high-volume concurrent processing."""
    
    def __init__(self, openai_api_key: Optional[str] = None, qdrant_url: Optional[str] = None, qdrant_api_key: Optional[str] = None):
        # Initialize OpenAI client
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key or Config.OPENAI_API_KEY)
        self.embedding_model = Config.OPENAI_MODEL
        
        # Initialize Qdrant client
        qdrant_url = qdrant_url or Config.QDRANT_URL
        if qdrant_url and ':6333' in qdrant_url:
            qdrant_url = qdrant_url.replace(':6333', '')
        
        try:
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key or Config.QDRANT_API_KEY,
                timeout=60,
                https=True,
            )
            logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise
        
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self.batch_size = Config.EMBEDDING_BATCH_SIZE
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure vector collection exists with proper configuration."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection created: {self.collection_name}")
            else:
                logger.debug(f"Collection exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Collection setup failed: {e}")
            raise
    
    def _generate_chunk_id(self, paper: ExamPaper, chunk_index: int) -> str:
        """Generate unique deterministic ID for a chunk."""
        content = f"{paper.url}_{chunk_index}_{hash(paper.chunks[chunk_index]) if chunk_index < len(paper.chunks) else 0}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_enhanced_payload(self, paper: ExamPaper, chunk_index: int, chunk_text: str) -> Dict[str, Any]:
        """Create comprehensive metadata payload for search optimization."""
        # Extract first 150 chars for preview
        chunk_preview = chunk_text[:150].replace('\n', ' ').strip()
        if len(chunk_text) > 150:
            chunk_preview += "..."
        
        payload = {
            # Paper identification
            "paper_url": paper.url,
            "paper_filename": paper.filename,
            "school_code": paper.school_code,
            "school_name": paper.school_name,
            "year": paper.year,
            
            # Chunk information
            "chunk_index": chunk_index,
            "chunk_preview": chunk_preview,
            "chunk_length": len(chunk_text),
            "chunk_word_count": len(chunk_text.split()),
            
            # Enhanced searchable fields
            "has_questions": "question" in chunk_text.lower(),
            "has_marks": "mark" in chunk_text.lower(),
            "is_instruction": "instruction" in chunk_text.lower(),
            
            # File metadata
            "file_size": paper.file_size or 0,
            "page_count": paper.page_count or 0,
            
            # Processing timestamps
            "processed_at": paper.processed_at.isoformat() if paper.processed_at else None,
            "embedded_at": datetime.now().isoformat(),
        }
        
        # Add course-specific metadata if available
        if paper.metadata:
            # Course information
            if 'course_code' in paper.metadata:
                payload['course_code'] = paper.metadata['course_code']
            if 'course_name' in paper.metadata:
                payload['course_name'] = paper.metadata['course_name']
            if 'course_prefix' in paper.metadata:
                payload['course_prefix'] = paper.metadata['course_prefix']
            
            # Academic context
            if 'degree_program' in paper.metadata:
                payload['degree_program'] = paper.metadata['degree_program']
            if 'semester' in paper.metadata:
                payload['semester'] = paper.metadata['semester']
            if 'study_year' in paper.metadata:
                payload['study_year'] = paper.metadata['study_year']
            
            # Exam specifics
            if 'exam_duration_hours' in paper.metadata:
                payload['exam_duration_hours'] = paper.metadata['exam_duration_hours']
            if 'total_questions' in paper.metadata:
                payload['total_questions'] = paper.metadata['total_questions']
        
        return payload
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text batch with error handling."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
    
    def _paper_needs_embedding(self, paper: ExamPaper) -> bool:
        """Check if paper requires embedding generation."""
        return (
            paper.processing_status == Status.COMPLETED and 
            paper.chunks and 
            len(paper.chunks) > 0 and
            paper.embedding_status != Status.COMPLETED
        )
    
    async def embed_paper(self, paper: ExamPaper) -> ExamPaper:
        """Generate embeddings for a single paper with optimizations."""
        if not self._paper_needs_embedding(paper):
            if not paper.chunks:
                paper.embedding_status = Status.FAILED
                paper.last_error = "No chunks available"
            return paper
        
        paper.embedding_status = Status.IN_PROGRESS
        
        try:
            points = []
            
            # Process chunks in batches
            for i in range(0, len(paper.chunks), self.batch_size):
                batch_chunks = paper.chunks[i:i + self.batch_size]
                batch_indices = list(range(i, min(i + self.batch_size, len(paper.chunks))))
                
                # Generate embeddings for batch
                embeddings = await self._generate_embeddings_batch(batch_chunks)
                
                # Create vector points
                for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = batch_indices[j]
                    point_id = self._generate_chunk_id(paper, chunk_index)
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=self._create_enhanced_payload(paper, chunk_index, chunk_text)
                    )
                    points.append(point)
            
            if points:
                # Batch upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                paper.embedding_status = Status.COMPLETED
                paper.embedded_at = datetime.now()
                
                logger.info(f"Embedded {paper.filename}: {len(points)} vectors")
            else:
                paper.embedding_status = Status.FAILED
                paper.last_error = "No points generated"
            
        except Exception as e:
            paper.embedding_status = Status.FAILED
            paper.last_error = str(e)
            logger.error(f"Embedding failed for {paper.filename}: {e}")
        
        return paper
    
    async def embed_papers_concurrent(self, papers: List[ExamPaper], max_concurrent: int = 3) -> List[ExamPaper]:
        """Embed multiple papers with controlled concurrency."""
        papers_to_embed = [p for p in papers if self._paper_needs_embedding(p)]
        
        if not papers_to_embed:
            return papers
        
        logger.info(f"Embedding {len(papers_to_embed)} papers")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def embed_with_semaphore(paper):
            async with semaphore:
                return await self.embed_paper(paper)
        
        # Process papers concurrently
        tasks = [embed_with_semaphore(paper) for paper in papers_to_embed]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        embedded_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Embedding task failed: {result}")
            else:
                embedded_papers.append(result)
        
        # Update original papers
        paper_dict = {p.url: p for p in papers}
        for embedded_paper in embedded_papers:
            paper_dict[embedded_paper.url] = embedded_paper
        
        successful = sum(1 for p in embedded_papers if p.embedding_status == Status.COMPLETED)
        failed = sum(1 for p in embedded_papers if p.embedding_status == Status.FAILED)
        
        logger.info(f"Embedding complete: {successful} successful, {failed} failed")
        
        return list(paper_dict.values())
    
    async def search_similar(self, query: str, limit: int = 10, school_filter: Optional[str] = None, 
                           year_filter: Optional[int] = None, course_filter: Optional[str] = None,
                           score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Enhanced semantic search with multiple filtering options."""
        try:
            # Generate query embedding
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Build filter conditions
            must_conditions = []
            if school_filter:
                must_conditions.append(
                    FieldCondition(key="school_code", match=MatchValue(value=school_filter))
                )
            if year_filter:
                must_conditions.append(
                    FieldCondition(key="year", match=MatchValue(value=year_filter))
                )
            if course_filter:
                must_conditions.append(
                    FieldCondition(key="course_code", match=MatchValue(value=course_filter))
                )
            
            query_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Perform vector search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "score": hit.score,
                    "metadata": hit.payload,
                    "chunk_id": hit.id
                })
            
            logger.info(f"Search completed: {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Sample points for statistics
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            stats = {
                "total_vectors": collection_info.vectors_count,
                "collection_status": collection_info.status,
                "school_distribution": {},
                "year_distribution": {},
                "course_distribution": {},
                "sample_size": 0
            }
            
            if scroll_result and len(scroll_result) > 0:
                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
                
                for point in points:
                    stats["sample_size"] += 1
                    payload = point.payload
                    
                    # Count by school
                    school = payload.get("school_code", "unknown")
                    stats["school_distribution"][school] = stats["school_distribution"].get(school, 0) + 1
                    
                    # Count by year
                    year = payload.get("year", "unknown")
                    stats["year_distribution"][str(year)] = stats["year_distribution"].get(str(year), 0) + 1
                    
                    # Count by course
                    course = payload.get("course_code", "unknown")
                    stats["course_distribution"][course] = stats["course_distribution"].get(course, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def delete_paper_embeddings(self, paper: ExamPaper) -> bool:
        """Remove all embeddings for a specific paper."""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(key="paper_url", match=MatchValue(value=paper.url))
                ]
            )
            
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"Deleted embeddings for: {paper.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for {paper.filename}: {e}")
            return False