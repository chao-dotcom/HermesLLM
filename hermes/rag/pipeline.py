"""RAG pipeline orchestrating retrieval, reranking, and generation."""

from typing import List, Optional

from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from hermes.core import Query, EmbeddedQuery, Prompt
from hermes.core.chunks import Chunk
from hermes.processing.embedders import SentenceTransformerEmbedder
from hermes.rag.retriever import VectorRetriever
from hermes.rag.reranker import CrossEncoderReranker
from hermes.rag.query_expander import LLMQueryExpander, MultiQueryExpander
from hermes.rag.self_query import SelfQueryExtractor, AuthorExtractor, MetadataEnricher


class RAGPipeline:
    """Complete RAG pipeline with self-query and metadata enrichment."""
    
    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        retriever: Optional[VectorRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        query_expander: Optional[LLMQueryExpander] = None,
        self_query_extractor: Optional[SelfQueryExtractor] = None,
        author_extractor: Optional[AuthorExtractor] = None,
        metadata_enricher: Optional[MetadataEnricher] = None,
        llm_model: str = "gpt-4o-mini"
    ) -> None:
        """
        Initialize RAG pipeline.
        
        Args:
            embedder: Query embedder
            retriever: Document retriever
            reranker: Chunk reranker
            query_expander: Query expander
            self_query_extractor: Self-query metadata extractor
            author_extractor: Author information extractor
            metadata_enricher: Query metadata enricher
            llm_model: LLM model for generation
        """
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.retriever = retriever or VectorRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.query_expander = query_expander
        self.self_query_extractor = self_query_extractor
        self.author_extractor = author_extractor
        self.metadata_enricher = metadata_enricher
        self.llm_model = llm_model
        
        if OPENAI_AVAILABLE:
            self.client = OpenAI()
        else:
            logger.warning("OpenAI not available. Install with: pip install openai")
            self.client = None
    
    def query(
        self,
        query: str | Query,
        use_query_expansion: bool = False,
        use_self_query: bool = False,
        use_author_extraction: bool = False,
        use_metadata_enrichment: bool = False,
        system_prompt: str | None = None
    ) -> str:
        """
        Execute RAG query with optional enhancements.
        
        Args:
            query: Query string or Query object
            use_query_expansion: Whether to expand query
            use_self_query: Whether to extract metadata
            use_author_extraction: Whether to extract author info
            use_metadata_enrichment: Whether to enrich with metadata
            system_prompt: Optional system prompt for LLM
            
        Returns:
            Generated response
        """
        # Convert to Query object
        if isinstance(query, str):
            query = Query(content=query)
        
        logger.info(f"Processing RAG query: {query.content}")
        
        # Extract author if requested
        if use_author_extraction and self.author_extractor:
            query = self.author_extractor.extract(query)
        
        # Extract metadata if requested
        if use_self_query and self.self_query_extractor:
            query = self.self_query_extractor.extract(query)
        
        # Enrich with metadata if requested
        if use_metadata_enrichment and self.metadata_enricher:
            query = self.metadata_enricher.enrich(query)
        
        # Log enriched metadata
        if query.metadata:
            logger.info(f"Query metadata: {query.metadata}")
        
        # Expand query if requested
        queries = [query]
        if use_query_expansion and self.query_expander:
            queries = self.query_expander.expand(query)
            logger.info(f"Expanded to {len(queries)} query variations")
        
        # Embed queries
        embedded_queries = []
        for q in queries:
            # Create chunk-like object for embedding
            from hermes.core import Chunk
            chunk = Chunk(content=q.content, index=0)
            embedded = self.embedder.handle([chunk])[0]
            embedded_query = EmbeddedQuery(
                content=q.content,
                embedding=embedded.embedding
            )
            embedded_queries.append(embedded_query)
        
        # Retrieve chunks for each query
        all_chunks = []
        for eq in embedded_queries:
            chunks = self.retriever.retrieve(eq)
            all_chunks.extend(chunks)
        
        # Deduplicate chunks by ID
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(unique_chunks)} unique chunks")
        
        # Rerank chunks
        if unique_chunks:
            reranked_chunks = self.reranker.rerank(query, unique_chunks)
        else:
            reranked_chunks = []
            logger.warning("No chunks retrieved")
        
        # Generate response
        response = self._generate_response(
            query=query,
            context_chunks=reranked_chunks,
            system_prompt=system_prompt
        )
        
        return response
    
    def _generate_response(
        self,
        query: Query,
        context_chunks: List[Chunk],
        system_prompt: str | None = None
    ) -> str:
        """
        Generate response using LLM.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        if not self.client:
            return "OpenAI client not available. Please install: pip install openai"
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the context doesn't contain enough information to answer the question, "
                "say so honestly rather than making up information."
            )
        
        # Build user message
        user_message = f"""Context:
{context}

Question: {query.content}

Please provide a helpful answer based on the context above."""
        
        logger.info("Generating LLM response")
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            logger.info("Generated response successfully")
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {e}"
