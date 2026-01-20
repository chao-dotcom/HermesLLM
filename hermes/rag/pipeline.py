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
from hermes.rag.query_expander import LLMQueryExpander


class RAGPipeline:
    """Complete RAG pipeline."""
    
    def __init__(
        self,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        retriever: Optional[VectorRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        query_expander: Optional[LLMQueryExpander] = None,
        llm_model: str = "gpt-4o-mini"
    ) -> None:
        """
        Initialize RAG pipeline.
        
        Args:
            embedder: Query embedder
            retriever: Document retriever
            reranker: Chunk reranker
            query_expander: Query expander
            llm_model: LLM model for generation
        """
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.retriever = retriever or VectorRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.query_expander = query_expander
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
        system_prompt: str | None = None
    ) -> str:
        """
        Execute RAG query.
        
        Args:
            query: Query string or Query object
            use_query_expansion: Whether to expand query
            system_prompt: Optional system prompt for LLM
            
        Returns:
            Generated response
        """
        # Convert to Query object
        if isinstance(query, str):
            query = Query(content=query)
        
        logger.info(f"Processing RAG query: {query.content}")
        
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
