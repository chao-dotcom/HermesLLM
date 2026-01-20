"""Query expansion for improved retrieval."""

from typing import List
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")

from hermes.core import Query


class MultiQueryExpander:
    """
    Advanced query expander that generates multiple perspectives.
    
    Based on multi-query retrieval technique to overcome distance-based
    similarity search limitations.
    """
    
    SEPARATOR = "#next-question#"
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        num_queries: int = 3,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize multi-query expander.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            num_queries: Total number of queries (including original)
            temperature: Sampling temperature
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.num_queries = num_queries
        self.temperature = temperature
        
        logger.info(f"MultiQueryExpander initialized: {num_queries} queries per expansion")
    
    def expand(self, query: Query) -> List[Query]:
        """
        Expand query into multiple perspectives.
        
        Args:
            query: Original query
            
        Returns:
            List of queries (original + expansions)
        """
        if self.num_queries <= 1:
            return [query]
        
        logger.info(f"Expanding query into {self.num_queries} perspectives")
        
        num_expansions = self.num_queries - 1
        
        prompt = f"""You are an AI language model assistant. Your task is to generate {num_expansions} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Provide these alternative questions separated by '{self.SEPARATOR}'.

Original question: {query.content}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates alternative phrasings of questions for better information retrieval.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=500,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Split by separator
            expanded_contents = result.split(self.SEPARATOR)
            
            # Create Query objects
            queries = [query]  # Start with original
            
            for content in expanded_contents:
                stripped = content.strip()
                if stripped:
                    # Preserve original query metadata in expansions
                    expanded_query = Query(
                        content=stripped,
                        metadata=query.metadata.copy()
                    )
                    expanded_query.metadata["is_expansion"] = True
                    expanded_query.metadata["original_query"] = query.content
                    queries.append(expanded_query)
            
            # Limit to requested number
            queries = queries[:self.num_queries]
            
            logger.info(f"Generated {len(queries)} query variations")
            return queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]


class LLMQueryExpander:
    """Query expander using LLM."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        num_expansions: int = 3
    ) -> None:
        """
        Initialize query expander.
        
        Args:
            model: LLM model name
            num_expansions: Number of query variations to generate
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Install with: pip install openai")
        
        self.model = model
        self.num_expansions = num_expansions
        self.client = OpenAI()
    
    def expand(self, query: Query) -> List[Query]:
        """
        Expand query into multiple variations.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries (includes original)
        """
        logger.info(f"Expanding query: {query.content}")
        
        prompt = f"""Given the following query, generate {self.num_expansions - 1} alternative ways to phrase it that would help retrieve relevant information.

Original query: {query.content}

Provide only the alternative queries, one per line, without numbering or additional explanation."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases queries for better information retrieval."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            expansions_text = response.choices[0].message.content
            expansions = [line.strip() for line in expansions_text.split("\n") if line.strip()]
            
            # Create Query objects
            expanded_queries = [query]  # Include original
            for expansion in expansions[:self.num_expansions - 1]:
                expanded_queries.append(Query(content=expansion))
            
            logger.info(f"Expanded to {len(expanded_queries)} query variations")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]  # Return original on failure


class SimpleQueryExpander:
    """Simple query expander using keyword synonyms."""
    
    def __init__(self, synonyms: dict[str, List[str]] | None = None) -> None:
        """
        Initialize simple expander.
        
        Args:
            synonyms: Dictionary mapping words to synonyms
        """
        self.synonyms = synonyms or {}
    
    def expand(self, query: Query) -> List[Query]:
        """
        Expand query by adding synonyms.
        
        Args:
            query: Original query
            
        Returns:
            List of queries (original + expanded)
        """
        logger.info(f"Simple expansion of query: {query.content}")
        
        words = query.content.lower().split()
        expanded_queries = [query]
        
        # For each word, check if we have synonyms
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded_content = query.content.lower().replace(word, synonym)
                    expanded_queries.append(Query(content=expanded_content))
        
        logger.info(f"Expanded to {len(expanded_queries)} variations")
        return expanded_queries
