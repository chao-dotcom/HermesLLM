"""Query expansion for improved retrieval."""

from typing import List

from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from hermes.core import Query


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
