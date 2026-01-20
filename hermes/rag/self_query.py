"""Self-query extraction for RAG - extracts metadata from queries."""

import json
from typing import Optional, Dict, Any
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")

from hermes.core import Query


class SelfQueryExtractor:
    """
    Extract metadata from queries using LLM.
    
    Extracts information like:
    - Author/user identification
    - Category/topic
    - Intent
    - Filters/constraints
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        extract_author: bool = True,
        extract_category: bool = True,
        extract_intent: bool = True,
    ) -> None:
        """
        Initialize self-query extractor.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
            extract_author: Extract author/user information
            extract_category: Extract category/topic
            extract_intent: Extract query intent
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.extract_author = extract_author
        self.extract_category = extract_category
        self.extract_intent = extract_intent
        
        logger.info("SelfQueryExtractor initialized")
    
    def extract(self, query: Query) -> Query:
        """
        Extract metadata from query.
        
        Args:
            query: Input query
            
        Returns:
            Query with enriched metadata
        """
        logger.info(f"Extracting metadata from query: {query.content[:50]}...")
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(query.content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from text. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            metadata = json.loads(response.choices[0].message.content)
            
            # Update query metadata
            if self.extract_author and metadata.get("author"):
                author_info = metadata["author"]
                if author_info.get("name"):
                    query.metadata["author_name"] = author_info["name"]
                    
                    # Parse name
                    first_name, last_name = self._parse_full_name(author_info["name"])
                    query.metadata["author_first_name"] = first_name
                    query.metadata["author_last_name"] = last_name
                
                if author_info.get("user_id"):
                    query.metadata["author_id"] = author_info["user_id"]
            
            if self.extract_category and metadata.get("category"):
                query.metadata["category"] = metadata["category"]
            
            if self.extract_intent and metadata.get("intent"):
                query.metadata["intent"] = metadata["intent"]
            
            # Store raw extracted metadata
            query.metadata["extracted_metadata"] = metadata
            
            logger.info(f"Extracted metadata: {metadata}")
            return query
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return query
    
    def _build_extraction_prompt(self, query_content: str) -> str:
        """Build prompt for metadata extraction."""
        
        fields = []
        
        if self.extract_author:
            fields.append("""
    "author": {
        "name": "<full name if mentioned, or null>",
        "user_id": "<user ID if mentioned, or null>"
    }""")
        
        if self.extract_category:
            fields.append('    "category": "<topic/category of the query, or null>"')
        
        if self.extract_intent:
            fields.append('    "intent": "<what the user wants to do: \'search\', \'generate\', \'summarize\', etc., or null>"')
        
        fields_str = ",\n".join(fields)
        
        return f"""Extract the following information from the user's query. If information is not present, use null.

Query: "{query_content}"

Respond with JSON in this format:
{{
{fields_str}
}}

Examples:
1. Query: "I am John Smith and I want to write about AI"
   Response: {{"author": {{"name": "John Smith", "user_id": null}}, "category": "AI", "intent": "generate"}}

2. Query: "My user ID is 12345. Search for machine learning papers"
   Response: {{"author": {{"name": null, "user_id": "12345"}}, "category": "machine learning", "intent": "search"}}

3. Query: "What are the best RAG techniques?"
   Response: {{"author": {{"name": null, "user_id": null}}, "category": "RAG techniques", "intent": "search"}}"""
    
    def _parse_full_name(self, full_name: str) -> tuple[str, str]:
        """
        Parse full name into first and last name.
        
        Args:
            full_name: Full name string
            
        Returns:
            Tuple of (first_name, last_name)
        """
        if not full_name:
            return "", ""
        
        name_parts = full_name.strip().split()
        
        if len(name_parts) == 0:
            return "", ""
        elif len(name_parts) == 1:
            return name_parts[0], name_parts[0]
        else:
            # First name is everything except last word, last name is last word
            first_name = " ".join(name_parts[:-1])
            last_name = name_parts[-1]
            return first_name, last_name


class AuthorExtractor:
    """
    Specialized extractor for author/user information.
    
    Simpler and faster than full SelfQueryExtractor when you only need author info.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """
        Initialize author extractor.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def extract(self, query: Query) -> Query:
        """
        Extract author information from query.
        
        Args:
            query: Input query
            
        Returns:
            Query with author metadata
        """
        logger.info(f"Extracting author from: {query.content[:50]}...")
        
        prompt = f"""You are an AI language model assistant. Your task is to extract information from a user question.
The required information that needs to be extracted is the user name or user id. 
Your response should consist of only the extracted user name (e.g., John Doe) or id (e.g. 1345256), nothing else.
If the user question does not contain any user name or id, you should return the following token: none.

For example:
QUESTION 1:
My name is Paul Iusztin and I want a post about...
RESPONSE 1:
Paul Iusztin

QUESTION 2:
I want to write a post about...
RESPONSE 2:
none

QUESTION 3:
My user id is 1345256 and I want to write a post about...
RESPONSE 3:
1345256

User question: {query.content}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts user information from queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            
            result = response.choices[0].message.content.strip()
            
            if result.lower() != "none":
                # Check if it's a numeric ID
                if result.isdigit():
                    query.metadata["author_id"] = result
                    logger.info(f"Extracted author ID: {result}")
                else:
                    # It's a name
                    query.metadata["author_name"] = result
                    first_name, last_name = self._parse_full_name(result)
                    query.metadata["author_first_name"] = first_name
                    query.metadata["author_last_name"] = last_name
                    logger.info(f"Extracted author name: {result}")
            else:
                logger.info("No author information found")
            
            return query
            
        except Exception as e:
            logger.error(f"Author extraction failed: {e}")
            return query
    
    def _parse_full_name(self, full_name: str) -> tuple[str, str]:
        """Parse full name into first and last name."""
        if not full_name:
            return "", ""
        
        name_parts = full_name.strip().split()
        
        if len(name_parts) == 0:
            return "", ""
        elif len(name_parts) == 1:
            return name_parts[0], name_parts[0]
        else:
            first_name = " ".join(name_parts[:-1])
            last_name = name_parts[-1]
            return first_name, last_name


class MetadataEnricher:
    """
    Enrich queries with additional metadata.
    
    Adds computed metadata like:
    - Query length
    - Complexity score
    - Language detection
    - Keywords extraction
    """
    
    def __init__(
        self,
        extract_keywords: bool = True,
        compute_complexity: bool = True,
    ) -> None:
        """
        Initialize metadata enricher.
        
        Args:
            extract_keywords: Extract keywords from query
            compute_complexity: Compute query complexity score
        """
        self.extract_keywords = extract_keywords
        self.compute_complexity = compute_complexity
    
    def enrich(self, query: Query) -> Query:
        """
        Enrich query with metadata.
        
        Args:
            query: Input query
            
        Returns:
            Query with enriched metadata
        """
        # Basic metadata
        query.metadata["query_length"] = len(query.content)
        query.metadata["word_count"] = len(query.content.split())
        
        # Extract keywords (simple version - most common words)
        if self.extract_keywords:
            keywords = self._extract_keywords(query.content)
            query.metadata["keywords"] = keywords
        
        # Compute complexity
        if self.compute_complexity:
            complexity = self._compute_complexity(query.content)
            query.metadata["complexity_score"] = complexity
        
        return query
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """Extract top keywords from text."""
        # Simple keyword extraction (remove common stop words)
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "have", "has", "had", "do", "does", "did", "will", "would",
            "should", "could", "may", "might", "can", "about", "what", "which",
            "who", "when", "where", "how", "why", "i", "you", "he", "she", "it",
            "we", "they", "my", "your", "his", "her", "its", "our", "their",
        }
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def _compute_complexity(self, text: str) -> float:
        """
        Compute query complexity score (0-1).
        
        Based on:
        - Word count
        - Average word length
        - Sentence count
        """
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / word_count
        
        # Sentence count (approximate)
        sentence_count = text.count('.') + text.count('?') + text.count('!')
        if sentence_count == 0:
            sentence_count = 1
        
        # Complexity score (normalized)
        # More words, longer words, more sentences = higher complexity
        complexity = min(1.0, (
            (word_count / 50) * 0.4 +  # Word count factor
            (avg_word_length / 10) * 0.3 +  # Word length factor
            (sentence_count / 3) * 0.3  # Sentence factor
        ))
        
        return round(complexity, 2)
