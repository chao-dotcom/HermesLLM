"""Output parsers for structured LLM responses."""

import json
from typing import Any, List, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from loguru import logger
from pydantic import BaseModel, ValidationError


class ListPydanticOutputParser(BaseOutputParser):
    """
    Parser that extracts and validates a list of Pydantic objects from LLM output.
    
    Handles JSON arrays and ensures each item conforms to the expected schema.
    """

    def __init__(self, pydantic_object: Type[BaseModel]) -> None:
        """
        Initialize parser with target Pydantic model.
        
        Args:
            pydantic_object: The Pydantic model class to validate against
        """
        super().__init__()
        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> List[BaseModel]:
        """
        Parse LLM output text into a list of Pydantic objects.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            List of validated Pydantic model instances
            
        Raises:
            OutputParserException: If parsing or validation fails
        """
        try:
            # Clean up the text
            text = text.strip()
            
            # Handle common markdown code block wrapping
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            try:
                json_objects = json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode failed: {e}. Attempting to extract JSON from text.")
                # Try to find JSON array in text
                start_idx = text.find("[")
                end_idx = text.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    json_text = text[start_idx:end_idx + 1]
                    json_objects = json.loads(json_text)
                else:
                    raise OutputParserException(f"Could not parse JSON from text: {text[:200]}...")
            
            # Ensure we have a list
            if not isinstance(json_objects, list):
                json_objects = [json_objects]
            
            # Validate each object
            parsed_objects = []
            for obj in json_objects:
                try:
                    validated_obj = self.pydantic_object(**obj)
                    parsed_objects.append(validated_obj)
                except ValidationError as e:
                    logger.warning(f"Validation failed for object {obj}: {e}")
                    # Skip invalid objects but continue processing
                    continue
            
            if not parsed_objects:
                raise OutputParserException("No valid objects could be parsed from the output")
            
            return parsed_objects
            
        except Exception as e:
            logger.error(f"Failed to parse output: {e}")
            raise OutputParserException(f"Failed to parse output: {str(e)}")

    def get_format_instructions(self) -> str:
        """
        Get instructions for the LLM on how to format its output.
        
        Returns:
            Format instructions string
        """
        schema = self.pydantic_object.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        return f"""Please provide your response as a JSON array of objects matching this schema:

{schema_str}

Format your response as:
[
    {{"field1": "value1", "field2": "value2", ...}},
    {{"field1": "value3", "field2": "value4", ...}}
]

Do not include any explanatory text, only the JSON array."""

    @property
    def _type(self) -> str:
        """Return the type of this parser."""
        return "list_pydantic"
