"""AI-powered dataset generators for creating training data from documents."""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from hermes.core.cleaned_documents import CleanedDocument
from hermes.core.datasets import InstructDataset, InstructDatasetSample, PreferenceDataset, PreferenceDatasetSample
from hermes.core.enums import DataCategory, DatasetType
from hermes.core.prompts import GenerateDatasetSamplesPrompt, Prompt
from hermes.datasets import constants
from hermes.datasets.output_parsers import ListPydanticOutputParser
from hermes.datasets.utils import batch_items, extract_substrings


class DatasetGenerator(ABC):
    """Base class for AI-powered dataset generation."""
    
    dataset_type: Optional[DatasetType] = None
    tokenizer = None
    
    # System prompt template
    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    
    # Main prompt template (to be defined by subclasses)
    prompt_template_str: Optional[str] = None
    
    def __init__(self, openai_api_key: Optional[str] = None, openai_model: str = "gpt-4o-mini"):
        """
        Initialize dataset generator.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            openai_model: OpenAI model to use
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        
        # Initialize tokenizer
        if self.tokenizer is None:
            try:
                DatasetGenerator.tokenizer = tiktoken.encoding_for_model(openai_model)
            except KeyError:
                # Fallback to cl100k_base encoding (used by GPT-4)
                DatasetGenerator.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @classmethod
    def get_system_prompt(cls) -> Prompt:
        """
        Get the system prompt for dataset generation.
        
        Returns:
            Formatted system prompt
        """
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"
        
        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION 
            else "instruction-answer triples with preferred and rejected responses"
        )
        
        input_variables = {"dataset_format": dataset_format}
        system_prompt = cls.system_prompt_template.format(**input_variables)
        
        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )
    
    @classmethod
    def get_prompts(
        cls, 
        documents: List[CleanedDocument],
        extract_chunks: bool = True
    ) -> Dict[DataCategory, List[GenerateDatasetSamplesPrompt]]:
        """
        Create generation prompts from documents.
        
        Args:
            documents: List of cleaned documents
            extract_chunks: Whether to extract substrings for better context
            
        Returns:
            Dictionary mapping categories to lists of prompts
        """
        if extract_chunks:
            documents = extract_substrings(documents)
        
        grouped_prompts = {}
        grouped_documents = CleanedDocument.group_by_category(documents)
        
        for category, category_documents in grouped_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts
        
        return grouped_prompts
    
    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        """
        Create a generation prompt for a single document.
        
        Args:
            document: Cleaned document to generate from
            
        Returns:
            Generation prompt
        """
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"
        
        data_category = document.get_category()
        
        # Create prompt from template
        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        
        input_variables = {"extract": document.content}
        prompt = prompt_template.format(**input_variables)
        
        # Truncate if needed
        prompt_tokens = cls.tokenizer.encode(prompt)
        max_tokens = 120000  # GPT-4o-mini max context
        
        if len(prompt_tokens) > max_tokens:
            logger.warning(f"Prompt too long ({len(prompt_tokens)} tokens), truncating to {max_tokens}")
            prompt_tokens = prompt_tokens[:max_tokens]
            prompt = cls.tokenizer.decode(prompt_tokens)
        
        return GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )
    
    def generate(
        self,
        prompts: Dict[DataCategory, List[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> Dict[str, Dict[DataCategory, InstructDataset | PreferenceDataset]]:
        """
        Generate datasets from prompts using LLM.
        
        Args:
            prompts: Dictionary of prompts grouped by category
            test_size: Proportion of data to use for test set
            mock: Whether to use mocked responses for testing
            
        Returns:
            Dictionary with 'train' and 'test' splits, each containing category-grouped datasets
        """
        assert self.dataset_type is not None, "Dataset type must be set before calling generate()"
        
        def _to_langchain(prompt: GenerateDatasetSamplesPrompt) -> List[BaseMessage]:
            """Convert prompt to LangChain message format."""
            messages = [
                SystemMessage(content=self.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]
            return messages
        
        # Initialize LLM
        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(self.dataset_type)])
        else:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key must be set to generate datasets")
            
            llm = ChatOpenAI(
                model=self.openai_model,
                api_key=self.openai_api_key,
                max_tokens=2000 if self.dataset_type == DatasetType.PREFERENCE else 1200,
                temperature=0.7,
            )
        
        # Initialize parser
        parser = ListPydanticOutputParser(pydantic_object=self._get_dataset_sample_type())
        chain = llm | parser
        
        # Generate for each category
        datasets = {}
        for category, category_prompts in prompts.items():
            logger.info(f"Generating {self.dataset_type.value} dataset for category '{category}' with {len(category_prompts)} prompts")
            
            # Convert to LangChain format
            langchain_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            
            # Batch the prompts
            batches = batch_items(langchain_prompts, constants.DEFAULT_BATCH_SIZE)
            
            # Generate samples
            all_samples = []
            for batch_idx, batch in enumerate(batches):
                logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")
                try:
                    batch_results = chain.batch(batch, stop=None)
                    
                    # Flatten results
                    for result in batch_results:
                        all_samples.extend(result)
                        
                except OutputParserException as e:
                    logger.warning(f"Failed to parse output for batch {batch_idx}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
            
            # Create dataset
            if self.dataset_type == DatasetType.INSTRUCTION:
                dataset = InstructDataset(category=category, samples=all_samples)
            else:
                dataset = PreferenceDataset(category=category, samples=all_samples)
            
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'")
        
        # Split into train/test
        return self._split_datasets(datasets, test_size)
    
    @classmethod
    def _get_dataset_sample_type(cls):
        """Get the Pydantic model type for dataset samples."""
        return (
            InstructDatasetSample 
            if cls.dataset_type == DatasetType.INSTRUCTION 
            else PreferenceDatasetSample
        )
    
    @abstractmethod
    def _split_datasets(
        self, 
        datasets: Dict[DataCategory, InstructDataset | PreferenceDataset], 
        test_size: float
    ) -> Dict[str, Dict[DataCategory, InstructDataset | PreferenceDataset]]:
        """
        Split datasets into train and test sets.
        
        Args:
            datasets: Dictionary of datasets by category
            test_size: Proportion for test set
            
        Returns:
            Dictionary with 'train' and 'test' keys
        """
        pass


class InstructionDatasetGenerator(DatasetGenerator):
    """Generator for instruction-following datasets."""
    
    dataset_type = DatasetType.INSTRUCTION
    
    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
    
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {{"instruction": "...", "answer": "..."}},
    ...
]

Extract:
{{extract}}
"""
    
    def _split_datasets(
        self, 
        datasets: Dict[DataCategory, InstructDataset], 
        test_size: float
    ) -> Dict[str, Dict[DataCategory, InstructDataset]]:
        """Split instruction datasets into train/test."""
        train_datasets = {}
        test_datasets = {}
        
        for category, dataset in datasets.items():
            train_dataset, test_dataset = dataset.train_test_split(test_size=test_size)
            train_datasets[category] = train_dataset
            test_datasets[category] = test_dataset
        
        return {
            "train": train_datasets,
            "test": test_datasets,
        }


class PreferenceDatasetGenerator(DatasetGenerator):
    """Generator for preference/RLHF datasets (DPO training)."""
    
    dataset_type = DatasetType.PREFERENCE
    
    prompt_template_str = """Based on the following extract, generate three instruction-answer triples. Each triple should contain:
1. An instruction asking about a topic from the context
2. A preferred answer that is clear, concise, and uses simple but technical language appropriate for blog posts
3. A rejected answer that is overly formal, uses complex academic language, or is too verbose

The preferred answer should:
- Use simple, direct language
- Be technically accurate but accessible
- Sound natural for a blog post or social media
- Avoid overly formal or academic tone

The rejected answer should:
- Use unnecessarily complex or formal language
- Include academic jargon or overly technical terms
- Be verbose or use complicated sentence structures
- Sound like it's from an academic paper

Both answers must be factually accurate and based on the context.

Example:
Instruction: "What is fine-tuning in machine learning?"
Preferred: "Fine-tuning takes a pre-trained model and trains it on your specific data. This lets the model adapt to your task while keeping the knowledge it already has. It's faster and works better than training from scratch."
Rejected: "Fine-tuning constitutes a sophisticated methodology whereby a pre-existing model undergoes additional optimization procedures utilizing domain-specific datasets, thereby facilitating the adaptation of learned representations to novel task specifications whilst preserving previously acquired knowledge structures."

Structure the answer in JSON format:
[
    {{"instruction": "...", "preferred_answer": "...", "rejected_answer": "..."}},
    ...
]

Extract:
{{extract}}
"""
    
    def _split_datasets(
        self, 
        datasets: Dict[DataCategory, PreferenceDataset], 
        test_size: float
    ) -> Dict[str, Dict[DataCategory, PreferenceDataset]]:
        """Split preference datasets into train/test."""
        train_datasets = {}
        test_datasets = {}
        
        for category, dataset in datasets.items():
            train_dataset, test_dataset = dataset.train_test_split(test_size=test_size)
            train_datasets[category] = train_dataset
            test_datasets[category] = test_dataset
        
        return {
            "train": train_datasets,
            "test": test_datasets,
        }
