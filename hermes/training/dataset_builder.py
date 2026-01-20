"""Dataset builders for training."""

import os
from typing import List, Dict, Any
from pathlib import Path

from loguru import logger

from hermes.core import (
    CleanedDocument, 
    InstructDataset, 
    PreferenceDataset,
    InstructDatasetSample,
    PreferenceDatasetSample,
    DatasetType,
)
from hermes.datasets.generators import InstructionDatasetGenerator, PreferenceDatasetGenerator
from hermes.storage.files import FileStorage


class InstructDatasetBuilder:
    """Builder for instruction-following datasets with AI generation support."""
    
    def __init__(
        self, 
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini"
    ) -> None:
        """
        Initialize builder.
        
        Args:
            openai_api_key: OpenAI API key for AI generation
            openai_model: Model to use for generation
        """
        self.file_storage = FileStorage()
        self.generator = InstructionDatasetGenerator(
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            openai_model=openai_model
        )
    
    def generate_from_documents(
        self,
        documents: List[CleanedDocument],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> Dict[str, Dict[str, InstructDataset]]:
        """
        Generate instruction dataset from documents using AI.
        
        Args:
            documents: Source documents to generate from
            test_size: Proportion for test split
            mock: Whether to use mocked responses for testing
            
        Returns:
            Dictionary with 'train' and 'test' splits containing datasets by category
        """
        logger.info(f"Generating instruction dataset from {len(documents)} documents")
        
        # Create prompts from documents
        prompts = self.generator.get_prompts(documents, extract_chunks=True)
        
        # Generate datasets using LLM
        split_datasets = self.generator.generate(
            prompts=prompts,
            test_size=test_size,
            mock=mock
        )
        
        total_train = sum(d.num_samples for d in split_datasets["train"].values())
        total_test = sum(d.num_samples for d in split_datasets["test"].values())
        
        logger.info(f"Generated {total_train} training samples and {total_test} test samples")
        
        return split_datasets
    
    def build_from_documents(
        self,
        documents: List[CleanedDocument],
        instruction_template: str = "Summarize the following content:"
    ) -> List[InstructDatasetSample]:
        """
        Build instruction dataset from documents (simple templating).
        
        This is a simple method that creates basic instruction-answer pairs
        without AI generation. Use generate_from_documents() for AI-powered generation.
        
        Args:
            documents: Source documents
            instruction_template: Instruction template
            
        Returns:
            List of instruction examples
        """
        logger.info(f"Building instruction dataset from {len(documents)} documents")
        
        samples = []
        for doc in documents:
            # Create instruction-output pair
            sample = InstructDatasetSample(
                instruction=instruction_template,
                answer=doc.content[:200],  # First 200 chars as summary
                input=doc.content[:1000],  # First 1000 chars as context
                metadata={
                    "document_id": str(doc.id),
                    "platform": doc.platform,
                    "author_id": str(doc.author_id)
                }
            )
            samples.append(sample)
        
        logger.info(f"Built {len(samples)} instruction examples")
        return samples
    Sample] | Dict[str, Dict[str, InstructDataset]],
        output_path: Path | str
    ) -> None:
        """
        Save datasets to JSON file.
        
        Args:
            datasets: Instruction datasets or samples
            output_path: Output file path
        """
        if isinstance(datasets, dict):
            # Handle split datasets
            data = {
                "train": {},
                "test": {}
            }
            for split in ["train", "test"]:
                for category, dataset in datasets[split].items():
                    data[split][category] = [
                        {
                            "instruction": sample.instruction,
                            "answer": sample.answer if hasattr(sample, "answer") else sample.output,
                            "metadata": sample.metadata if hasattr(sample, "metadata") else {}
                        }
                        for sample in dataset.samples
                    ]
        else:
            # Handle list of samples
            data = [
                {
                    "instruction": sample.instruction,
                    "answer": sample.answer,
                    "input": sample.input,
                    "metadata": sample.metadata
                }
                for sample in datasets
            ]
        
        self.file_storage.save_Sample]:
        """
        Build preference dataset from response pairs (simple method).
        
        Use generate_from_documents() for AI-powered generation.
        
        Args:
            prompts: List of prompts
            chosen_responses: Preferred responses
            rejected_responses: Rejected responses
            
        Returns:
            List of preference examples
        """
        logger.info(f"Building preference dataset from {len(prompts)} pairs")
        
        samples = []
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            sample = PreferenceDatasetSample(
                instruction=prompt,
                preferred_answer=chosen,
                rejected_answer=rejected
            )
            samples.append(sample)
        
        logger.info(f"Built {len(samples)} preference examples")
        return samples
    
    def save_to_file(
        self,
        datasets: List[PreferenceDatasetSample] | Dict[str, Dict[str, PreferenceDataset]],
        output_path: Path | str
    ) -> None:
        """
        Save datasets to JSON file.
        
        Args:
            datasets: Preference datasets or samples
            output_path: Output file path
        """
        if isinstance(datasets, dict):
            # Handle split datasets
            data = {
                "train": {},
                "test": {}
            }
            for split in ["train", "test"]:
                for category, dataset in datasets[split].items():
                    data[split][category] = [
                        {
                            "instruction": sample.instruction,
                            "preferred_answer": sample.preferred_answer if hasattr(sample, "preferred_answer") else sample.chosen,
                            "rejected_answer": sample.rejected_answer if hasattr(sample, "rejected_answer") else sample.rejected,
                            "metadata": sample.metadata if hasattr(sample, "metadata") else {}
                        }
                        for sample in dataset.samples
                    ]
        else:
            # Handle list of samples
            data = [
                {
                    "instruction": sample.instruction,
                    "preferred_answer": sample.preferred_answer,
                    "rejected_answer": sample.rejected_answer,
                    "metadata": sample.metadata
                }
                for sample in datasets
            ]
        
        self.file_storage.save_json(data, output_path)
        logger.info(f"Saved dataset
            test_size=test_size,
            mock=mock
        )
        
        total_train = sum(d.num_samples for d in split_datasets["train"].values())
        total_test = sum(d.num_samples for d in split_datasets["test"].values())
        
        logger.info(f"Generated {total_train} training samples and {total_test} test samples")
        
        return split_datasetsts."""
    
    def __init__(self) -> None:
        self.file_storage = FileStorage()
    
    def build_from_pairs(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str]
    ) -> List[PreferenceDataset]:
        """
        Build preference dataset from response pairs.
        
        Args:
            prompts: List of prompts
            chosen_responses: Preferred responses
            rejected_responses: Rejected responses
            
        Returns:
            List of preference examples
        """
        logger.info(f"Building preference dataset from {len(prompts)} pairs")
        
        datasets = []
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            dataset = PreferenceDataset(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected
            )
            datasets.append(dataset)
        
        logger.info(f"Built {len(datasets)} preference examples")
        return datasets
    
    def save_to_file(
        self,
        datasets: List[PreferenceDataset],
        output_path: Path | str
    ) -> None:
        """
        Save datasets to JSON file.
        
        Args:
            datasets: Preference datasets
            output_path: Output file path
        """
        data = [
            {
                "prompt": ds.prompt,
                "chosen": ds.chosen,
                "rejected": ds.rejected,
                "metadata": ds.metadata
            }
            for ds in datasets
        ]
        
        self.file_storage.save_json(data, output_path)
        logger.info(f"Saved {len(datasets)} examples to {output_path}")
