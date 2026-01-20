"""Dataset builders for training."""

from typing import List, Dict, Any
from pathlib import Path

from loguru import logger

from hermes.core import CleanedDocument, InstructDataset, PreferenceDataset
from hermes.storage.files import FileStorage


class InstructDatasetBuilder:
    """Builder for instruction-following datasets."""
    
    def __init__(self) -> None:
        self.file_storage = FileStorage()
    
    def build_from_documents(
        self,
        documents: List[CleanedDocument],
        instruction_template: str = "Summarize the following content:"
    ) -> List[InstructDataset]:
        """
        Build instruction dataset from documents.
        
        Args:
            documents: Source documents
            instruction_template: Instruction template
            
        Returns:
            List of instruction examples
        """
        logger.info(f"Building instruction dataset from {len(documents)} documents")
        
        datasets = []
        for doc in documents:
            # Create instruction-output pair
            dataset = InstructDataset(
                instruction=instruction_template,
                input=doc.content[:1000],  # First 1000 chars as context
                output=doc.content[:200],  # First 200 chars as summary
                metadata={
                    "document_id": str(doc.id),
                    "platform": doc.platform,
                    "author_id": str(doc.author_id)
                }
            )
            datasets.append(dataset)
        
        logger.info(f"Built {len(datasets)} instruction examples")
        return datasets
    
    def save_to_file(
        self,
        datasets: List[InstructDataset],
        output_path: Path | str
    ) -> None:
        """
        Save datasets to JSON file.
        
        Args:
            datasets: Instruction datasets
            output_path: Output file path
        """
        data = [
            {
                "instruction": ds.instruction,
                "input": ds.input,
                "output": ds.output,
                "metadata": ds.metadata
            }
            for ds in datasets
        ]
        
        self.file_storage.save_json(data, output_path)
        logger.info(f"Saved {len(datasets)} examples to {output_path}")


class PreferenceDatasetBuilder:
    """Builder for preference/RLHF datasets."""
    
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
