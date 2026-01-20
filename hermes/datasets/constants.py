"""Constants for dataset generation."""

from hermes.core.enums import DatasetType


# Mocked responses for testing
MOCKED_INSTRUCTION_RESPONSE = """[
    {
        "instruction": "Explain what a Large Language Model (LLM) is.",
        "answer": "A Large Language Model is a type of AI model trained on vast amounts of text data. These models can understand and generate human-like text, making them useful for tasks like writing, translation, and question-answering. They work by predicting the next word in a sequence based on patterns learned during training."
    },
    {
        "instruction": "Describe the concept of fine-tuning in machine learning.",
        "answer": "Fine-tuning is the process of taking a pre-trained model and training it further on a specific dataset. This allows the model to adapt to a particular task or domain while retaining the general knowledge it gained during initial training. It's more efficient than training from scratch and often yields better results."
    },
    {
        "instruction": "What are embeddings in natural language processing?",
        "answer": "Embeddings are numerical representations of text that capture semantic meaning. They convert words, sentences, or documents into vectors of numbers that similar meanings are close together in vector space. This makes it easier for machine learning models to process and understand text data."
    }
]"""

MOCKED_PREFERENCE_RESPONSE = """[
    {
        "instruction": "Explain what RAG (Retrieval-Augmented Generation) means.",
        "preferred_answer": "RAG is a technique that combines retrieval and generation. It first searches a knowledge base for relevant information, then uses that context to generate accurate responses. This approach helps reduce hallucinations and keeps answers grounded in real data.",
        "rejected_answer": "RAG is a complicated AI system that uses advanced algorithms to retrieve information from databases and then generates responses using complex neural networks and transformer architectures with attention mechanisms."
    },
    {
        "instruction": "What is the purpose of vector databases?",
        "preferred_answer": "Vector databases store and search embeddings efficiently. They let you find similar items quickly using vector similarity rather than exact matches. This is essential for applications like semantic search and recommendation systems.",
        "rejected_answer": "Vector databases constitute an innovative paradigm in data storage solutions, leveraging multidimensional vector representations to facilitate high-dimensional similarity searches through sophisticated indexing mechanisms and approximate nearest neighbor algorithms."
    }
]"""


def get_mocked_response(dataset_type: DatasetType) -> str:
    """
    Get a mocked response for testing dataset generation.
    
    Args:
        dataset_type: Type of dataset being generated
        
    Returns:
        Mocked JSON response string
    """
    if dataset_type == DatasetType.INSTRUCTION:
        return MOCKED_INSTRUCTION_RESPONSE
    elif dataset_type == DatasetType.PREFERENCE:
        return MOCKED_PREFERENCE_RESPONSE
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# Default chunk sizes for extraction
DEFAULT_MIN_CHUNK_LENGTH = 500
DEFAULT_MAX_CHUNK_LENGTH = 2000

# Number of samples to request per generation
DEFAULT_SAMPLES_PER_GENERATION = 5

# Batch size for LLM requests
DEFAULT_BATCH_SIZE = 24
