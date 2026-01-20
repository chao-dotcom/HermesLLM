"""Model inference and prediction."""

from typing import List, Dict, Any, Optional
from pathlib import Path

from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LocalPredictor:
    """Predictor for local fine-tuned models."""
    
    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto"
    ) -> None:
        """
        Initialize local predictor.
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to use (auto/cpu/cuda)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required. Install: pip install transformers torch")
        
        self.model_path = Path(model_path)
        logger.info(f"Loading model from {self.model_path}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        
        # Create pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
    
    def predict(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate predictions for prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        logger.info(f"Generating prediction for prompt: {prompt[:50]}...")
        
        results = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True
        )
        
        predictions = [result["generated_text"] for result in results]
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def batch_predict(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[str]]:
        """
        Generate predictions for multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List of prediction lists
        """
        logger.info(f"Batch predicting for {len(prompts)} prompts")
        
        all_predictions = []
        for prompt in prompts:
            predictions = self.predict(prompt, **kwargs)
            all_predictions.append(predictions)
        
        return all_predictions


class OpenAIPredictor:
    """Predictor using OpenAI API."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ) -> None:
        """
        Initialize OpenAI predictor.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or use env var)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required. Install: pip install openai")
        
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info(f"Initialized OpenAI predictor with model: {model}")
    
    def predict(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate prediction using OpenAI.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        logger.info(f"Generating OpenAI prediction for: {prompt[:50]}...")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        prediction = response.choices[0].message.content
        logger.info("Generated OpenAI prediction")
        
        return prediction
    
    def batch_predict(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate predictions for multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List of predictions
        """
        logger.info(f"Batch predicting for {len(prompts)} prompts")
        
        predictions = []
        for prompt in prompts:
            prediction = self.predict(prompt, **kwargs)
            predictions.append(prediction)
        
        return predictions


class StreamingPredictor:
    """Predictor with streaming support."""
    
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """
        Initialize streaming predictor.
        
        Args:
            model: OpenAI model name
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required for streaming")
        
        self.model = model
        self.client = OpenAI()
    
    def predict_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Generate streaming prediction.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Yields:
            Streamed text chunks
        """
        logger.info("Starting streaming prediction")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
