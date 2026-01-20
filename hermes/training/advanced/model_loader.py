"""Model loading utilities with LoRA and quantization support."""

from typing import Optional, Tuple

from loguru import logger

from hermes.training.advanced.config import TrainingConfig

# Try importing Unsloth
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not installed. Install with: pip install unsloth")

# Fallback to regular transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed")


class ModelLoader:
    """Load models with LoRA and quantization support."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize model loader.
        
        Args:
            config: Training configuration
        """
        self.config = config
    
    def load_model_and_tokenizer(self) -> Tuple:
        """
        Load model and tokenizer with LoRA and quantization.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.config.use_unsloth and UNSLOTH_AVAILABLE:
            return self._load_with_unsloth()
        elif TRANSFORMERS_AVAILABLE:
            return self._load_with_transformers()
        else:
            raise ImportError("Neither Unsloth nor Transformers available. Install one of them.")
    
    def _load_with_unsloth(self) -> Tuple:
        """
        Load model using Unsloth (optimized, recommended).
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model with Unsloth: {self.config.model_name}")
        
        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,  # Auto-detect
        )
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            use_gradient_checkpointing=self.config.lora.use_gradient_checkpointing,
            random_state=self.config.seed,
            use_rslora=self.config.lora.use_rslora,
        )
        
        # Set chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.config.chat_template.value,
        )
        
        logger.info(f"Model loaded successfully with Unsloth")
        logger.info(f"LoRA config: r={self.config.lora.rank}, alpha={self.config.lora.alpha}")
        logger.info(f"4-bit quantization: {self.config.load_in_4bit}")
        
        return model, tokenizer
    
    def _load_with_transformers(self) -> Tuple:
        """
        Load model using standard Transformers (fallback).
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model with Transformers: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.load_in_4bit if self.config.load_in_4bit else None,
            device_map="auto",
        )
        
        # Prepare for k-bit training if quantized
        if self.config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        
        logger.info("Model loaded successfully with Transformers")
        
        return model, tokenizer
    
    @staticmethod
    def supports_bfloat16() -> bool:
        """Check if bfloat16 is supported."""
        if UNSLOTH_AVAILABLE:
            return is_bfloat16_supported()
        return False
