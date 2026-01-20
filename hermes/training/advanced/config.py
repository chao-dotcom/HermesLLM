"""Training configuration models."""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class FinetuningType(str, Enum):
    """Types of fine-tuning."""
    SFT = "sft"  # Supervised Fine-Tuning
    DPO = "dpo"  # Direct Preference Optimization
    ORPO = "orpo"  # Odds Ratio Preference Optimization


class ChatTemplate(str, Enum):
    """Chat template formats."""
    CHATML = "chatml"
    ALPACA = "alpaca"
    LLAMA3 = "llama-3"
    MISTRAL = "mistral"
    ZEPHYR = "zephyr"


class LoRAConfig(BaseModel):
    """Configuration for LoRA/QLoRA training."""
    
    rank: int = Field(32, description="LoRA rank")
    alpha: int = Field(32, description="LoRA alpha")
    dropout: float = Field(0.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", 
            "up_proj", "down_proj", "o_proj", "gate_proj"
        ],
        description="Target modules for LoRA"
    )
    use_gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing")
    use_rslora: bool = Field(False, description="Use rank-stabilized LoRA")
    
    class Config:
        use_enum_values = True


class DPOConfig(BaseModel):
    """Configuration for DPO training."""
    
    beta: float = Field(0.1, description="DPO beta parameter")
    label_smoothing: float = Field(0.0, description="Label smoothing")
    loss_type: str = Field("sigmoid", description="DPO loss type")
    
    class Config:
        use_enum_values = True


class TrainingConfig(BaseModel):
    """Comprehensive training configuration."""
    
    # Model configuration
    model_name: str = Field("meta-llama/Llama-3.2-1B", description="Base model name")
    max_seq_length: int = Field(2048, description="Maximum sequence length")
    load_in_4bit: bool = Field(True, description="Load model in 4-bit quantization")
    
    # LoRA configuration
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    
    # DPO configuration (only used if finetuning_type=DPO)
    dpo: Optional[DPOConfig] = Field(default_factory=DPOConfig)
    
    # Training type
    finetuning_type: FinetuningType = Field(FinetuningType.SFT, description="Type of fine-tuning")
    chat_template: ChatTemplate = Field(ChatTemplate.CHATML, description="Chat template format")
    
    # Training hyperparameters
    learning_rate: float = Field(3e-4, description="Learning rate")
    num_train_epochs: int = Field(3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(2, description="Training batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Gradient accumulation steps")
    warmup_steps: int = Field(10, description="Warmup steps")
    max_grad_norm: float = Field(1.0, description="Max gradient norm")
    weight_decay: float = Field(0.01, description="Weight decay")
    
    # Optimizer configuration
    optim: str = Field("adamw_8bit", description="Optimizer type")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler")
    
    # Output configuration
    output_dir: Path = Field(Path("models/fine-tuned"), description="Output directory")
    logging_steps: int = Field(10, description="Logging frequency")
    save_steps: int = Field(100, description="Save checkpoint frequency")
    save_total_limit: int = Field(3, description="Maximum number of checkpoints to keep")
    
    # Dataset configuration
    dataset_path: Optional[str] = Field(None, description="HuggingFace dataset path")
    dataset_text_field: str = Field("text", description="Text field in dataset")
    
    # Advanced options
    use_unsloth: bool = Field(True, description="Use Unsloth for optimization")
    bf16: bool = Field(True, description="Use bfloat16 precision")
    fp16: bool = Field(False, description="Use float16 precision")
    max_steps: int = Field(-1, description="Maximum training steps (-1 for unlimited)")
    
    # Evaluation
    eval_strategy: str = Field("steps", description="Evaluation strategy")
    eval_steps: int = Field(100, description="Evaluation frequency")
    
    # Misc
    seed: int = Field(42, description="Random seed")
    push_to_hub: bool = Field(False, description="Push model to HuggingFace Hub")
    hub_model_id: Optional[str] = Field(None, description="HuggingFace Hub model ID")
    
    class Config:
        use_enum_values = True
    
    def to_training_arguments(self) -> dict:
        """Convert to HuggingFace TrainingArguments dict."""
        return {
            "output_dir": str(self.output_dir),
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "max_steps": self.max_steps,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "seed": self.seed,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
        }
