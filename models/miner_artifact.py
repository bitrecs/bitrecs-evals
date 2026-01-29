from datetime import datetime, timezone
import os
from typing import List, Optional, Dict, Any
from pydantic import UUID4, BaseModel, Field
import uuid

import yaml

class SamplingParams(BaseModel):
    temperature: float = Field(ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    max_tokens: Optional[int] = Field(None, gt=0)
    stop_sequences: Optional[List[str]] = None


class MessageExample(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(max_length=8192)


# Main input into the evolution system
class Artifact(BaseModel):    
    artifact_id: UUID4 = Field(default_factory=uuid.uuid4, description="Unique artifact ID")
    parent_id: Optional[UUID4] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO8601 timestamp of submission creation"
    )
    name: str = Field(description="Name of the artifact")
    miner_hotkey: str = Field(description="Hotkey of the submitting miner")
    miner_uid: int = Field(gt=0, description="Active UID on Subnet")
    provider: str = Field(description="LLM provider name")
    model: str = Field(description="LLM model name")
    system_prompt_template: str = Field(description="Jinja2 template for system prompt")
    user_prompt_template: str = Field(description="Jinja2 template for user prompt")
    sampling_params: SamplingParams = Field(description="Sampling parameters for LLM")
    fewshot_examples: Optional[List[MessageExample]] = Field(None, max_length=64)
    eval_scores: Dict[str, float] = Field(description="Evaluation scores claimed by the miner", default_factory=dict)
    status: Optional[str] = Field(None, description="Status of the artifact in the evaluation pipeline")


    @staticmethod
    def from_yaml(input_path=None) -> Optional["Artifact"]:
        """Load miner input YAML and convert to Artifact object."""    
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"Error: {input_path} not found.")
        with open(input_path, 'r') as f:
            data = yaml.safe_load(f)
        try:
            artifact = Artifact(**data)
            return artifact
        except Exception as e:
            raise ValueError(f"Failed to parse miner input into Artifact: {e}")
