from typing import Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
import torch

class VLMDistanceResponse(BaseModel):
    """
    VLM response model for evaluating the similarity between a single input face image 
    and the target face image.
    """
    reasoning: str = Field(
        description="A concise summary of the visual comparison, strictly limited to 1 or 2 sentences in a single line without any newlines or line breaks."
    )
    score: int = Field(
        ge=1,
        le=100,
        description="The similarity score between the input face and the target face as an integer. Ranges from 1 (completely different) to 100 (perfect match)."
    )

class VLMBatchDistanceResponse(BaseModel):
    """
    VLM response model for evaluating whether the existing face image or the new face image
    is closer to the target face image.
    """
    reasoning: str = Field(
        description="A concise summary explaining which face is closer to the target, strictly limited to 1 or 2 sentences in a single line without any newlines or line breaks."
    )
    selection: Literal["existing", "new"] = Field(
        description="The selected face image that is closer to the target face. Must be strictly either 'existing' or 'new'."
    )
    score: int = Field(
        ge=1,
        le=100,
        description="The similarity score between the selected face and the target face as an integer. Ranges from 1 (completely different) to 100 (perfect match)."
    )

@dataclass
class SampleState:
    """
    State of a single sample.
    """
    distance: float
    latent: torch.Tensor
    iteration: int = 0

@dataclass
class MCMCTracker:
    """
    Tracker for MCMC.
    """
    best: Optional[SampleState] = None
    cur: Optional[SampleState] = None
    init: Optional[SampleState] = None
    cur_best: Optional[SampleState] = None