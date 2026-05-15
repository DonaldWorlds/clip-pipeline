from pydantic import BaseModel

from pydantic import BaseModel, Field


class ClipRequest(BaseModel):
    input_video: str
    start_time: float
    end_time: float
    layout: str



class ClipDecision(BaseModel):
    clip_start: float = Field(ge=0)
    clip_end: float = Field(gt=0)
    layout: str
    title: str = Field(max_length=60)