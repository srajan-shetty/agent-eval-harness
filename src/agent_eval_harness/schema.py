from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator


class SpanType(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    HANDOFF = "handoff"


class TokenUsage(BaseModel):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)

    @computed_field
    @property
    def total(self) -> int:
        return self.input + self.output


class Span(BaseModel):
    span_id: str
    parent_id: str | None = None  # None = root span
    agent_id: str
    type: SpanType
    input: Any = None  # prompt / tool args / handoff payload
    output: Any = None  # completion / tool result / handoff result
    latency_ms: float = Field(ge=0)
    tokens: TokenUsage | None = None  # only meaningful for type == "llm"


class RunResult(BaseModel):
    question: str
    answer: str
    spans: list[Span] = Field(default_factory=list)
    total_latency: float = Field(ge=0)  # wall-clock, in ms

    @computed_field
    @property
    def total_tokens(self) -> int:
        return sum(s.tokens.total for s in self.spans if s.tokens)

    @model_validator(mode="after")
    def _check_span_tree(self) -> "RunResult":
        ids = [s.span_id for s in self.spans]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate span_id(s): {dupes}")

        known = set(ids)
        for s in self.spans:
            if s.parent_id is not None and s.parent_id not in known:
                raise ValueError(
                    f"span {s.span_id!r} has unknown parent_id {s.parent_id!r}"
                )
        return self
