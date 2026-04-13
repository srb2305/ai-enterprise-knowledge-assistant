# LangGraph state schema

from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class LangGraphState:
	query: str
	context_chunks: List[str] = field(default_factory=list)
	sources: List[dict] = field(default_factory=list)
	answer: Optional[str] = None
	faithfulness_score: Optional[float] = None
	relevance_score: Optional[float] = None
	iteration_count: int = 0
	trace: List[dict] = field(default_factory=list)
	# Optionally, add more fields as needed (e.g., logs, metadata)
