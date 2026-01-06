"""
Schema definitions for Phase 2 and Phase 3 refinement pipeline.

This module provides structured dataclasses for type-safe agent communication
during the multi-phase refinement process.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class RefinementApproach(Enum):
    """Approach for improving content quality"""
    INCREMENTAL_EDIT = "incremental_edit"
    SECTION_REWRITE = "section_rewrite"
    EVIDENCE_SWAP = "evidence_swap"


class RiskLevel(Enum):
    """Risk level for proposed changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RootCause:
    """A specific issue causing low score"""
    issue: str  # Description of the problem
    quote: str  # Exact text from the document
    score_impact: float  # Estimated score penalty (e.g., -0.8)
    location: str  # Where in the document (e.g., "2문단 전체")


@dataclass
class SideEffectRisk:
    """Risk of degrading other modules when fixing target module"""
    affected_module: str  # Which module might be hurt
    risk: str  # Description of the risk
    mitigation: str  # How to prevent the side effect


@dataclass
class EvidenceNeed:
    """Specification for additional evidence needed"""
    type: str  # Type of evidence needed (e.g., "personality_shaping_experience")
    query: str  # Search query for KB


@dataclass
class Phase2DiagnosticResult:
    """Output from Phase 2 WHY (diagnostic) step"""
    root_causes: List[RootCause]
    improvement_approach: RefinementApproach
    approach_rationale: str
    side_effect_risks: List[SideEffectRisk]
    requires_new_evidence: bool
    evidence_need: Optional[EvidenceNeed] = None
    confidence: float = 0.0  # 0.0-1.0


@dataclass
class ContentChange:
    """A specific change to be made to the text"""
    location: str  # Where to make the change
    action: str  # "replace", "delete", "insert"
    before: str  # Current text
    after: str  # New text
    rationale: str  # Why this change improves the score
    new_evidence_needed: Optional[EvidenceNeed] = None


@dataclass
class Phase2ImprovementPlan:
    """Output from Phase 2 WHAT (planning) step"""
    changes: List[ContentChange]
    expected_score_delta: Dict[str, float]  # Module name -> score change
    risk_level: RiskLevel
    approach: RefinementApproach


@dataclass
class Phase2RefinerInput:
    """Structured input for Phase 2 structural refinement"""

    # Context
    purpose: str = "Structural quality improvement to 9.0-9.2 baseline"
    question: str = ""
    question_type: str = ""  # "자기소개", "지원동기", etc.

    # Current state
    current_text: str = ""
    current_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    # Target module
    target_module: str = ""
    target_module_score: float = 0.0
    target_module_goal: float = 9.0
    module_weight: float = 1.0

    # Diagnostic info
    scorer_rationale: str = ""
    failed_criteria: List[str] = field(default_factory=list)

    # Constraints
    preserve_strengths: List[str] = field(default_factory=list)
    forbidden_changes: List[str] = field(default_factory=list)

    # Evidence pool
    available_evidence: str = ""
    used_evidence_ids: List[str] = field(default_factory=list)

    # Meta
    iteration_num: int = 1
    previous_attempts: List[Dict] = field(default_factory=list)

    # Resources
    current_char_count: int = 0
    target_char_min: int = 950
    target_char_max: int = 1000


@dataclass
class Phase3PolisherInput:
    """Structured input for Phase 3 final polishing"""

    # Context
    purpose: str = "Final convergence to 9.5+ submission quality"
    question: str = ""

    # Current state
    current_text: str = ""
    current_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    # Target gaps
    failing_modules: List[str] = field(default_factory=list)
    primary_target: str = ""

    # Specific issues to fix
    format_violations: List[str] = field(default_factory=list)
    redundant_phrases: List[str] = field(default_factory=list)
    length_delta: int = 0  # Chars to add/remove

    # What to preserve (from Phase 2)
    locked_content: List[str] = field(default_factory=list)

    # Meta
    iteration_num: int = 1
    phase2_final_score: float = 0.0


@dataclass
class RefineIteration:
    """Record of a single refinement iteration"""
    iteration: int
    module: str
    module_score_before: float
    module_score_after: float
    score_before: float
    score_after: float
    improvements_made: List[Any]  # ContentChange or similar
    strategy: str
    diagnostics: Dict[str, Any]
    text_before: str
    text_after: str
    timestamp: Optional[str] = None

    @property
    def module_delta(self) -> float:
        return self.module_score_after - self.module_score_before

    @property
    def total_delta(self) -> float:
        return self.score_after - self.score_before


@dataclass
class PhaseResult:
    """Result from a complete phase (Phase 2 or Phase 3)"""
    phase_name: str  # "Phase 2" or "Phase 3"
    initial_score: float
    final_score: float
    iterations: List[RefineIteration]
    target_reached: bool
    exit_reason: str  # "target_reached", "max_iterations", "stagnation", etc.

    @property
    def total_improvement(self) -> float:
        return self.final_score - self.initial_score

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)
