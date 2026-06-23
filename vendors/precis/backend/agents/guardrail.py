"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re


class GuardrailAction(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    action: GuardrailAction
    issues_found: List[str] = field(default_factory=list)
    redacted_content: List[str] = field(default_factory=list)
    redacted_response: Optional[str] = None   # ← response with PII scrubbed (when action=REDACT)
    confidence: float = 1.0
    requires_human_review: bool = False


class GuardrailAgent:
    """Validates outputs before delivery. PII detection, hallucination check, content policy.

    Layer 1 — PII Detection & Redaction:
        Scans the generated response for SSNs, credit-card numbers, email
        addresses, and phone numbers.  When PII is found the response is
        *redacted* in-place rather than blocked outright, so the user still
        receives useful content.

    Layer 2 — Prompt Injection:
        Checks the original query for injection markers.  These are ALWAYS
        blocked — no response is returned.

    Layer 3 — Hallucination Threshold:
        If the VeriScore hallucination rate exceeds 30 %, the response is
        flagged for human review but still delivered (with a warning).
    """

    # ── PII patterns: (regex, human-readable type, replacement mask) ─
    _PII_PATTERNS: List[Tuple[str, str, str]] = [
        (r'\b\d{3}-\d{2}-\d{4}\b',                              "SSN",           "[REDACTED-SSN]"),
        (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',            "Credit Card",   "[REDACTED-CC]"),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email",        "[REDACTED-EMAIL]"),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',                       "Phone",         "[REDACTED-PHONE]"),
    ]

    # ── Prompt injection markers: any match → immediate BLOCK ──────
    _INJECTION_MARKERS: List[str] = [
        "ignore previous", "you are now", "system prompt",
        "[/INST]", "<|im_start|>", "<|im_end|>",
        "forget all", "new instructions", "pretend you are",
    ]

    # ── Thresholds ─────────────────────────────────────────────────
    HALLUCINATION_BLOCK_THRESHOLD: float = 0.8   # > 80 % → block (extreme cases only)
    HALLUCINATION_FLAG_THRESHOLD: float = 0.3    # > 30 % → flag

    # ── Public API ──────────────────────────────────────────────────

    async def validate(
        self,
        generated_response: str,
        retrieved_sources: List[Dict[str, Any]],
        original_query: str,
        veriscore_report: Optional[Any] = None,
    ) -> GuardrailResult:
        """Run all safety layers.  Returns a GuardrailResult with the
        most restrictive action warranted by any layer.
        """
        issues: List[str] = []
        redacted: List[str] = []
        redacted_text: Optional[str] = None
        confidence: float = 1.0
        needs_review: bool = False

        # ── Layer 1: PII Detection & Redaction ──────────────────
        redacted_text, pii_found = self._redact_pii(generated_response)
        if pii_found:
            for _, pii_type, _ in self._PII_PATTERNS:
                if re.search(self._PII_PATTERNS[0][0], generated_response):  # quick re-check
                    pass
            issues.extend(pii_found)
            redacted.extend(pii_found)
            confidence = 0.85
            needs_review = True

        # ── Layer 2: Prompt Injection ──────────────────────────
        query_lower = original_query.lower()
        for marker in self._INJECTION_MARKERS:
            if marker.lower() in query_lower:
                issues.append(f"Prompt injection detected: '{marker}'")
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    issues_found=issues,
                    redacted_content=redacted,
                    confidence=1.0,
                    requires_human_review=True,
                )

        # ── Layer 3: Hallucination Rate Threshold ──────────────
        if veriscore_report is not None:
            hall_rate = getattr(veriscore_report, "hallucination_rate", 0.0)
            if hall_rate > self.HALLUCINATION_BLOCK_THRESHOLD:
                issues.append(
                    f"Critical hallucination rate ({hall_rate:.0%} > "
                    f"{self.HALLUCINATION_BLOCK_THRESHOLD:.0%}) — response blocked"
                )
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    issues_found=issues,
                    redacted_content=redacted,
                    confidence=0.6,
                    requires_human_review=True,
                )
            elif hall_rate > self.HALLUCINATION_FLAG_THRESHOLD:
                issues.append(
                    f"High hallucination rate ({hall_rate:.0%} > "
                    f"{self.HALLUCINATION_FLAG_THRESHOLD:.0%})"
                )
                confidence = 0.7
                needs_review = True

        # ── Also check flagged_issues from VeriScore ───────────
        if veriscore_report is not None:
            for fi in getattr(veriscore_report, "flagged_issues", []):
                if fi not in issues:
                    issues.append(fi)

        # ── Decide final action ────────────────────────────────
        if not issues:
            return GuardrailResult(
                action=GuardrailAction.PASS,
                redacted_response=redacted_text or generated_response,
            )

        if redacted:
            # PII was found → redact and deliver
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                issues_found=issues,
                redacted_content=redacted,
                redacted_response=redacted_text,
                confidence=confidence,
                requires_human_review=needs_review,
            )

        # Non-PII issues → flag for review but still deliver
        return GuardrailResult(
            action=GuardrailAction.FLAG,
            issues_found=issues,
            redacted_response=generated_response,
            confidence=confidence,
            requires_human_review=needs_review,
        )

    # ── PII Redaction ──────────────────────────────────────────────

    def _redact_pii(self, text: str) -> Tuple[str, List[str]]:
        """Scan *text* for PII patterns and replace matches with safe tokens.

        Returns (redacted_text, list_of_types_found).
        """
        found: List[str] = []
        result = text
        for pattern, pii_type, replacement in self._PII_PATTERNS:
            matches = re.findall(pattern, result)
            if matches:
                found.append(f"PII redacted: {pii_type} ({len(matches)} instance(s))")
                result = re.sub(pattern, replacement, result)
        return result, found
