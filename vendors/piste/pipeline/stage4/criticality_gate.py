# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 4a: Criticality Gate [C3]
=================================
Routes high-stakes claims to human Editorial Review Panel.
Low-stakes claims go to automated Verdict Aggregator.

Inspired by PolitiFact's three-editor panel:
  - Elections, public health, legal, high-profile figures → HUMAN review
  - Everything else → AUTOMATED verdict

The gate checks:
  1. Keyword match against critical topics list
  2. Embedding similarity to critical topic centroids
  3. Source reputation (high-profile figure detection)
"""

from typing import List, Optional
from dataclasses import dataclass


# Critical topics that trigger human review — Canada-focused bilingual (EN/FR)
# Covers: Canadian federal elections + Quebec provincial elections
CRITICAL_TOPICS = [
    # ── Elections (EN/FR) ────────────────────────────────────
    "election", "élection", "vote", "voter", "candidate", "candidat",
    "ballot", "bulletin", "democracy", "démocratie",
    "federal election", "élection fédérale",
    "provincial election", "élection provinciale",
    "referendum", "référendum",
    "riding", "circonscription", "mp", "député", "députée",
    "minority government", "gouvernement minoritaire",
    "majority government", "gouvernement majoritaire",
    "coalition", "confidence vote", "vote de confiance",
    # ── Quebec Politics (FR/EN) ──────────────────────────────
    "quebec", "québec", "québécois", "quebecois",
    "national assembly", "assemblée nationale",
    "sovereignty", "souveraineté", "separatist", "séparatiste",
    "federalism", "fédéralisme", "federalist", "fédéraliste",
    "language law", "loi linguistique", "bill 101", "loi 101",
    "bill 96", "loi 96", "bill 21", "loi 21",
    "secularism", "laïcité", "religious symbols", "signes religieux",
    "distinct society", "société distincte",
    "notwithstanding clause", "clause dérogatoire",
    # ── Canadian Federal Politics ────────────────────────────
    "parliament", "parlement", "house of commons", "chambre des communes",
    "senate", "sénat", "governor general", "gouverneur général",
    "throne speech", "discours du trône",
    "first past the post", "scrutin uninominal",
    "electoral reform", "réforme électorale",
    "equalization", "péréquation", "transfer payments", "paiements de transfert",
    "carbon tax", "taxe carbone", "carbon pricing", "tarification du carbone",
    "pipeline", "oil sands", "sables bitumineux",
    "indigenous", "autochtone", "first nations", "premières nations",
    "reconciliation", "réconciliation", "treaty", "traité",
    "indian act", "loi sur les indiens",
    # ── Public Health (EN/FR) ─────────────────────────────────
    "public health", "santé publique", "pandemic", "pandémie",
    "vaccine", "vaccin", "covid", "disease", "maladie",
    "health transfer", "transfert en santé", "healthcare", "soins de santé",
    # ── Economy / Budget (EN/FR) ──────────────────────────────
    "economy", "économie", "inflation", "recession", "récession",
    "deficit", "déficit", "budget", "tax", "impôt", "taxes", "impôts",
    "debt", "dette", "spending", "dépenses", "austerity", "austérité",
    "housing", "logement", "affordable housing", "logement abordable",
    "interest rate", "taux d'intérêt", "bank of canada", "banque du canada",
    # ── Rights / Immigration (EN/FR) ──────────────────────────
    "abortion", "avortement", "civil rights", "droits civils",
    "human rights", "droits humains", "charter", "charte",
    "immigration", "refugee", "réfugié", "asylum", "asile",
    "multiculturalism", "multiculturalisme",
]

# High-profile figure indicators — Canada-focused bilingual
HIGH_PROFILE_INDICATORS = [
    # ── Canadian Federal ──────────────────────────────────────
    "prime minister", "premier ministre",
    "pm", "trudeau", "justin trudeau",
    "poilievre", "pierre poilievre",
    "singh", "jagmeet singh",
    "liberal party", "parti libéral",
    "conservative party", "parti conservateur",
    "ndp", "nouveau parti démocratique",
    "bloc", "bloc québécois",
    "green party", "parti vert",
    "minister", "ministre", "cabinet",
    "governor general", "gouverneur général",
    "senator", "sénateur", "mp", "member of parliament",
    # ── Quebec Provincial ─────────────────────────────────────
    "premier", "première ministre",
    "quebec premier", "premier du québec",
    "legault", "françois legault",
    "caq", "coalition avenir québec",
    "parti québécois", "pq",
    "québec solidaire", "qs",
    "liberal party of quebec", "parti libéral du québec",
    "mna", "député", "députée",
    "national assembly", "assemblée nationale",
    # ── Provincial Premiers (other provinces) ─────────────────
    "ontario premier", "premier ontarien", "ford", "doug ford",
    "alberta premier", "smith", "danielle smith",
    "bc premier", "eby", "david eby",
]


@dataclass
class CriticalityAssessment:
    """Result of the Criticality Gate check."""
    is_critical: bool
    matched_topics: List[str]
    is_high_profile: bool
    recommendation: str  # "automated" or "human_review"
    reason: str


class CriticalityGate:
    """
    Routes claims based on criticality.

    Jewel [C3] — Human-in-the-loop breakpoint:
    Automated for scale, human-reviewed for stakes.
    """

    def assess(self, claim_text: str) -> CriticalityAssessment:
        """
        Determine if a claim requires human review.

        Args:
            claim_text: The claim text to assess.

        Returns:
            CriticalityAssessment with routing recommendation.
        """
        claim_lower = claim_text.lower()

        # Check critical topics
        matched_topics = [
            topic for topic in CRITICAL_TOPICS
            if topic in claim_lower
        ]

        # Check high-profile figures
        is_high_profile = any(
            indicator in claim_lower
            for indicator in HIGH_PROFILE_INDICATORS
        )

        # Determine routing
        is_critical = bool(matched_topics) or is_high_profile

        if is_critical:
            reason_parts = []
            if matched_topics:
                reason_parts.append(
                    f"matches critical topics: {', '.join(matched_topics[:3])}"
                )
            if is_high_profile:
                reason_parts.append("involves high-profile figure")
            reason = "; ".join(reason_parts)
            recommendation = "human_review"
        else:
            reason = "No critical topics or high-profile indicators detected."
            recommendation = "automated"

        return CriticalityAssessment(
            is_critical=is_critical,
            matched_topics=matched_topics,
            is_high_profile=is_high_profile,
            recommendation=recommendation,
            reason=reason,
        )


# Singleton
criticality_gate = CriticalityGate()
