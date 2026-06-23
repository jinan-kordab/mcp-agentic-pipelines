"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT

Extracts structured fields from semi-structured work orders without OCR training,
without ML models, without vector databases. Uses regex patterns + stemming for
field label detection and value extraction.

Common aviation work order fields:
  Tail Number, Work Order #, Date, Aircraft Model, Part Number, Serial Number,
  Mechanic ID, Station, Work Performed, Hours, AD/SB Compliance, Inspector Stamp
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class ExtractedField:
    field_name: str          # e.g., "tail_number"
    raw_label: str           # e.g., "Tail #:", "REG:", "A/C REG NO"
    value: str               # e.g., "N737AG"
    confidence: float        # 0.0 - 1.0
    page: int
    line_number: int
    context: str = ""        # surrounding text for verification


@dataclass
class WorkOrder:
    """Structured work order record extracted from a document."""
    source_file: str
    tail_number: Optional[str] = None
    work_order_number: Optional[str] = None
    date: Optional[str] = None
    aircraft_model: Optional[str] = None
    part_numbers: List[str] = field(default_factory=list)
    part_descriptions: List[str] = field(default_factory=list)
    serial_numbers: List[str] = field(default_factory=list)
    mechanic_id: Optional[str] = None
    station: Optional[str] = None
    work_performed: Optional[str] = None
    hours_worked: Optional[str] = None
    ad_sb_references: List[str] = field(default_factory=list)
    inspector_stamp: Optional[str] = None
    extracted_fields: List[ExtractedField] = field(default_factory=list)
    raw_text_snippet: str = ""


class WorkOrderExtractor:
    """Extracts structured aviation work order fields using token-pattern matching."""

    # Field label patterns — key is the canonical field name, value is list of regex patterns
    FIELD_PATTERNS: Dict[str, List[str]] = {
        "tail_number": [
            r"(?:tail|a/?c|aircraft|registration|reg|n-?number)[\s#:.-]*([\s]*[nN]\d{1,6}[a-zA-Z]{0,2})",
            r"(?:tail|a/?c|aircraft|registration|reg)[\s#:.-]*([\s]*\d{1,6}[a-zA-Z]{0,2})",
            r"\b([nN]\d{1,6}[a-zA-Z]{0,2})\b",
        ],
        "work_order_number": [
            r"(?:wo|w/?o|work\s*order|job\s*card|task\s*order)[\s#:.-]*([\s]*[a-zA-Z0-9]{4,20})",
            r"\b(?:WO|W/O)\s*[:#.-]*\s*([a-zA-Z0-9]{4,20})",
        ],
        "date": [
            r"(?:date|dated|performed|completed)[\s:.-]*([\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:date|dated)[\s:.-]*([\s]*\d{4}-\d{2}-\d{2})",
            r"(?:date|dated)[\s:.-]*([\s]*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4})",
        ],
        "aircraft_model": [
            r"(?:model|a/?c\s*type|aircraft\s*type)[\s:.-]*([\s]*(?:B|b)(?:7[3-9]\d|7[3-9]\d[-\s]*[a-zA-Z0-9]{0,6}))",
            r"(?:model|a/?c\s*type|aircraft\s*type)[\s:.-]*([\s]*(?:A|a)(?:3[12]\d|3[12]\d[-\s]*[a-zA-Z0-9]{0,4}))",
            r"(?:model|a/?c\s*type|aircraft\s*type)[\s:.-]*([\s]*[a-zA-Z]{1,4}[-\s]*\d{2,4}[a-zA-Z0-9]{0,4})",
        ],
        "part_number": [
            r"(?:p/?n|part\s*(?:no|number|#)|p/n)[\s:.-]*([\s]*[a-zA-Z0-9]{3,30})",
            r"\b([a-zA-Z]{2,6}\d{3,10}[a-zA-Z]{0,4})\b",
            r"\b([a-zA-Z]{1,3}\d{5,12})\b",
        ],
        "serial_number": [
            r"(?:s/?n|s/n|serial\s*(?:no|number|#))[\s:.-]*([\s]*[a-zA-Z0-9]{3,30})",
            r"\b(?:SN|S/N)\s*[:#.-]*\s*([a-zA-Z0-9]{3,30})",
        ],
        "mechanic_id": [
            r"(?:mechanic|tech|technician|a&p|a/?p|inspector|performed\s*by)[\s:.-]*([\s]*[a-zA-Z0-9]{2,20})",
            r"(?:mechanic|tech)[\s:.-]*\#?\s*([a-zA-Z]{1,3}\d{2,8})",
        ],
        "station": [
            r"(?:station|location|facility|base|gate)[\s:.-]*([\s]*[a-zA-Z]{3,6})",
            r"\b(?:ATL|DFW|ORD|LAX|JFK|MIA|SEA|SFO|DEN|IAH|MCO|BOS|EWR|PHX|MSP|DTW|CLT|LAS|HNL)\b",
        ],
        "hours_worked": [
            r"(?:hours|man[-\s]*hours|labor\s*hrs|labor\s*hours)[\s:.-]*([\s]*\d+\.?\d*)",
            r"(?:total\s*hrs|total\s*hours)[\s:.-]*([\s]*\d+\.?\d*)",
        ],
        "ad_sb_reference": [
            r"\b(AD\s*\d{4}[-\s]*\d{2}[-\s]*\d{2,4})\b",
            r"\b(SB\s*[a-zA-Z0-9]{2,6}[-\s]*\d{2,5})\b",
            r"(?:airworthiness\s*directive|a\.?d\.?|service\s*bulletin|s\.?b\.?)[\s:.-]*([a-zA-Z0-9]{4,20})",
        ],
        "inspector_stamp": [
            r"(?:inspector|inspected\s*by|stamp|signed\s*by|approved\s*by)[\s:.-]*([\s]*[a-zA-Z\s]{3,30})",
            r"(?:RII|R\.?I\.?I\.?|buy\s*back)[\s:.-]*([\s]*[a-zA-Z]{2,20})",
        ],
        "work_performed": [
            r"(?:work\s*performed|description|task|action\s*taken|corrective\s*action)[\s:.-]*([\s]*[a-zA-Z0-9\s,;()]{20,500})",
        ],
    }

    def __init__(self) -> None:
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for field_name, patterns in self.FIELD_PATTERNS.items():
            self._compiled_patterns[field_name] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def extract(self, text: str, source_file: str = "unknown") -> WorkOrder:
        """Extract all known fields from work order text. Returns structured WorkOrder."""
        wo = WorkOrder(source_file=source_file)
        lines = text.split("\n")

        for line_num, line in enumerate(lines):
            for field_name, patterns in self._compiled_patterns.items():
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        value = match.group(1).strip() if match.groups() else match.group(0).strip()
                        # Skip if value is too short or looks like a false positive
                        if len(value) < 2:
                            continue
                        if field_name == "tail_number" and not self._looks_like_tail(value):
                            continue
                        if field_name == "part_number" and len(value) < 4:
                            continue

                        field = ExtractedField(
                            field_name=field_name,
                            raw_label=match.group(0)[:50],
                            value=value,
                            confidence=self._confidence(field_name, value),
                            page=1,
                            line_number=line_num,
                            context=self._get_context(lines, line_num),
                        )
                        wo.extracted_fields.append(field)

                        # Populate the structured fields
                        self._assign_field(wo, field_name, value)

        # Capture raw text for the "work performed" field if not explicitly extracted
        if not wo.work_performed:
            # Take the longest paragraph as potential work description
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
            if paragraphs:
                wo.work_performed = max(paragraphs, key=len)[:500]

        return wo

    def _looks_like_tail(self, value: str) -> bool:
        """Check if a value looks like an N-number."""
        # N-number: N followed by 1-5 digits, optionally 1-2 letters
        return bool(re.match(r'^[nN]\d{1,5}[a-zA-Z]{0,2}$', value.strip()))

    def _confidence(self, field_name: str, value: str) -> float:
        """Heuristic confidence score based on value quality."""
        base = 0.7
        if len(value) > 20:
            base -= 0.2  # Too long, might be grabbing extra text
        if len(value) < 3:
            base -= 0.3
        if field_name == "tail_number" and self._looks_like_tail(value):
            base = 0.95  # High confidence for valid N-numbers
        if field_name == "date" and re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value):
            base = 0.90
        return min(max(base, 0.3), 1.0)

    def _get_context(self, lines: List[str], line_num: int, window: int = 1) -> str:
        """Get surrounding lines for context."""
        start = max(0, line_num - window)
        end = min(len(lines), line_num + window + 1)
        return " | ".join(lines[start:end])

    def _assign_field(self, wo: WorkOrder, field_name: str, value: str) -> None:
        """Assign extracted value to the structured WorkOrder."""
        if field_name == "tail_number" and not wo.tail_number:
            wo.tail_number = value
        elif field_name == "work_order_number" and not wo.work_order_number:
            wo.work_order_number = value
        elif field_name == "date" and not wo.date:
            wo.date = value
        elif field_name == "aircraft_model" and not wo.aircraft_model:
            wo.aircraft_model = value
        elif field_name == "part_number" and value not in wo.part_numbers:
            wo.part_numbers.append(value)
        elif field_name == "serial_number" and value not in wo.serial_numbers:
            wo.serial_numbers.append(value)
        elif field_name == "mechanic_id" and not wo.mechanic_id:
            wo.mechanic_id = value
        elif field_name == "station" and not wo.station:
            wo.station = value
        elif field_name == "work_performed" and not wo.work_performed:
            wo.work_performed = value[:500]
        elif field_name == "hours_worked" and not wo.hours_worked:
            wo.hours_worked = value
        elif field_name == "ad_sb_reference" and value not in wo.ad_sb_references:
            wo.ad_sb_references.append(value)
        elif field_name == "inspector_stamp" and not wo.inspector_stamp:
            wo.inspector_stamp = value
