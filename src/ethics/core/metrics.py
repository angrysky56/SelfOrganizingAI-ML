# Ethical Metrics Module: Core Measurement Framework
# Implements foundational metric structures for ethical analysis

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

@dataclass
class TimeSeriesMetric:
    """Time series measurement with confidence scoring."""
    value: float
    timestamp: datetime
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StatisticalProperties:
    """Core statistical properties of ethical metrics."""
    mean: float
    variance: float
    sample_size: int
    confidence_interval: tuple[float, float]
