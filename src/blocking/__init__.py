"""
src/blocking/__init__.py

Public API for Phase 3: Multi-Index Candidate Generation & Blocking Engine.
"""

from src.blocking.address_structural import AddressStructuralBlocker
from src.blocking.address_token import AddressTokenBlocker
from src.blocking.base import BaseBlocker
from src.blocking.compact_name import CompactNameBlocker
from src.blocking.composite import CompositeBlocker
from src.blocking.evaluator import BlockingEvaluator, BlockingMetrics
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.name_token import NameTokenBlocker
from src.blocking.ngram_blocker import CharNgramBlocker

__all__ = [
    "BaseBlocker",
    "ExactNameBlocker",
    "CompactNameBlocker",
    "NameTokenBlocker",
    "CharNgramBlocker",
    "AddressTokenBlocker",
    "AddressStructuralBlocker",
    "CompositeBlocker",
    "BlockingEvaluator",
    "BlockingMetrics",
]
