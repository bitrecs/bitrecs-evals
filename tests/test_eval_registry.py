import os
import sys
project_root = os.getcwd() 
sys.path.insert(0, project_root)
import pytest
import logging
from evals.eval_factory import EvalFactory
from models.eval_type import BitrecsEvaluationType
from common import constants as CONST

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

def test_eval_factory_registry_unique_types():
    registry = EvalFactory._registry
    
    eval_types = list(registry.keys())
    assert len(eval_types) == len(set(eval_types)), "Duplicate eval types found in EvalFactory registry"
    #logger.debug(f"Registered eval types: {[et.name for et in eval_types]}")
    
    eval_classes = list(registry.values())
    assert len(eval_classes) == len(set(eval_classes)), "Duplicate eval classes found in EvalFactory registry"
    #logger.info(f"Registered eval classes: {[ec.__name__ for ec in eval_classes]}")


def test_eval_factory_missing_types():
    registry = EvalFactory._registry
    all_types = set(BitrecsEvaluationType)
    registered_types = set(registry.keys())
    missing_types = all_types - registered_types
    if missing_types:
        print("Missing BitrecsEvaluationType(s):", [t.name for t in missing_types])
        logger.debug(f"Missing BitrecsEvaluationType(s): {[t.name for t in missing_types]}")
    assert not missing_types, f"Some BitrecsEvaluationType(s) are not registered: {[t.name for t in missing_types]}"