import os
import re
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_LEVEL = logging.DEBUG

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 40
MIN_RECS_PER_REQUEST = 1
MAX_RECS_PER_REQUEST = 20

CATALOG_DUPE_THRESHOLD = 0.00

RE_PRODUCT_NAME = re.compile(r"[^A-Za-z0-9 |-]")
RE_REASON = re.compile(r"[^A-Za-z0-9 ]")
RE_MODEL_NAME = re.compile(r"[^A-Za-z0-9._/: +-]")


MINER_ARTIFACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Bitrecs Miner Eval Input",
    "description": "Simplified input format for miners",
    "type": "object",
    "required": [
        "miner_hotkey", "miner_uid", "provider", "model",
        "system_prompt_template", "user_prompt_template", "sampling_params"
    ],
    "additionalProperties": False,
    "properties": {
        "miner_hotkey": {"type": "string"},
        "miner_uid": {"type": "integer", "minimum": 1},
        "provider": {"type": "string"},
        "model": {"type": "string"},
        "system_prompt_template": {"type": "string", "minLength": 1, "maxLength": 16000},
        "user_prompt_template": {"type": "string", "minLength": 1, "maxLength": 500000},
        "sampling_params": {
            "type": "object",
            "required": ["temperature"],
            "additionalProperties": False,
            "properties": {
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                "top_p": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "max_tokens": {"type": ["integer", "null"], "minimum": 1},
                "stop_sequences": {"type": ["array", "null"], "items": {"type": "string"}}
            }
        },
        "fewshot_examples": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "additionalProperties": False,
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                    "content": {"type": "string", "maxLength": 8192}
                }
            },
            "maxItems": 64
        }
    }
}
