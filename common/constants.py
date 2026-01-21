import os
import re
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_LEVEL = logging.DEBUG

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 40
MIN_RECS_PER_REQUEST = 1
MAX_RECS_PER_REQUEST = 20
MIN_PRODUCTS_PER_CONTEXT = 10
MAX_PRODUCTS_PER_CONTEXT = 50_000

CATALOG_DUPE_THRESHOLD = 0.00

RE_PRODUCT_NAME = re.compile(r"[^A-Za-z0-9 |-]")
RE_REASON = re.compile(r"[^A-Za-z0-9 ]")
RE_MODEL_NAME = re.compile(r"[^A-Za-z0-9._/: +-]")



TOP_RECORDS = 3

MIN_PROMPT_TOKENS = 100
MAX_PROMPT_TOKENS = 50_000
MAX_SYSTEM_PROMPT_TOKENS = 10_000