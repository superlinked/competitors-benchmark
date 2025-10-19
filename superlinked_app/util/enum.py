from enum import Enum


class QueryMode(Enum):
    USE_FULL_QUERY_TEXT = "full_text"
    USE_GROUND_TRUTH_QUERY_INPUTS = "gt_params"
    USE_NLQ = "nlq"
