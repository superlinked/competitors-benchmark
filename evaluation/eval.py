from collections import defaultdict


def _evaluate_single_query(qid, query_preds, gt_tuples, k=10, debug=False):
    """
    Compute metrics for a single query.

    Args:
        qid (hashable): query identifier
        query_preds (list[tuple]): [(doc_id, rank, query_type|None), ...]
        gt_tuples (list[tuple]): [(doc_id, rank), ...]
        k (int): cutoff
        debug (bool): debug flag

    Returns:
        dict | None: Per-query metrics:
            {
              'query_id': qid,
              'query_type': query_type or 'unknown',
              f'precision@{k}': float,
              f'recall@{k}': float,
              f'mrr@{k}': float,
              f'ndcg@{k}': float,
              'pred_docs': list[str|int],
              'gt_size': int
            }
            Returns None if there are no predictions or no ground truth for the query.
    """
    # Sort predictions by rank and take top-k
    sorted_preds = sorted(query_preds, key=lambda x: x[1])[:k]
    pred_docs = [doc_id for doc_id, _, _ in sorted_preds]

    # Sort GT by rank for consistency; build helpers
    gt_sorted = sorted(gt_tuples, key=lambda x: x[1])
    gt_docs = set(doc_id for doc_id, _ in gt_sorted)
    gt_docs_with_ranks = {doc_id: rank for doc_id, rank in gt_sorted}

    if debug:
        print(f"\n--- Query {qid} ---")
        print(f"Predictions (top {k}): {pred_docs}")
        print(f"Ground truth docs: {gt_docs}")

    # Skip if no predictions or no ground truth
    if not pred_docs or not gt_docs:
        if debug:
            print(
                f"Skipping - no predictions ({len(pred_docs)}) or ground truth ({len(gt_docs)})"
            )
        return None

    # Determine query type (most frequent among the top-k predictions), default to 'unknown'
    query_types = [qt for _, _, qt in sorted_preds if qt is not None]
    query_type = (
        max(set(query_types), key=query_types.count) if query_types else "unknown"
    )

    # Retrieved relevant
    retrieved_relevant = sum(1 for doc in pred_docs if doc in gt_docs)
    if debug:
        print(f"Retrieved relevant: {retrieved_relevant}")
        print(
            f"Relevant docs in predictions: {[doc for doc in pred_docs if doc in gt_docs]}"
        )

    # Precision@k
    precision_denominator = min(k, len(pred_docs), len(gt_docs))
    precision = (
        retrieved_relevant / precision_denominator if precision_denominator > 0 else 0.0
    )

    # Recall@k
    recall = retrieved_relevant / len(gt_docs) if len(gt_docs) > 0 else 0.0

    # MRR@k
    mrr = 0.0
    for idx, doc in enumerate(pred_docs, start=1):
        if doc in gt_docs:
            mrr = 1.0 / idx
            break

    # NDCG@k (uses existing calculate_ndcg)
    ndcg = calculate_ndcg(pred_docs, gt_docs_with_ranks, k)

    # Round to 4 decimals
    result = {
        "query_id": qid,
        "query_type": query_type,
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4),
        f"mrr@{k}": round(mrr, 4),
        f"ndcg@{k}": round(ndcg, 4),
        "pred_docs": pred_docs,
        "gt_size": len(gt_docs),
    }
    return result


def evaluate_rankings(predictions, ground_truth, k=10, debug=False):
    """
    Evaluates rankings using Precision@k, Recall@k, MRR@k, and NDCG@k.

    Args:
        predictions (list of tuples):
            Accepts either:
              - (query_id, document_id, rank)
              - (query_id, document_id, rank, query_type)
        ground_truth (list of tuples):
            Accepts tuples where the first three items are (query_id, document_id, rank).
            Any extra items are ignored.
        k (int): cutoff for precision and recall
        debug (bool): if True, prints intermediate results for debugging

    Returns:
        Tuple[dict, dict]:
          (
            aggregated,  # dict with "<query_type>" and "overall" at the same level
            by_query     # dict keyed by query_id with per-query metrics
          )
    """
    k = int(k)

    # Group predictions by query_id; handle 3- or 4-tuple inputs
    pred_by_query = defaultdict(list)
    for item in predictions:
        if not isinstance(item, (list, tuple)):
            raise ValueError("Each prediction must be a tuple/list")
        if len(item) == 4:
            qid, doc_id, rank, query_type = item
        elif len(item) == 3:
            qid, doc_id, rank = item
            query_type = None  # default if not provided
        else:
            raise ValueError(
                f"Prediction tuple must have 3 or 4 items, got {len(item)}: {item}"
            )
        pred_by_query[qid].append((doc_id, rank, query_type))

    # Group ground truth by query_id; allow extra fields but take first three
    gt_by_query = defaultdict(list)
    for item in ground_truth:
        qid, doc_id, rank = item[0], item[1], item[2]
        gt_by_query[qid].append((doc_id, rank))

    # Collect per-query results
    by_query = {}

    # Temporary collectors for aggregated metrics
    metrics = defaultdict(
        lambda: defaultdict(list)
    )  # metrics[metric_name][query_type] -> list[float]

    metric_names = [
        f"precision@{k}",
        f"recall@{k}",
        f"mrr@{k}",
        f"ndcg@{k}",
    ]

    # Process each query via the single-query evaluator
    for qid in set(pred_by_query.keys()) | set(gt_by_query.keys()):
        result = _evaluate_single_query(
            qid, pred_by_query[qid], gt_by_query[qid], k, debug=debug
        )
        if result is None:
            continue

        by_query[qid] = result
        qtype = result["query_type"] or "unknown"

        # Accumulate for aggregation
        for m in metric_names:
            metrics[m][qtype].append(result[m])

    # Build aggregated results:
    # Keys for each query type and "overall" at the same level.
    aggregated = {
        "overall": {},
    }

    # Per-type averages
    for m in metric_names:
        for qtype, values in metrics[m].items():
            aggregated.setdefault(qtype, {})
            aggregated[qtype][m] = (
                round(sum(values) / len(values), 4) if values else 0.0
            )

    # Overall averages across all query types
    for m in metric_names:
        all_vals = []
        for values in metrics[m].values():
            all_vals.extend(values)
        aggregated["overall"][m] = (
            round(sum(all_vals) / len(all_vals), 4) if all_vals else 0.0
        )

    return aggregated, by_query


import math


def calculate_ndcg(pred_docs, gt_docs_with_ranks, k):
    """
    Calculate NDCG@k using rank-based relevance scores.

    This approach converts ground truth ranks into relevance scores using:
    relevance_score = max_rank - ground_truth_rank + 1

    Scoring example with 5 relevant documents:
    - GT rank 1 (best) → relevance = 5
    - GT rank 2 → relevance = 4
    - GT rank 3 → relevance = 3
    - GT rank 4 → relevance = 2
    - GT rank 5 (worst) → relevance = 1
    - Non-relevant → relevance = 0

    This creates a linear decreasing relevance scale where earlier ground truth
    ranks receive higher relevance scores, rewarding systems that retrieve
    the most important documents first.

    Args:
        pred_docs: List of predicted document IDs in order
        gt_docs_with_ranks: Dict mapping doc_id to ground truth rank
        k: Cutoff value

    Returns:
        float: NDCG@k score
    """
    if not pred_docs or not gt_docs_with_ranks:
        return 0.0

    # Convert ranks to relevance scores (higher rank = lower relevance)
    max_rank = max(gt_docs_with_ranks.values()) if gt_docs_with_ranks else 1

    # Calculate DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(pred_docs):
        if doc_id in gt_docs_with_ranks:
            gt_rank = gt_docs_with_ranks[doc_id]
            # Convert rank to relevance score (rank 1 gets highest relevance)
            relevance = max_rank - gt_rank + 1
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Calculate IDCG@k (best possible DCG)
    # Sort ground truth by rank (best first)
    sorted_gt_docs = sorted(gt_docs_with_ranks.items(), key=lambda x: x[1])
    idcg = 0.0
    for i, (doc_id, gt_rank) in enumerate(sorted_gt_docs[:k]):
        relevance = max_rank - gt_rank + 1
        idcg += relevance / math.log2(i + 2)

    # Return NDCG
    return dcg / idcg if idcg > 0 else 0.0
