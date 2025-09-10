from collections import Counter
from typing import Callable, Iterable, List, Set, Sequence

# We keep interfaces simple and compatible with existing helper in recent_code/calculate_jaccard.py

def calculate_jaccard(set_a: Iterable, set_b: Iterable, return_dissimilarity: bool = False) -> float:
    """
    Compute Jaccard similarity (or dissimilarity if return_dissimilarity=True) between two iterables.
    """
    A, B = set(set_a), set(set_b)
    union = len(A | B)
    if union == 0:
        sim = 0.0
    else:
        inter = len(A & B)
        sim = inter / union
    if return_dissimilarity:
        return 1.0 - sim
    return sim


def calculate_mean_pairwise_metric(list_of_sets: Sequence[Iterable], metric_func: Callable[[Iterable, Iterable], float]) -> float:
    """
    Compute the mean of metric_func over all unique pairs among list_of_sets.
    This is the U-statistic average over nC2 pairs.
    """
    n = len(list_of_sets)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += metric_func(list_of_sets[i], list_of_sets[j])
            count += 1
    return total / count if count > 0 else 0.0


def construct_representative_portfolio(list_of_sets: Sequence[Iterable], K: int = 30) -> List:
    """
    From repeated recommendations (each iterable is a list/set of tickers),
    pick top-K most frequently recommended tickers.
    Returns a list of tickers (length up to K if unique items are fewer than K).
    """
    counter = Counter()
    for s in list_of_sets:
        counter.update(set(s))  # count unique occurrences per repetition
    top = counter.most_common(K)
    return [item for item, _ in top]
