from typing import List

import numpy as np


def kld(p: List[float], q: List[float]) -> float:
    """Calculate the Kullback-Leibler Divergence between two distributions."""
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    # Ensure the distributions are normalized.
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate KLD.
    kld = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    return kld
