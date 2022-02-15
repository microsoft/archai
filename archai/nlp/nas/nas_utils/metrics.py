# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Calculates set of metrics related to NAS.
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr


# def spearman_ranking(top_k: int,
#                      sorted_gt: List[int],
#                      sorted_target: List[int],
#                      val_ppl_gt: List[float],
#                      val_ppl_target: List[float]) -> Tuple[float, float]:
#     """Calculates the common ratio, as well as Spearman Correlation and its ranking.

#     Args:
#         top_k: Number of top-k rankings.
#         sorted_gt: Sorted ground-truth labels.
#         sorted_target: Sorted target labels.
#         val_ppl_gt: Validation perplexity ground-truths.
#         val_ppl_target: Validation perplexity targets.

#     Returns:
#         (Tuple[float, float]): Common ratio and Spearman Ranking.

#     """

#     idx = int(top_k / 100 * len(sorted_gt))

#     sorted_gt_binned = sorted_gt[:idx].astype(np.int32)
#     sorted_target_binned = sorted_target[:idx].astype(np.int32)

#     correct = len(np.intersect1d(sorted_target_binned, sorted_gt_binned))
#     total = len(sorted_target_binned)
#     common_ratio = correct * 1 / total

#     top_k_val_ppl_gt = [val_ppl_gt[i] for i in range(len(val_ppl_gt)) if i in sorted_gt_binned]
#     top_k_val_ppl_target = [val_ppl_target[i] for i in range(len(val_ppl_target)) if i in sorted_gt_binned]
#     spr_rank, _ = spearmanr(top_k_val_ppl_gt, top_k_val_ppl_target)

#     print('Correctly ranked top %d %% (%d) with %.2f accuracy' %(top_k, total, common_ratio))
#     print('Spearman Correlation on top %d %% (%d): %.3f' % (top_k, len(top_k_val_ppl_gt), spr_rank))

#     return common_ratio, spr_rank
