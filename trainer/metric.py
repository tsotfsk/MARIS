import numpy as np


class TopKMetric(object):

    def __init__(self):
        pass

    def __call__(self, *args, **kwgs):
        return self.calculate(*args, **kwgs)

    def calculate(self, *args, **kwgs):
        raise NotImplementedError


class HR(TopKMetric):
    r"""Hit_ (also known as hit ratio at :math:`N`) is a way of calculating how many 'hits' you have
    in an n-sized list of ranked items.
``
    .. _Hit: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    .. math::
        \mathrm {HR@K} =\frac{Number \space of \space Hits @K}{|GT|}

    :math:`HR` is the number of users with a positive sample in the recommendation list.
    :math:`GT` is the total number of samples in the test set.

    """

    def calculate(self, pos_idx, *args):
        result = np.cumsum(pos_idx, axis=1)
        return (result > 0).astype(int)


class MRR(TopKMetric):
    r"""The MRR_ (also known as mean reciprocal rank) is a statistic measure for evaluating any process
    that produces a list of possible responses to a sample of queries, ordered by probability of correctness.

    .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    .. math::
        \mathrm {MRR} = \frac{1}{|{U}|} \sum_{i=1}^{|{U}|} \frac{1}{rank_i}

    :math:`U` is the number of users, :math:`rank_i` is the rank of the first item in the recommendation list
    in the test set results for user :math:`i`.

    """

    def calculate(self, pos_idx, *args):
        idxs = pos_idx.argmax(axis=1)
        result = np.zeros_like(pos_idx, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_idx[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class MAP(TopKMetric):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.

    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.

    .. _map: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}

    """

    def calculate(self, pos_idx, pos_len):
        pre = Precision()(pos_idx, pos_len)
        sum_pre = np.cumsum(pre * pos_idx.astype(np.float), axis=1)
        len_rank = np.full_like(pos_len, pos_idx.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_idx, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_idx.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


class Recall(TopKMetric):
    r"""Recall_ (also known as sensitivity) is the fraction of the total amount of relevant instances
    that were actually retrieved

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
        \mathrm {Recall@K} = \frac{|Rel_u\cap Rec_u|}{Rel_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Recall@K` of each user.

    """

    def calculate(self, pos_idx, pos_len):
        return np.cumsum(pos_idx, axis=1) / pos_len.reshape(-1, 1)


class NDCG(TopKMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in U^{te}NDCG_u@K}}{|U^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`{rel_i}` equals to 1 if the item is ground truth otherwise 0.
    :math:`U^{te}` stands for all users in the test set.

    """
    def calculate(self, pos_idx, pos_len):
        len_rank = np.full_like(pos_len, pos_idx.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_idx, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_idx.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_idx, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_idx.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_idx, dcg, 0), axis=1)

        result = dcg / idcg
        return result


class Precision(TopKMetric):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances

    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

    :math:`Rel_u` is the set of items relevant to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.

    """

    def calculate(self, pos_idx, *args):
        return pos_idx.cumsum(axis=1) / np.arange(1, pos_idx.shape[1] + 1)
