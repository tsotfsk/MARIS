import torch
import numpy as np
from collections import namedtuple
import sys
import os
from utils import NEG_ITEM_ID


class Evaluator(object):

    def __init__(self, metrics=None, topk=[5]):
        self.topk = topk
        self.metrics = metrics

    def collect(self, scores, input):
        # if use test sample, just rank the test samples
        if hasattr(input, NEG_ITEM_ID):
            cand_id = torch.cat((input.pos_item_id.unsqueeze(1), input.neg_item_id), dim=1)
            scores.scatter_add_(1, cand_id, torch.full_like(
                cand_id, 1000, device=cand_id.device, dtype=torch.float32))

        # mask history items
        # scores.scatter_(1, input.user_seqs, -np.inf)

        # mask padding items
        scores[:, 0] = -np.inf

        # topk
        _, topk_idx = torch.topk(scores, max(self.topk), dim=-1)  # nusers x k

        # pos_idx
        target_mask = torch.zeros_like(scores, device=scores.device)
        target_mask.scatter_(1, input.pos_item_id.unsqueeze(1), 1)
        pos_idx = target_mask.gather(dim=1, index=topk_idx)

        # pos_len
        pos_len = (target_mask > 0).sum(dim=1)

        return pos_idx.cpu().numpy(), pos_len.cpu().numpy()

    def evaluate(self, pos_idx_lst, pos_len_lst):
        pos_idx = np.concatenate(pos_idx_lst, axis=0).astype(bool)
        pos_len = np.concatenate(pos_len_lst, axis=0)
        # get metrics
        metrics_dict = {}
        result_matrix = self._calculate_metrics(pos_idx, pos_len)
        result_lst = np.stack(result_matrix, axis=0)
        result_lst = result_lst.mean(axis=1)
        for metric, value in zip(self.metrics, result_lst):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metrics_dict[key] = round(value[k - 1], 4)
        return metrics_dict

    def _calculate_metrics(self, pos_idx, pos_len):
        result_list = []
        from trainer import metric_dict
        for metric in self.metrics:
            metric_fuc = metric_dict[metric.lower()]
            result = metric_fuc(pos_idx, pos_len)
            result_list.append(result)
        return result_list


if __name__ == "__main__":
    evaluator = Evaluator(metrics=['HR', 'Recall', 'NDCG'], topk=(2, ))
    scores = torch.randn((2, 5))
    Input = namedtuple("input", ["user_seqs", "target_ids"])
    input = Input(
        torch.LongTensor([[2, 3], [0, 1]]),
        torch.LongTensor([[1], [4]]),
    )
    pos_idx, pos_len = evaluator.collect(scores, input)
    result = evaluator.evaluate([pos_idx], [pos_len])
    print(scores)
    print(pos_idx)
    print(pos_len)
    print(result)
