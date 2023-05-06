from trainer.metric import HR, MAP, NDCG, MRR, Recall, Precision
from trainer.learner import BaseLearner, RLLearner
from trainer.dataloader import SequentialDataLoader, SequentialDataset, SequentialInput

metric_dict = {
    'hr': HR(),
    'map': MAP(),
    'ndcg': NDCG(),
    'mrr': MRR(),
    'recall': Recall(),
    'precision': Precision()
}

learner_dict = {
    'GRU4Rec': BaseLearner,
    'GRU4RecF': BaseLearner,
    'MARIS': RLLearner,
}
