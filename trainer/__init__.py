from trainer.metric import HR, MAP, NDCG, MRR, Recall, Precision
from trainer.learner import BaseLearner, RACELearner
from trainer.dataloader import SequentialDataLoader
from trainer.evaluator import Evaluator

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
}
