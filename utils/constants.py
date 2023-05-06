
# feature names
SAMPLE_ID = "sample_id"
USER_ID = "user_id"
ITEM_SEQ_ID = "item_seq_id"
POS_ITEM_ID = "pos_item_id"
NEG_ITEM_ID = "neg_item_id"
ACTION_SEQ = "action_seq"

# path
CONFIG_PATH = "config/{dataset}/{model}.json"
DATA_INFO_PATH = "dataset/{dataset}/{dataset}_info.pkl"
SEQ_DATA_PATH = "dataset/{dataset}/seqdata/{dataset}_{phase}.csv"
STATIC_SAMPLE_PATH = "dataset/{dataset}/seqdata/{dataset}_{phase}_samples.pkl"
SAVE_ROOT_PATH = "saved/{dataset}/{model}/{commit}"


MODAL_PATH_DICT = {
    'ent': "dataset/{dataset}/pretrain/{dataset}_ent.npy",
    'txt': "dataset/{dataset}/pretrain/{dataset}_txt.npy",
    'img': "dataset/{dataset}/pretrain/{dataset}_img.npy"
}

# dynamic: for train sample
# static: for eval sample
sample_strategy_dict = {
    "full_sort_ce": {
        "dynamic_sample_num": 0,
        "static_sample_num": 0},
    "full_sort_bpr": {
        "dynamic_sample_num": 32,
        "static_sample_num": 0},
    "sample_sort_ce": {
        "dynamic_sample_num": 0,
        "static_sample_num": 1000},
    "sample_sort_bpr": {
        "dynamic_sample_num": 32,
        "static_sample_num": 1000},
}
