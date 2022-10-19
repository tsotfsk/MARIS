import argparse
import gc
import os
import pickle
import random
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.append('../')

class AmazonSeq(object):

    def __init__(self, dataset='Beauty', root='.',
                 kcore=(5, 5), window_size=5, logger=None):
        """dataset class

        Args:
            sample (int, optional): sample a litte part of data. Defaults to -1.
            dataset (str, optional): dataset name. Defaults to 'Beauty'.
            kcore (tuple, optional): k-core parms. Defaults to (5, 5).
            logger ([type], optional): logger. Defaults to None.
        """

        self.logger = logger
        self.dataset = dataset
        self.loadpath = os.path.join(root, dataset)
        self.low_u, self.low_i = kcore
        self.phase = ['train', 'valid', 'test']
        self.window_size = window_size

        # data to save is save is True
        self.df = None
        self.reviews = []
        self.info = {}
        self.kg = nx.DiGraph()

        # build_data
        self._build_data()
        self._save_data()

    def _build_data(self):
        self._load_data()
        self._drop_dups()
        self._kcore()

        self._sort()
        self._remap_id()
        self._pad_items()
        self._split_data()

    def _load_data(self):
        users, items, timestamps = [], [], []

        for user, item, timestamp in tqdm(self._read_json_file(), leave=False):
            users.append(user)
            items.append(item)
            timestamps.append(timestamp)

        self.df = pd.DataFrame(
            {'user_id': users, 'item_id': items, 'timestamp': timestamps})
        self.logger.info('build_dataset done')

    def _read_json_file(self):
        pathname = os.path.join(self.loadpath, f'reviews_{self.dataset}.json')
        with open(pathname) as f:
            for line in f:
                inter = eval(line)
                user = inter['reviewerID']
                item = inter['asin']
                timestamp = inter['unixReviewTime']
                yield user, item, timestamp

    def _drop_dups(self):
        self.df.drop_duplicates(
            subset=['user_id', 'item_id'], keep='last', inplace=True)

    def _kcore(self):
        self.logger.info('process k core start...')
        df = self.df
        while True:
            df['user_count'] = df.groupby(
                'user_id')['item_id'].transform('count')
            df['item_count'] = df.groupby(
                'item_id')['user_id'].transform('count')
            user_mask = df['user_count'] >= self.low_u
            item_mask = df['item_count'] >= self.low_i
            mask = user_mask & item_mask
            self.logger.info('user_sum {} item_sum {}'.format(
                user_mask.sum(), item_mask.sum()))
            if all(mask):
                break
            df = df[mask].copy()
            gc.collect()
        df.drop(columns=['user_count', 'item_count'], inplace=True)
        self.logger.info('process k core done...')
        self.df = df

    def _pad_items(self):
        # XXX items pad at 0
        self.df['item_id'] += 1

    def _remap_id(self):
        # pathname = os.path.join(self.loadpath, f'{self.dataset}.inter')
        # header = ['user_id:token', 'item_id:token', 'timestamp:float']
        # self.df.to_csv(pathname, index=False, sep='\t', header=header)

        self.df['user_id'], raw_users = pd.factorize(self.df['user_id'])
        self.df['item_id'], raw_items = pd.factorize(self.df['item_id'])

        self.info['user_ids'] = raw_users.tolist()
        self.info['item_ids'] = raw_items.tolist()
        self.info['user_num'] = self.df['user_id'].max() + 1
        self.info['item_num'] = self.df['item_id'].max() + 1
        sparsity = 1 - self.df.shape[0] / (self.info['user_num'] * self.info['item_num'])

        self.logger.info("remap id done...")
        self.logger.info("number of users: {}".format(
            self.info['user_num']))
        self.logger.info("number of items: {}".format(
            self.info['item_num']))
        self.logger.info("number of inters: {}".format(self.df.shape[0]))
        self.logger.info("avg len: {}".format(self.df.shape[0] / self.info['user_num']))
        self.logger.info("sparsity: {}".format(self.df.shape[0] / self.info['user_num']))


    def _split_data(self):
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        for user_id, item_ids in self.df.groupby('user_id')['item_id']:
            item_ids = item_ids.tolist()
            seqs = []
            for idx in range(1, len(item_ids)):
                if idx < self.window_size:
                    seq = item_ids[:idx] + [0] * (self.window_size - idx)
                else:
                    seq = item_ids[idx - self.window_size:idx]
                target = item_ids[idx]
                seqs.append((user_id, seq, target))
            for seq in seqs[:-2]:
                self.train_data.append(seq)
            self.valid_data.append(seqs[-2])
            self.test_data.append(seqs[-1])

    def _sort(self):
        self.df.sort_values(by=['user_id', 'timestamp'], inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.logger.info("sort data by [user_id, timestamp] done...")

    def _save_data(self):
        # save seq data
        for name in self.phase:
            data = getattr(self, name + '_data')
            pathname = os.path.join(self.loadpath, 'seqdata', f'{self.dataset}_{name}.csv')
            with open(pathname, 'w') as f:
                for user_id, seq, target in data:
                    f.write(','.join([str(user_id), '|'.join(map(str, seq)), str(target)]))
                    f.write('\n')

        # save df
        pathname = os.path.join(self.loadpath, f'{self.dataset}.csv')
        self.df.to_csv(pathname, index=False)

        # save pkl
        pathname = os.path.join(self.loadpath, 'seqdata', f'{self.dataset}_item_sequences.pkl')
        seqs_for_kerl = self.df.groupby('user_id')['item_id'].agg(list).tolist()
        with open(pathname, 'wb') as f:
            pickle.dump(seqs_for_kerl, f)

        # save info
        pathname = os.path.join(self.loadpath, f'{self.dataset}_info.pkl')
        with open(pathname, 'wb') as f:
            pickle.dump(self.info, f)


class AmazonKG(object):

    def __init__(self, dataset='Beauty', root='.', device='1', logger=None, save=True):

        self.logger = logger
        self.dataset = dataset
        self.loadpath = os.path.join(root, dataset)
        self.device = device

        # data to save
        self.df = None
        self.heads = []
        self.tails = []
        self.rels = []
        self.others = []
        self.kg = nx.DiGraph()

        self._build_kg()

    def _build_kg(self):
        self._load_data()  # 加载kg数据为df
        self._set_priority()  # 设置kg关系的优先级，因为有重复的边, 比如 also bought 和 also view
        self._drop_dups()  # 删除重复的rel
        self._remap_id()  # remap_id, 这个要与seq保持一致，即商品id总是最小的

        self._build_graph()
        self._save_data()
        self._train_transe()
        self._save_embedding()

    def _set_priority(self):
        # priority >= 1
        self.priority = {
            'also_viewed': 1,
            'also_bought': 2,
            'bought_together': 3,
            'product': 4,
            'belong_to': 5,
        }

    def _drop_dups(self):

        # 设置优先级，然后保留优先级最高的
        self.df['priority'] = 0  # 初始化
        self.df['priority'] = self.df['rel'].map(
            self.priority, na_action='ignore')
        self.df.sort_values(by='priority', inplace=True)
        self.df.drop_duplicates(
            subset=['head_id', 'tail_id'], keep='last', inplace=True)
        self.df.drop(columns=['priority'], inplace=True)

        self.logger.info('the shape of dataframe after drop dups{}'.format(self.df.shape))
        self.logger.info('the edge of different_rels {}'.format(
            self.df.groupby('rel')['rel'].count()))

    def _load_data(self):
        # save info
        pathname = os.path.join(self.loadpath, f'{self.dataset}_info.pkl')
        with open(pathname, 'rb') as f:
            self.info = pickle.load(f)
        self.item_ids = self.info['item_ids']
        self.item_set = set(self.item_ids)

        # load kg
        self._read_json_file()
        self.df = pd.DataFrame(
            {'head_id': self.heads, 'tail_id': self.tails, 'rel': self.rels})
        self.logger.info('base kg edges {}'.format(self.df.shape[0]))

    def _read_json_file(self):
        """
            asin - ID of the product, e.g. 0000031852
            title - name of the product
            price - price in US dollars (at time of crawl)
            imUrl - url of the product image
            related - related products (also bought, also viewed, bought together, buy after viewing)
            salesRank - sales rank information
            brand - brand name
            categories - list of categories the product belongs to

        """
        pathname = os.path.join(self.loadpath, f'meta_{self.dataset}.json')
        with open(pathname) as f:
            for line in tqdm(f, leave=False):
                meta_info = eval(line)
                self._extract_line(meta_info)

    def _extract_line(self, line):
        head_e = line['asin']
        # not in seq items
        if head_e not in self.item_set:
            return

        self._extract_related(line)
        self._extract_brand(line)
        self._extract_categories(line)

    def _extract_related(self, line):
        head_e = line['asin']
        if 'related' in line:
            related = line['related']
            for rel in ['also_bought', 'also_viewed', 'bought_together']:
                if rel in related:
                    for tail_e in related[rel]:
                        if tail_e in self.item_set:
                            self._add_tuple(head_e, tail_e, rel)

    def _extract_brand(self, line):
        head_e = line['asin']
        rel = 'product'
        if 'brand' in line:
            assert isinstance(line['brand'], str)
            tail_e = line['brand']
            self._add_tuple(head_e, tail_e, rel)

    def _extract_categories(self, line):
        head_e = line['asin']
        rel = 'belong_to'
        if 'categories' in line:
            categories = line['categories']
            assert isinstance(categories, list)
            for categorie in categories:
                for tail_id in categorie:
                    if tail_id != self.dataset:  # all items belong to datasets
                        self._add_tuple(head_e, tail_id, rel)

    def _add_tuple(self, head_e, tail_e, rel, double=True):
        # forward
        self.heads.append(head_e)
        self.tails.append(tail_e)
        self.rels.append(rel)

        if double is False:
            return

        # backward
        self.heads.append(tail_e)
        self.tails.append(head_e)
        self.rels.append(rel)

    def _remap_id(self):
        # item map
        assert len(self.item_ids) == self.info['item_num']
        self.item_map = dict(
            zip(self.item_ids, range(self.info['item_num'])))
        self.logger.info(
            'the number of item entitys {}'.format(len(self.item_map)))

        # brand cates words map
        id_number = self.info['item_num']
        self.other_map = {}
        all_entity_ids = np.concatenate((self.df['head_id'], self.df['tail_id']))
        other_entity = list(set(all_entity_ids) - self.item_set)
        for idx, head in enumerate(other_entity):
            self.other_map[head] = idx + id_number
        self.info['other_num'] = len(self.other_map)
        self.logger.info(
            'the number of other entitys {}'.format(len(self.other_map)))

        # rel map
        self.rel_map = {}
        for idx, rel in enumerate(set(self.df['rel'])):
            self.rel_map[rel] = idx
        self.info['rel_num'] = len(self.rel_map)
        self.logger.info('the number of rels {}'.format(len(self.rel_map)))

        # merge maps
        self.entity_map = {**self.item_map, **self.other_map}

        # TODO map
        self.df['head_id'] = self.df['head_id'].map(self.entity_map)
        self.df['tail_id'] = self.df['tail_id'].map(self.entity_map)
        self.df['rel'] = self.df['rel'].map(self.rel_map)

    def _build_graph(self):
        for edge in self.df.values:
            head_id, tail_id, rel = edge
            self._add_edge(head_id, tail_id, rel)

        n_entitys = self.kg.number_of_nodes()
        n_edges = self.kg.number_of_edges()
        self.info['entity_num'] = n_entitys
        self.info['edge_num'] = n_edges
        self.logger.info('the number of all entitys {}'.format(
            n_entitys))
        self.logger.info('the number of edges {}'.format(
            n_edges))
        self.logger.info('the average degree {}'.format(
            n_edges / n_entitys))

    def _add_edge(self, head_id, tail_id, rel, double=True):
        self.kg.add_edge(head_id, tail_id, relation=rel)
        if double is False:
            return
        self.kg.add_edge(tail_id, head_id, relation=rel)

    def _save_data(self):
        # trans to recbole format
        # self.recbole_format()

        # trans to openke format
        self.openke_format()

    # def recbole_format(self):
    #     link = {}
    #     entity_ids = list(self.entity_map.keys())[:len(self.item_map)]
    #     link['item_id:token'] = entity_ids
    #     link['entity_id:token'] = entity_ids
    #     link_df = pd.DataFrame(link)

    #     pathname = os.path.join(self.loadpath, f'{self.dataset}.link')
    #     link_df.to_csv(pathname, sep='\t', index=False)

    def openke_format(self):
        kg_root = os.path.join(self.loadpath, 'openke')
        if not os.path.exists(kg_root):
            os.makedirs(kg_root, exist_ok=True)

        pathname = os.path.join(kg_root, 'train2id.txt')
        with open(pathname, 'w') as f:
            f.write(str(self.info['edge_num']) + '\n')
            for line in nx.generate_edgelist(self.kg, delimiter='\t', data=['relation']):
                line += "\n"
                f.write(line)

        pathname = os.path.join(kg_root, 'relation2id.txt')
        with open(pathname, 'w') as f:
            id_map = sorted(self.rel_map.items(), key=lambda x: x[1])
            f.write(str(len(id_map)) + '\n')
            for rel, rel_id in id_map:
                f.write(f'{rel}\t{rel_id}\n')

        pathname = os.path.join(kg_root, 'entity2id.txt')
        with open(pathname, 'w') as f:
            id_map = sorted(self.entity_map.items(), key=lambda x: x[1])
            f.write(str(len(id_map)) + '\n')
            for entity, entity_id in id_map:
                f.write(f'{entity}\t{entity_id}\n')

    def _train_transe(self):
        from openke.config import Trainer
        from openke.data import TrainDataLoader
        from openke.module.loss import MarginLoss
        from openke.module.model import TransE
        from openke.module.strategy import NegativeSampling

        # dataloader for training
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        train_dataloader = TrainDataLoader(
            in_path=os.path.join(self.loadpath, 'openke') + '/',  # XXX 大无语事件
            nbatches=100,
            threads=8,
            sampling_mode="cross",
            bern_flag=1,
            filter_flag=1,
            neg_ent=5,
            neg_rel=0)

        # define the model
        transe = TransE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=128,
            p_norm=1,
            norm_flag=True)

        # define the loss function
        model = NegativeSampling(
            model=transe,
            loss=MarginLoss(margin=5.0),
            batch_size=train_dataloader.get_batch_size()
        )

        # train the model
        trainer = Trainer(model=model, data_loader=train_dataloader,
                          train_times=200, alpha=1.0, use_gpu=True)
        trainer.run()
        transe.save_checkpoint(os.path.join(self.loadpath, 'openke', 'transe.ckpt'))

    def _save_embedding(self):
        pathname = os.path.join(self.loadpath, 'openke', 'transe.ckpt')
        parm_dict = torch.load(pathname)
        rel_embed = parm_dict['rel_embeddings.weight'].cpu().numpy()
        ent_embed = parm_dict['ent_embeddings.weight'].cpu().numpy()

        pathname = os.path.join(self.loadpath, 'pretrain', f'{self.dataset}_ent.npy')
        np.save(pathname, ent_embed)

        pathname = os.path.join(self.loadpath, 'pretrain', f'{self.dataset}_rel.npy')
        np.save(pathname, rel_embed)


class AmazonImage(object):

    def __init__(self, dataset='Beauty', root='.',
                 logger=None, save=True):

        self.logger = logger
        self.dataset = dataset
        self.loadpath = os.path.join(root, dataset)

        self._extract_data()

    def _read_binary_file(self):
        import array
        pathname = os.path.join(self.loadpath, f'image_features_{self.dataset}.b')
        f = open(pathname, 'rb')
        while True:
            asin = f.read(10)
            if asin == '':
                break
            a = array.array('f')
            try:
                a.fromfile(f, 4096)
            except Exception:
                break
            yield asin.decode('utf-8'), a.tolist()

    def _extract_data(self):
        pathname = os.path.join(self.loadpath, f'{self.dataset}_info.pkl')
        with open(pathname, 'rb') as f:
            self.info = pickle.load(f)
        self.item_ids = self.info['item_ids']
        self.item_set = set(self.item_ids)
        self.item_map = dict(
            zip(self.item_ids, range(self.info['item_num'])))

        items, images = [], []
        for item, image in tqdm(self._read_binary_file(), leave=False):
            if item in self.item_set:
                items.append(self.item_map[item])
                images.append(image)

        self._save_embedding(items, images)

        self.logger.info('load image done')
        self.logger.info('the number of items with image {}'.format(
            len(items)))

    def _save_embedding(self, items, images):
        image_embed = np.zeros((self.info['item_num'], 128))
        items = np.array(items)
        images = np.array(images)
        model = PCA(n_components=128).fit(images)
        new_embed = model.transform(images)
        image_embed[items] = new_embed

        pathname = os.path.join(self.loadpath, 'pretrain', f'{self.dataset}_img.npy')
        np.save(pathname, image_embed)


class Callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


class AmazonText(object):
    def __init__(self, dataset='Beauty', root='.',
                 logger=None, save=True):

        self.logger = logger
        self.dataset = dataset
        self.loadpath = os.path.join(root, dataset)
        self.save = save

        self._extract_data()

    def _extract_data(self):
        pathname = os.path.join(self.loadpath, f'{self.dataset}_info.pkl')
        with open(pathname, 'rb') as f:
            self.info = pickle.load(f)
        self.item_ids = self.info['item_ids']
        self.item_set = set(self.item_ids)
        self.item_map = dict(
            zip(self.item_ids, range(self.info['item_num'])))

        def _clean_text(text):
            text = text.strip().lower()
            stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
            text = re.sub(stop, '', text)
            return text.split(' ')

        # load text
        items, texts = [], []
        for item, text in tqdm(self._read_json_file(), leave=False):
            if item in self.item_set:
                items.append(self.item_map[item])
                texts.append(_clean_text(text))

        self.logger.info('load text done')
        self.logger.info('the number of items with description {}'.format(
            len(items)))

        self.logger.info('train word2vec...')
        word_embed = self._train_word2vec_model(texts)
        self._save_embedding(word_embed, items, texts)
        self.logger.info('train word2vec done.')

    def _train_word2vec_model(self, titles):
        model = Word2Vec(sentences=titles,
                         min_count=1,
                         size=128,
                         window=5,
                         iter=40,
                         compute_loss=True,
                         callbacks=[Callback()])
        print("# words in the space: {}".format(len(model.wv.index2word)))
        word_list = model.wv.index2word
        embeddings = np.array([model[word] for word in word_list])
        return dict(zip(word_list, embeddings))

    def _read_json_file(self):
        """
            asin - ID of the product, e.g. 0000031852
            title - name of the product
            price - price in US dollars (at time of crawl)
            imUrl - url of the product image
            related - related products (also bought, also viewed, bought together, buy after viewing)
            salesRank - sales rank information
            brand - brand name
            categories - list of categories the product belongs to

        """
        pathname = os.path.join(self.loadpath, f'meta_{self.dataset}.json')
        with open(pathname) as f:
            for line in f:
                meta_info = eval(line)
                try:
                    asin, text = self._extract_text(meta_info)
                    yield asin, text
                except Exception:
                    pass

    def _extract_text(self, line):
        asin = line['asin']
        text = []
        if 'title' in line:
            assert isinstance(line['title'], str)
            title = line['title']
            text.append(title)
        if 'description' in line:
            assert isinstance(line['description'], str)
            desc = line['description']
            text.append(desc)
        if not text:
            raise ValueError
        return asin, " ".join(text)

    def _save_embedding(self, word_embed, items, texts):
        text_embed = np.zeros((self.info['item_num'], 128))
        for item, text in zip(items, texts):
            text_vec = [word_embed[word] for word in text]
            text_vec = sum(text_vec) / len(text_vec)
            text_embed[item] = text_vec

        pathname = os.path.join(self.loadpath, 'pretrain', f'{self.dataset}_txt.npy')
        np.save(pathname, text_embed)


def get_logger(filename=None):
    from utils import Logger
    return Logger(filename)


def mkdirs(root, dataset):
    for dirname in ['openke', 'pretrain', 'seqdata']:
        if not os.path.exists(os.path.join(root, dataset, dirname)):
            os.makedirs(os.path.join(root, dataset, dirname), exist_ok=True)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Beauty')
    parser.add_argument('--root', default='.', help='the root path of all datasets')
    parser.add_argument('--device', default='1', help='transe device')
    args = parser.parse_args()

    seed_everything(2048)
    mkdirs(args.root, args.dataset)
    logger = get_logger(f"{args.dataset}/preprocess.log")
    seq_data = AmazonSeq(dataset=args.dataset, root=args.root, logger=logger)
    kg_data = AmazonKG(dataset=args.dataset, root=args.root, device=args.device, logger=logger)
    image_data = AmazonImage(dataset=args.dataset, root=args.root, logger=logger)
    text_data = AmazonText(dataset=args.dataset, root=args.root, logger=logger)
