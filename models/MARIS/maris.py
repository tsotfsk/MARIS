from copy import deepcopy

import torch
import torch.nn as nn
from box import Box

from models import GRU4Rec
from models.abstract_model import SequentialModel
from utils import ACTION_SEQ

from .agent import ModalAgent
from .qmix import QMixer


class MARIS(SequentialModel):

    def __init__(self, config, logger):
        super().__init__(config, logger)

        self.maris = MARISModule(config, logger)
        self.target_maris = deepcopy(self.maris)
        self.gamma = self.model_param.gamma

    def forward(self, input):
        return self.maris(input)

    def calculate_loss(self, input):
        pos_item_id = input.pos_item_id
        neg_item_id = input.neg_item_id
        item_seq_mask = input.item_seq_mask
        item_seq_len = input.item_seq_len

        output = self.maris.imitate_episodes(input)  # [B, agent_num/1, T, ...]
        target_output = self.target_maris.imitate_episodes(input)

        chosen_action_qvals = output.qvals.gather(dim=-1, index=input.action_seq.unsqueeze(-1)).squeeze(-1)  # [B, agent_num, T]
        target_max_qvals = target_output.qvals.max(dim=-1)[0]  # [B, agent_num, T]
      
        chosen_action_qvals = self.maris.mixer(chosen_action_qvals, output.states)  # [B, T, 1]

        target_max_qvals = self.target_maris.mixer(target_max_qvals, target_output.states)  # [B, T - 1, 1]
        target_max_qvals = torch.cat([target_max_qvals[:, 1:], torch.zeros_like(target_max_qvals[:, :1])], dim=1)  # [B, T, 1]

        scatter_rewards = torch.zeros_like(chosen_action_qvals).scatter_(
            dim=1, index=item_seq_len.reshape(-1, 1, 1) - 1,
            src=output.rewards.reshape(-1, 1, 1))  # [B, T, 1]

        # calculate_rec_loss
        rec_loss = self.maris.calculate_rec_loss(output.enhts, pos_item_id, neg_item_id)

        # calculate_td_loss
        targets = scatter_rewards + self.gamma * target_max_qvals
        td_error = (chosen_action_qvals - targets.detach())
        # 0-out the targets that came from padded data
        mask = item_seq_mask.unsqueeze(-1)
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        return rec_loss, td_loss

    def sample_episodes(self, input, sample_num=5):
        assert not hasattr(input, ACTION_SEQ)
        assert input.test_mode is False
        return self.maris.sample_episodes(input, sample_num=sample_num)

    def imitate_episodes(self, input):
        assert hasattr(input, ACTION_SEQ)
        assert input.test_mode is False
        return self.maris.imitate_episodes(input)

    def predict(self, input):
        assert input.test_mode is True
        return self.maris.predict(input)

    def update_targets(self):
        self.target_maris.load_state_dict(self.maris.state_dict())


class MARISModule(SequentialModel):

    def __init__(self, config, logger):
        super(MARISModule, self).__init__(config, logger)

        # load embedding

        self.embedding_size = self.model_param.embedding_size
        self.hidden_size = self.model_param.hidden_size
        self.num_layers = self.model_param.num_layers
        self.dropout = self.model_param.dropout
        self.agent_num = self.model_param.agent_num  # XXX ent, txt, img

        self.ent_agent = ModalAgent(config, logger, modal='ent')
        self.txt_agent = ModalAgent(config, logger, modal='txt')
        self.img_agent = ModalAgent(config, logger, modal='img')

        self.encoder = GRU4Rec(self.config, self.logger)
        
        self.img_agent.shift_embedding = self.encoder.bsc_embedding
        self.txt_agent.shift_embedding = self.encoder.bsc_embedding
        self.ent_agent.shift_embedding = self.encoder.bsc_embedding

        self.proj = nn.Sequential(
            nn.Linear(self.embedding_size * 3, self.embedding_size),
        )
        self.dropout = nn.Dropout(self.dropout)
        self.mixer = QMixer(config, logger)
        self.apply(self._init_weights)

        # load embedding
        for agent in [self.ent_agent, self.txt_agent, self.img_agent]:
            agent.load_embedding()

    def init_hidden(self, input):
        out = self.encoder.get_user_embedding(input.item_seq_id, input.item_seq_len)  # [B, D]

        B, D = out.shape
        out = out.reshape(1, B, 1, D)  # [1, B, 1, D]
        out = out.repeat(self.num_layers, 1, self.agent_num, 1)  # [num_layers, B, agent_num, D]
        return out
    
    def get_final_score(self, user_emb, item_emb):
        bsc_score = torch.matmul(user_emb[:, 0, :], item_emb[:, 0, :].t())
        ent_score = torch.matmul(user_emb[:, 1, :], item_emb[:, 1, :].t())
        txt_score = torch.matmul(user_emb[:, 2, :], item_emb[:, 2, :].t())
        img_score = torch.matmul(user_emb[:, 3, :], item_emb[:, 3, :].t())

        bsc_img_score = (bsc_score + img_score)
        bsc_txt_score = (bsc_score + txt_score)
        bsc_ent_score = (bsc_score + ent_score)
        img_txt_score = (bsc_score + img_score + txt_score)
        img_ent_score = (bsc_score + img_score + ent_score)
        txt_ent_score = (bsc_score + txt_score + ent_score)
        img_txt_ent_score = (bsc_score + img_score + txt_score + ent_score)

        score = torch.stack((bsc_score,
                            bsc_img_score,
                            bsc_txt_score,
                            bsc_ent_score,
                            img_txt_score,
                            img_ent_score,
                            txt_ent_score,
                            img_txt_ent_score), dim=0)               
        return score.max(dim=0)[0]

    def get_state(self, input, hidden, step):
        item_id = input.item_seq_id[:, step, None]  # [B, 1]
        ent_obs = self.ent_agent.get_observation(hidden[:, :, 0], item_id)  # [B, 1, 2D]
        txt_obs = self.txt_agent.get_observation(hidden[:, :, 1], item_id)  # [B, 1, 2D]
        img_obs = self.img_agent.get_observation(hidden[:, :, 2], item_id)  # [B, 1, 2D]
        state = ent_obs + txt_obs + img_obs  # [B, 1, 2D]

        return state  # [B, 1, 2D]

    def select_action(self, input, hidden, step):
        ent_qval, ent_action = self.ent_agent.select_action(input, hidden[:, :, 0], step)  # [B, 1]
        txt_qval, txt_action = self.txt_agent.select_action(input, hidden[:, :, 1], step)
        img_qval, img_action = self.img_agent.select_action(input, hidden[:, :, 2], step)

        step_action = torch.cat([ent_action, txt_action, img_action], dim=1)
        qval = torch.cat([ent_qval, txt_qval, img_qval], dim=1)
        return step_action, qval

    def imitate_action(self, input, hidden, step):
        step_action = input.action_seq[:, :, step]  # [B, 3]
        ent_qval = self.ent_agent.imitate_action(input, hidden[:, :, 0], step)
        txt_qval = self.txt_agent.imitate_action(input, hidden[:, :, 1], step)
        img_qval = self.img_agent.imitate_action(input, hidden[:, :, 2], step)

        qval = torch.cat([ent_qval, txt_qval, img_qval], dim=1)
        return step_action, qval

    def move_one_step(self, input, hidden, step, step_action):
        ent_hidden = self.ent_agent.move_one_step(input, hidden[:, :, 0], step, step_action[:, 0, None])
        txt_hidden = self.txt_agent.move_one_step(input, hidden[:, :, 1], step, step_action[:, 1, None])
        img_hidden = self.img_agent.move_one_step(input, hidden[:, :, 2], step, step_action[:, 2, None])

        hidden = torch.stack([ent_hidden, txt_hidden, img_hidden], dim=-2)  # [num_layers, B, agent_num, D]
        return hidden

    def imitate_episodes(self, input):
        pos_item_id = input.pos_item_id

        output = self.forward(input)
        output.rewards = self.get_reward(output.enhts, pos_item_id)  # [B]
        return output

    def sample_episodes(self, input, sample_num=5):
        pos_item_id = input.pos_item_id

        outputs = Box(enhts=[], qvals=[], actions=[], states=[], rewards=[])
        for _ in range(sample_num):
            output = self.forward(input)
            for key, val in output.items():
                outputs[key].append(val)
            reward = self.get_reward(output.enhts, pos_item_id)  # [B]
            outputs.rewards.append(reward)

        for key, val in outputs.items():
            val = torch.stack(val, dim=1)  # [B, sample_num, ...]
            outputs[key] = val.reshape(-1, *val.shape[2:])  # [B * sample_num, ...]

        return outputs

    def proj_layer(self, hidden):
        shift_embed, modal_embed = torch.chunk(hidden, 2, dim=-1)  # [B, agent_num, D]
        modal_embed = modal_embed.reshape(modal_embed.size(0), -1)
        shift_embed = self.proj(shift_embed.reshape(shift_embed.size(0), -1))  # [B, agent_num * 3D] -> [B, D]
        return torch.cat([shift_embed, modal_embed], dim=1)  # [B, (1 + agent_num) * D]

    def forward(self, input, output=None):
        item_seq_id = input.item_seq_id  # [B, T]
        item_seq_len = input.item_seq_len  # [B]

        hidden = self.init_hidden(input)  # [num_layers, B, agent_num, D]
        
        ht = hidden[-1, :, 0, :]  # [B, D]
        
        output = Box(enhts=[], qvals=[], actions=[], states=[])
        
        for step in range(item_seq_id.size(1)):
            if hasattr(input, ACTION_SEQ):
                step_action, qvals = self.imitate_action(input, hidden, step)  # [B, 1]
            else:
                step_action, qvals = self.select_action(input, hidden, step)  # [B, 1]
            state = self.get_state(input, hidden, step)  # [B, 1, 2D]
            hidden = self.move_one_step(input, hidden, step, step_action)  # [num_layers, B, agent_num, D]

            output.actions.append(step_action)
            output.enhts.append(hidden[-1])
            output.states.append(state)
            output.qvals.append(qvals)
            
        for key, val in output.items():
            output[key] = torch.stack(val, dim=2)  # [B, agent_num/1, T, ...]    
            
        if hasattr(input, ACTION_SEQ):
            assert torch.equal(output.actions, getattr(input, ACTION_SEQ))
        
        enhts = self.dense_gather_layer(input, output.enhts)  # [B, agent_num, T, D]
        output.enhts = self.proj_layer(enhts) # + ht  # [B, D]
        return output

    def dense_gather_layer(self, input, enhts):
        ent_ht = self.gather_indexes(enhts[:, 0], input.item_seq_len - 1) 
        ent_ht = self.ent_agent.dense(ent_ht)
        
        txt_ht = self.gather_indexes(enhts[:, 1], input.item_seq_len - 1)
        txt_ht = self.txt_agent.dense(txt_ht)
        
        img_ht = self.gather_indexes(enhts[:, 2], input.item_seq_len - 1)
        img_ht = self.img_agent.dense(img_ht)
        
        return torch.stack([ent_ht, txt_ht, img_ht], dim=1)  # [B, 3, 2D]
        
    def calculate_rec_loss(self, enht, pos_item_id, neg_item_id):
        user_embed = enht.reshape(-1, (1 + self.agent_num), self.hidden_size)  # [B, 1+agent_num, D]

        combo_item_embedding = self.combination_item_embedding()  # [I, 2^agent_num, D]

        scores = self.get_final_score(user_embed, combo_item_embedding)  # [B, I]
        
        pos_scores = scores.gather(1, pos_item_id.unsqueeze(1)).squeeze(1)
        neg_scores = scores.gather(1, neg_item_id)

        loss = self.loss_fct(pos_scores, neg_scores)
        return loss

    def get_reward(self, enht, item_id):
        user_embed = enht.reshape(-1, (1 + self.agent_num), self.hidden_size)  # [B, 1+agent_num, D]
        combo_item_embedding = self.combination_item_embedding()  # [I, 2^agent_num, D]
        scores = self.get_final_score(user_embed, combo_item_embedding)  # [B, I]
        pos_scores = scores.gather(1, item_id.unsqueeze(1)).squeeze(1)
        return pos_scores  # [B]

    def combination_item_embedding(self):
        bsc_embed = self.dropout(self.encoder.get_item_embedding())  # [I, D]
        ent_emebd = self.ent_agent.get_item_embedding()  # [I, D]
        txt_embed = self.txt_agent.get_item_embedding()  # [I, D]
        img_embed = self.img_agent.get_item_embedding()  # [I, D]
        return torch.stack([bsc_embed, ent_emebd, txt_embed, img_embed], dim=1)  # [I, 1 + agent_num, D]

    def calculate_loss(self, input):
        raise NotImplementedError
    
    def calculate_td_loss(self, input):
        raise NotImplementedError

    def predict(self, input):
        enhts = self.forward(input).enhts  # [B, D]
        user_embed = enhts.reshape(-1, (1 + self.agent_num), self.hidden_size)  # [B, 1+agent_num, D]
        
        combo_item_embedding = self.combination_item_embedding()  # [I, 1+agent_num, D]

        scores = self.get_final_score(user_embed, combo_item_embedding)  # [B, I]
        return scores
