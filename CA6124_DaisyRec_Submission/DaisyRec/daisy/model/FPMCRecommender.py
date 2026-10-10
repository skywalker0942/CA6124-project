"""FPMC sequential item ranker for integration into DaisyRec.

Training batch contract: (user, previous_item, positive_item, negative_item).
Evaluation CandidatesDataset batches remain (users, candidate_item_ids); the
previous item is supplied through config['fpmc_eval_prev'] for each split.
"""

import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender


class FPMC(GeneralRecommender):
    tunable_param_names = ["factors", "lr", "reg_2", "epochs", "batch_size", "num_ng"]

    def __init__(self, config):
        super().__init__(config)
        self.user_num = int(config["user_num"])
        self.item_num = int(config["item_num"])
        self.factors = int(config["factors"])
        self.lr = float(config["lr"])
        self.epochs = int(config["epochs"])
        self.reg_2 = float(config.get("reg_2", 1e-4))
        self.topk = int(config.get("topk", 10))
        self.loss_type = config.get("loss_type", "BPR")
        self.optimizer = config.get("optimizer", "adam")
        init_method = config.get("init_method", "default")
        self.initializer = "xavier_uniform" if init_method == "default" else init_method
        self.early_stop = bool(config.get("early_stop", False))
        self.fpmc_eval_prev = {
            int(user): int(item)
            for user, item in config.get("fpmc_eval_prev", {}).items()
        }

        self.user_pref = nn.Embedding(self.user_num, self.factors)
        self.item_pref = nn.Embedding(self.item_num, self.factors)
        self.prev_item = nn.Embedding(self.item_num, self.factors)
        self.next_item = nn.Embedding(self.item_num, self.factors)
        self.apply(self._init_weight)

    def forward(self, users, previous_items, item_ids):
        u = self.user_pref(users)
        q = self.item_pref(item_ids)
        p = self.prev_item(previous_items)
        v = self.next_item(item_ids)
        return (u * q).sum(dim=-1) + (p * v).sum(dim=-1)

    def calc_loss(self, batch):
        users, previous, positive, negative = [x.to(self.device).long() for x in batch]
        pos_score = self(users, previous, positive)
        neg_score = self(users, previous, negative)
        loss = self.criterion(pos_score, neg_score)
        penalty = (
            self.user_pref(users).square().sum()
            + self.item_pref(positive).square().sum()
            + self.prev_item(previous).square().sum()
            + self.next_item(positive).square().sum()
        )
        return loss + self.reg_2 * penalty / max(1, users.shape[0])

    def _context_for(self, user):
        # Item 0 is the deterministic fallback for a user with no history.
        return self.fpmc_eval_prev.get(int(user), 0)

    @torch.no_grad()
    def predict(self, u, i):
        user = int(u)
        device = self.user_pref.weight.device
        users = torch.tensor([user], dtype=torch.long, device=device)
        previous = torch.tensor([self._context_for(user)], dtype=torch.long, device=device)
        items = torch.tensor([int(i)], dtype=torch.long, device=device)
        return float(self(users, previous, items).item())

    @torch.no_grad()
    def full_rank(self, u):
        user = int(u)
        device = self.user_pref.weight.device
        users = torch.full((self.item_num,), user, dtype=torch.long, device=device)
        previous = torch.full((self.item_num,), self._context_for(user), dtype=torch.long, device=device)
        items = torch.arange(self.item_num, dtype=torch.long, device=device)
        scores = self(users, previous, items)
        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()

    @torch.no_grad()
    def rank(self, test_loader):
        self.eval()
        ranked_batches = []
        device = self.user_pref.weight.device
        for users, candidates in test_loader:
            users = users.to(device).long()
            candidates = candidates.to(device).long()
            contexts = torch.tensor(
                [self._context_for(user) for user in users.cpu().tolist()],
                dtype=torch.long, device=device
            )
            user_vec = self.user_pref(users)[:, None, :]
            item_user_vec = self.item_pref(candidates)
            prev_vec = self.prev_item(contexts)[:, None, :]
            item_seq_vec = self.next_item(candidates)
            scores = (user_vec * item_user_vec).sum(-1) + (prev_vec * item_seq_vec).sum(-1)
            order = torch.argsort(scores, dim=1, descending=True)[:, :self.topk]
            ranked_batches.append(torch.gather(candidates, 1, order).cpu())
        return torch.cat(ranked_batches, dim=0).numpy()
