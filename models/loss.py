import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Softmax
import torch.nn as nn

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class SupConTwoViewLoss(torch.nn.Module):
    """
    Supervised contrastive learning (Khosla et al., 2020) with two views per sample.
    All representations in the batch with the *same class label* are treated as
    mutual positives (not only the two augmentations of the same window), while
    still excluding self-similarity. Reduces to instance-only SimCLR when every
    label in the batch is unique.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1, eps=1e-8)
        z2 = F.normalize(z2, p=2, dim=1, eps=1e-8)
        z = torch.cat([z1, z2], dim=0)
        n = int(z.size(0))
        lab = torch.cat([labels, labels], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        # Same-class pairs (excluding self on diagonal)
        pos_mask = (lab.view(1, n) == lab.view(n, 1)).to(dtype=z.dtype)
        pos_mask = pos_mask * (1.0 - torch.eye(n, device=z.device, dtype=z.dtype))

        m_inf = -float("inf")
        # Denominator: all j != i
        sim_den = sim.clone()
        sim_den = sim_den.masked_fill(torch.eye(n, device=z.device, dtype=torch.bool), m_inf)
        log_den = torch.logsumexp(sim_den, dim=1)
        # Numerator: j in same class (excludes self via pos_mask)
        sim_num = sim.clone()
        sim_num = sim_num.masked_fill(pos_mask < 0.5, m_inf)
        log_num = torch.logsumexp(sim_num, dim=1)
        per_anchor = log_den - log_num
        has_pos = pos_mask.sum(dim=1) > 0
        if not bool(has_pos.any()):
            return z1.new_tensor(0.0)
        return per_anchor[has_pos].mean()
