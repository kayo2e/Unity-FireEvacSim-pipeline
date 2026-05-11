"""
policy_joint.py — JointPPO 정책 (TransformerEncoder, K_MAX 토큰)
==================================================================
obs (B, K_MAX*8 + 15)  ← env_joint.py의 경로 셀 K_MAX개 + 글로벌
  ↓
global token (CLS) + K_MAX cell tokens → TransformerEncoder
  d_model=64, nhead=4, layers=2, seq=K_MAX+1=65
  → 1회 추론 ~3ms/CPU (전체 611 토큰 대비 100× 빠름)
  ↓
latent_pi = cells[:, 1:, :].flatten (B, K_MAX*d)
latent_vf = cells[:, 0, :]          (B, d)        CLS 토큰
  ↓
PerCellHead: Linear(d, 4) 공유 → (B, K_MAX*4)
MultiCategorical([4]*K_MAX): K_MAX=64개 분포만 처리 (611→64 10×↓)
"""

import sys
import os

import torch
import torch.nn as nn

_STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE2)

from joint_ppo.env_joint import CELL_FEAT, GLOBAL_FEAT, K_MAX

try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
except ImportError:
    raise SystemExit("sb3_contrib 미설치. pip install sb3-contrib")


# ══════════════════════════════════════════════
# Transformer 특징 추출기
# ══════════════════════════════════════════════
class PathTransformerExtractor(nn.Module):
    """
    SB3 MlpExtractor 호환:
      forward(obs) → (latent_pi, latent_vf)
      latent_dim_pi = d_model * k_max
      latent_dim_vf = d_model
    """

    def __init__(self, k_max: int = K_MAX,
                 cell_feat: int = CELL_FEAT,
                 global_feat: int = GLOBAL_FEAT,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2):
        super().__init__()
        self.k_max       = k_max
        self.cell_feat   = cell_feat
        self.global_feat = global_feat
        self.d_model     = d_model

        self.latent_dim_pi = d_model * k_max
        self.latent_dim_vf = d_model

        self.cell_proj   = nn.Linear(cell_feat, d_model)
        self.global_proj = nn.Linear(global_feat, d_model)
        # k_max개 셀 + 1개 CLS = k_max+1 위치
        self.pos_embed   = nn.Embedding(k_max + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, obs: torch.Tensor):
        B  = obs.shape[0]
        k  = self.k_max
        cf = self.cell_feat

        cell_obs = obs[:, :k * cf].view(B, k, cf)           # (B, K, 8)
        glob_obs = obs[:, k * cf:]                           # (B, 15)

        c_tok = self.cell_proj(cell_obs)                     # (B, K, d)
        g_tok = self.global_proj(glob_obs).unsqueeze(1)      # (B, 1, d)
        tokens = torch.cat([g_tok, c_tok], dim=1)            # (B, K+1, d)

        pos_ids = torch.arange(k + 1, device=obs.device)
        tokens  = tokens + self.pos_embed(pos_ids)

        out       = self.transformer(tokens)                 # (B, K+1, d)
        latent_vf = out[:, 0, :]                             # CLS
        latent_pi = out[:, 1:, :].reshape(B, -1)            # (B, K*d)
        return latent_pi, latent_vf

    def forward_actor(self, obs: torch.Tensor):
        return self.forward(obs)[0]

    def forward_critic(self, obs: torch.Tensor):
        return self.forward(obs)[1]


# ══════════════════════════════════════════════
# 셀별 분류 헤드
# ══════════════════════════════════════════════
class PerCellHead(nn.Module):
    """(B, K*d) → (B, K*4). 가중치 슬롯 간 공유."""

    def __init__(self, d_model: int, n_slots: int, n_classes: int = 4):
        super().__init__()
        self.d_model   = d_model
        self.n_slots   = n_slots
        self.n_classes = n_classes
        self.head      = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.head(
            x.view(B, self.n_slots, self.d_model)
        ).view(B, self.n_slots * self.n_classes)


# ══════════════════════════════════════════════
# MaskablePPO 정책
# ══════════════════════════════════════════════
class PathTransformerPolicy(MaskableActorCriticPolicy):
    """
    TransformerEncoder 기반 JointPPO 정책.
    seq = K_MAX+1 = 65 토큰 → CPU에서도 ~3ms/forward.

    policy_kwargs 예시:
        dict(k_max=64, d_model=64, nhead=4, num_layers=2)
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 k_max: int = K_MAX,
                 cell_feat: int = CELL_FEAT,
                 global_feat: int = GLOBAL_FEAT,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 **kwargs):
        self._k_max       = k_max
        self._cell_feat   = cell_feat
        self._global_feat = global_feat
        self._d_model     = d_model
        self._nhead       = nhead
        self._num_layers  = num_layers

        kwargs.setdefault('net_arch', [])
        kwargs.setdefault('ortho_init', True)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = PathTransformerExtractor(
            k_max      = self._k_max,
            cell_feat  = self._cell_feat,
            global_feat= self._global_feat,
            d_model    = self._d_model,
            nhead      = self._nhead,
            num_layers = self._num_layers,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = PerCellHead(self._d_model, self._k_max, 4)
