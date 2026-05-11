"""
policy_joint.py — TransformerEncoder 기반 MaskablePPO 정책
===========================================================
아키텍처:
  obs (B, n_lights*8 + 15)
    ↓ PathTransformerExtractor
    ┌─ cell_feats (B, n_lights, 8) → cell_proj → (B, n_lights, d_model)
    └─ global    (B, 15)           → global_proj → (B, 1, d_model)
    → concat + pos_embed → TransformerEncoder → (B, n_lights+1, d_model)
    → latent_vf = out[:,0,:]          (B, d_model)   CLS 토큰 → 가치함수
    → latent_pi = out[:,1:,:].flatten (B, n_lights*d_model) → per-cell 분류헤드

  PerCellHead (action_net):
    (B, n_lights*d_model) → reshape (B, n_lights, d_model)
                          → Linear(d_model, 4)
                          → (B, n_lights*4)   ← MultiCategorical 입력

  value_net (SB3 기본):
    (B, d_model) → Linear(d_model, 1)
"""

import sys
import os

import torch
import torch.nn as nn
import numpy as np

_STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE2)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from joint_ppo.env_joint import CELL_FEAT, GLOBAL_FEAT

try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
except ImportError:
    raise SystemExit("sb3_contrib 미설치. pip install sb3-contrib")


# ══════════════════════════════════════════════
# Transformer 특징 추출기
# ══════════════════════════════════════════════
class PathTransformerExtractor(nn.Module):
    """
    SB3 MlpExtractor 호환 인터페이스:
      forward(obs) → (latent_pi, latent_vf)
      latent_dim_pi, latent_dim_vf 속성 필요
    """

    def __init__(self, n_lights: int, cell_feat: int = CELL_FEAT,
                 global_feat: int = GLOBAL_FEAT,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.n_lights    = n_lights
        self.cell_feat   = cell_feat
        self.global_feat = global_feat
        self.d_model     = d_model

        self.latent_dim_pi = d_model * n_lights
        self.latent_dim_vf = d_model

        self.cell_proj   = nn.Linear(cell_feat, d_model)
        self.global_proj = nn.Linear(global_feat, d_model)
        # +1 : global(CLS) 토큰 위치
        self.pos_embed   = nn.Embedding(n_lights + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
            norm_first=True,   # Pre-LN: 학습 안정성 향상
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 num_layers=num_layers)

    def _encode(self, obs: torch.Tensor):
        B  = obs.shape[0]
        n  = self.n_lights
        cf = self.cell_feat
        cell_obs  = obs[:, :n * cf].view(B, n, cf)
        glob_obs  = obs[:, n * cf:]

        c_tok = self.cell_proj(cell_obs)                  # (B, n, d)
        g_tok = self.global_proj(glob_obs).unsqueeze(1)   # (B, 1, d)
        tokens = torch.cat([g_tok, c_tok], dim=1)         # (B, n+1, d)

        pos_ids = torch.arange(n + 1, device=obs.device)
        tokens  = tokens + self.pos_embed(pos_ids)        # pos embedding 추가
        return self.transformer(tokens)                   # (B, n+1, d)

    def forward(self, obs: torch.Tensor):
        out       = self._encode(obs)
        latent_vf = out[:, 0, :]                          # CLS 토큰
        latent_pi = out[:, 1:, :].reshape(out.shape[0], -1)
        return latent_pi, latent_vf

    def forward_actor(self, obs: torch.Tensor):
        return self.forward(obs)[0]

    def forward_critic(self, obs: torch.Tensor):
        return self.forward(obs)[1]


# ══════════════════════════════════════════════
# 셀별 분류 헤드
# ══════════════════════════════════════════════
class PerCellHead(nn.Module):
    """
    (B, n_cells * d_model) → (B, n_cells * n_classes)
    가중치는 모든 셀에서 공유 (파라미터 수 = d_model * n_classes).
    """

    def __init__(self, d_model: int, n_cells: int, n_classes: int = 4):
        super().__init__()
        self.d_model   = d_model
        self.n_cells   = n_cells
        self.n_classes = n_classes
        self.head      = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        logits = self.head(x.view(B, self.n_cells, self.d_model))
        return logits.view(B, self.n_cells * self.n_classes)


# ══════════════════════════════════════════════
# MaskablePPO 정책
# ══════════════════════════════════════════════
class PathTransformerPolicy(MaskableActorCriticPolicy):
    """
    TransformerEncoder 기반 MaskablePPO 정책.

    policy_kwargs 예시:
        dict(n_lights=307, d_model=64, nhead=4, num_layers=2)
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 n_lights: int = 307,
                 cell_feat: int = CELL_FEAT,
                 global_feat: int = GLOBAL_FEAT,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 **kwargs):
        # _build() 전에 저장해야 _build_mlp_extractor()에서 사용 가능
        self._n_lights    = n_lights
        self._cell_feat   = cell_feat
        self._global_feat = global_feat
        self._d_model     = d_model
        self._nhead       = nhead
        self._num_layers  = num_layers

        kwargs.setdefault('net_arch', [])       # 기본 MLP 비활성화
        kwargs.setdefault('ortho_init', True)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = PathTransformerExtractor(
            n_lights   = self._n_lights,
            cell_feat  = self._cell_feat,
            global_feat= self._global_feat,
            d_model    = self._d_model,
            nhead      = self._nhead,
            num_layers = self._num_layers,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # SB3가 생성한 Linear(latent_dim_pi, sum_action_dims) 를
        # 파라미터 효율적인 PerCellHead로 교체
        self.action_net = PerCellHead(self._d_model, self._n_lights, 4)
        # value_net은 SB3가 생성한 Linear(d_model, 1) 그대로 사용
