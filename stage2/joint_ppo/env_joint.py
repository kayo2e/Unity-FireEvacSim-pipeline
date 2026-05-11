"""
env_joint.py — JointPPO용 환경 래퍼
======================================
FireEvacEnv를 감싸서:
  - action_space: MultiDiscrete([4] * n_lights) — 셀별 화살표 방향 직접 제어
  - obs: (n_lights * CELL_FEAT + GLOBAL_FEAT,) — 셀별 로컬 피처 + 글로벌 F1~F15
  - action_masks(): MaskablePPO용 — 생존자 경로 셀만 활성화

기존 env_core.py / ppo_train.py / recurrent_ppo_train.py 와 완전 분리.
env_core.py 수정 없음 — step() 내부 _compute_dirs_for_strategy를 임시 교체해
light_dirs를 직접 주입.

셀별 피처 (CELL_FEAT = 8):
  [0] fire_near    : 3×3 이웃에 화재 존재 여부
  [1] smoke_near   : 3×3 이웃에 연기 존재 여부
  [2] density      : 3×3 이웃 점유 인원 / 9
  [3] on_path      : 생존자 A* 경로상 셀 여부
  [4] dist_A       : 출구A 까지 구조적 BFS 거리 / 50
  [5] dist_B       : 출구B 까지 구조적 BFS 거리 / 50
  [6] row_norm     : 행 위치 / ROWS
  [7] col_norm     : 열 위치 / COLS
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE2)

from env_core import (FireEvacEnv, SCENARIO_CONFIGS,
                      WALKABLE, DELTA, EXIT_A_POS, EXIT_B_POS)

CELL_FEAT   = 8
GLOBAL_FEAT = 15


class JointEvacEnv(gym.Env):
    """
    화살표 방향을 셀마다 직접 제어하는 JointPPO 전용 환경.

    Action  : MultiDiscrete([4] * n_lights)  — 0=N, 1=S, 2=E, 3=W
    Obs     : (n_lights * CELL_FEAT + GLOBAL_FEAT,) float32
    Masks   : action_masks() → (n_lights * 4,) bool
              경로 셀 = [T,T,T,T], 비경로 셀 = [T,F,F,F]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: int = 1, n_agents: int = 10,
                 render_mode=None):
        super().__init__()
        self._base       = FireEvacEnv(scenario=scenario, n_agents=n_agents,
                                       render_mode=render_mode)
        self.n_lights    = self._base.n_lights
        self.light_cells = self._base.light_cells
        self.light_idx   = self._base.light_idx
        self.cfg         = self._base.cfg
        self.scenario    = scenario
        self.n_agents    = n_agents

        self.action_space = spaces.MultiDiscrete([4] * self.n_lights)
        obs_dim = self.n_lights * CELL_FEAT + GLOBAL_FEAT
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._last_global_obs = np.zeros(GLOBAL_FEAT, dtype=np.float32)
        self._cached_path     = None   # step() 내 중복 계산 방지

    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        global_obs, info = self._base.reset(seed=seed, options=options)
        self._last_global_obs = global_obs
        self._cached_path     = None
        # 초기 light_dirs: 기본 BFS 방향
        self._base.light_dirs = self._base._compute_dirs_for_strategy(
            10.0, 10.0, 1.0)
        return self._build_obs(), info

    # ──────────────────────────────────────────
    def step(self, action: np.ndarray):
        dirs = np.asarray(action, dtype=np.int32)

        # monkey-patch: env_core.step() 내 _compute_dirs_for_strategy 호출을
        # 우리가 준 dirs를 반환하도록 임시 교체 (env_core.py 수정 불필요)
        _dirs_snapshot = dirs.copy()
        def _fixed_dirs(*args, **kwargs):
            return _dirs_snapshot

        orig = self._base._compute_dirs_for_strategy
        self._base._compute_dirs_for_strategy = _fixed_dirs

        dummy = np.array([10.0, 10.0, 1.0], dtype=np.float32)
        global_obs, reward, terminated, truncated, info = self._base.step(dummy)

        self._base._compute_dirs_for_strategy = orig
        self._last_global_obs = global_obs
        self._cached_path     = self._compute_path_indices()
        obs = self._build_obs(self._cached_path)
        self._cached_path = None
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        """
        MaskablePPO용 마스크.  shape: (n_lights * 4,)
        경로 셀: 4방향 모두 허용.
        비경로 셀: N(인덱스 0)만 허용 — 더미 액션, 환경에 영향 없음.
        """
        path_set = self._compute_path_indices()
        mask = np.zeros(self.n_lights * 4, dtype=bool)
        for i in range(self.n_lights):
            if i in path_set:
                mask[i * 4: i * 4 + 4] = True
            else:
                mask[i * 4] = True   # N만 허용 (더미)
        return mask

    # ──────────────────────────────────────────
    def _compute_path_indices(self) -> set:
        """
        각 생존자에서 출구까지 gradient-descent로 추적한 경로상
        light_cell 인덱스 집합.
        fire-aware BFS 거리 (_bfs_dist) 기준으로 내리막 추적.
        """
        base = self._base
        if not base.people_data or base._bfs_dist is None:
            return set()

        path_set = set()
        for p in base.people_data:
            cur     = p["pos"]
            visited = set()
            for _ in range(100):
                if cur in visited:
                    break
                visited.add(cur)
                if cur in self.light_idx:
                    path_set.add(self.light_idx[cur])
                if cur in EXIT_A_POS or cur in EXIT_B_POS:
                    break
                cr, cc   = cur
                cur_cost = float(base._bfs_dist[cr, cc])
                best_next, best_cost = None, float('inf')
                for d, (dr, dc) in DELTA.items():
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < base.ROWS and 0 <= nc < base.COLS):
                        continue
                    if base.grid[nr, nc] not in WALKABLE:
                        continue
                    cost = float(base._bfs_dist[nr, nc])
                    if cost < best_cost:
                        best_cost, best_next = cost, (nr, nc)
                if best_next is None or best_cost >= cur_cost:
                    break
                cur = best_next
        return path_set

    # ──────────────────────────────────────────
    def _build_obs(self, path_set: set = None) -> np.ndarray:
        if path_set is None:
            path_set = self._compute_path_indices()
        base = self._base
        rows, cols = base.ROWS, base.COLS

        cell_feats = np.zeros((self.n_lights, CELL_FEAT), dtype=np.float32)
        for i, (r, c) in enumerate(self.light_cells):
            fire_n = smoke_n = density = 0.0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if base.fire_map[nr, nc] > 0:
                            fire_n = 1.0
                        if base.smoke_map[nr, nc] > 0:
                            smoke_n = 1.0
                        density += base._occupancy.get((nr, nc), 0)
            density = min(density / 9.0, 1.0)
            da = (min(float(base._dist_to_exit_A[r, c]) / 50.0, 1.0)
                  if base._dist_to_exit_A is not None else 0.5)
            db = (min(float(base._dist_to_exit_B[r, c]) / 50.0, 1.0)
                  if base._dist_to_exit_B is not None else 0.5)
            cell_feats[i] = [
                fire_n, smoke_n, density,
                1.0 if i in path_set else 0.0,
                da, db,
                r / rows, c / cols,
            ]

        return np.concatenate([cell_feats.flatten(), self._last_global_obs])

    # ──────────────────────────────────────────
    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()
