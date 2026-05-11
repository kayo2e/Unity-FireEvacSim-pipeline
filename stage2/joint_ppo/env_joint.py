"""
env_joint.py — JointPPO용 환경 래퍼 (K_MAX 고정 슬롯 방식)
=============================================================
핵심 아이디어:
  - 전체 n_lights(611) 중 경로 셀만 제어 (LLM next-token 영감)
  - action_space = MultiDiscrete([4] * K_MAX)  ← K_MAX=64 고정
  - 슬롯 i < 실제경로셀 수: 활성 (4방향), 슬롯 i ≥: 비활성 (N 더미)
  - 비경로 셀: A* 기본 방향 유지 (env_core._compute_dirs_for_strategy 호출)

기존 env_core.py 수정 없음 — step() 내 monkey-patch로 light_dirs 주입.

슬롯 정렬 기준: 생존자에게 가장 가까운 경로 셀부터 (BFS 거리 오름차순).

관측 (K_MAX * CELL_FEAT + GLOBAL_FEAT = 64*8 + 15 = 527):
  각 슬롯 i: [fire_near, smoke_near, density, active, dist_A, dist_B, row, col]
  글로벌   : F1~F15
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
K_MAX       = 64    # 제어할 최대 경로 셀 수. 줄이면 빠르고, 늘리면 표현력 증가.


class JointEvacEnv(gym.Env):
    """
    경로 셀 K_MAX개를 직접 제어하는 JointPPO 전용 환경.

    Action  : MultiDiscrete([4] * K_MAX)  — 0=N,1=S,2=E,3=W
    Obs     : (K_MAX * CELL_FEAT + GLOBAL_FEAT,) float32
    Masks   : 활성 슬롯=[T,T,T,T], 비활성=[T,F,F,F]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: int = 1, n_agents: int = 10,
                 render_mode=None, k_max: int = K_MAX):
        super().__init__()
        self._base       = FireEvacEnv(scenario=scenario, n_agents=n_agents,
                                       render_mode=render_mode)
        self.n_lights    = self._base.n_lights
        self.light_cells = self._base.light_cells
        self.light_idx   = self._base.light_idx
        self.cfg         = self._base.cfg
        self.scenario    = scenario
        self.n_agents    = n_agents
        self.k_max       = k_max

        self.action_space = spaces.MultiDiscrete([4] * k_max)
        obs_dim = k_max * CELL_FEAT + GLOBAL_FEAT
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._last_global_obs = np.zeros(GLOBAL_FEAT, dtype=np.float32)
        self._last_path_slots: list = []   # [(cell_idx, (r,c)), ...] 슬롯 매핑

    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        global_obs, info = self._base.reset(seed=seed, options=options)
        self._last_global_obs = global_obs
        # 초기 light_dirs: A* 기본 방향
        self._base.light_dirs = self._base._compute_dirs_for_strategy(10.0, 10.0, 1.0)
        self._last_path_slots = self._compute_path_slots()
        return self._build_obs(self._last_path_slots), info

    # ──────────────────────────────────────────
    def step(self, action: np.ndarray):
        slots = self._last_path_slots
        n_active = len(slots)

        # 활성 슬롯 행동 → light_dirs 패치
        base_dirs = self._base._compute_dirs_for_strategy(10.0, 10.0, 1.0)
        for i, (cell_idx, _) in enumerate(slots):
            if i < len(action):
                base_dirs[cell_idx] = int(action[i])

        _dirs_snap = base_dirs.copy()
        def _fixed(*a, **kw): return _dirs_snap

        orig = self._base._compute_dirs_for_strategy
        self._base._compute_dirs_for_strategy = _fixed

        dummy = np.array([10.0, 10.0, 1.0], dtype=np.float32)
        global_obs, reward, terminated, truncated, info = self._base.step(dummy)

        self._base._compute_dirs_for_strategy = orig
        self._last_global_obs = global_obs
        self._last_path_slots = self._compute_path_slots()
        return self._build_obs(self._last_path_slots), reward, terminated, truncated, info

    # ──────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        """
        활성 슬롯 (실제 경로 셀): 4방향 모두 True.
        비활성 슬롯 (패딩): N(0)만 True.
        """
        n_active = len(self._last_path_slots)
        mask = np.zeros(self.k_max * 4, dtype=bool)
        for i in range(self.k_max):
            if i < n_active:
                mask[i * 4: i * 4 + 4] = True
            else:
                mask[i * 4] = True
        return mask

    # ──────────────────────────────────────────
    def _compute_path_slots(self) -> list:
        """
        생존자 경로상 light_cell 목록.
        각 생존자에서 BFS gradient descent로 경로 추적.
        결과: [(cell_idx, (r,c)), ...] BFS 거리 오름차순, 중복 제거, 최대 k_max개.
        """
        base = self._base
        if not base.people_data or base._bfs_dist is None:
            return []

        cell_dist: dict = {}   # cell_idx → min BFS dist from any survivor
        for p in base.people_data:
            cur = p["pos"]
            visited = set()
            for _ in range(100):
                if cur in visited:
                    break
                visited.add(cur)
                if cur in self.light_idx:
                    idx = self.light_idx[cur]
                    d   = float(base._bfs_dist[cur[0], cur[1]])
                    if idx not in cell_dist or d < cell_dist[idx]:
                        cell_dist[idx] = d
                if cur in EXIT_A_POS or cur in EXIT_B_POS:
                    break
                cr, cc   = cur
                cur_cost = float(base._bfs_dist[cr, cc])
                best_next, best_cost = None, float('inf')
                for _, (dr, dc) in DELTA.items():
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

        # BFS 거리 오름차순 정렬 후 k_max개 선택
        sorted_cells = sorted(cell_dist.items(), key=lambda x: x[1])
        return [(idx, self.light_cells[idx])
                for idx, _ in sorted_cells[:self.k_max]]

    # ──────────────────────────────────────────
    def _build_obs(self, slots: list) -> np.ndarray:
        base = self._base
        rows, cols = base.ROWS, base.COLS
        n_active   = len(slots)

        cell_feats = np.zeros((self.k_max, CELL_FEAT), dtype=np.float32)
        for i in range(self.k_max):
            if i >= n_active:
                break
            _, (r, c) = slots[i]
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
            cell_feats[i] = [fire_n, smoke_n, density,
                             1.0,     # active flag
                             da, db,
                             r / rows, c / cols]

        return np.concatenate([cell_feats.flatten(), self._last_global_obs])

    # ──────────────────────────────────────────
    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()
