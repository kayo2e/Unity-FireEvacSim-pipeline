"""
env_autoregressive.py — AutoregressivePPO용 환경 (next-token 방식)
====================================================================
핵심 아이디어:
  - 시뮬레이션 1 tick = 최대 K_MAX개의 Discrete(2) gym step으로 분해
  - RecurrentPPO의 LSTM이 셀 간 일관성과 틱 간 맥락을 자연스럽게 학습
  - JointPPO의 MultiDiscrete 비율 폭발 문제 없음
      Discrete(2) 단일 액션 → ratio = exp(Δlogit) → clip이 정상 작동

Action  : Discrete(2)   — 0=Exit A 방향, 1=Exit B 방향
Obs     : (OBS_DIM,) = (CELL_FEAT + GLOBAL_FEAT + 2,) = 25 float32
           셀 8 피처 + 글로벌 F1~F15 + [슬롯위치, 활성슬롯비율]
Reward  : 0.0 (중간 서브스텝) | 시뮬 보상 (틱 마지막 서브스텝)
엔트로피 : ln(2) ≈ 0.69  (JointPPO: 64×ln(2) ≈ 44.4)

기존 env_core.py / env_joint.py 수정 없음.
"""

import heapq
import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

_STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE2)

from env_core import (FireEvacEnv, SCENARIO_CONFIGS,
                      WALKABLE, DELTA, EXIT_A_POS, EXIT_B_POS)
from joint_ppo.env_joint import CELL_FEAT, GLOBAL_FEAT, K_MAX

# 슬롯 위치(normalized) + 활성슬롯비율 2개 추가
OBS_DIM = CELL_FEAT + GLOBAL_FEAT + 2   # 8 + 15 + 2 = 25


class AutoregressiveEvacEnv(gym.Env):
    """
    경로 셀을 순차적으로 결정하는 next-token 방식 환경.

    시뮬레이션 1 tick이 K_MAX개의 gym step으로 분해된다.
    LSTM이 이전 결정을 기억하며 출구 방향 일관성을 학습.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: int = 1, n_agents: int = None,
                 render_mode=None, k_max: int = K_MAX):
        super().__init__()
        if n_agents is None:
            n_agents = SCENARIO_CONFIGS[scenario]["n_agents"]
        self._base       = FireEvacEnv(scenario=scenario, n_agents=n_agents,
                                       render_mode=render_mode)
        self.cfg         = self._base.cfg
        self.scenario    = scenario
        self.n_agents    = n_agents
        self.k_max       = k_max
        self.n_lights    = self._base.n_lights
        self.light_cells = self._base.light_cells
        self.light_idx   = self._base.light_idx

        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        # 서브스텝 상태
        self._slot_idx:           int       = 0
        self._path_slots:         list      = []
        self._accumulated_actions: list     = []
        self._global_obs:         np.ndarray = np.zeros(GLOBAL_FEAT, dtype=np.float32)
        self._last_info:          dict      = {}

    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        global_obs, info = self._base.reset(seed=seed, options=options)
        self._global_obs          = global_obs
        self._last_info           = info
        self._base.light_dirs     = self._base._compute_dirs_for_strategy(10.0, 10.0, 1.0)
        self._path_slots          = self._compute_path_slots()
        self._slot_idx            = 0
        self._accumulated_actions = [0] * self.k_max
        return self._build_obs(0), info

    # ──────────────────────────────────────────
    def step(self, action: int):
        i = self._slot_idx
        if i < self.k_max:
            self._accumulated_actions[i] = int(action)
        self._slot_idx += 1

        n_active  = len(self._path_slots)
        # 활성 슬롯을 모두 처리했거나 k_max 한도에 도달하면 틱 완료
        tick_done = (self._slot_idx >= min(n_active, self.k_max)
                     or self._slot_idx >= self.k_max)

        if not tick_done:
            # 중간 서브스텝: 보상 없이 다음 셀 관측만 반환
            return self._build_obs(self._slot_idx), 0.0, False, False, {}

        # ── 틱 완료: 누적 액션으로 시뮬레이션 1스텝 실행 ──
        global_obs, reward, terminated, truncated, info = self._advance_sim()
        self._global_obs          = global_obs
        self._last_info           = info
        self._path_slots          = self._compute_path_slots()
        self._slot_idx            = 0
        self._accumulated_actions = [0] * self.k_max

        obs = self._build_obs(0)
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────
    def _advance_sim(self):
        """누적 액션을 light_dirs에 주입하고 시뮬레이션 1스텝 실행."""
        base      = self._base
        base_dirs = base._compute_dirs_for_strategy(10.0, 10.0, 1.0)

        for i, (cell_idx, (r, c)) in enumerate(self._path_slots):
            if i >= self.k_max:
                break
            act              = self._accumulated_actions[i]
            dist_a, dist_b   = base._dist_to_exit_A, base._dist_to_exit_B
            primary, fallback = (dist_a, dist_b) if act == 0 else (dist_b, dist_a)

            if primary is not None and primary[r, c] < 9999:
                base_dirs[cell_idx] = self._dir_toward(r, c, primary)
            elif fallback is not None and fallback[r, c] < 9999:
                base_dirs[cell_idx] = self._dir_toward(r, c, fallback)

        _snap = base_dirs.copy()
        orig  = base._compute_dirs_for_strategy
        base._compute_dirs_for_strategy = lambda *a, **kw: _snap

        dummy  = np.array([10.0, 10.0, 1.0], dtype=np.float32)
        result = base.step(dummy)

        base._compute_dirs_for_strategy = orig
        return result

    # ──────────────────────────────────────────
    def _dir_toward(self, r: int, c: int, dist_map: np.ndarray) -> int:
        base     = self._base
        best_d   = float(dist_map[r, c])
        best_dir = 0
        for dir_idx, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, 1), (0, -1)]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < base.ROWS and 0 <= nc < base.COLS:
                d = float(dist_map[nr, nc])
                if d < best_d:
                    best_d, best_dir = d, dir_idx
        return best_dir

    # ──────────────────────────────────────────
    def _compute_path_slots(self) -> list:
        """생존자 경로상 light_cell 목록 — forward A* (화재 회피) 경로 기반."""
        base = self._base
        if not base.people_data or base._bfs_dist is None:
            return []

        goals = {pos for pos in EXIT_A_POS + EXIT_B_POS
                 if base.grid[pos[0], pos[1]] in WALKABLE
                 and pos not in base.blocked_exits}

        def _h(r: int, c: int) -> float:
            return min(abs(r - gr) + abs(c - gc) for gr, gc in goals) if goals else 0.0

        cell_dist: dict = {}

        for p in base.people_data:
            sr, sc    = p["pos"]
            start     = (sr, sc)
            visited:  set  = set()
            parent:   dict = {}
            g_map          = {start: 0}
            heap           = [(_h(sr, sc), 0, sr, sc)]
            goal_cell      = None

            while heap:
                f, g, r, c = heapq.heappop(heap)
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                if (r, c) in goals:
                    goal_cell = (r, c)
                    break
                for _, (dr, dc) in DELTA.items():
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < base.ROWS and 0 <= nc < base.COLS):
                        continue
                    if base.grid[nr, nc] not in WALKABLE:
                        continue
                    if base.fire_map[nr, nc] > 0:
                        continue
                    if (nr, nc) in visited:
                        continue
                    new_g = g + 1
                    if new_g < g_map.get((nr, nc), float('inf')):
                        g_map[(nr, nc)] = new_g
                        parent[(nr, nc)] = (r, c)
                        heapq.heappush(heap, (new_g + _h(nr, nc), new_g, nr, nc))

            if goal_cell is None:
                continue

            # 경로 역추적 → light cell 수집
            cur = goal_cell
            while cur != start:
                if cur in self.light_idx:
                    idx = self.light_idx[cur]
                    d   = float(base._bfs_dist[cur[0], cur[1]])
                    if idx not in cell_dist or d < cell_dist[idx]:
                        cell_dist[idx] = d
                cur = parent[cur]
            if start in self.light_idx:
                idx = self.light_idx[start]
                d   = float(base._bfs_dist[sr, sc])
                if idx not in cell_dist or d < cell_dist[idx]:
                    cell_dist[idx] = d

        sorted_cells = sorted(cell_dist.items(), key=lambda x: x[1])
        return [(idx, self.light_cells[idx]) for idx, _ in sorted_cells[:self.k_max]]

    # ──────────────────────────────────────────
    def _build_obs(self, slot_idx: int) -> np.ndarray:
        base       = self._base
        rows, cols = base.ROWS, base.COLS
        slots      = self._path_slots
        n_active   = len(slots)

        if slot_idx < n_active:
            _, (r, c) = slots[slot_idx]
            fire_n = smoke_n = density = 0.0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if base.fire_map[nr, nc] > 0:
                            fire_n  = 1.0
                        if base.smoke_map[nr, nc] > 0:
                            smoke_n = 1.0
                        density += base._occupancy.get((nr, nc), 0)
            density = min(density / 9.0, 1.0)
            da = (min(float(base._dist_to_exit_A[r, c]) / 50.0, 1.0)
                  if base._dist_to_exit_A is not None else 0.5)
            db = (min(float(base._dist_to_exit_B[r, c]) / 50.0, 1.0)
                  if base._dist_to_exit_B is not None else 0.5)
            cell_feat = np.array(
                [fire_n, smoke_n, density, 1.0, da, db, r / rows, c / cols],
                dtype=np.float32)
        else:
            cell_feat = np.zeros(CELL_FEAT, dtype=np.float32)

        # 슬롯 위치 정보: LSTM이 시퀀스 내 위치를 인식할 수 있도록
        pos_feat = np.array([
            slot_idx / self.k_max,
            n_active  / self.k_max,
        ], dtype=np.float32)

        return np.concatenate([cell_feat, self._global_obs, pos_feat])

    # ──────────────────────────────────────────
    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()
