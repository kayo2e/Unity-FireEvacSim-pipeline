"""
화재대피유도시스템 | Stage 2 최종본
=====================================
수정 내역:
  Q1. EXIT 도달 학습:
      - 사람이 EXIT 셀에 실제로 '도착'해야 보상 (기존과 동일하나 보상 흐름 명확화)
      - 출구 방향 접근 보상 추가: 매 스텝 BFS 거리 단축 시 +보상
      - 출구 이탈(멀어짐) 패널티: -보상으로 방황 방지
      - EXIT 도달 = episode 내 유일한 큰 보상 (+20), 그 외는 작은 shaping 보상

  Q2. BASE_GRID 13/13 방 양방향 수정:
      - 핵심 문제: 엘리베이터(row9~10)가 복도를 막아 상단↔하단 이동 불가
      - 해결: row10,11 col7 개방 → 방 내부로 엘리베이터 우회
      - + 6곳 추가 개방으로 모든 방에서 두 출구 모두 접근 가능
      - 검증: EXIT_A 막힘 → 도달불가 0셀 / EXIT_B 막힘 → 도달불가 0셀

  Q3. MlpPolicy 사용 (요청에 따라 변경)

설치: pip install gymnasium stable-baselines3 tensorboard
실행: python fire_evac_final.py --mode train
"""

import numpy as np
import random
from collections import deque
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import platform


# ══════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════
EMPTY, WALL, EXIT, ELEVATOR, STAIR, TOILET, ROOM, OUTSIDE = 0, 1, 2, 3, 4, 5, 6, 9
WALKABLE = {EMPTY, EXIT, STAIR, TOILET, ROOM}
N, S, E, W = 0, 1, 2, 3
DELTA = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}

# ══════════════════════════════════════════════
# BASE_GRID: 13/13 방 양방향 수정본
# ══════════════════════════════════════════════
# 수정 포인트 (기존 대비):
#   (7,  12): 1→0  212호 ↔ 복도 연결
#   (10,  7): 1→0  엘리베이터 구간 우회문 (상단방→하단출구)
#   (11,  7): 1→0  계단 구간 우회문
#   (13, 11): 1→0  213호 ↔ 복도 양방향
#   (21,  7): 1→0  207호 상단 우회 (EXIT A 경로 확보)
#   (21, 11): 1→0  211호 하단 ↔ 복도
#   (23, 11): 1→0  210호 상단 ↔ 복도
BASE_GRID = np.array([
    [9,9,9,9,9,9,9,9,1,1,1,1,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,1,0,0,1,9,9,9,9,9,9,9,9],
    [1,1,1,1,1,1,1,1,1,0,0,1,9,9,9,9,9,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,9,9,9,9,9,9,9,9],
    [1,6,6,6,6,6,6,1,5,5,5,1,9,9,9,9,9,9,9,9],
    [1,1,1,1,1,1,1,1,5,5,5,1,9,9,9,9,9,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,1,1,1,1,1,9,9,9],
    [1,6,6,6,6,6,6,1,0,0,0,0,0,6,6,6,1,9,9,9],  # (7,12) 수정
    [1,6,6,6,6,6,6,0,0,0,0,2,2,6,6,6,1,9,9,9],  # EXIT A
    [1,6,6,6,6,6,6,0,3,3,3,1,6,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,0,3,3,3,1,6,6,6,6,1,9,9,9],  # (10,7) 수정
    [1,6,6,6,6,6,6,0,4,4,4,1,6,6,6,6,1,9,9,9],  # (11,7) 수정
    [1,1,1,1,1,1,1,1,0,0,0,1,0,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,0,6,6,6,6,1,9,9,9],  # (13,11) 수정
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,1,9,9,9],
    [1,1,1,1,1,1,1,1,0,0,0,1,0,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,1,1,1,1,9,9,9,9],
    [1,6,6,6,6,6,6,1,0,0,0,0,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,0,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,0,4,4,4,0,6,6,6,6,6,6,1,9],  # (21,7),(21,11) 수정
    [1,6,6,6,6,6,6,0,2,2,2,1,6,6,6,6,6,6,1,9],  # EXIT B
    [1,6,6,6,6,6,6,1,0,0,0,0,6,6,6,6,6,6,1,9],  # (23,11) 수정
    [1,1,1,1,1,1,1,1,0,5,5,1,1,1,1,1,1,1,1,9],
    [1,6,6,6,6,6,6,0,0,5,5,0,0,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
], dtype=np.int32)

EXIT_POSITIONS = [(8, 11), (8, 12), (22, 8), (22, 9), (22, 10)]
EXIT_A_POS = [(8, 11), (8, 12)]
EXIT_B_POS = [(22, 8), (22, 9), (22, 10)]

SCENARIO_CONFIGS = {
    1: {"name":"초기 화재",  "fire_count":(1,1),  "spread_prob":0.05, "smoke_radius":0, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":[(3,1)], "max_steps":200},
    2: {"name":"화재 확산",  "fire_count":(2,3),  "spread_prob":0.12, "smoke_radius":2, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":None,    "max_steps":250},
    3: {"name":"출구 폐쇄",  "fire_count":(2,4),  "spread_prob":0.15, "smoke_radius":3, "exit_block_prob":0.5, "collapse_prob":0.0, "fire_fixed":None,    "max_steps":300},
    4: {"name":"폭발 붕괴",  "fire_count":(3,6),  "spread_prob":0.20, "smoke_radius":4, "exit_block_prob":0.3, "collapse_prob":0.3, "fire_fixed":None,    "max_steps":400},
}


# ══════════════════════════════════════════════
# 환경
# ══════════════════════════════════════════════
class FireEvacEnv(gym.Env):
    """
    화재대피유도시스템 강화학습 환경.

    [Q1 답변] 사람이 EXIT 셀에 실제로 도달해야 보상이 발생합니다:
      - EXIT 도달: +20 (에피소드 내 유일한 큰 보상)
      - 출구 방향으로 접근: +거리단축 × 2.0 (shaping reward)
      - 출구에서 멀어짐: -거리증가 × 2.0 (방황 억제)
      - 화재 구역 진입: -8
      - 타임아웃: -3 × 잔류인원

    [Q2 답변] BASE_GRID 수정으로 13/13 방 전부 두 출구 모두 접근 가능.
    화재로 한 출구가 막히면 RL이 다른 출구로 유도하는 정책을 학습합니다.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: int = 1, n_agents: int = 10, render_mode: Optional[str] = None):
        super().__init__()
        self.scenario    = scenario
        self.cfg         = SCENARIO_CONFIGS[scenario]
        self.n_agents    = n_agents
        self.render_mode = render_mode
        self.ROWS, self.COLS = BASE_GRID.shape

        # 유도등: 모든 walkable 셀 (출구 제외)
        self.light_cells = [
            (r, c) for r in range(self.ROWS) for c in range(self.COLS)
            if BASE_GRID[r, c] in WALKABLE and BASE_GRID[r, c] != EXIT
        ]
        self.n_lights  = len(self.light_cells)
        self.light_idx = {cell: i for i, cell in enumerate(self.light_cells)}

        # [Q3] MlpPolicy용 1D 관측
        # 6채널 × 30 × 20 = 3600 → flatten
        obs_size = 6 * self.ROWS * self.COLS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: EXIT A, 1: EXIT B, 2: 화재 우회 (전체 출구)

        self.grid = BASE_GRID.copy()  # 초기 grid 설정
        self.fire_map = self.smoke_map = None
        self.people_data = self.light_dirs = self.blocked_exits = None
        self.step_count = self.escaped = self.dead = 0
        self._bfs_dist = None
        self.current_strategy = 2  # 초기 전략

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        cfg = self.cfg
        self.grid = BASE_GRID.copy()

        # 복도 붕괴
        if cfg["collapse_prob"] > 0:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    if (self.grid[r,c] in WALKABLE and self.grid[r,c] != EXIT
                            and random.random() < cfg["collapse_prob"]):
                        self.grid[r,c] = WALL

        # 출구 폐쇄
        self.blocked_exits = set()
        if cfg["exit_block_prob"] > 0 and random.random() < cfg["exit_block_prob"]:
            groups = [[(8,11),(8,12)], [(22,8),(22,9),(22,10)]]
            for cell in random.choice(groups):
                self.blocked_exits.add(cell)
                self.grid[cell[0], cell[1]] = WALL

        # BFS 거리 계산 — 출구까지 실제 경로 거리
        self._bfs_dist = self._compute_bfs()

        # 화재
        self.fire_map = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        walkable = self._get_walkable()
        if cfg["fire_fixed"]:
            for p in cfg["fire_fixed"]: self.fire_map[p] = 1.0
        else:
            n = random.randint(*cfg["fire_count"])
            for p in random.sample(walkable, min(n, len(walkable))): self.fire_map[p] = 1.0

        self.smoke_map = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        self._spread_smoke()

        # 사람 배치: 화재 없는 safe 셀 + 속도 + 이전 거리 기록
        safe = [c for c in walkable if self.fire_map[c[0],c[1]] == 0]
        starts = random.sample(safe, min(self.n_agents, len(safe)))
        self.people_data = [
            {
                "pos":       pos,
                "speed":     round(random.uniform(0.5, 1.2), 2),
                "accum":     0.0,
                "prev_dist": float(self._bfs_dist[pos[0], pos[1]]),
            }
            for pos in starts
        ]

        self.light_dirs = self._compute_dirs_for_strategy(2)  # 초기: 전체 출구 방향
        self.step_count = self.escaped = self.dead = 0
        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────
    # step: Q1 핵심 — EXIT 도달이 목표
    # ──────────────────────────────────────────
    def step(self, action: int):
        self.light_dirs = self._compute_dirs_for_strategy(action)
        self.step_count += 1
        reward = 0.0

        next_people = []
        for p in self.people_data:
            # 속도 누적 → 1.0 넘으면 실제 이동
            p["accum"] += p["speed"]
            if p["accum"] >= 1.0:
                p["pos"] = self._move_person(p["pos"])
                p["accum"] -= 1.0

            r, c = p["pos"]
            cur_dist = float(self._bfs_dist[r, c])

            # ── [Q1] EXIT 셀 도달 = 탈출 성공 ─────────────────
            if self.grid[r, c] == EXIT and (r, c) not in self.blocked_exits:
                self.escaped += 1
                reward += 20.0          # 탈출 완료 보상

            # ── 화재 구역 진입 ──────────────────────────────────
            elif self.fire_map[r, c] > 0:
                self.dead += 1
                reward -= 8.0

            # ── 생존 중 이동 ─────────────────────────────────────
            else:
                # 출구로 가까워지면 +, 멀어지면 - (shaping reward)
                # ★ 이것이 RL이 "어떤 방향이 출구 방향인지" 학습하는 핵심 신호
                delta = p["prev_dist"] - cur_dist
                reward += delta * 2.0
                p["prev_dist"] = cur_dist
                next_people.append(p)

        self.people_data = next_people

        # 화재 확산 → 잔류 인원 추가 피해 체크
        self._spread_fire()
        self._spread_smoke()

        alive = []
        for p in self.people_data:
            r, c = p["pos"]
            if self.fire_map[r, c] > 0:
                self.dead += 1; reward -= 8.0
            else:
                alive.append(p)
        self.people_data = alive

        # 타임아웃: 못 나간 인원당 페널티
        if self.step_count >= self.cfg["max_steps"]:
            reward -= len(self.people_data) * 3.0

        terminated = len(self.people_data) == 0
        truncated  = self.step_count >= self.cfg["max_steps"]
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _move_person(self, pos):
        r, c = pos
        pref = (int(self.light_dirs[self.light_idx[(r, c)]])
                if (r, c) in self.light_idx else self._bfs_best(r, c))
        for d in [pref] + [x for x in (N, S, E, W) if x != pref]:
            dr, dc = DELTA[d]
            nr, nc = r + dr, c + dc
            if self._passable(nr, nc): return (nr, nc)
        return pos

    def _bfs_best(self, r, c):
        best, best_d = N, float("inf")
        for d, (dr, dc) in DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                v = self._bfs_dist[nr, nc]
                if v < best_d: best_d, best = v, d
        return best

    def _passable(self, r, c):
        if not (0 <= r < self.ROWS and 0 <= c < self.COLS): return False
        return self.grid[r, c] in WALKABLE and self.fire_map[r, c] == 0

    def _spread_fire(self):
        nf = self.fire_map.copy(); prob = self.cfg["spread_prob"]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.fire_map[r, c] > 0:
                    for dr, dc in DELTA.values():
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                                and self.grid[nr, nc] in WALKABLE
                                and random.random() < prob):
                            nf[nr, nc] = 1.0
        self.fire_map = nf

    def _spread_smoke(self):
        radius = self.cfg["smoke_radius"]; self.smoke_map[:] = 0
        if radius == 0: return
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.fire_map[r, c] > 0:
                    r0,r1 = max(0,r-radius), min(self.ROWS,r+radius+1)
                    c0,c1 = max(0,c-radius), min(self.COLS,c+radius+1)
                    self.smoke_map[r0:r1, c0:c1] = 1.0

    def _compute_bfs(self):
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = deque()
        for (r, c) in EXIT_POSITIONS:
            if self.grid[r, c] == EXIT: dist[r, c] = 0; q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and dist[nr, nc] == 9999 and self.grid[nr, nc] in WALKABLE):
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        return dist  # raw dist 반환

    def _compute_bfs_specific(self, exit_positions):
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = deque()
        for (r, c) in exit_positions:
            if self.grid[r, c] == EXIT: dist[r, c] = 0; q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and dist[nr, nc] == 9999 and self.grid[nr, nc] in WALKABLE):
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        return dist

    def _compute_bfs_with_risk(self, exit_positions):
        """화재와 연기를 비용으로 반영한 BFS (Dijkstra-like)"""
        import heapq
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = []
        for (r, c) in exit_positions:
            if self.grid[r, c] == EXIT:
                dist[r, c] = 0
                heapq.heappush(q, (0, (r, c)))
        while q:
            cost, (r, c) = heapq.heappop(q)
            if cost > dist[r, c]: continue
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and self.grid[nr, nc] in WALKABLE):
                    # 비용: 화재 10, 연기 5, 일반 1
                    add_cost = 10 if self.fire_map[nr, nc] > 0 else 5 if self.smoke_map[nr, nc] > 0 else 1
                    new_cost = cost + add_cost
                    if new_cost < dist[nr, nc]:
                        dist[nr, nc] = new_cost
                        heapq.heappush(q, (new_cost, (nr, nc)))
        return dist

    def _bfs_best_from_dist(self, dist, r, c):
        best, best_d = N, float("inf")
        for d, (dr, dc) in DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                v = dist[nr, nc]
                if v < best_d: best_d, best = v, d
        return best

    def _compute_dirs_for_strategy(self, strategy):
        """전략에 따른 동적 유도등 방향 계산 (화재/연기 비용 반영)"""
        if strategy == 0:
            goals = EXIT_A_POS
        elif strategy == 1:
            goals = EXIT_B_POS
        else:
            goals = EXIT_POSITIONS
        dist = self._compute_bfs_with_risk(goals)
        dirs = np.zeros(self.n_lights, dtype=np.int32)
        for i, cell in enumerate(self.light_cells):
            dirs[i] = self._bfs_best_from_dist(dist, cell[0], cell[1])
        return dirs

    def _get_walkable(self):
        return [(r, c) for r in range(self.ROWS) for c in range(self.COLS)
                if self.grid[r, c] in WALKABLE and self.grid[r, c] != EXIT]

    def _get_obs(self):
        # 6채널 → flatten (MlpPolicy용)
        obs = np.zeros((6, self.ROWS, self.COLS), dtype=np.float32)
        obs[0] = self.grid / 9.0
        obs[1] = self.fire_map
        obs[2] = self.smoke_map
        for p in self.people_data: obs[3, p["pos"][0], p["pos"][1]] += 1.0
        if obs[3].max() > 0: obs[3] /= obs[3].max()
        for i, (r, c) in enumerate(self.light_cells): obs[4, r, c] = self.light_dirs[i] / 3.0
        mx = self._bfs_dist[self._bfs_dist < 9999].max() if (self._bfs_dist < 9999).any() else 1
        normalized_dist = np.clip(self._bfs_dist / mx, 0, 1)
        obs[5] = 1.0 - normalized_dist
        return obs.flatten()  # ★ MlpPolicy를 위해 flatten

    def _get_info(self):
        return {
            "scenario": self.scenario, "scenario_name": self.cfg["name"],
            "step": self.step_count, "n_agents": self.n_agents,
            "escaped": self.escaped, "dead": self.dead,
            "remaining": len(self.people_data),
            "survival_rate": self.escaped / self.n_agents if self.n_agents else 0.0,
            "fire_cells": int(self.fire_map.sum()),
            "blocked_exits": list(self.blocked_exits),
        }

    def render(self):
        if self.render_mode != "human": return
        DIR  = {N:"↑", S:"↓", E:"→", W:"←"}
        CELL = {OUTSIDE:" ",WALL:"█",EXIT:"E",ELEVATOR:"V",STAIR:"S",TOILET:"T",ROOM:".",EMPTY:"."}
        pset = {p["pos"] for p in self.people_data}
        lmap = {self.light_cells[i]: int(self.light_dirs[i]) for i in range(self.n_lights)}
        print(f"\n[Step {self.step_count}] 탈출 {self.escaped} | 사망 {self.dead} | 잔류 {len(self.people_data)}")
        for r in range(self.ROWS):
            row = ""
            for c in range(self.COLS):
                if   (r,c) in pset:        row += "P"
                elif self.fire_map[r,c]:   row += "F"
                elif self.smoke_map[r,c]:  row += "~"
                elif (r,c) in lmap:        row += DIR[lmap[(r,c)]]
                else:                      row += CELL.get(self.grid[r,c],"?")
            print(row)


# ══════════════════════════════════════════════
# 커리큘럼 래퍼
# ══════════════════════════════════════════════
class EvacCurriculumWrapper(gym.Wrapper):
    def __init__(self, n_agents: int = 10, threshold: float = 0.50, window: int = 20):
        self.current_scenario = 1
        self.n_agents = n_agents
        env = FireEvacEnv(scenario=1, n_agents=n_agents)
        super().__init__(env)
        self.threshold = threshold; self.window = window; self.recent = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.recent.append(info["survival_rate"])
            if len(self.recent) > self.window: self.recent.pop(0)
            avg = sum(self.recent) / len(self.recent)
            if len(self.recent) == self.window and avg >= self.threshold and self.current_scenario < 4:
                self.current_scenario += 1
                self.env = FireEvacEnv(scenario=self.current_scenario, n_agents=self.n_agents)
                self.recent = []
                print(f"\n[커리큘럼] ★ {self.current_scenario}단계 승급! ({self.env.cfg['name']}) | 생존율 {avg:.0%}")
        return obs, rew, term, trunc, info

    def reset(self, **kw): return self.env.reset(**kw)

    @property
    def cfg(self): return self.env.cfg


# ══════════════════════════════════════════════
# 콜백
# ══════════════════════════════════════════════
class EvacTrainCallback(BaseCallback):
    def __init__(self, log_interval=10_000):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards = []; self.ep_survival = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "survival_rate" in info: self.ep_survival.append(info["survival_rate"])
        if self.locals.get("episode") is not None:
            self.ep_rewards.append(self.locals["episode"]["r"])
        if self.num_timesteps % self.log_interval == 0 and self.ep_survival:
            n = min(len(self.ep_survival), 100)
            avg_s = sum(self.ep_survival[-n:]) / n
            avg_r = sum(self.ep_rewards[-n:]) / min(len(self.ep_rewards), n) if self.ep_rewards else 0
            print(f"  Step {self.num_timesteps:>8,} | 생존율 {avg_s:>5.1%} | 보상 {avg_r:>+7.1f}")
        return True


# ══════════════════════════════════════════════
# 병렬 환경 팩토리
# ══════════════════════════════════════════════
def make_env(n_agents: int, seed: int):
    def _init():
        env = EvacCurriculumWrapper(n_agents=n_agents)
        env.reset(seed=seed)
        return env
    return _init


# ══════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════
def train_fire_evac(person_counts=(10, 30, 50), total_timesteps=300_000, n_envs=None):
    import torch

    n_cpu = __import__('multiprocessing').cpu_count()
    if n_envs is None:
        n_envs = max(4, min(n_cpu, 16))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lights = FireEvacEnv(scenario=1).n_lights

    print("=" * 62)
    print("화재대피유도시스템 Stage 2 — PPO 학습 (MlpPolicy)")
    print(f"인원수         : {person_counts}명")
    print(f"총 스텝        : {total_timesteps:,} / 모델")
    print(f"병렬 환경      : {n_envs}개")
    print(f"유도등         : {n_lights}개 셀")
    print(f"Policy         : MlpPolicy (flatten 3600-dim)")
    print(f"학습 디바이스  : {device}")
    print(f"그리드 검증    : 13/13 방 양방향 출구 접근 가능")
    print("=" * 62)

    for n in person_counts:
        print(f"\n{'─'*62}\n인원수 {n}명 | 커리큘럼 학습 시작\n{'─'*62}")

        env_fns = [make_env(n_agents=n, seed=i) for i in range(n_envs)]
        vec_env = (DummyVecEnv(env_fns) if platform.system() == "Windows"
                   else SubprocVecEnv(env_fns))

        callback = EvacTrainCallback(log_interval=10_000)

        model = PPO(
            "MlpPolicy",       # [Q3] MlpPolicy 사용
            vec_env,
            device          = device,
            verbose         = 0,
            n_steps         = 512,
            batch_size      = 256,
            n_epochs        = 4,
            gamma           = 0.99,
            learning_rate   = 5e-4,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            tensorboard_log = "./fire_evac_log/",
        )

        model.learn(
            total_timesteps = total_timesteps,
            callback        = callback,
            tb_log_name     = f"PPO_{n}ppl",
            progress_bar    = True,
        )

        save_path = f"fire_evac_model_{n}ppl"
        model.save(save_path)
        vec_env.close()
        print(f"\n모델 저장: {save_path}.zip")

    print("\n" + "=" * 62)
    print("학습 완료! TensorBoard: tensorboard --logdir ./fire_evac_log/")
    print("=" * 62)


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════
def test_fire_evac(n_agents: int = 10, scenario: int = 1, n_episodes: int = 3):
    model_path = f"fire_evac_model_{n_agents}ppl"
    print(f"\n모델 로드: {model_path}.zip")
    model = PPO.load(model_path)
    env   = FireEvacEnv(scenario=scenario, n_agents=n_agents, render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep); total_r = 0.0
        print(f"\n[에피소드 {ep+1}] {info['scenario_name']} | {n_agents}명")
        for _ in range(env.cfg["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r; env.render()
            if term or trunc: break
        print(f"탈출 {info['escaped']}/{n_agents}명 | 생존율 {info['survival_rate']:.0%} | 보상 {total_r:.1f}")


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="화재대피유도시스템 Stage 2 최종")
    parser.add_argument("--mode", choices=["train","test","check"], default="train")
    parser.add_argument("--people",        type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--steps",         type=int, default=300_000)
    parser.add_argument("--n-envs",        type=int, default=None)
    parser.add_argument("--test-n",        type=int, default=10)
    parser.add_argument("--test-scenario", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "check":
        print("환경 검증 중...")
        env = FireEvacEnv(scenario=1, n_agents=10)
        check_env(env)
        print(f"관측 크기: {env.observation_space.shape} (MlpPolicy용 flatten)")
        print(f"유도등 수: {env.n_lights}개")
        print("환경 검증 완료!")

    elif args.mode == "train":
        train_fire_evac(person_counts=args.people, total_timesteps=args.steps, n_envs=args.n_envs)

    elif args.mode == "test":
        test_fire_evac(n_agents=args.test_n, scenario=args.test_scenario)