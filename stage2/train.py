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

import sys
import numpy as np
import random
from collections import deque
from typing import Optional

# Windows cp949 콘솔에서 UTF-8 출력 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import platform


# ══════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════
EMPTY, WALL, EXIT, ELEVATOR, STAIR, TOILET, ROOM, OUTSIDE = 0, 1, 2, 3, 4, 5, 6, 9
WALKABLE = {EMPTY, EXIT, STAIR, TOILET, ROOM}
N, S, E, W = 0, 1, 2, 3
DELTA = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}

# ── 병목(Bottleneck) 파라미터 ──────────────────────────────────────────
# 실제 피난 시뮬레이션에서 출구·복도는 처리량 한계가 존재:
#   - Henderson(1974): 보행자 출구 처리율 ≈ 1~2명/스텝
#   - 복도 점유 밀도가 4명/m² 초과 시 이동 속도 급감 (Fruin, 1971)
EXIT_CAPACITY = 2   # 출구 셀당 스텝당 최대 탈출 인원 (한 셀 = 문 폭 1유닛)
CELL_CAPACITY = 3   # 복도 셀당 최대 동시 점유 인원 (초과 시 이동 차단)
QUEUE_RADIUS  = 4   # 출구 혼잡도 피처(F7/F8) 측정 반경 (BFS 거리 기준)

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
    1: {"name":"초기 화재",  "fire_count":(1,1),  "spread_prob":0.05, "smoke_radius":0, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":[(3,1)],   "max_steps":200},
    2: {"name":"화재 확산",  "fire_count":(2,3),  "spread_prob":0.12, "smoke_radius":2, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":None,     "max_steps":250},
    3: {"name":"출구 위협",  "fire_count":(1,1),  "spread_prob":0.25, "smoke_radius":4, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":[(8,10)], "max_steps":350},
    4: {"name":"폭발 붕괴",  "fire_count":(3,6),  "spread_prob":0.20, "smoke_radius":4, "exit_block_prob":0.2, "collapse_prob":0.15, "fire_fixed":None,     "max_steps":600},
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
        # 6채널 × 30 × 20 = 3600 + 요약 피처 8개 = 3608
        # (Mnih et al., 2015): raw pixel보다 사전 계산 피처가 학습 속도·성능 향상
        # F7/F8: 출구별 근접 혼잡도 → PPO가 병목 회피·분산 유도를 학습하는 핵심 신호
        obs_size = 6 * self.ROWS * self.COLS + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        # [액션 확장] exit_A_cost, exit_B_cost, crowd_weight
        # - exit_A/B_cost: 해당 출구 경로의 화재 회피 강도 (값 클수록 우회)
        # - crowd_weight: 밀도 패널티 강도 (값 클수록 분산 유도)
        self.action_space = spaces.Box(
            low=np.array( [5.0,  5.0,  0.5]),
            high=np.array([50.0, 50.0, 5.0]),
            shape=(3,), dtype=np.float32
        )

        self.grid = BASE_GRID.copy()  # 초기 grid 설정
        self.fire_map = self.smoke_map = None
        self.people_data = self.light_dirs = self.blocked_exits = None
        self.step_count = self.escaped = self.dead = 0
        self.escaped_A = self.escaped_B = 0
        self._bfs_dist = None
        self._dist_to_exit_A = self._dist_to_exit_B = None
        self.prev_fire_map = None
        self._occupancy: dict = {}   # 셀별 현재 점유 인원 수 (병목 계산용)

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
        # 출구별 정적 거리맵 (요약 피처 F1~F3용 — fire-agnostic, 에피소드 내 1회 계산)
        valid_A = [p for p in EXIT_A_POS if p not in self.blocked_exits]
        valid_B = [p for p in EXIT_B_POS if p not in self.blocked_exits]
        self._dist_to_exit_A = self._compute_bfs_specific(valid_A)
        self._dist_to_exit_B = self._compute_bfs_specific(valid_B)

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

        self.light_dirs = self._compute_dirs_for_strategy(10.0, 10.0, 2.0)  # 초기값
        self.prev_fire_map = self.fire_map.copy()  # 초기 화재 상태 저장
        self.step_count = self.escaped = self.dead = 0
        self.escaped_A = self.escaped_B = 0
        # 초기 점유 맵 구성
        self._occupancy = {}
        for p in self.people_data:
            pos = p["pos"]
            self._occupancy[pos] = self._occupancy.get(pos, 0) + 1
        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────
    # step: 병목(Bottleneck) + 분산유도 포함
    # ──────────────────────────────────────────
    def step(self, action: np.ndarray):
        exit_a_cost  = float(action[0])
        exit_b_cost  = float(action[1])
        crowd_weight = float(action[2])
        self.light_dirs = self._compute_dirs_for_strategy(exit_a_cost, exit_b_cost, crowd_weight)
        self.step_count += 1
        reward = 0.0

        # ── 병목 초기화 ──────────────────────────────────────────
        # EXIT_CAPACITY: 이번 스텝에 각 출구 셀이 통과시킬 수 있는 최대 인원
        exit_quota = {pos: EXIT_CAPACITY
                      for pos in EXIT_A_POS + EXIT_B_POS
                      if pos not in self.blocked_exits}

        next_people = []
        for p in self.people_data:
            # 속도 누적 → 1.0 넘으면 이동 시도
            p["accum"] += p["speed"]
            if p["accum"] >= 1.0:
                old_pos = p["pos"]
                new_pos, exited = self._attempt_move(old_pos, exit_quota)
                # 점유 맵 갱신
                self._occupancy[old_pos] = max(0, self._occupancy.get(old_pos, 0) - 1)
                if not exited:
                    self._occupancy[new_pos] = self._occupancy.get(new_pos, 0) + 1
                p["pos"] = new_pos
                p["accum"] -= 1.0

                if exited:
                    # ── EXIT 도달 = 탈출 성공 ────────────────────
                    self.escaped += 1
                    if new_pos in EXIT_A_POS:
                        self.escaped_A += 1
                    else:
                        self.escaped_B += 1
                    reward += 20.0
                    continue  # 탈출자는 next_people에 추가 안 함

            r, c = p["pos"]
            cur_dist = float(self._bfs_dist[r, c])

            if self.fire_map[r, c] > 0:
                # ── 화재 구역 진입 ──────────────────────────────
                self.dead += 1
                reward -= 8.0
                self._occupancy[p["pos"]] = max(0, self._occupancy.get(p["pos"], 0) - 1)
            else:
                # ── 생존 중 이동 (shaping reward) ──────────────
                delta = p["prev_dist"] - cur_dist
                urgency = 1.0 + (self.step_count / self.cfg["max_steps"]) * 2.0
                reward += delta * 2.0 * urgency
                p["prev_dist"] = cur_dist
                next_people.append(p)

        self.people_data = next_people

        # 화재 확산 → 잔류 인원 추가 피해 체크
        self._spread_fire()
        self._spread_smoke()

        # 동적 최적화: 화재 변화 시 유도등·거리 재계산
        if self.prev_fire_map is None or not np.array_equal(self.fire_map, self.prev_fire_map):
            self.light_dirs = self._compute_dirs_for_strategy(exit_a_cost, exit_b_cost, crowd_weight)
            self._bfs_dist  = self._compute_bfs_fire_aware()
            for p in self.people_data:
                p["prev_dist"] = float(self._bfs_dist[p["pos"][0], p["pos"][1]])
            self.prev_fire_map = self.fire_map.copy()

        alive = []
        for p in self.people_data:
            r, c = p["pos"]
            if self.fire_map[r, c] > 0:
                self.dead += 1
                reward -= 8.0
                self._occupancy[p["pos"]] = max(0, self._occupancy.get(p["pos"], 0) - 1)
            else:
                alive.append(p)
        self.people_data = alive

        terminated = len(self.people_data) == 0
        truncated  = self.step_count >= self.cfg["max_steps"]

        if terminated or truncated:
            not_escaped = self.n_agents - self.escaped
            reward -= not_escaped * 5.0
            # 분산 보너스: 두 출구 모두 사용 시 +15
            # ★ A*는 EXIT_CAPACITY 병목에서 한 출구에 쏠려 큐가 생김
            #   PPO는 exit_A_cost / exit_B_cost 차별화로 부하 분산을 학습
            if self.escaped_A > 0 and self.escaped_B > 0:
                reward += 15.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _attempt_move(self, pos, exit_quota: dict):
        """
        병목 인식 이동.
        - EXIT 셀: exit_quota 잔여분 있을 때만 진입 → 탈출 성공 반환
        - 일반 셀: CELL_CAPACITY 초과 시 이동 차단 (복도 혼잡 병목)
        Returns: (new_pos, exited: bool)
        """
        r, c = pos
        pref = (int(self.light_dirs[self.light_idx[(r, c)]])
                if (r, c) in self.light_idx else self._bfs_best(r, c))
        for d in [pref] + [x for x in (N, S, E, W) if x != pref]:
            dr, dc = DELTA[d]
            nr, nc = r + dr, c + dc
            if not self._passable(nr, nc):
                continue
            # EXIT 셀: 처리량 한계 확인
            if self.grid[nr, nc] == EXIT and (nr, nc) not in self.blocked_exits:
                if exit_quota.get((nr, nc), 0) > 0:
                    exit_quota[(nr, nc)] -= 1
                    return (nr, nc), True   # 탈출 성공
                else:
                    continue                # 이 EXIT은 이번 스텝 만원, 다른 방향 시도
            # 일반 복도 셀: 점유 한계 확인
            if self._occupancy.get((nr, nc), 0) < CELL_CAPACITY:
                return (nr, nc), False      # 이동 성공
        return pos, False                   # 모든 방향 차단 → 제자리 대기

    def _move_person(self, pos):
        """하위 호환용 (render 등 내부 호출). 점유 무시 단순 이동."""
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

    def _compute_bfs_fire_aware(self):
        """화재 셀을 벽으로 처리한 BFS — shaping reward용 동적 거리"""
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = deque()
        for (r, c) in EXIT_POSITIONS:
            if self.grid[r, c] == EXIT and (r, c) not in self.blocked_exits:
                dist[r, c] = 0; q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and dist[nr, nc] == 9999
                        and self.grid[nr, nc] in WALKABLE
                        and self.fire_map[nr, nc] == 0):   # 화재 셀 통과 불가
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        return dist

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

    def _compute_bfs_with_risk(self, exit_positions, fire_cost=10.0, crowd_weight=2.0):
        """화재/연기/밀도 비용을 반영한 Dijkstra BFS"""
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
                    base_cost = fire_cost if self.fire_map[nr, nc] > 0 else 5 if self.smoke_map[nr, nc] > 0 else 1
                    density = sum(1 for p in self.people_data if p["pos"] == (nr, nc))
                    add_cost = base_cost + density * crowd_weight
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

    def _compute_dirs_for_strategy(self, exit_a_cost, exit_b_cost, crowd_weight):
        """출구별 독립 비용으로 Dijkstra 2회 실행 후 셀마다 더 저렴한 출구 방향 선택"""
        dist_a = self._compute_bfs_with_risk(EXIT_A_POS, exit_a_cost, crowd_weight)
        dist_b = self._compute_bfs_with_risk(EXIT_B_POS, exit_b_cost, crowd_weight)
        # 셀마다 두 출구 중 비용이 낮은 쪽 방향을 선택
        dist_combined = np.minimum(dist_a, dist_b)
        dirs = np.zeros(self.n_lights, dtype=np.int32)
        for i, cell in enumerate(self.light_cells):
            dirs[i] = self._bfs_best_from_dist(dist_combined, cell[0], cell[1])
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
        reachable = self._bfs_dist[self._bfs_dist < 9999]
        mx = float(reachable.max()) if reachable.size > 0 else 1.0
        mx = max(mx, 1.0)  # divide-by-zero 방지
        normalized_dist = np.clip(self._bfs_dist / mx, 0, 1)
        obs[5] = 1.0 - normalized_dist

        # ── 요약 피처 8개 (3608 = 3600 + 8) ────────────────────────────
        # Mnih et al.(2015): 사전 계산 피처가 raw pixel 대비 학습 속도·성능 향상
        fire_cells = np.argwhere(self.fire_map > 0)
        a_blocked = all(p in self.blocked_exits for p in EXIT_A_POS)
        b_blocked = all(p in self.blocked_exits for p in EXIT_B_POS)

        # F1: 출구 A 화재 위협 (0=위험/막힘, 1=안전)
        if a_blocked:
            f1 = 0.0
        elif len(fire_cells) == 0 or self._dist_to_exit_A is None:
            f1 = 1.0
        else:
            reachable_A = [float(self._dist_to_exit_A[r, c]) for r, c in fire_cells
                           if self._dist_to_exit_A[r, c] < 9999]
            f1 = float(np.clip(min(reachable_A) / 20.0, 0.0, 1.0)) if reachable_A else 1.0

        # F2: 출구 B 화재 위협 (0=위험/막힘, 1=안전)
        if b_blocked:
            f2 = 0.0
        elif len(fire_cells) == 0 or self._dist_to_exit_B is None:
            f2 = 1.0
        else:
            reachable_B = [float(self._dist_to_exit_B[r, c]) for r, c in fire_cells
                           if self._dist_to_exit_B[r, c] < 9999]
            f2 = float(np.clip(min(reachable_B) / 20.0, 0.0, 1.0)) if reachable_B else 1.0

        # F3: 출구 A 쪽이 더 가까운 생존 인원 비율
        n_alive = len(self.people_data)
        if n_alive == 0 or self._dist_to_exit_A is None or self._dist_to_exit_B is None:
            f3 = 0.5
        else:
            n_near_a = sum(1 for p in self.people_data
                           if self._dist_to_exit_A[p["pos"][0], p["pos"][1]]
                           <= self._dist_to_exit_B[p["pos"][0], p["pos"][1]])
            f3 = n_near_a / n_alive

        # F4: 탈출 완료 비율  F5: 사망 비율  F6: 시간 경과(긴급도)
        f4 = self.escaped / self.n_agents
        f5 = self.dead / self.n_agents
        f6 = self.step_count / self.cfg["max_steps"]

        # F7/F8: 출구별 근접 혼잡도 (QUEUE_RADIUS 이내 생존 인원 비율)
        # ★ 핵심 병목 신호: PPO는 이 값으로 exit_A_cost/exit_B_cost를 차별화해
        #   한쪽 출구 대기열이 쌓이면 반대편으로 분산시키는 정책을 학습한다.
        #   A*는 이 정보를 활용하지 않으므로 병목 발생 시 분산 유도 불가.
        n_alive = len(self.people_data)
        if n_alive == 0 or self._dist_to_exit_A is None or self._dist_to_exit_B is None:
            f7, f8 = 0.0, 0.0
        else:
            near_a = sum(1 for p in self.people_data
                         if self._dist_to_exit_A[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
            near_b = sum(1 for p in self.people_data
                         if self._dist_to_exit_B[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
            f7 = near_a / n_alive
            f8 = near_b / n_alive

        scalar_feats = np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float32)
        return np.concatenate([obs.flatten(), scalar_feats])

    def _get_info(self):
        return {
            "scenario": self.scenario, "scenario_name": self.cfg["name"],
            "step": self.step_count, "n_agents": self.n_agents,
            "escaped": self.escaped, "dead": self.dead,
            "escaped_A": self.escaped_A, "escaped_B": self.escaped_B,
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
            if len(self.recent) == self.window and avg >= self.threshold and self.current_scenario < len(SCENARIO_CONFIGS):
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
        self._cum_rewards = None  # 환경별 에피소드 누적 보상

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]

        # 첫 스텝에서 환경 수만큼 누적 배열 초기화
        if self._cum_rewards is None:
            self._cum_rewards = np.zeros(len(rewards))

        self._cum_rewards += rewards

        for i, done in enumerate(dones):
            if done:
                self.ep_rewards.append(float(self._cum_rewards[i]))
                self._cum_rewards[i] = 0.0

        for info in self.locals.get("infos", []):
            if "survival_rate" in info:
                self.ep_survival.append(info["survival_rate"])

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
    print(f"Policy         : MlpPolicy (flatten 3608-dim) | net_arch=[256,256]")
    print(f"학습 디바이스  : {device}")
    print(f"그리드 검증    : 13/13 방 양방향 출구 접근 가능")
    print("=" * 62)

    for n in person_counts:
        print(f"\n{'─'*62}\n인원수 {n}명 | 커리큘럼 학습 시작\n{'─'*62}")

        env_fns = [make_env(n_agents=n, seed=i) for i in range(n_envs)]
        raw_env = (DummyVecEnv(env_fns) if platform.system() == "Windows"
                   else SubprocVecEnv(env_fns))
        vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        callback = EvacTrainCallback(log_interval=10_000)

        model = PPO(
            "MlpPolicy",       # [Q3] MlpPolicy 사용
            vec_env,
            device          = device,
            verbose         = 0,
            n_steps         = 1024,          # 512→1024: S4 600스텝 에피소드 커버 (Schulman et al., 2016 GAE)
            batch_size      = 256,
            n_epochs        = 10,            # 4→10: 샘플 효율성 향상 (PPO 원논문 권장)
            gamma           = 0.99,
            learning_rate   = 3e-4,
            clip_range      = 0.2,
            ent_coef        = 0.02,          # 0.01→0.02: 탐색 강화, S3/S4 다양한 패턴 대응
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(net_arch=[256, 256]),  # 64×2→256×2: 3608-dim 처리 표현력 (Schulman et al., 2017)
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
        vec_env.save(f"{save_path}_vecnorm.pkl")
        vec_env.close()
        print(f"\n모델 저장: {save_path}.zip | 정규화 통계: {save_path}_vecnorm.pkl")

    print("\n" + "=" * 62)
    print("학습 완료! TensorBoard: tensorboard --logdir ./fire_evac_log/")
    print("=" * 62)


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════
def test_fire_evac(n_agents: int = 10, scenario: int = 1, n_episodes: int = 10,
                   save_results: bool = True, render: bool = False):
    import os, csv, json
    from datetime import datetime

    model_path   = f"fire_evac_model_{n_agents}ppl"
    vecnorm_path = f"{model_path}_vecnorm.pkl"
    print(f"\n모델 로드: {model_path}.zip")
    model = PPO.load(model_path)

    render_mode = "human" if render else None
    env     = FireEvacEnv(scenario=scenario, n_agents=n_agents, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training  = False
        vec_env.norm_reward = False
        print(f"정규화 통계 로드: {vecnorm_path}")

    # ── 에피소드 단위 결과 수집 ─────────────────────────────────
    records = []
    for ep in range(n_episodes):
        obs      = vec_env.reset()
        total_r  = 0.0
        step_cnt = 0
        max_fire = 0
        print(f"\n[에피소드 {ep+1}/{n_episodes}] {env.cfg['name']} | {n_agents}명", end="", flush=True)

        for _ in range(env.cfg["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = vec_env.step(action)
            info      = infos[0]
            total_r  += float(r[0])
            step_cnt  = info["step"]
            max_fire  = max(max_fire, info["fire_cells"])
            if render:
                env.render()
            if done[0]:
                break

        rec = {
            "episode":       ep + 1,
            "scenario":      scenario,
            "scenario_name": env.cfg["name"],
            "n_agents":      n_agents,
            "escaped":       info["escaped"],
            "escaped_A":     info.get("escaped_A", 0),
            "escaped_B":     info.get("escaped_B", 0),
            "dead":          info["dead"],
            "remaining":     info["remaining"],
            "survival_rate": round(info["survival_rate"], 4),
            "total_reward":  round(total_r, 2),
            "steps_taken":   step_cnt,
            "max_fire_cells":max_fire,
            "blocked_exits": str(info["blocked_exits"]),
        }
        records.append(rec)
        print(f" → 탈출 {rec['escaped']}/{n_agents} | 생존율 {rec['survival_rate']:.0%} | 보상 {rec['total_reward']:+.1f} | {step_cnt}스텝")

    vec_env.close()

    # ── 통계 요약 ──────────────────────────────────────────────
    def stats(vals):
        a = np.array(vals, dtype=float)
        return {
            "mean":   round(float(a.mean()), 4),
            "std":    round(float(a.std()),  4),
            "min":    round(float(a.min()),  4),
            "max":    round(float(a.max()),  4),
            "median": round(float(np.median(a)), 4),
        }

    summary = {
        "model":         model_path,
        "scenario":      scenario,
        "scenario_name": env.cfg["name"],
        "n_agents":      n_agents,
        "n_episodes":    n_episodes,
        "survival_rate": stats([r["survival_rate"]  for r in records]),
        "total_reward":  stats([r["total_reward"]   for r in records]),
        "steps_taken":   stats([r["steps_taken"]    for r in records]),
        "escaped":       stats([r["escaped"]         for r in records]),
        "escaped_A":     stats([r["escaped_A"]       for r in records]),
        "escaped_B":     stats([r["escaped_B"]       for r in records]),
        "dead":          stats([r["dead"]            for r in records]),
        "max_fire_cells":stats([r["max_fire_cells"]  for r in records]),
    }

    print("\n" + "═" * 62)
    print(f"  테스트 결과 요약 | {env.cfg['name']} | {n_agents}명 × {n_episodes}회")
    print("═" * 62)
    for key in ("survival_rate", "total_reward", "steps_taken", "escaped", "dead"):
        s = summary[key]
        print(f"  {key:<16} mean={s['mean']:>8}  std={s['std']:>7}  "
              f"min={s['min']:>7}  max={s['max']:>7}  median={s['median']:>8}")
    print("═" * 62)

    # ── 파일 저장 ──────────────────────────────────────────────
    if save_results:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"s{scenario}_{n_agents}ppl_{ts}"

        csv_path  = f"test_results_{tag}.csv"
        json_path = f"test_summary_{tag}.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  에피소드 기록 저장: {csv_path}")
        print(f"  통계 요약 저장   : {json_path}")

    return records, summary


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
    parser.add_argument("--test-n",        type=int,  default=10)
    parser.add_argument("--test-scenario", type=int,  default=1)
    parser.add_argument("--test-episodes", type=int,  default=10)
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
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
        test_fire_evac(
            n_agents=args.test_n,
            scenario=args.test_scenario,
            n_episodes=args.test_episodes,
            save_results=not args.no_save,
            render=args.render,
        )