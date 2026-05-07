"""
env_core.py — 화재대피유도시스템 공통 환경
=============================================
FireEvacEnv, 상수, BASE_GRID를 단일 모듈로 관리.
train.py (PPO), astar.py (A*), unity_bridge.py 모두 이 파일을 import.

군중 물리 추가:
  - 밀도 기반 속도 감소 (Fruin, 1971): 주변 밀도 높을수록 이동 속도 감소
  - 공황 레벨     (Helbing, 2000): 화재 근접 + 시간 경과 → 공황 → 비합리적 이동
  - get_snapshot(): Unity JSON export용 스텝별 상태 스냅샷
"""

import sys
import numpy as np
import random
from collections import deque
from typing import Optional, List, Dict

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import gymnasium as gym
from gymnasium import spaces


# ══════════════════════════════════════════════
# 셀 타입 상수  (4종으로 단순화)
# ══════════════════════════════════════════════
HALL, WALL, EXIT, ROOM = 0, 1, 2, 3
WALKABLE = {HALL, EXIT, ROOM}
N, S, E, W = 0, 1, 2, 3
DELTA = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}

# ── 병목(Bottleneck) 파라미터 ──────────────────────────────────────────
# Henderson(1974): 보행자 출구 처리율 ≈ 1~2명/스텝
# Fruin(1971): 복도 점유 밀도가 4명/m² 초과 시 이동 속도 급감
EXIT_CAPACITY = 1   # 출구 셀당 스텝당 최대 탈출 인원 (셀당 1명 → 명수별 병목 차이 강화)
CELL_CAPACITY = 1   # 복도 셀당 최대 동시 점유 인원 (셀당 1명 → 실제 혼잡 발생)
QUEUE_RADIUS  = 4   # 출구 혼잡도 피처 측정 반경 (BFS 거리 기준)

# ── 군중 물리 파라미터 ─────────────────────────────────────────────────
DENSITY_RADIUS     = 1    # 밀도 측정 반경 (체비쇼프 거리)
DENSITY_SLOW_MAX   = 0.80 # 최대 속도 감소율 (Fruin, 1971)
PANIC_FIRE_DIST    = 15.0 # 이 거리 이내 화재 시 공황 발생 (셀 단위)
PANIC_RANDOM_MAX   = 0.30 # 공황 최대치일 때 랜덤 이동 확률


# ══════════════════════════════════════════════
# BASE_GRID: HALL=0, WALL=1, EXIT=2, ROOM=3
# Unity 원본 인코딩(0=Hall,1=Wall,2=Room,3=Exit)에서 변환:
#   Unity 2(Room) → 3(ROOM), Unity 3(Door) → 0(HALL)
#   EXIT_A(7,10-11), EXIT_B(34,10-11) 만 2(EXIT) 유지
# ══════════════════════════════════════════════
BASE_GRID = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,3,1,1,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,3,3,3,1,1,1],
    [1,1,1,1,0,0,0,0,1,3,3,3,3,1,0,0,0,0,1,3,3,3,1,1,1],
    [1,3,3,1,0,0,0,0,1,3,3,3,3,1,0,0,0,0,1,3,3,3,1,1,1],
    [1,3,3,1,0,0,0,0,1,3,3,3,3,1,0,0,0,0,1,3,3,3,1,1,1],
    [1,3,3,0,0,0,0,0,1,3,3,3,3,1,0,0,0,0,1,3,3,3,1,1,1],
    [1,1,1,1,0,0,0,0,1,1,2,2,1,1,0,0,0,0,1,3,3,3,1,1,1],  # EXIT A
    [1,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,1,1,1,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,1,1,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1],
    [1,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,3,1],
    [1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,3,3,3,3,1],
    [1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,3,1],
    [1,3,3,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,1],
    [1,3,3,0,0,0,0,0,1,1,2,2,1,1,0,0,0,1,3,3,3,3,3,3,1],  # EXIT B
    [1,3,3,1,0,0,0,0,1,3,3,3,3,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,1,0,0,0,0,1,3,3,3,3,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,1,1,1,0,0,0,0,1,3,3,3,3,1,0,0,0,1,3,3,3,3,3,3,1],
    [1,3,3,0,0,0,0,0,1,3,3,3,3,1,0,0,0,0,3,3,3,3,3,3,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int32)

EXIT_POSITIONS = [(7, 10), (7, 11), (34, 10), (34, 11)]
EXIT_A_POS     = [(7, 10), (7, 11)]
EXIT_B_POS     = [(34, 10), (34, 11)]

_EXIT_THREAT_ZONE_A = [
    (r, c) for r in range(3, 7) for c in range(9, 13)
]  # Exit A(7,10-11) 위협 구역 — rows 3-6, cols 9-12, 16/16 walkable, 거리 1~5
_EXIT_THREAT_ZONE_B = [
    (r, c) for r in range(35, 39) for c in range(9, 13)
]  # Exit B(34,10-11) 위협 구역 — rows 35-38, cols 9-12, 16/16 walkable, 거리 1~5 (A와 대칭)
_CENTER_ZONE = [
    (r, c) for r in range(17, 23) for c in range(4, 7)
]  # 건물 중앙 복도 구역 — rows 17-22, cols 4-6, 18셀 모두 HALL, 양쪽 출구에서 균등 거리

SCENARIO_CONFIGS = {
    # S1: 기본 탈출 — 고정 화재, 단순 경로. Sanity check용. 둘 다 ~100% 예상.
    1: {
        "name":            "기본 탈출",
        "fire_count":      (1, 1),
        "spread_prob":     0.05,
        "smoke_radius":    0,
        "exit_block_prob": 0.0,
        "collapse_prob":   0.0,
        "fire_fixed":      [(2, 1)],
        "fire_zone":       None,
        "fire_zone_multi": None,
        "max_steps":       200,
        "n_agents":        10,
    },
    # S2: 점진적 위협 — 중앙 화재가 출구 방향으로 천천히 확산.
    # F14/F15(위협 변화 속도)를 활용해 PPO가 선제적으로 경로 전환할 수 있음.
    # A*는 F1/F2가 실제로 낮아진 뒤에야 반응 → PPO 소폭 유리.
    2: {
        "name":            "점진적 위협",
        "fire_count":      (1, 2),
        "spread_prob":     0.10,
        "smoke_radius":    3,
        "exit_block_prob": 0.0,
        "collapse_prob":   0.0,
        "fire_fixed":      None,
        "fire_zone":       _CENTER_ZONE,
        "fire_zone_multi": None,
        "max_steps":       250,
        "n_agents":        10,
    },
    # S3: 출구 혼잡 분산 — 화재가 출구에서 멀어 양쪽 모두 안전.
    # 20명이 가까운 출구에 몰리면 병목 발생. A*는 F7/F8 미사용으로 혼잡 방치.
    # PPO는 F7/F8로 출구별 혼잡 감지 → 반대편 분산 유도 → 명확히 PPO 유리.
    3: {
        "name":            "출구 혼잡",
        "fire_count":      (1, 1),
        "spread_prob":     0.05,
        "smoke_radius":    2,
        "exit_block_prob": 0.0,
        "collapse_prob":   0.0,
        "fire_fixed":      None,
        "fire_zone":       _CENTER_ZONE,
        "fire_zone_multi": None,
        "max_steps":       180,
        "n_agents":        20,
    },
    # S4: 부분 위협 — 출구 A·B 구역 각각 60% 확률로 독립 점화.
    # 양쪽 동시 위협(36%), 단일 위협(48%), 무위협(16%)이 혼재.
    # 이진 결정이 아니므로 A*의 단순 threshold 규칙이 실패.
    # PPO는 F1/F2 위협 + F7/F8 혼잡을 함께 고려한 trade-off 학습 → PPO 유리.
    4: {
        "name":            "부분 위협",
        "fire_count":      (1, 1),
        "spread_prob":     0.18,
        "smoke_radius":    3,
        "exit_block_prob": 0.0,
        "collapse_prob":   0.0,
        "fire_fixed":      None,
        "fire_zone":       None,
        "fire_zone_multi": [
            {"zone": _EXIT_THREAT_ZONE_A, "prob": 0.6},
            {"zone": _EXIT_THREAT_ZONE_B, "prob": 0.6},
        ],
        "max_steps":       280,
        "n_agents":        15,
    },
}


# ══════════════════════════════════════════════
# 환경
# ══════════════════════════════════════════════
class FireEvacEnv(gym.Env):
    """
    화재대피유도시스템 강화학습 환경.

    군중 물리:
      - 밀도 기반 속도 감소 (Fruin, 1971)
      - 공황 레벨 (Helbing, 2000): 화재 근접 + 시간 → 비합리적 이동
    Unity 연동:
      - get_snapshot(): 스텝별 상태 딕셔너리 반환 (JSON 직렬화 가능)
      - get_grid_info(): 초기 씬 설정용 그리드 정보
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: int = 1, n_agents: int = 10,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.scenario    = scenario
        self.cfg         = SCENARIO_CONFIGS[scenario]
        self.n_agents    = n_agents
        self.render_mode = render_mode
        self.ROWS, self.COLS = BASE_GRID.shape

        self.light_cells = [
            (r, c) for r in range(self.ROWS) for c in range(self.COLS)
            if BASE_GRID[r, c] in WALKABLE and BASE_GRID[r, c] != EXIT
        ]
        self.n_lights  = len(self.light_cells)
        self.light_idx = {cell: i for i, cell in enumerate(self.light_cells)}

        # 관측: 스칼라 피처 15개 (F1~F15) — 그리드 크기 독립, Unity 이식 가능
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array( [5.0,  5.0,  0.5]),
            high=np.array([50.0, 50.0, 5.0]),
            shape=(3,), dtype=np.float32
        )

        self.grid        = BASE_GRID.copy()
        self.fire_map    = self.smoke_map = None
        self.people_data = self.light_dirs = self.blocked_exits = None
        self.step_count  = self.escaped = self.dead = 0
        self.escaped_A   = self.escaped_B = 0
        self._bfs_dist   = None
        self._dist_to_exit_A = self._dist_to_exit_B = None
        self.prev_fire_map   = None
        self._occupancy: dict = {}
        self._prev_f1 = self._prev_f2 = 1.0  # F14/F15 계산용 이전 스텝 위협값

    # ──────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        cfg = self.cfg
        self.grid = BASE_GRID.copy()

        # 복도 붕괴 (S4)
        if cfg["collapse_prob"] > 0:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    if (self.grid[r, c] in WALKABLE and self.grid[r, c] != EXIT
                            and random.random() < cfg["collapse_prob"]):
                        self.grid[r, c] = WALL

        # 출구 폐쇄 — EXIT 셀 자체를 WALL로 전환해 F1/F2에 즉시 반영
        self.blocked_exits = set()
        if cfg["exit_block_prob"] > 0 and random.random() < cfg["exit_block_prob"]:
            target = random.choice([EXIT_A_POS, EXIT_B_POS])
            for cell in target:
                self.blocked_exits.add(cell)
                self.grid[cell[0], cell[1]] = WALL

        # BFS 거리맵
        self._bfs_dist = self._compute_bfs()
        valid_A = [p for p in EXIT_A_POS if p not in self.blocked_exits]
        valid_B = [p for p in EXIT_B_POS if p not in self.blocked_exits]
        self._dist_to_exit_A = self._compute_bfs_specific(valid_A)
        self._dist_to_exit_B = self._compute_bfs_specific(valid_B)

        # 화재
        self.fire_map = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        walkable = self._get_walkable()
        walkable_set = set(walkable)
        if cfg["fire_fixed"]:
            for p in cfg["fire_fixed"]: self.fire_map[p] = 1.0
        elif cfg.get("fire_zone_multi"):
            # 각 구역을 독립 확률로 점화 — 동시 다중 위협 구현
            for zone_cfg in cfg["fire_zone_multi"]:
                if random.random() < zone_cfg["prob"]:
                    zone = [p for p in zone_cfg["zone"] if p in walkable_set]
                    if zone:
                        self.fire_map[random.choice(zone)] = 1.0
        elif cfg["fire_zone"]:
            raw = cfg["fire_zone"]
            zone = [p for p in raw if p in walkable_set]
            n = random.randint(*cfg["fire_count"])
            for p in random.sample(zone, min(n, len(zone))):
                self.fire_map[p] = 1.0
        else:
            n = random.randint(*cfg["fire_count"])
            for p in random.sample(walkable, min(n, len(walkable))):
                self.fire_map[p] = 1.0

        self.smoke_map = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        self._spread_smoke()

        # 사람 배치 — panic 필드 추가
        safe   = [c for c in walkable if self.fire_map[c[0], c[1]] == 0]
        starts = random.sample(safe, min(self.n_agents, len(safe)))
        self.people_data = [
            {
                "pos":       pos,
                "speed":     round(random.uniform(0.5, 1.2), 2),
                "accum":     0.0,
                "prev_dist": float(self._bfs_dist[pos[0], pos[1]]),
                "panic":     0.0,  # Helbing(2000): 초기 공황 없음
            }
            for pos in starts
        ]

        self.light_dirs    = self._compute_dirs_for_strategy(10.0, 10.0, 2.0)
        self.prev_fire_map = self.fire_map.copy()
        self.step_count    = self.escaped = self.dead = 0
        self.escaped_A     = self.escaped_B = 0
        self._prev_f1 = self._prev_f2 = 1.0

        self._occupancy = {}
        for p in self.people_data:
            pos = p["pos"]
            self._occupancy[pos] = self._occupancy.get(pos, 0) + 1

        return self._get_obs(), self._get_info()

    # ──────────────────────────────────────────
    # step
    # ──────────────────────────────────────────
    def step(self, action: np.ndarray):
        exit_a_cost  = float(action[0])
        exit_b_cost  = float(action[1])
        crowd_weight = float(action[2])
        self.light_dirs = self._compute_dirs_for_strategy(
            exit_a_cost, exit_b_cost, crowd_weight)
        self.step_count += 1
        reward = 0.0

        exit_quota = {pos: EXIT_CAPACITY
                      for pos in EXIT_A_POS + EXIT_B_POS
                      if pos not in self.blocked_exits}

        next_people = []
        for p in self.people_data:
            # ── 군중 물리 업데이트 ──────────────────────────────────
            self._update_panic(p)
            eff_speed = self._effective_speed(p)
            p["accum"] += eff_speed

            if p["accum"] >= 1.0:
                old_pos = p["pos"]
                new_pos, exited = self._attempt_move(
                    old_pos, exit_quota, panic=p["panic"])
                self._occupancy[old_pos] = max(
                    0, self._occupancy.get(old_pos, 0) - 1)
                if not exited:
                    self._occupancy[new_pos] = \
                        self._occupancy.get(new_pos, 0) + 1
                p["pos"] = new_pos
                p["accum"] -= 1.0

                if exited:
                    self.escaped += 1
                    if new_pos in EXIT_A_POS: self.escaped_A += 1
                    else:                     self.escaped_B += 1
                    reward += 20.0
                    continue

            r, c    = p["pos"]
            cur_dist = float(self._bfs_dist[r, c])

            if self.fire_map[r, c] > 0:
                self.dead += 1
                reward    -= 8.0
                self._occupancy[p["pos"]] = max(
                    0, self._occupancy.get(p["pos"], 0) - 1)
            else:
                delta    = float(np.clip(p["prev_dist"] - cur_dist, -20.0, 20.0))
                urgency  = 1.0 + (self.step_count / self.cfg["max_steps"]) * 2.0
                reward  += delta * 2.0 * urgency
                p["prev_dist"] = cur_dist
                next_people.append(p)

        self.people_data = next_people
        self._spread_fire()
        self._spread_smoke()

        if self.prev_fire_map is None or not np.array_equal(
                self.fire_map, self.prev_fire_map):
            self.light_dirs = self._compute_dirs_for_strategy(
                exit_a_cost, exit_b_cost, crowd_weight)
            self._bfs_dist  = self._compute_bfs_fire_aware()
            for p in self.people_data:
                p["prev_dist"] = float(
                    self._bfs_dist[p["pos"][0], p["pos"][1]])
            self.prev_fire_map = self.fire_map.copy()

        alive = []
        for p in self.people_data:
            r, c = p["pos"]
            if self.fire_map[r, c] > 0:
                self.dead += 1
                reward    -= 8.0
                self._occupancy[p["pos"]] = max(
                    0, self._occupancy.get(p["pos"], 0) - 1)
            else:
                alive.append(p)
        self.people_data = alive

        # 스텝별 출구 불균형 패널티 — 두 출구 모두 안전할 때만 적용
        # 한쪽 출구가 화재에 가까우면 전원을 반대 출구로 보내는 것이 정답이므로 패널티 제거
        n_alive = len(self.people_data)
        if (n_alive > 1
                and self._dist_to_exit_A is not None
                and self._dist_to_exit_B is not None):
            near_a = sum(1 for p in self.people_data
                         if self._dist_to_exit_A[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
            near_b = sum(1 for p in self.people_data
                         if self._dist_to_exit_B[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
            fire_cells_pos = np.argwhere(self.fire_map > 0)
            def _min_fire_dist(dist_map):
                if len(fire_cells_pos) == 0:
                    return 999.0
                d = [dist_map[r, c] for r, c in fire_cells_pos if dist_map[r, c] < 9999]
                return float(min(d)) if d else 999.0
            a_fire_dist = _min_fire_dist(self._dist_to_exit_A)
            b_fire_dist = _min_fire_dist(self._dist_to_exit_B)
            if a_fire_dist > 5 and b_fire_dist > 5:
                reward -= abs(near_a - near_b) / n_alive * 1.0

        terminated = len(self.people_data) == 0
        truncated  = self.step_count >= self.cfg["max_steps"]

        if terminated or truncated:
            not_escaped = self.n_agents - self.escaped
            reward -= not_escaped * 5.0
            if self.escaped_A > 0 and self.escaped_B > 0:
                reward += 15.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ──────────────────────────────────────────
    # 군중 물리
    # ──────────────────────────────────────────
    def _effective_speed(self, p) -> float:
        """밀도 기반 속도 감소 (Fruin, 1971).
        반경 DENSITY_RADIUS 내 점유 인원 수에 비례해 속도 감소.
        최대 DENSITY_SLOW_MAX(80%) 감소.
        """
        r, c = p["pos"]
        local = 0
        for dr in range(-DENSITY_RADIUS, DENSITY_RADIUS + 1):
            for dc in range(-DENSITY_RADIUS, DENSITY_RADIUS + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                    local += self._occupancy.get((nr, nc), 0)
        # 반경 1 → 주변 최대 8칸
        max_neighbors = (2 * DENSITY_RADIUS + 1) ** 2 - 1
        density_ratio = min(local / max(max_neighbors, 1), 1.0)
        return p["speed"] * max(1.0 - DENSITY_SLOW_MAX, 1.0 - DENSITY_SLOW_MAX * density_ratio)

    def _update_panic(self, p) -> None:
        """공황 레벨 업데이트 (Helbing, 2000).
        화재 근접도 + 시간 경과 → panic 상승 (빠름) / 감소 (느림).
        """
        r, c = p["pos"]
        dist_to_fire = float(self._bfs_dist[r, c])
        fire_threat  = float(np.clip(1.0 - dist_to_fire / PANIC_FIRE_DIST, 0.0, 1.0))
        time_pressure = self.step_count / self.cfg["max_steps"] * 0.4
        target = min(1.0, fire_threat + time_pressure)
        # 공황은 빠르게 오르고 천천히 가라앉음
        alpha      = 0.3 if target > p["panic"] else 0.05
        p["panic"] = float(np.clip(
            p["panic"] * (1.0 - alpha) + target * alpha, 0.0, 1.0))

    # ──────────────────────────────────────────
    # 이동
    # ──────────────────────────────────────────
    def _attempt_move(self, pos, exit_quota: dict, panic: float = 0.0):
        """
        병목 인식 이동.
        panic > 0 이면 PANIC_RANDOM_MAX 확률로 유도등 무시 랜덤 이동 (Helbing, 2000).
        """
        r, c = pos

        # 공황 상태: 랜덤 방향 시도
        if panic > 0 and random.random() < panic * PANIC_RANDOM_MAX:
            dirs = list(DELTA.keys())
            random.shuffle(dirs)
        else:
            pref = (int(self.light_dirs[self.light_idx[(r, c)]])
                    if (r, c) in self.light_idx else self._bfs_best(r, c))
            dirs = [pref] + [x for x in (N, S, E, W) if x != pref]

        for d in dirs:
            dr, dc = DELTA[d]
            nr, nc = r + dr, c + dc
            if not self._passable(nr, nc):
                continue
            if self.grid[nr, nc] == EXIT and (nr, nc) not in self.blocked_exits:
                if exit_quota.get((nr, nc), 0) > 0:
                    exit_quota[(nr, nc)] -= 1
                    return (nr, nc), True
                else:
                    continue
            if self._occupancy.get((nr, nc), 0) < CELL_CAPACITY:
                return (nr, nc), False
        return pos, False

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

    # ──────────────────────────────────────────
    # 화재 / 연기
    # ──────────────────────────────────────────
    def _spread_fire(self):
        nf = self.fire_map.copy()
        prob = self.cfg["spread_prob"]
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
        radius = self.cfg["smoke_radius"]
        self.smoke_map[:] = 0
        if radius == 0: return
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.fire_map[r, c] > 0:
                    r0 = max(0, r - radius); r1 = min(self.ROWS, r + radius + 1)
                    c0 = max(0, c - radius); c1 = min(self.COLS, c + radius + 1)
                    self.smoke_map[r0:r1, c0:c1] = 1.0

    # ──────────────────────────────────────────
    # BFS / Dijkstra
    # ──────────────────────────────────────────
    def _compute_bfs(self):
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = deque()
        for (r, c) in EXIT_POSITIONS:
            if self.grid[r, c] == EXIT:
                dist[r, c] = 0; q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and dist[nr, nc] == 9999
                        and self.grid[nr, nc] in WALKABLE):
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        return dist

    def _compute_bfs_fire_aware(self):
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
                        and self.fire_map[nr, nc] == 0):
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        return dist

    def _compute_bfs_specific(self, exit_positions):
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = deque()
        for (r, c) in exit_positions:
            if self.grid[r, c] == EXIT:
                dist[r, c] = 0; q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and dist[nr, nc] == 9999
                        and self.grid[nr, nc] in WALKABLE):
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        return dist

    def _compute_bfs_with_risk(self, exit_positions, exit_cost=10.0,
                               crowd_weight=2.0):
        import heapq
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q = []
        for (r, c) in exit_positions:
            if self.grid[r, c] == EXIT:
                dist[r, c] = exit_cost          # 출구 선호도 비용 — 화재 없어도 항상 유효
                heapq.heappush(q, (exit_cost, (r, c)))
        while q:
            cost, (r, c) = heapq.heappop(q)
            if cost > dist[r, c]: continue
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and self.grid[nr, nc] in WALKABLE):
                    base = (10.0 if self.fire_map[nr, nc] > 0  # 화재 통과 페널티 고정
                            else 5 if self.smoke_map[nr, nc] > 0 else 1)
                    new_cost = cost + base + \
                        self._occupancy.get((nr, nc), 0) * crowd_weight
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
        dist_a = self._compute_bfs_with_risk(EXIT_A_POS, exit_a_cost, crowd_weight)
        dist_b = self._compute_bfs_with_risk(EXIT_B_POS, exit_b_cost, crowd_weight)
        dist_combined = np.minimum(dist_a, dist_b)
        dirs = np.zeros(self.n_lights, dtype=np.int32)
        for i, cell in enumerate(self.light_cells):
            dirs[i] = self._bfs_best_from_dist(dist_combined, cell[0], cell[1])
        return dirs

    def _get_walkable(self):
        return [(r, c) for r in range(self.ROWS) for c in range(self.COLS)
                if self.grid[r, c] in WALKABLE and self.grid[r, c] != EXIT]

    # ──────────────────────────────────────────
    # 관측 / 정보
    # ──────────────────────────────────────────
    def _get_obs(self):
        fire_cells = np.argwhere(self.fire_map > 0)
        a_blocked  = all(p in self.blocked_exits for p in EXIT_A_POS)
        b_blocked  = all(p in self.blocked_exits for p in EXIT_B_POS)

        def _exit_threat(blocked, dist_map, fire_cs):
            if blocked: return 0.0
            if len(fire_cs) == 0 or dist_map is None: return 1.0
            reach = [float(dist_map[r, c]) for r, c in fire_cs
                     if dist_map[r, c] < 9999]
            return float(np.clip(min(reach) / 20.0, 0.0, 1.0)) if reach else 1.0

        # F1/F2: 출구별 화재 위협 (1=안전, 0=위험)
        f1 = _exit_threat(a_blocked, self._dist_to_exit_A, fire_cells)
        f2 = _exit_threat(b_blocked, self._dist_to_exit_B, fire_cells)

        n_alive = len(self.people_data)
        near_a = near_b = 0
        if n_alive == 0 or self._dist_to_exit_A is None or self._dist_to_exit_B is None:
            f3, f7, f8 = 0.5, 0.0, 0.0
        else:
            n_near_a = sum(1 for p in self.people_data
                           if self._dist_to_exit_A[p["pos"][0], p["pos"][1]]
                           <= self._dist_to_exit_B[p["pos"][0], p["pos"][1]])
            f3 = n_near_a / n_alive  # A가 더 가까운 사람 비율

            near_a = sum(1 for p in self.people_data
                         if self._dist_to_exit_A[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
            near_b = sum(1 for p in self.people_data
                         if self._dist_to_exit_B[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
            f7 = near_a / n_alive  # 출구 A 혼잡도
            f8 = near_b / n_alive  # 출구 B 혼잡도

        f4  = self.escaped / self.n_agents
        f5  = self.dead    / self.n_agents
        f6  = self.step_count / self.cfg["max_steps"]
        f9  = float(np.mean([p["panic"] for p in self.people_data])) if n_alive > 0 else 0.0
        f10 = near_a / self.n_agents  # 출구 A 혼잡 절댓값
        f11 = near_b / self.n_agents  # 출구 B 혼잡 절댓값

        # F12/F13: 화재 무게중심 위치 (공간 위치 — 그리드 크기 정규화)
        if len(fire_cells) > 0:
            f12 = float(np.mean(fire_cells[:, 0])) / self.ROWS
            f13 = float(np.mean(fire_cells[:, 1])) / self.COLS
        else:
            f12 = f13 = 0.5  # 화재 없음 → 중립

        # F14/F15: 화재의 출구 접근 속도 (이전 스텝 대비 위협 증가량)
        f14 = float(np.clip(self._prev_f1 - f1, 0.0, 1.0))
        f15 = float(np.clip(self._prev_f2 - f2, 0.0, 1.0))
        self._prev_f1 = f1
        self._prev_f2 = f2

        return np.array(
            [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15],
            dtype=np.float32
        )

    def _get_info(self):
        return {
            "scenario":      self.scenario,
            "scenario_name": self.cfg["name"],
            "step":          self.step_count,
            "n_agents":      self.n_agents,
            "escaped":       self.escaped,
            "dead":          self.dead,
            "escaped_A":     self.escaped_A,
            "escaped_B":     self.escaped_B,
            "remaining":     len(self.people_data),
            "survival_rate": self.escaped / self.n_agents if self.n_agents else 0.0,
            "fire_cells":    int(self.fire_map.sum()),
            "blocked_exits": list(self.blocked_exits),
            "mean_panic":    float(np.mean([p["panic"] for p in self.people_data])
                                   if self.people_data else 0.0),
        }

    def render(self):
        if self.render_mode != "human": return
        DIR  = {N: "↑", S: "↓", E: "→", W: "←"}
        CELL = {HALL: "·", WALL: "█", EXIT: "E", ROOM: "R"}
        pset = {p["pos"] for p in self.people_data}
        lmap = {self.light_cells[i]: int(self.light_dirs[i])
                for i in range(self.n_lights)}
        print(f"\n[Step {self.step_count}] 탈출 {self.escaped} | "
              f"사망 {self.dead} | 잔류 {len(self.people_data)}")
        for r in range(self.ROWS):
            row = ""
            for c in range(self.COLS):
                if   (r, c) in pset:       row += "P"
                elif self.fire_map[r, c]:  row += "F"
                elif self.smoke_map[r, c]: row += "~"
                elif (r, c) in lmap:       row += DIR[lmap[(r, c)]]
                else:                      row += CELL.get(self.grid[r, c], "?")
            print(row)

    # ──────────────────────────────────────────
    # Unity 연동 — JSON 스냅샷
    # ──────────────────────────────────────────
    def get_snapshot(self) -> dict:
        """스텝별 상태 딕셔너리. JSON 직렬화 가능. Unity replay용."""
        fire_cells  = [[int(r), int(c)]
                       for r, c in zip(*np.where(self.fire_map > 0))] \
                      if self.fire_map.any() else []
        smoke_cells = [[int(r), int(c)]
                       for r, c in zip(*np.where(self.smoke_map > 0))] \
                      if self.smoke_map.any() else []
        return {
            "step":    self.step_count,
            "people":  [
                {
                    "id":    i,
                    "row":   int(p["pos"][0]),
                    "col":   int(p["pos"][1]),
                    "panic": round(float(p["panic"]), 3),
                    "speed": round(float(p["speed"]), 2),
                }
                for i, p in enumerate(self.people_data)
            ],
            "fire_cells":    fire_cells,
            "smoke_cells":   smoke_cells,
            "escaped":       self.escaped,
            "escaped_A":     self.escaped_A,
            "escaped_B":     self.escaped_B,
            "dead":          self.dead,
            "blocked_exits": [[int(r), int(c)] for r, c in self.blocked_exits],
            "light_dirs":    {
                f"{r},{c}": int(self.light_dirs[self.light_idx[(r, c)]])
                for r, c in self.light_cells
            },
        }

    def get_grid_info(self) -> dict:
        """초기 씬 설정용 그리드 정보. 에피소드 시작 시 1회 호출."""
        return {
            "rows":         self.ROWS,
            "cols":         self.COLS,
            "grid":         self.grid.tolist(),
            "exit_a":       EXIT_A_POS,
            "exit_b":       EXIT_B_POS,
            "scenario":     self.scenario,
            "scenario_name":self.cfg["name"],
            "n_agents":     self.n_agents,
        }


# ══════════════════════════════════════════════
# BFS 연결성 검증
# ══════════════════════════════════════════════
def verify_connectivity():
    grid = BASE_GRID.copy()
    ROWS, COLS = grid.shape

    def bfs(targets):
        dist = np.full((ROWS, COLS), 9999)
        q = deque()
        for (r, c) in targets:
            if grid[r, c] == EXIT:
                dist[r, c] = 0; q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < ROWS and 0 <= nc < COLS
                        and dist[nr, nc] == 9999
                        and grid[nr, nc] in WALKABLE):
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    dist_a = bfs(EXIT_A_POS)
    dist_b = bfs(EXIT_B_POS)
    walkable = [(r, c) for r in range(ROWS) for c in range(COLS)
                if grid[r, c] in WALKABLE and grid[r, c] != EXIT]
    ua = [(r, c) for (r, c) in walkable if dist_a[r, c] == 9999]
    ub = [(r, c) for (r, c) in walkable if dist_b[r, c] == 9999]

    print(f"[연결성 검증]")
    print(f"  WALKABLE 셀 수: {len(walkable)}")
    print(f"  EXIT_A 도달 불가: {len(ua)}셀 {ua[:5]}")
    print(f"  EXIT_B 도달 불가: {len(ub)}셀 {ub[:5]}")
    if ua or ub:
        print("  ⚠️ 도달 불가 셀 존재")
    else:
        print("  ✅ 모든 셀에서 두 출구 모두 도달 가능")
    return ua, ub
