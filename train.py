"""
화재대피유도시스템 | Stage 2 (수정본)
======================================
수정 내역:
  1. 모든 함수/파일명 fire_evac 계열로 변경
  2. 승급 조건 완화: threshold 0.75 → 0.50, window 30 → 20
  3. 보상 재설계:
     - 탈출 성공: 10 → 20
     - 매 스텝 생존 보너스 제거 (서 있어도 보상 먹는 문제)
     - 출구 접근 보상 추가 (한 스텝 이동할 때마다 거리 단축되면 +보상)
     - 화재 방향 유도 페널티 강화: -5 → -8
  4. learning_rate 3e-4 → 5e-4
  5. 잘못된 import (duckdb) 제거

설치: pip install gymnasium stable-baselines3 tensorboard
실행: python fire_evac_train.py --mode train --people 10 30 50 --steps 500000
"""

import numpy as np
import random
from collections import deque
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


# ══════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════
EMPTY, WALL, EXIT, ELEVATOR, STAIR, TOILET, ROOM, OUTSIDE = 0, 1, 2, 3, 4, 5, 6, 9
WALKABLE = {EMPTY, EXIT, STAIR, TOILET, ROOM}
N, S, E, W = 0, 1, 2, 3
DELTA = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}

# 상상관 2층 Grid Map (30 x 20) - 수정 반영본
BASE_GRID = np.array([
    [9,9,9,9,9,9,9,9,1,1,1,1,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,1,0,0,1,9,9,9,9,9,9,9,9],
    [1,1,1,1,1,1,1,1,1,0,0,1,9,9,9,9,9,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,9,9,9,9,9,9,9,9],
    [1,6,6,6,6,6,6,1,5,5,5,1,9,9,9,9,9,9,9,9],
    [1,1,1,1,1,1,1,1,5,5,5,1,9,9,9,9,9,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,1,1,1,1,1,9,9,9],
    [1,6,6,6,6,6,6,1,0,0,0,0,1,9,9,9,9,9,9,9],  # (7, 11) 1 -> 0 수정
    [1,6,6,6,6,6,6,0,0,0,0,2,2,6,6,6,1,9,9,9],  # (8, 7)  1 -> 0 수정
    [1,6,6,6,6,6,6,0,3,3,3,1,6,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,1,3,3,3,1,6,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,1,4,4,4,1,6,6,6,6,1,9,9,9],
    [1,1,1,1,1,1,1,1,0,0,0,1,0,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,6,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,1,9,9,9],
    [1,1,1,1,1,1,1,1,0,0,0,1,0,6,6,6,1,9,9,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,1,1,1,1,9,9,9,9],
    [1,6,6,6,6,6,6,1,0,0,0,0,6,6,6,6,6,6,1,9],  # (17, 11) 1 -> 0 수정
    [1,1,1,1,1,1,1,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,0,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,4,4,4,1,6,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,0,2,2,2,1,6,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,0,5,5,1,1,1,1,1,1,1,1,9],
    [1,6,6,6,6,6,6,0,0,5,5,0,0,6,6,6,6,6,1,9],  # (25, 11) 1 -> 0 수정
    [1,6,6,6,6,6,6,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,6,6,6,6,6,6,0,0,0,0,1,6,6,6,6,6,6,1,9],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9],
], dtype=np.int32)

EXIT_POSITIONS = [(8, 11), (8, 12), (22, 8), (22, 9), (22, 10)]

SCENARIO_CONFIGS = {
    1: {
        "name": "초기 화재",
        "fire_count": (1, 1),
        "spread_prob": 0.05,
        "smoke_radius": 0,
        "exit_block_prob": 0.0,
        "collapse_prob": 0.0,
        "fire_fixed": [(3, 1)],
        "max_steps": 200,
    },
    2: {
        "name": "화재 확산",
        "fire_count": (2, 3),
        "spread_prob": 0.12,
        "smoke_radius": 2,
        "exit_block_prob": 0.0,
        "collapse_prob": 0.0,
        "fire_fixed": None,
        "max_steps": 250,
    },
    3: {
        "name": "출구 폐쇄",
        "fire_count": (2, 4),
        "spread_prob": 0.15,
        "smoke_radius": 3,
        "exit_block_prob": 0.5,
        "collapse_prob": 0.0,
        "fire_fixed": None,
        "max_steps": 300,
    },
    4: {
        "name": "폭발 붕괴",
        "fire_count": (3, 6),
        "spread_prob": 0.20,
        "smoke_radius": 4,
        "exit_block_prob": 0.3,
        "collapse_prob": 0.3,
        "fire_fixed": None,
        "max_steps": 400,
    },
}


# ══════════════════════════════════════════════
# 환경 (FireEvacEnv)
# ══════════════════════════════════════════════
class FireEvacEnv(gym.Env):
    """
    화재대피유도시스템 강화학습 환경.
    에이전트: 건물 내 각 셀의 유도등 방향 제어 (N/S/E/W)
    목표: 대피 인원의 생존율 최대화
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: int = 1,
        n_agents: int = 10,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert scenario in SCENARIO_CONFIGS, f"시나리오는 1~4여야 합니다. 입력값: {scenario}"

        self.scenario    = scenario
        self.cfg         = SCENARIO_CONFIGS[scenario]
        self.n_agents    = n_agents
        self.render_mode = render_mode
        self.ROWS, self.COLS = BASE_GRID.shape

        # 유도등 설치 가능 셀 (복도/방/계단, 출구 제외)
        self.light_cells = [
            (r, c)
            for r in range(self.ROWS)
            for c in range(self.COLS)
            if BASE_GRID[r, c] in WALKABLE and BASE_GRID[r, c] != EXIT
        ]
        self.n_lights  = len(self.light_cells)
        self.light_idx = {cell: i for i, cell in enumerate(self.light_cells)}

        # 액션 공간: 유도등 셀 수 × 4방향
        self.action_space = spaces.MultiDiscrete([4] * self.n_lights)

        # 관측 공간: (6채널, ROWS, COLS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(6, self.ROWS, self.COLS),
            dtype=np.float32,
        )

        # 내부 상태 (reset에서 초기화)
        self.grid         = None
        self.fire_map     = None
        self.smoke_map    = None
        self.people_data  = None   # [{"pos", "speed", "accum", "prev_dist"}, ...]
        self.light_dirs   = None
        self.blocked_exits= None
        self.step_count   = 0
        self.escaped      = 0
        self.dead         = 0
        self._bfs_dist    = None

    # ──────────────────────────────────
    # reset
    # ──────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cfg = self.cfg
        self.grid = BASE_GRID.copy()

        # 복도 붕괴 (4단계)
        if cfg["collapse_prob"] > 0:
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    if (self.grid[r, c] in WALKABLE
                            and self.grid[r, c] != EXIT
                            and random.random() < cfg["collapse_prob"]):
                        self.grid[r, c] = WALL

        # 출구 폐쇄
        self.blocked_exits = set()
        if cfg["exit_block_prob"] > 0 and random.random() < cfg["exit_block_prob"]:
            groups = [
                [(8, 11), (8, 12)],
                [(22, 8), (22, 9), (22, 10)],
            ]
            for cell in random.choice(groups):
                self.blocked_exits.add(cell)
                self.grid[cell[0], cell[1]] = WALL

        # BFS 거리 계산 (출구까지)
        self._bfs_dist = self._compute_bfs()

        # 화재 초기화
        self.fire_map = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        walkable = self._get_walkable()

        if cfg["fire_fixed"]:
            for p in cfg["fire_fixed"]:
                self.fire_map[p] = 1.0
        else:
            n = random.randint(*cfg["fire_count"])
            for p in random.sample(walkable, min(n, len(walkable))):
                self.fire_map[p] = 1.0

        # 연기 초기화
        self.smoke_map = np.zeros((self.ROWS, self.COLS), dtype=np.float32)
        self._spread_smoke()

        # 사람 배치 (속도 + 이전 출구 거리 기록)
        safe = [c for c in walkable if self.fire_map[c[0], c[1]] == 0]
        starts = random.sample(safe, min(self.n_agents, len(safe)))
        self.people_data = [
            {
                "pos":       pos,
                "speed":     round(random.uniform(0.5, 1.2), 2),
                "accum":     0.0,
                "prev_dist": float(self._bfs_dist[pos[0], pos[1]]),  # 접근 보상용
            }
            for pos in starts
        ]

        # 유도등 방향 초기화 (랜덤)
        self.light_dirs = np.random.randint(0, 4, size=self.n_lights)

        self.step_count = 0
        self.escaped    = 0
        self.dead       = 0

        return self._get_obs(), self._get_info()

    # ──────────────────────────────────
    # step
    # ──────────────────────────────────
    def step(self, action: np.ndarray):
        self.light_dirs = np.array(action, dtype=np.int32)
        self.step_count += 1
        reward = 0.0

        next_people = []
        for p in self.people_data:
            p["accum"] += p["speed"]

            if p["accum"] >= 1.0:
                new_pos = self._move_person(p["pos"])
                p["pos"]  = new_pos
                p["accum"] -= 1.0

            r, c = p["pos"]
            cur_dist = float(self._bfs_dist[r, c])

            # ── 탈출 성공 ──────────────────────────────────
            if self.grid[r, c] == EXIT and (r, c) not in self.blocked_exits:
                self.escaped += 1
                reward += 20.0                          # ★ 10 → 20

            # ── 화재 구역 진입 ──────────────────────────────
            elif self.fire_map[r, c] > 0:
                self.dead += 1
                reward -= 8.0                           # ★ -5 → -8

            # ── 생존 중: 출구 접근 보상 ─────────────────────
            else:
                approach = p["prev_dist"] - cur_dist    # ★ 접근 거리만큼 보상
                reward += approach * 2.0                #   멀어지면 음수 패널티
                p["prev_dist"] = cur_dist
                next_people.append(p)

        self.people_data = next_people

        # 화재 확산
        self._spread_fire()
        self._spread_smoke()

        # 확산된 화재에 남은 인원 피해
        alive = []
        for p in self.people_data:
            r, c = p["pos"]
            if self.fire_map[r, c] > 0:
                self.dead += 1
                reward -= 8.0
            else:
                alive.append(p)
        self.people_data = alive

        # 타임아웃 페널티
        if self.step_count >= self.cfg["max_steps"]:
            reward -= len(self.people_data) * 3.0       # ★ 2 → 3

        terminated = len(self.people_data) == 0
        truncated  = self.step_count >= self.cfg["max_steps"]

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ──────────────────────────────────
    # 사람 이동
    # ──────────────────────────────────
    def _move_person(self, pos):
        r, c = pos
        # 현재 셀 유도등 방향 우선
        pref = (int(self.light_dirs[self.light_idx[(r, c)]])
                if (r, c) in self.light_idx else self._bfs_best(r, c))

        for d in [pref] + [x for x in (N, S, E, W) if x != pref]:
            dr, dc = DELTA[d]
            nr, nc = r + dr, c + dc
            if self._passable(nr, nc):
                return (nr, nc)
        return pos

    def _bfs_best(self, r, c):
        best, best_d = N, float("inf")
        for d, (dr, dc) in DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
                v = self._bfs_dist[nr, nc]
                if v < best_d:
                    best_d, best = v, d
        return best

    def _passable(self, r, c):
        if not (0 <= r < self.ROWS and 0 <= c < self.COLS):
            return False
        return self.grid[r, c] in WALKABLE and self.fire_map[r, c] == 0

    # ──────────────────────────────────
    # 화재 / 연기 확산
    # ──────────────────────────────────
    def _spread_fire(self):
        nf   = self.fire_map.copy()
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
        if radius == 0:
            return
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.fire_map[r, c] > 0:
                    r0 = max(0, r - radius);  r1 = min(self.ROWS, r + radius + 1)
                    c0 = max(0, c - radius);  c1 = min(self.COLS, c + radius + 1)
                    self.smoke_map[r0:r1, c0:c1] = 1.0

    # ──────────────────────────────────
    # BFS 거리 계산
    # ──────────────────────────────────
    def _compute_bfs(self):
        dist = np.full((self.ROWS, self.COLS), 9999.0)
        q    = deque()
        for (r, c) in EXIT_POSITIONS:
            if self.grid[r, c] == EXIT:
                dist[r, c] = 0
                q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.ROWS and 0 <= nc < self.COLS
                        and dist[nr, nc] == 9999
                        and self.grid[nr, nc] in WALKABLE):
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        mx = dist[dist < 9999].max() if (dist < 9999).any() else 1
        return np.clip(dist / mx, 0, 1)

    def _get_walkable(self):
        return [
            (r, c)
            for r in range(self.ROWS)
            for c in range(self.COLS)
            if self.grid[r, c] in WALKABLE and self.grid[r, c] != EXIT
        ]

    # ──────────────────────────────────
    # 관측 생성 (6채널)
    # ──────────────────────────────────
    def _get_obs(self):
        obs = np.zeros((6, self.ROWS, self.COLS), dtype=np.float32)
        obs[0] = self.grid / 9.0           # 지형
        obs[1] = self.fire_map             # 화재
        obs[2] = self.smoke_map            # 연기
        for p in self.people_data:         # 사람 밀도
            obs[3, p["pos"][0], p["pos"][1]] += 1.0
        if obs[3].max() > 0:
            obs[3] /= obs[3].max()
        for i, (r, c) in enumerate(self.light_cells):  # 유도등 방향
            obs[4, r, c] = self.light_dirs[i] / 3.0
        obs[5] = 1.0 - self._bfs_dist     # 출구 접근성 (가까울수록 1)
        return obs

    def _get_info(self):
        return {
            "scenario":      self.scenario,
            "scenario_name": self.cfg["name"],
            "step":          self.step_count,
            "n_agents":      self.n_agents,
            "escaped":       self.escaped,
            "dead":          self.dead,
            "remaining":     len(self.people_data),
            "survival_rate": self.escaped / self.n_agents if self.n_agents else 0.0,
            "fire_cells":    int(self.fire_map.sum()),
            "blocked_exits": list(self.blocked_exits),
        }

    # ──────────────────────────────────
    # ASCII 렌더링
    # ──────────────────────────────────
    def render(self):
        if self.render_mode != "human":
            return
        DIR  = {N: "↑", S: "↓", E: "→", W: "←"}
        CELL = {OUTSIDE:" ", WALL:"█", EXIT:"E",
                ELEVATOR:"V", STAIR:"S", TOILET:"T", ROOM:".", EMPTY:"."}
        pset = {p["pos"] for p in self.people_data}
        lmap = {self.light_cells[i]: int(self.light_dirs[i])
                for i in range(self.n_lights)}
        print(f"\n[Step {self.step_count}] "
              f"탈출 {self.escaped} | 사망 {self.dead} | 잔류 {len(self.people_data)}")
        for r in range(self.ROWS):
            row = ""
            for c in range(self.COLS):
                if   (r, c) in pset:        row += "P"
                elif self.fire_map[r, c]:   row += "F"
                elif self.smoke_map[r, c]:  row += "~"
                elif (r, c) in lmap:        row += DIR[lmap[(r, c)]]
                else:                       row += CELL.get(self.grid[r, c], "?")
            print(row)


# ══════════════════════════════════════════════
# 커리큘럼 래퍼 (EvacCurriculumWrapper)
# ══════════════════════════════════════════════
class EvacCurriculumWrapper(gym.Wrapper):
    """
    평균 생존율이 threshold 이상이면 다음 시나리오로 자동 승급.
    ★ threshold: 0.75 → 0.50, window: 30 → 20
    """

    def __init__(self, n_agents: int = 10, threshold: float = 0.50, window: int = 20):
        self.current_scenario = 1
        self.n_agents         = n_agents
        env = FireEvacEnv(scenario=1, n_agents=n_agents)
        super().__init__(env)
        self.threshold = threshold
        self.window    = window
        self.recent    = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.recent.append(info["survival_rate"])
            if len(self.recent) > self.window:
                self.recent.pop(0)
            avg = sum(self.recent) / len(self.recent)
            if (len(self.recent) == self.window
                    and avg >= self.threshold
                    and self.current_scenario < 4):
                self.current_scenario += 1
                self.env = FireEvacEnv(
                    scenario=self.current_scenario,
                    n_agents=self.n_agents,
                )
                self.recent = []
                print(f"\n[커리큘럼] ★ {self.current_scenario}단계 승급! "
                      f"({self.cfg['name']}) | 인원 {self.n_agents}명 "
                      f"| 평균 생존율 {avg:.0%}")
        return obs, rew, term, trunc, info

    def reset(self, **kw):
        return self.env.reset(**kw)

    @property
    def cfg(self):
        return self.env.cfg


# ══════════════════════════════════════════════
# 학습 진행 콜백 (EvacTrainCallback)
# ══════════════════════════════════════════════
class EvacTrainCallback(BaseCallback):
    """주기적으로 생존율 & 평균 보상 출력."""

    def __init__(self, log_interval: int = 10_000):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards   = []
        self.ep_survival  = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "survival_rate" in info:
                self.ep_survival.append(info["survival_rate"])
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_interval == 0 and self.ep_survival:
            n      = min(len(self.ep_survival), 50)
            avg_s  = sum(self.ep_survival[-n:]) / n
            avg_r  = (sum(self.ep_rewards[-n:]) / min(len(self.ep_rewards), n)
                      if self.ep_rewards else 0.0)
            print(f"  Step {self.num_timesteps:>8,} | "
                  f"평균 생존율 {avg_s:>5.1%} | "
                  f"평균 보상 {avg_r:>+7.1f}")
        return True


# ══════════════════════════════════════════════
# PPO 학습 (train_fire_evac)
# ══════════════════════════════════════════════
def train_fire_evac(person_counts=(10, 30, 50), total_timesteps=500_000):
    """
    인원수별 PPO 학습 후 모델 저장.
    저장 파일명: fire_evac_model_10ppl.zip / _30ppl.zip / _50ppl.zip
    """
    import torch

    print("=" * 62)
    print("화재대피유도시스템 Stage 2 — PPO 학습")
    print(f"인원수 조건    : {person_counts}명")
    print(f"스텝 수        : {total_timesteps:,} / 모델")
    print(f"승급 조건      : 생존율 50% (window=20)")
    print(f"보상 설계      : 탈출+20 / 접근+거리 / 화재-8 / 타임아웃-3/인")
    print("=" * 62)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"학습 디바이스  : {device}\n")

    for n in person_counts:
        print(f"\n{'─'*62}")
        print(f"인원수 {n}명 | 커리큘럼 학습 (시나리오 1 → 4)")
        print(f"{'─'*62}")

        env      = EvacCurriculumWrapper(n_agents=n, threshold=0.50, window=20)
        callback = EvacTrainCallback(log_interval=10_000)

        model = PPO(
            "CnnPolicy",
            env,
            device          = device,
            verbose         = 0,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            learning_rate   = 5e-4,    # ★ 3e-4 → 5e-4
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
        print(f"\n모델 저장 완료: {save_path}.zip")

    print("\n" + "=" * 62)
    print("모든 학습 완료!")
    print("TensorBoard: tensorboard --logdir ./fire_evac_log/")
    print("=" * 62)


# ══════════════════════════════════════════════
# 테스트 시각화 (test_fire_evac)
# ══════════════════════════════════════════════
def test_fire_evac(n_agents: int = 10, scenario: int = 1, n_episodes: int = 3):
    """
    저장된 모델 로드 후 ASCII 시각화.
    모델 파일: fire_evac_model_{n_agents}ppl.zip
    """
    model_path = f"fire_evac_model_{n_agents}ppl"
    print(f"\n모델 로드: {model_path}.zip")
    model = PPO.load(model_path)

    env = FireEvacEnv(scenario=scenario, n_agents=n_agents, render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        total_r   = 0.0
        print(f"\n{'='*52}")
        print(f"[에피소드 {ep+1}] 시나리오 {scenario} — {info['scenario_name']}")
        print(f"인원수: {n_agents}명")

        for _ in range(env.cfg["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            env.render()
            if term or trunc:
                break

        print(f"\n결과 | 탈출 {info['escaped']}/{n_agents}명 "
              f"| 생존율 {info['survival_rate']:.0%} "
              f"| 총 보상 {total_r:.1f}")


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="화재대피유도시스템 Stage 2")
    parser.add_argument(
        "--mode", choices=["train", "test", "check"],
        default="train",
        help="train: PPO 학습 | test: 결과 시각화 | check: 환경 검증",
    )
    parser.add_argument("--people", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--steps",  type=int, default=500_000)
    parser.add_argument("--test-n", type=int, default=10)
    parser.add_argument("--test-scenario", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "check":
        print("환경 검증 중...")
        env = FireEvacEnv(scenario=1, n_agents=10)
        check_env(env)
        print("환경 검증 완료!")

    elif args.mode == "train":
        train_fire_evac(person_counts=args.people, total_timesteps=args.steps)

    elif args.mode == "test":
        test_fire_evac(n_agents=args.test_n, scenario=args.test_scenario)