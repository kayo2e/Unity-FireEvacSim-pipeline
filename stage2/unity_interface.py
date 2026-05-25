import socket
import json
import numpy as np
import threading
import os
import struct
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_core import FireEvacEnv, SCENARIO_CONFIGS

# GIF 시연 때 사용한 시나리오별 고정 seed (make_comparison_gifs.py SC_SEED 기준)
DEMO_SEEDS = {1: 42, 2: 5, 3: 1, 4: 4, 5: 42, 6: 20}

# ═══════════════════════════════════════════════════════════
# 모델 로드
# ═══════════════════════════════════════════════════════════
def load_model(scenario: int):
    n_agents = SCENARIO_CONFIGS[scenario]["n_agents"]
    model_dir = os.path.join("model", "ppo")
    model_name = f"fire_evac_model_{n_agents}ppl"
    model_path = os.path.join(model_dir, model_name)
    vecnorm_path = f"{model_path}_vecnorm.pkl"

    print(f"📋 시나리오 {scenario} → {n_agents}명 모델 로드 시도")

    try:
        model = PPO.load(model_path)
        print(f"✅ 모델 로드 성공: {model_path}.zip")

        vecnorm = None
        if os.path.exists(vecnorm_path):
            with open(vecnorm_path, "rb") as f:
                vecnorm = pickle.load(f)
            vecnorm.training = False
            vecnorm.norm_reward = False
            print(f"✅ 정규화 로드 성공: {vecnorm_path}")
        else:
            print(f"⚠️ 정규화 파일 없음 (정규화 미적용): {vecnorm_path}")

        return model, vecnorm, n_agents

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print(f"🔍 찾으려고 시도한 경로: {os.path.abspath(model_path)}.zip")
        return None, None, n_agents

# ═══════════════════════════════════════════════════════════
# 환경 관리자 (관측값 유지를 위해 self.obs 저장 위치 수정)
# ═══════════════════════════════════════════════════════════
class EnvironmentManager:
    def __init__(self, scenario=4):
        self.scenario = scenario
        self.n_agents = SCENARIO_CONFIGS[scenario]["n_agents"]
        self.demo_seed = DEMO_SEEDS.get(scenario)
        self.env = FireEvacEnv(scenario=scenario, n_agents=self.n_agents)
        self.obs, self.info = self.env.reset(seed=self.demo_seed)
        self.step_count = 0
        print(f"🎮 환경 초기화 | Scenario: {scenario}, Agents: {self.n_agents}, Seed: {self.demo_seed}")
    
    def step(self, action):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.step_count += 1
        return self.obs, reward, terminated, truncated, self.info
    
    def reset(self, seed=None):
        self.obs, self.info = self.env.reset(seed=seed)
        self.step_count = 0
        return self.obs
    
    def get_snapshot(self):
        return self.env.get_snapshot()
    
    def get_grid_info(self):
        return self.env.get_grid_info()

# ═══════════════════════════════════════════════════════════
# 소켓 서버
# ═══════════════════════════════════════════════════════════
class FireEvacServer:
    def __init__(self, host='0.0.0.0', port=5555, scenario=4):
        self.host = host
        self.port = port
        self.scenario = scenario

        self.model, self.vecnorm, self.n_agents = load_model(scenario)
        self.env_manager = EnvironmentManager(scenario=scenario)
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        
        print(f"🚀 서버 시작 | {host}:{port}")
    
    def recv_data(self, sock, length):
        data = b''
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise ConnectionError("연결이 끊어졌습니다")
            data += chunk
        return data
    
    def send_message(self, sock, message_dict):
        json_str = json.dumps(message_dict, default=str)
        json_bytes = json_str.encode('utf-8')
        length = struct.pack('>I', len(json_bytes))
        sock.sendall(length + json_bytes)
    
    def recv_message(self, sock):
        length_bytes = self.recv_data(sock, 4)
        length = struct.unpack('>I', length_bytes)[0]
        json_bytes = self.recv_data(sock, length)
        return json.loads(json_bytes.decode('utf-8'))
    
    def handle_client(self, client_socket, addr):
        print(f"\n✅ 클라이언트 연결: {addr}")
        
        # ── 녹화 파일 준비 ──
        rec_dir = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        rec_path = os.path.join(
            rec_dir,
            f"recording_s{self.scenario}_seed{self.env_manager.demo_seed}.jsonl"
        )
        rec_file = open(rec_path, "w", encoding="utf-8")
        print(f"🎬 녹화 시작: {rec_path}")

        try:
            print("🔄 환경 초기화 중 (새 에피소드 준비)...")
            self.env_manager.reset(seed=self.env_manager.demo_seed)
            self._smooth_action = None       # 에피소드마다 EMA 초기화
            self._prev_snapshot_pos = {}     # 원거리 점프 감지용 초기화

            # 1. 초기 맵 정보와 시작 스냅샷을 유니티로 전송
            init_msg = {
                "message_type": "init",
                "grid_info": self.env_manager.get_grid_info(),
                "initial_snapshot": self.env_manager.get_snapshot()
            }
            rec_file.write(json.dumps(init_msg, default=str) + "\n")
            self.send_message(client_socket, init_msg)

            print("\n🎯 [에피소드 시작] 유니티의 진행 요청을 대기합니다.")
            
            while True:
                # 2. 유니티가 이전 애니메이션을 마치고 다음 스텝을 요청할 때까지 대기
                # 이제 유니티는 observation을 보낼 필요 없이 {"request": "next_step"} 정도만 보내면 됩니다.
                request = self.recv_message(client_socket)
                
                if request.get("request") == "disconnect":
                    print("클라이언트에서 종료를 요청했습니다.")
                    break
                
                # 3. 파이썬 내부 환경의 관측값(self.env_manager.obs)을 이용해 행동 예측
                obs = self.env_manager.obs.reshape(1, -1)
                if self.vecnorm is not None:
                    obs = self.vecnorm.normalize_obs(obs)
                
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                    action = action[0].tolist()
                else:
                    action = [20.0, 20.0, 2.0]

                # BFS cost 급변으로 인한 경로 진동 방지: EMA smoothing (alpha=0.35)
                if self._smooth_action is None:
                    self._smooth_action = list(action)
                self._smooth_action = [0.35 * a + 0.65 * s
                                       for a, s in zip(action, self._smooth_action)]
                action = list(self._smooth_action)

                # 4. 파이썬 환경 스텝 진행
                env_obs, reward, terminated, truncated, info = self.env_manager.step(action)
                snapshot = self.env_manager.get_snapshot()

                # ── 디버그: 연속 스텝 간 2셀 이상 이동하는 원거리 점프 감지 ──
                cur_pos = {p["id"]: p for p in snapshot["people"]}
                if hasattr(self, '_prev_snapshot_pos'):
                    for pid, p in cur_pos.items():
                        prev = self._prev_snapshot_pos.get(pid)
                        if prev:
                            dist = abs(p["row"] - prev["row"]) + abs(p["col"] - prev["col"])
                            if dist > 2:
                                print(f"⚠️  [Step {self.env_manager.step_count}] 원거리 점프 "
                                      f"id={pid}: ({prev['row']},{prev['col']}) dr={prev['dr']:+d} accum={prev['accum']:.3f}"
                                      f" → ({p['row']},{p['col']}) dr={p['dr']:+d} accum={p['accum']:.3f}  dist={dist}")
                self._prev_snapshot_pos = cur_pos

                # 5. 결과 및 갱신된 스냅샷 전송
                response = {
                    "message_type": "step_snapshot",
                    "exit_A_cost": float(action[0]),
                    "exit_B_cost": float(action[1]),
                    "crowd_weight": float(action[2]),
                    "directions": self._get_directions_for_arrows(),
                    "snapshot": snapshot,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                    "step": self.env_manager.step_count
                }
                rec_file.write(json.dumps(response, default=str) + "\n")
                self.send_message(client_socket, response)

                if terminated or truncated:
                    print(f"\n🏁 에피소드 종료 | 생존율: {info['survival_rate']:.1%}")
                    end_msg = {
                        "message_type": "episode_end",
                        "final_info": info
                    }
                    rec_file.write(json.dumps(end_msg, default=str) + "\n")
                    self.send_message(client_socket, end_msg)
                    print(f"💾 녹화 저장 완료: {rec_path}")
                    break

        except Exception as e:
            print(f"❌ 클라이언트 에러: {e}")
        finally:
            rec_file.close()
            client_socket.close()
    
    def _get_directions_for_arrows(self):
        snapshot = self.env_manager.get_snapshot()
        light_dirs = snapshot.get('light_dirs', {})
        directions = []
        for r, c in self.env_manager.env.light_cells:
            key = f"{r},{c}"
            directions.append(float(light_dirs.get(key, 0)))
        return directions
    
    def run(self):
        self.server_socket.settimeout(1.0) 
        print("⏳ Unity 클라이언트 대기 중... (서버 종료: Ctrl+C)")
        try:
            while True:
                try:
                    client_socket, addr = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    pass
        except KeyboardInterrupt:
            print("\n\n🛑 [Ctrl+C] 감지됨: 서버를 안전하게 종료합니다.")
        finally:
            self.server_socket.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--scenario", type=int, default=4,
                        help="시나리오 번호 (1~6). 시나리오에 맞는 에이전트 수와 모델이 자동 선택됩니다.")
    args = parser.parse_args()

    server = FireEvacServer(host=args.host, port=args.port, scenario=args.scenario)
    server.run()