"""
녹화된 JSON Lines 파일을 Unity에 재생하는 서버.
Unity 코드 변경 없이 기존 TCP 프로토콜 그대로 사용.

사용법:
  python playback_server.py --scenario 4
  python playback_server.py --file recordings/recording_s4_seed4.jsonl
"""
import socket
import json
import struct
import argparse
import os

DEMO_SEEDS = {1: 42, 2: 5, 3: 1, 4: 4, 5: 42, 6: 20}


def recv_data(sock, length):
    data = b''
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("연결 끊김")
        data += chunk
    return data


def send_message(sock, msg):
    data = json.dumps(msg, default=str).encode('utf-8')
    sock.sendall(struct.pack('>I', len(data)) + data)


def recv_message(sock):
    length = struct.unpack('>I', recv_data(sock, 4))[0]
    return json.loads(recv_data(sock, length).decode('utf-8'))


def run(file_path, host='0.0.0.0', port=5555):
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(1.0)
    print(f"🎬 재생 서버 시작 | {host}:{port}")
    print(f"📂 파일: {file_path}")
    print("⏳ Unity 연결 대기 중... (종료: Ctrl+C)")

    try:
        while True:
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue

            print(f"\n✅ 연결됨: {addr}")

            with open(file_path, encoding='utf-8') as f:
                lines = [json.loads(line) for line in f if line.strip()]

            # 첫 줄은 init 메시지
            init_msg = lines[0]
            total_steps = sum(1 for l in lines if l.get("message_type") == "step_snapshot")
            print(f"📊 총 {total_steps} 스텝 재생 시작")

            try:
                send_message(client, init_msg)

                step_lines = [l for l in lines[1:]]
                for msg in step_lines:
                    if msg.get("message_type") == "episode_end":
                        # episode_end는 마지막 step_snapshot 전송 후 Unity 요청 받고 보냄
                        break

                    # Unity의 next_step 요청 대기
                    req = recv_message(client)
                    if req.get("request") == "disconnect":
                        print("Unity에서 종료 요청")
                        break

                    send_message(client, msg)

                    if msg.get("terminated") or msg.get("truncated"):
                        end_msg = next(
                            (l for l in step_lines if l.get("message_type") == "episode_end"),
                            {"message_type": "episode_end", "final_info": msg.get("info", {})}
                        )
                        send_message(client, end_msg)
                        survival = msg.get("info", {}).get("survival_rate", 0)
                        print(f"🏁 재생 완료 | 생존율: {survival:.1%}")
                        break

            except Exception as e:
                print(f"❌ 에러: {e}")
            finally:
                client.close()

    except KeyboardInterrupt:
        print("\n🛑 서버 종료")
    finally:
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument("--file", type=str, default=None,
                        help="직접 파일 경로 지정 (없으면 시나리오 번호로 자동 선택)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    if args.file:
        file_path = args.file
    else:
        seed = DEMO_SEEDS.get(args.scenario, 42)
        file_path = os.path.join("recordings", f"recording_s{args.scenario}_seed{seed}.jsonl")

    run(file_path, host=args.host, port=args.port)
