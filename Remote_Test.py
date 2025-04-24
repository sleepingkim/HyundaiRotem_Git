# Remote_Test.py (컴퓨터 2에서 실행)

import requests # requests 라이브러리 필요 (pip install requests)
import json

# 컴퓨터 1의 IP 주소와 Server.py 에서 설정한 포트 번호 입력
# <COMPUTER1_IP> 부분을 실제 컴퓨터 1의 IP 주소로 변경해야 합니다!
SERVER_IP = "192.168.0.118" # 예: "192.168.0.15"
SERVER_PORT = 5051
LIDAR_DATA_URL = f"http://{SERVER_IP}:{SERVER_PORT}/lidar_data"

def fetch_lidar_data(url):
    """지정된 URL에서 LiDAR 데이터를 가져옵니다."""
    print(f"📡 Attempting to fetch LiDAR data from: {url}")
    try:
        # 서버에 GET 요청 보내기 (timeout 설정 권장)
        response = requests.get(url, timeout=20) # 20초 이상 응답 없으면 타임아웃

        # HTTP 에러 체크 (예: 404 Not Found, 500 Internal Server Error)
        response.raise_for_status()

        # 응답이 성공적이면 JSON 데이터 파싱
        data = response.json()
        print("✅ Successfully received data from server.")
        return data

    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to the server at {url}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ Error: Request timed out while connecting to {url}.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err} (Status code: {response.status_code})")
        # 서버 에러 메시지 출력 시도
        try:
            error_details = response.json()
            print(f"   Server error details: {error_details}")
        except json.JSONDecodeError:
            print(f"   Could not parse server error response body: {response.text}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON response from the server.")
        print(f"   Response content: {response.text[:500]}...") # 받은 내용 일부 출력
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # 서버로부터 LiDAR 데이터 가져오기 시도
    lidar_response = fetch_lidar_data(LIDAR_DATA_URL)

    if lidar_response and 'lidar_data' in lidar_response:
        all_lidar_points = lidar_response['lidar_data']
        print(f"📊 Received a total of {len(all_lidar_points)} LiDAR data points.")

        if all_lidar_points:
            print("\n--- First 5 LiDAR data points ---")
            for i, point in enumerate(all_lidar_points[:5]):
                # 'distnace' -> 'distance' 로 수정
                print(f"[{i+1}] Angle: {point.get('angle')}, Dist: {point.get('distance')}, Pos: {point.get('position')}")

        else:
            print("ℹ️ The 'lidar_data' list is empty.")
    else:
        print("📉 Failed to retrieve or parse LiDAR data from the server.")