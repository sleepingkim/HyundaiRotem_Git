# -*- coding: utf-8 -*-
import sys
import os
import time
import math
import requests  # 데이터 로딩 위해 추가
import json      # 데이터 로딩 위해 추가

try:
    # Scikit-learn: DBSCAN 클러스터링 알고리즘 사용
    from sklearn.cluster import DBSCAN
    # NumPy: 다차원 배열 및 수학 연산 지원 (필수)
    import numpy as np
    # Matplotlib: 데이터 시각화 (특히 3D 플롯)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # 3D 플롯 위해 추가
except ImportError as e:
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] 오류: 필수 라이브러리 임포트 실패: {e}")
    print(f"[{t_now}] Scikit-learn, NumPy, Matplotlib, Requests 라이브러리가 설치되어 있는지 확인하세요.")
    print(f"[{t_now}] 설치 명령어 예시: pip install scikit-learn numpy matplotlib requests")
    sys.exit(1)
except Exception as e:
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] 임포트 중 예기치 않은 오류 발생: {e}")
    sys.exit(1)

# --- 서버 및 데이터 설정 ---
# 컴퓨터 1의 IP 주소와 Server.py 에서 설정한 포트 번호 입력
# 제공해주신 IP 주소를 사용합니다.
SERVER_IP = "192.168.0.118"
SERVER_PORT = 5051 # Server.py 에서 사용하는 포트
LIDAR_DATA_URL = f"http://{SERVER_IP}:{SERVER_PORT}/lidar_data"

# --- 클러스터링 알고리즘 (DBSCAN) 설정값 ---
DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 15

# --- 시각화 설정 ---
PLOT_AREA_LIMIT = 50
PLOT_HEIGHT_LIMIT = 10

# === 데이터 로딩 함수 (제공해주신 Remote_Test.py 버전 사용) ===
def fetch_lidar_data(url):
    """지정된 URL에서 LiDAR 데이터를 가져옵니다."""
    print(f"📡 Attempting to fetch LiDAR data from: {url}")
    try:
        # 서버에 GET 요청 보내기 (timeout 설정 권장)
        response = requests.get(url, timeout=20) # 20초 이상 응답 없으면 타임아웃

        # HTTP 에러 체크 (예: 404 Not Found, 500 Internal Server Error)
        response.raise_for_status()

        # 응답이 성공적이면 JSON 데이터 파싱 시도
        # 파싱 오류는 여기서 잡힙니다 (json.JSONDecodeError)
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
        # 서버에서 4xx 또는 5xx 응답 코드를 보낸 경우
        print(f"❌ HTTP error occurred: {http_err} (Status code: {response.status_code})")
        # 서버 에러 메시지 출력 시도 (서버가 JSON 에러 메시지를 보냈다면 여기서 파싱될 수 있음)
        try:
            error_details = response.json()
            print(f"   Server error details: {error_details}")
        except json.JSONDecodeError:
            # 서버가 JSON이 아닌 응답(예: HTML 오류 페이지)을 보낸 경우
            print(f"   Could not parse server error response body (it might be HTML or plain text):")
            print(f"   Response text (first 500 chars): {response.text[:500]}...")
        return None
    except json.JSONDecodeError:
        # 서버가 200 OK 응답을 보냈지만, 본문 내용이 유효한 JSON이 아닌 경우
        print(f"❌ Error: Failed to decode JSON response from the server (Status code was OK, but content is not valid JSON).")
        print(f"   Response content (first 500 chars): {response.text[:500]}...") # 받은 내용 일부 출력
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ An unexpected error occurred during the request: {e}")
        return None

# === 데이터 처리 함수 (이전과 동일) ===
def extract_points_from_data(lidar_data_list):
    """서버에서 받은 LiDAR 데이터 리스트에서 'isDetected'가 True인 포인트들의 3D 월드 좌표 (x, z, y)를 추출하여 NumPy 배열로 반환합니다."""
    points_3d = []
    if not isinstance(lidar_data_list, list):
        print("⚠️ 경고: 입력 데이터가 리스트 형식이 아닙니다.")
        return np.empty((0, 3))

    detected_count = 0
    processed_count = 0
    for point_data in lidar_data_list:
        if (isinstance(point_data, dict) and
                'position' in point_data and
                isinstance(point_data['position'], dict) and
                all(k in point_data['position'] for k in ('x', 'y', 'z')) and
                'isDetected' in point_data): # isDetected 키 존재 확인
            processed_count += 1
            # isDetected가 True인 포인트만 사용
            if point_data['isDetected'] is True: # 명시적으로 True와 비교
                try:
                    x = float(point_data['position']['x'])
                    y = float(point_data['position']['y']) # 실제 높이값
                    z = float(point_data['position']['z'])
                    # Matplotlib 플롯팅 순서 (X, Z, Y(높이)) 에 맞춰 저장
                    points_3d.append([x, z, y])
                    detected_count += 1
                except (ValueError, TypeError) as e:
                     print(f"⚠️ 경고: 좌표 값 변환 중 오류 발생 (데이터 건너뜀): {e} - 데이터: {point_data['position']}")
        #else:
        #    # 필요하다면 형식 오류 로그 활성화
        #    print(f"⚠️ 경고: 필요한 키가 없거나 형식이 잘못된 데이터 포인트 (건너뜀): {point_data}")

    print(f"ℹ️ 처리된 총 포인트 수: {processed_count}")
    if processed_count > 0:
        print(f"ℹ️ 'isDetected'=True 인 포인트 수 (시각화/클러스터링 대상): {detected_count}")
    #else:
    #    print(f"ℹ️ 유효한 포인트 데이터가 없습니다.") # 데이터 없을 때 메시지는 메인에서 처리

    return np.array(points_3d) if points_3d else np.empty((0, 3))


# === 3D 경계 상자 그리기 함수 (이전과 동일) ===
def draw_cuboid(ax, min_coords, max_coords, color='r', alpha=0.1, linewidth=1, label=None):
    """ Matplotlib 3D 축(ax)에 주어진 최소/최대 좌표로 정의되는 직육면체를 그립니다. 좌표 순서는 (X, Z, Y(높이)) 를 가정합니다. """
    min_x, min_z, min_y = min_coords; max_x, max_z, max_y = max_coords
    vertices = np.array([
        (min_x, min_z, min_y), (max_x, min_z, min_y), (max_x, max_z, min_y), (min_x, max_z, min_y),
        (min_x, min_z, max_y), (max_x, min_z, max_y), (max_x, max_z, max_y), (min_x, max_z, max_y)
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    plotted_label = False
    for i, edge in enumerate(edges):
        p1_idx, p2_idx = edge
        xs, zs, ys = vertices[[p1_idx, p2_idx], 0], vertices[[p1_idx, p2_idx], 1], vertices[[p1_idx, p2_idx], 2]
        current_label = label if not plotted_label else None
        ax.plot(xs, zs, ys, color=color, alpha=max(alpha * 3, 0.5), linewidth=linewidth, label=current_label)
        if current_label: plotted_label = True

# === 메인 시각화 및 클러스터링 함수 (Y축 시각화 조정 버전) ===
def visualize_and_cluster(points_3d):
    """
    주어진 3D 포인트 클라우드를 시각화하고, DBSCAN 클러스터링을 수행하여
    결과를 3D 경계 상자와 함께 표시합니다.
    **시각화 시 Y(높이) 좌표에서 10을 빼서 표시합니다.**
    points_3d: (N, 3) 형태의 NumPy 배열 [X, Z, Y(원본 높이)]
    """
    print("📊 3D 포인트 클라우드 시각화 및 DBSCAN 클러스터링 시작 (Y축 시각화 조정됨)...")

    if points_3d.shape[0] == 0:
        print("⚠️ 시각화할 포인트 데이터가 없습니다.")
        return

    print(f" - 입력 포인트 개수: {points_3d.shape[0]}")

    # --- Matplotlib Figure 및 3D 서브플롯 생성 ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. 원본 포인트 클라우드 시각화 (Y좌표 조정) ---
    # 시각화를 위한 조정된 Y 좌표 계산 (원본 Y - 10)
    adjusted_y_coords = points_3d[:, 2] - 10

    # scatter 플롯: X, Z 좌표는 원본 사용, Y(높이) 좌표는 조정된 값 사용
    # 점 색상은 원본 Y(높이) 값 기준
    scatter = ax.scatter(points_3d[:, 0],  # 원본 X
                         points_3d[:, 1],  # 원본 Z
                         adjusted_y_coords, # 조정된 Y (시각적 높이)
                         s=2,  # 점 크기
                         c=points_3d[:, 2], # 색상 기준: 원본 Y(높이)
                         cmap='viridis',
                         alpha=0.5, # 투명도
                         label='LiDAR Points (Y adjusted for vis)') # 범례 텍스트 수정
    print(" - 포인트 클라우드 플롯 완료 (시각적 Y축 -10 조정됨).")

    # --- 2. DBSCAN 클러스터링 수행 (원본 좌표 사용) ---
    detected_obstacle_count = 0
    if points_3d.shape[0] >= DBSCAN_MIN_SAMPLES:
        try:
            print(f" - DBSCAN 클러스터링 실행 (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
            t_start = time.time()
            # DBSCAN은 원본 좌표(points_3d)로 수행해야 공간적 의미가 유지됨
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points_3d)
            t_end = time.time()
            labels = db.labels_
            unique_labels = set(labels)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            num_noise = np.sum(labels == -1)
            print(f" - DBSCAN 완료 ({t_end - t_start:.2f}초). {num_clusters}개의 클러스터 발견 (노이즈 포인트: {num_noise}개).")

            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            bbox_label_printed = False
            # 고유한 클러스터 레이블 각각에 대해 반복
            for k, col in zip(unique_labels, colors):
                if k == -1: continue # 노이즈 제외

                cluster_mask = (labels == k)
                # 클러스터에 속하는 원본 포인트들
                cluster_points = points_3d[cluster_mask]

                if cluster_points.shape[0] > 0:
                    # 경계 상자는 원본 좌표 기준으로 계산
                    min_coords_orig = np.min(cluster_points, axis=0) # (min_x, min_z, min_y_orig)
                    max_coords_orig = np.max(cluster_points, axis=0) # (max_x, max_z, max_y_orig)

                    # 시각화를 위한 경계 상자 좌표 조정 (Y 좌표만 -10)
                    min_coords_vis = (min_coords_orig[0], min_coords_orig[1], min_coords_orig[2] - 10)
                    max_coords_vis = (max_coords_orig[0], max_coords_orig[1], max_coords_orig[2] - 10)

                    # 조정된 좌표로 3D Bounding Box 그리기
                    box_label = 'Detected Object (BBox)' if not bbox_label_printed else None
                    draw_cuboid(ax, min_coords_vis, max_coords_vis, color=col, alpha=0.15, linewidth=1.5, label=box_label)
                    if not bbox_label_printed: bbox_label_printed = True

                    detected_obstacle_count += 1

            if detected_obstacle_count > 0:
                print(f' - 감지된 객체 플롯: {detected_obstacle_count}개의 3D 경계 상자 그림 (시각적 Y축 -10 조정됨).')

        except Exception as e:
            print(f"❌ 오류: DBSCAN 클러스터링 또는 3D Box 플로팅 중 오류 발생: {e}")
            ax.text2D(0.05, 0.95, 'DBSCAN/Plotting Error', transform=ax.transAxes, color='red')

    else:
        print("ℹ️ DBSCAN을 위한 포인트 개수가 부족합니다.")
        ax.text2D(0.05, 0.95, 'Not Enough Points for DBSCAN', transform=ax.transAxes, color='gray')

    # --- 플롯 설정 (Z축 라벨 및 범위 조정) ---
    ax.set_title(f"LiDAR Point Cloud & DBSCAN Clustering ({detected_obstacle_count} Objects Detected)")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Z (m)") # Matplotlib 3D 기준 Y축
    # Z축(높이) 라벨에 조정 정보 명시
    ax.set_zlabel("Adjusted World Y (Height - 10, m)")

    # 축 범위 설정 (조정된 Y 좌표 기준)
    if points_3d.shape[0] > 0:
        # Y(높이) 축 범위는 adjusted_y_coords 기준으로 설정
        y_min_adj, y_max_adj = np.min(adjusted_y_coords), np.max(adjusted_y_coords)
        ax.set_zlim(y_min_adj - 1, max(y_max_adj + 1, PLOT_HEIGHT_LIMIT - 10)) # 조정된 Y 최소/최대값 사용
        # X, Z 축은 원본 데이터 기준
        x_min, x_max = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
        z_min, z_max = np.min(points_3d[:, 1]), np.max(points_3d[:, 1])
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(z_min - 1, z_max + 1)
    else:
         # 기본값도 Y축 조정 고려
         ax.set_xlim(-PLOT_AREA_LIMIT/2, PLOT_AREA_LIMIT/2)
         ax.set_ylim(-PLOT_AREA_LIMIT/2, PLOT_AREA_LIMIT/2)
         ax.set_zlim(-10, PLOT_HEIGHT_LIMIT - 10) # 기본 높이 범위도 조정

    ax.view_init(elev=25., azim=-75) # 초기 3D 뷰 각도
    ax.grid(True)
    ax.legend(loc='upper left', fontsize='small')

    # 컬러바 추가 (원본 높이 정보 표시)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    # 컬러바 라벨에 원본 높이임을 명시
    cbar.set_label('Original Height (Y)')

    plt.tight_layout()
    print("✅ 플롯 생성 완료 (Y축 -10 조정됨). 창을 닫으면 프로그램이 종료됩니다.")
    plt.show() # Matplotlib 창 표시

# === 메인 실행 블록 ===
if __name__ == '__main__':
    print("--- LiDAR 데이터 처리 및 시각화 시작 ---")

    # 1. 서버에서 LiDAR 데이터 가져오기 (제공해주신 함수 사용)
    lidar_response = fetch_lidar_data(LIDAR_DATA_URL)

    # 2. 데이터 확인 및 처리
    # fetch_lidar_data 함수가 None을 반환하거나, 반환된 딕셔너리에 'lidar_data' 키가 없는 경우 실패로 간주
    if lidar_response and isinstance(lidar_response, dict) and 'lidar_data' in lidar_response:
        raw_lidar_points = lidar_response['lidar_data']
        # lidar_data 값이 리스트인지 한번 더 확인
        if isinstance(raw_lidar_points, list):
            print(f" - 서버로부터 받은 총 데이터 포인트 수: {len(raw_lidar_points)}")

            # 3. 필요한 3D 좌표 추출 및 필터링
            points_for_processing = extract_points_from_data(raw_lidar_points)

            # 4. 추출된 포인트로 시각화 및 클러스터링 수행
            if points_for_processing.shape[0] > 0:
                 visualize_and_cluster(points_for_processing)
            else:
                 # extract_points_from_data 함수 내에서 이미 로그 출력됨
                 print("ℹ️ 처리할 유효한 LiDAR 포인트가 없습니다. ('isDetected'=True 인 포인트 없음).")

        else:
             print(f"❌ 오류: 서버 응답의 'lidar_data' 키 값이 리스트가 아닙니다 (타입: {type(raw_lidar_points)}).")
             print(f"   받은 값 일부: {str(raw_lidar_points)[:500]}...")

    else:
        # fetch_lidar_data 함수 내에서 이미 상세 오류 메시지가 출력되었을 것임
        print("❌ 서버로부터 LiDAR 데이터를 가져오지 못했거나 형식이 잘못되었습니다. 위 에러 메시지를 확인하세요.")

    print("--- 프로그램 종료 ---")