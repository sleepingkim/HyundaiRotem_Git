import sys
import os
import time # time 모듈 추가

# (Ursina 경로 설정 코드는 필요시 유지)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# ursina_path = os.path.join(current_dir, '..')
# sys.path.append(ursina_path)

try:
    from ursina import *
    # --- Required for Clustering ---
    from sklearn.cluster import DBSCAN
    import numpy as np # NumPy 사용 필수
except ImportError as e:
    print(f"Error importing Ursina or Scikit-learn/NumPy: {e}")
    print("Make sure Ursina, Scikit-learn, NumPy are installed (`pip install ursina scikit-learn numpy pandas matplotlib`)")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Required Libraries ---
import random
import math
# import numpy as np # Already imported above
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For drawing obstacles
import pandas as pd # For DataFrame storage

# --- Simulation Parameters ---
NUM_OBSTACLES = 100 # 기본 장애물 수
AREA_SIZE = 60      # 시뮬레이션 영역 크기
AGENT_SPEED = 5     # 에이전트 이동 속도
ROTATION_SPEED = 100 # 에이전트 회전 속도
AGENT_HEIGHT = 0.5  # 에이전트 높이
LIDAR_RANGE = 15    # 라이다 최대 측정 거리
LIDAR_FOV = 150     # 라이다 측정 각도 범위 (Field of View)
NUM_LIDAR_RAYS = 90 # 라이다 빔(Ray) 개수
LIDAR_VIS_HEIGHT = 0.6 # 라이다 시각화 높이 오프셋
LIDAR_COLOR = color.cyan # 라이다 빔 기본 색상
OBSTACLE_TAG = "obstacle" # 장애물 엔티티 태그
SCAN_INTERVAL = 1.0 # 스캔 및 데이터 기록 간격 (초)

# --- Clustering Parameters ---
DBSCAN_EPS = 0.5  # DBSCAN epsilon (같은 클러스터로 간주할 최대 거리) - 필요시 조정
DBSCAN_MIN_SAMPLES = 10 # DBSCAN min_samples (클러스터 핵심점이 되기 위한 최소 주변 샘플 수) - 필요시 조정


# --- Application Setup ---
app = Ursina(
    title="LiDAR Mapping & Detected Obstacles",
    borderless=False,
)

# --- Mouse Setup ---
mouse.visible = True
mouse.locked = False

# --- Global Variables ---
scan_timer = 0.0
# scan_history: {'pose': (x, z, rot_rad), 'relative_points': list of [rx, rz]}
scan_history = []
obstacle_df = None          # 장애물 실제 정보 DataFrame
ground_info = {'size': AREA_SIZE, 'center_x': 0, 'center_z': 0} # 지면 정보

# --- Environment Setup (Ground, Sky, Obstacles) ---
ground = Entity(model='plane', scale=(AREA_SIZE, 1, AREA_SIZE), color=color.light_gray,
                texture='white_cube', texture_scale=(AREA_SIZE/2, AREA_SIZE/2), collider='box')
sky = Sky()
obstacles_list_for_df = [] # DataFrame 생성용 임시 리스트
obstacles = []             # Ursina 장애물 Entity 리스트
min_obstacle_distance_from_spawn = 3.0 # 스폰 지점과 장애물 최소 거리

# 장애물 생성 로직
for i in range(NUM_OBSTACLES):
    while True:
        pos_x = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5)
        pos_z = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5)
        distance_from_spawn = math.sqrt(pos_x**2 + pos_z**2)
        if distance_from_spawn >= min_obstacle_distance_from_spawn:
            break # 스폰 지점에서 충분히 떨어졌으면 위치 확정
    # 장애물 크기 및 색상 랜덤 설정
    scale_x = random.uniform(0.5, 3)
    scale_y = random.uniform(1, 4) # 높이는 클러스터링에 직접 영향 없음
    scale_z = random.uniform(0.5, 3)
    pos_y = scale_y / 2 # 지면 위에 놓이도록 y 위치 조정
    obstacle_color = color.random_color()

    # Ursina Entity 생성
    obs_entity = Entity(model='cube', position=(pos_x, pos_y, pos_z), scale=(scale_x, scale_y, scale_z),
                        color=obstacle_color, collider='box', tag=OBSTACLE_TAG, name=f"obstacle_{i}")
    obstacles.append(obs_entity)

    # DataFrame 에 저장할 정보 추가
    obstacles_list_for_df.append({
        'name': f"obstacle_{i}",
        'center_x': pos_x,
        'center_z': pos_z,
        'width': scale_x,
        'depth': scale_z,
        'height': scale_y,
        'rotation_y': 0 # 초기 장애물은 축 정렬 상태로 가정
    })

# --- Obstacle DataFrame 생성 ---
obstacle_df = pd.DataFrame(obstacles_list_for_df)
print(f"Created DataFrame with {len(obstacle_df)} obstacles.")

# --- Agent Setup ---
agent = Entity(model='sphere', color=color.blue, position=(0, AGENT_HEIGHT, 0),
               collider='sphere', scale=1) # 에이전트 Entity 생성

# --- Camera Setup ---
camera.parent = agent # 카메라를 에이전트에 부착 (따라다니도록)
camera.position = (0, 10, -12) # 에이전트 기준 카메라 위치
camera.rotation_x = 45 # 카메라 기울기
camera.rotation_y = 0
camera.fov = 75 # 카메라 시야각

# --- LiDAR Visualization (Lines in Ursina) ---
lidar_lines = [] # 시각화용 라인 Entity 저장 리스트
def update_lidar_visualization():
    """ 현재 라이다 스캔 결과를 Ursina 씬에 선으로 그림 """
    global lidar_lines
    # 이전 라인 제거
    for line in lidar_lines: destroy(line)
    lidar_lines.clear()

    # 라이다 파라미터 기반 계산
    start_angle = agent.world_rotation_y - LIDAR_FOV / 2
    angle_step = LIDAR_FOV / (NUM_LIDAR_RAYS - 1) if NUM_LIDAR_RAYS > 1 else 0
    origin = agent.world_position + Vec3(0, LIDAR_VIS_HEIGHT - AGENT_HEIGHT, 0) # 빔 시작점

    # 각 Ray 에 대해 Raycasting 수행
    for i in range(NUM_LIDAR_RAYS):
        current_angle_deg = start_angle + i * angle_step
        current_angle_rad = math.radians(current_angle_deg)
        # 빔 방향 계산 (XZ 평면)
        direction = Vec3(math.sin(current_angle_rad), 0, math.cos(current_angle_rad)).normalized()
        # Raycast 실행
        hit_info = raycast(origin, direction, distance=LIDAR_RANGE, ignore=[agent,], debug=False, traverse_target=scene) # traverse_target 추가

        if hit_info.hit: # 충돌 시
            end_point = hit_info.world_point # 충돌 지점
            # 충돌 객체가 Ground 가 아니면 빨간색, Ground 면 기본 색상
            line_color = color.red if hit_info.entity != ground else LIDAR_COLOR
        else: # 미충돌 시 (최대 거리 도달)
            end_point = origin + direction * LIDAR_RANGE # 최대 거리 지점
            line_color = LIDAR_COLOR

        # 매우 짧은 라인 제외하고 Entity 생성
        if distance(origin, end_point) > 0.01:
            line = Entity(model=Mesh(vertices=[origin, end_point], mode='line', thickness=2), color=line_color)
            lidar_lines.append(line)


# --- Function to Perform Scan and Generate Relative Map Data ---
def generate_relative_lidar_map():
    """ 에이전트 현재 자세 기준, 상대 좌표 라이다 스캔 데이터 생성 """
    relative_points = [] # 상대 좌표 (x, z) 저장 리스트
    agent_pos_world = agent.world_position # 현재 에이전트 월드 위치
    agent_rot_y_rad = math.radians(agent.world_rotation_y) # 현재 에이전트 월드 회전 (라디안)
    scan_origin = agent_pos_world + Vec3(0, LIDAR_VIS_HEIGHT - AGENT_HEIGHT, 0) # 스캔 시작점

    # 각도 계산 파라미터
    angle_step_rad = math.radians(LIDAR_FOV / (NUM_LIDAR_RAYS - 1)) if NUM_LIDAR_RAYS > 1 else 0
    start_angle_world_rad = math.radians(agent.world_rotation_y - LIDAR_FOV / 2) # 시작 각도 (월드 기준)

    # 각 Ray 에 대해 Raycasting 수행
    for i in range(NUM_LIDAR_RAYS):
        ray_angle_world_rad = start_angle_world_rad + i * angle_step_rad
        ray_direction_world = Vec3(math.sin(ray_angle_world_rad), 0, math.cos(ray_angle_world_rad)).normalized()
        # Raycast 실행 (에이전트 제외)
        hit_info = raycast(scan_origin, ray_direction_world, distance=LIDAR_RANGE, ignore=[agent,], debug=False, traverse_target=scene)

        # 장애물(Ground 제외)에 충돌한 경우만 처리
        if hit_info.hit and hit_info.entity != ground:
            hit_point_world = hit_info.world_point # 월드 충돌 지점
            # 월드 충돌 지점 -> 에이전트 기준 상대 좌표로 변환
            world_vec = hit_point_world - agent_pos_world # 에이전트 위치 기준 벡터
            world_vec_xz = Vec2(world_vec.x, world_vec.z) # XZ 평면만 고려
            # 2D 회전 변환 (World -> Agent 좌표계)
            cos_a = math.cos(-agent_rot_y_rad); sin_a = math.sin(-agent_rot_y_rad)
            relative_x = world_vec_xz.x * cos_a - world_vec_xz.y * sin_a
            relative_z = world_vec_xz.x * sin_a + world_vec_xz.y * cos_a # 이전 코드 y -> z 로 수정
            relative_points.append([relative_x, relative_z]) # 리스트로 추가

    # 결과를 NumPy 배열로 반환 (데이터 처리 용이)
    return np.array(relative_points) if relative_points else np.empty((0, 2))

# --- Function to Build and Visualize GLOBAL LiDAR Map (Points) ---
def plot_global_lidar_map(history):
    """ 누적된 scan_history 로부터 전역 점 구름 지도 시각화 """
    print(f"Building global LiDAR map from {len(history)} scans...")
    global_map_points_x, global_map_points_z = [], []
    agent_trajectory_x, agent_trajectory_z = [], []
    if not history: print("No scan history to plot."); return

    # 기록된 각 스캔 데이터 처리
    for scan_record in history:
        pose = scan_record['pose'] # 기록된 (GT) 자세
        relative_points = scan_record['relative_points'] # 기록된 상대 좌표 점들
        # 데이터 로딩 시 리스트일 수 있으므로 NumPy 배열로 변환
        if not isinstance(relative_points, np.ndarray):
             relative_points = np.array(relative_points)
        # 빈 스캔 데이터 건너뛰기
        if relative_points.shape[0] == 0: continue

        scan_pos_x, scan_pos_z, scan_rot_rad = pose
        # 에이전트 궤적 기록
        agent_trajectory_x.append(scan_pos_x); agent_trajectory_z.append(scan_pos_z)
        # 상대 좌표 -> 월드 좌표 변환
        cos_a, sin_a = math.cos(scan_rot_rad), math.sin(scan_rot_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # NumPy 연산으로 모든 점 한 번에 변환
        world_points = np.dot(relative_points, rot_matrix.T) + np.array([[scan_pos_x, scan_pos_z]])
        # 플로팅을 위해 X, Z 좌표 분리 저장
        global_map_points_x.extend(world_points[:, 0].tolist())
        global_map_points_z.extend(world_points[:, 1].tolist())

    print(f"Total global points accumulated: {len(global_map_points_x)}")
    # Matplotlib 사용하여 플롯 생성
    fig_lidar, ax_lidar = plt.subplots(figsize=(10, 10))
    if global_map_points_x: # 점 데이터가 있을 경우 Scatter 플롯
        ax_lidar.scatter(global_map_points_x, global_map_points_z, s=2, c='blue', label='Map Points (World)')
    if agent_trajectory_x: # 궤적 데이터가 있을 경우 Line 플롯
        ax_lidar.plot(agent_trajectory_x, agent_trajectory_z, marker='o', markersize=3, linestyle='-', color='red', label='Agent Trajectory (GT)')
        ax_lidar.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], s=50, c='magenta', marker='*', label='Last Scan Pose (GT)') # 마지막 위치 표시
    # 플롯 설정
    ax_lidar.set_title("Accumulated LiDAR Map (World Coordinates)"); ax_lidar.set_xlabel("World X"); ax_lidar.set_ylabel("World Z")
    ax_lidar.grid(True, linestyle='--', alpha=0.6); ax_lidar.set_aspect('equal', adjustable='box'); ax_lidar.legend()
    plt.show(block=True) # 창을 닫을 때까지 코드 실행 중지

# --- Function to Visualize FULL Ground Truth Map ---
def plot_full_map(obs_df, gnd_info):
    """ 실제 장애물 정보(Ground Truth)를 Matplotlib으로 시각화 """
    print("Building ground truth map...")
    if obs_df is None or obs_df.empty: print("No obstacle data available."); return

    fig_full, ax_full = plt.subplots(figsize=(10, 10))
    # 지면 영역 표시
    gnd_size = gnd_info['size']
    ax_full.add_patch(patches.Rectangle((-gnd_size/2, -gnd_size/2), gnd_size, gnd_size,
                                        edgecolor='gray', facecolor='none', linestyle='--', label='Ground Area'))
    # 저장된 DataFrame 에서 장애물 정보 읽어와 사각형 그리기
    for index, row in obs_df.iterrows():
        x, z, w, d, rot = row['center_x'], row['center_z'], row['width'], row['depth'], row['rotation_y']
        bottom_left_x = x - w / 2; bottom_left_z = z - d / 2
        # 사각형 패치 추가 (회전 각도 적용 가능)
        ax_full.add_patch(patches.Rectangle((bottom_left_x, bottom_left_z), w, d,
                                            edgecolor='black', facecolor='darkgray', angle=rot))
    # 에이전트 시작 지점 표시
    ax_full.scatter(0, 0, s=100, c='red', marker='x', label='Agent Start (0, 0)')
    # 플롯 설정
    ax_full.set_title("Ground Truth Map (Obstacle Layout)"); ax_full.set_xlabel("World X"); ax_full.set_ylabel("World Z")
    ax_full.grid(True, linestyle='--', alpha=0.6); ax_full.set_aspect('equal', adjustable='box'); ax_full.legend()
    plt.show(block=True) # 창 닫을 때까지 중지

# --- *** NEW FUNCTION for Detected Obstacle Plot *** ---
def plot_detected_obstacles(history):
    """ 누적된 LiDAR 포인트들을 클러스터링하여 감지된 장애물을 사각형으로 시각화 """
    print(f"Detecting obstacles from {len(history)} scans using DBSCAN...")
    global_map_points = [] # 모든 월드 좌표 점들을 저장할 리스트
    if not history: print("No scan history to detect obstacles."); return

    # 1. 모든 스캔 기록으로부터 전역 지도 점(Point Cloud) 생성
    for scan_record in history:
        pose = scan_record['pose'] # 스캔 당시의 (GT) 자세
        relative_points = scan_record['relative_points'] # 해당 스캔의 상대 좌표 점들
        # Load 시 list 일 수 있으므로 array 변환
        if not isinstance(relative_points, np.ndarray):
            relative_points = np.array(relative_points)
        # 빈 스캔 건너뛰기
        if relative_points.shape[0] == 0: continue

        scan_pos_x, scan_pos_z, scan_rot_rad = pose
        # 상대 좌표 -> 월드 좌표 변환 (NumPy 사용)
        cos_a, sin_a = math.cos(scan_rot_rad), math.sin(scan_rot_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        world_points = np.dot(relative_points, rot_matrix.T) + np.array([[scan_pos_x, scan_pos_z]])
        global_map_points.append(world_points) # 변환된 월드 좌표 점들 추가

    # 모든 점들을 하나의 NumPy 배열로 합치기
    if not global_map_points:
        print("No valid map points generated.")
        return
    map_points_np = np.concatenate(global_map_points, axis=0)
    print(f"Total global points for clustering: {len(map_points_np)}")

    # 클러스터링을 위한 최소 포인트 수 확인
    if len(map_points_np) < DBSCAN_MIN_SAMPLES:
        print("Not enough points for DBSCAN clustering.")
        return

    # 2. DBSCAN 클러스터링 수행
    try:
        # DBSCAN 객체 생성 및 학습/예측
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(map_points_np)
        labels = db.labels_ # 각 포인트에 할당된 클러스터 레이블 (-1은 노이즈)
        # 유니크한 레이블 찾기 (노이즈 제외)
        unique_labels = set(labels)
        n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0) # 노이즈 제외 클러스터 수
        n_noise_ = list(labels).count(-1) # 노이즈 포인트 수
        print(f'Estimated number of clusters: {n_clusters_}')
        print(f'Estimated number of noise points: {n_noise_}')
    except Exception as e:
        print(f"Error during DBSCAN: {e}")
        return # 오류 발생 시 종료

    # 3. 클러스터링 결과 시각화
    fig, ax = plt.subplots(figsize=(10, 10))

    # 배경으로 모든 점 표시 (선택 사항)
    # ax.scatter(map_points_np[:, 0], map_points_np[:, 1], s=1, c='lightgray', alpha=0.3, label='_nolegend_')

    # 각 클러스터(감지된 장애물)를 사각형으로 표시
    # 클러스터별 색상 지정을 위한 컬러맵 사용
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    detected_obstacle_count = 0
    # 각 유니크 레이블(클러스터 ID)에 대해 반복
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 레이블 -1은 노이즈 포인트 -> 여기서는 무시 (필요시 다르게 표시 가능)
            continue

        # 현재 클러스터(k)에 속하는 포인트들만 선택
        cluster_mask = (labels == k)
        cluster_points = map_points_np[cluster_mask]

        # 클러스터에 포인트가 존재하면
        if len(cluster_points) > 0:
            # 해당 클러스터의 축 정렬 경계 상자(AABB) 계산
            min_x, min_z = np.min(cluster_points, axis=0) # 최소 x, z 좌표
            max_x, max_z = np.max(cluster_points, axis=0) # 최대 x, z 좌표
            width = max_x - min_x # 경계 상자 너비
            height = max_z - min_z # 경계 상자 높이 (Z축 방향)

            # Matplotlib 사각형 패치 생성 및 추가
            rect = patches.Rectangle((min_x, min_z), width, height, linewidth=1, edgecolor=col, facecolor=col, alpha=0.5)
            ax.add_patch(rect)
            detected_obstacle_count += 1 # 감지된 장애물 수 카운트

    # 플롯 설정
    ax.set_title(f"Detected Obstacles ({detected_obstacle_count} clusters found via DBSCAN)")
    ax.set_xlabel("World X")
    ax.set_ylabel("World Z")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') # X, Y 축 비율 동일하게
    # 플롯 범위 설정 (시뮬레이션 영역 기준)
    ax.set_xlim(-AREA_SIZE / 2, AREA_SIZE / 2)
    ax.set_ylim(-AREA_SIZE / 2, AREA_SIZE / 2)
    # ax.legend() # 범례는 필요 없을 수 있음

    plt.show(block=True) # 창 닫을 때까지 중지


# --- UI Display ---
info_text = Text(origin=(0.5, 0.5), # 기준점 (화면 중앙=0,0 / 왼쪽 위=-0.5,0.5)
                 scale=(0.8, 0.8),   # 텍스트 크기
                 x=0.5 * window.aspect_ratio - 0.02, # x 위치 (오른쪽 끝)
                 y=0.48,            # y 위치 (상단)
                 text="Initializing...")

# --- Input Handling ---
def input(key):
    """ 키보드 입력 처리 """
    global scan_history, obstacle_df # 전역 변수 사용 선언

    # 'M' 키: 누적된 LiDAR 점 지도 플롯
    if key == 'm' or key == 'M':
        print("Plotting global LiDAR map...")
        plot_global_lidar_map(scan_history)

    # 'N' 키: 실제 장애물(Ground Truth) 지도 플롯
    if key == 'n' or key == 'N':
        print("Plotting full ground truth map...")
        if obstacle_df is not None:
             plot_full_map(obstacle_df, ground_info)
        else:
             print("Obstacle DataFrame not ready.")

    # 'C' 키: LiDAR 스캔 기록 삭제
    if key == 'c' or key == 'C':
        print("Clearing LiDAR map history.")
        scan_history = []

    # *** NEW 'O' Key *** : 감지된 장애물(클러스터) 플롯
    if key == 'o' or key == 'O':
        print("Plotting detected obstacles based on LiDAR clusters...")
        plot_detected_obstacles(scan_history)

    # 'P' 키: 데이터 저장 (선택 사항)
    if key == 'p' or key == 'P':
        print("Saving data...")
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # 스캔 기록 저장 (Pickle 형식 권장)
            if scan_history:
                # 주의: scan_history 내 relative_points가 list 형태여야 함 (NumPy array는 pickle 시 문제 가능성)
                history_df_to_save = pd.DataFrame(scan_history) # DataFrame 변환 시도 (복잡 구조면 어려울 수 있음)
                filename_hist = f"scan_history_{timestamp}.pkl"
                # history_df_to_save.to_pickle(filename_hist) # DataFrame 저장
                # 또는 리스트 자체를 pickle 로 저장
                import pickle
                with open(filename_hist, 'wb') as f:
                    pickle.dump(scan_history, f)
                print(f"Scan history saved to {filename_hist}")
            else: print("No scan history to save.")
            # 장애물 정보 저장 (CSV)
            if obstacle_df is not None:
                filename_obs = f"obstacles_{timestamp}.csv"
                obstacle_df.to_csv(filename_obs, index=False)
                print(f"Obstacle data saved to {filename_obs}")
            else: print("No obstacle data to save.")
        except Exception as e: print(f"Error saving data: {e}")


    # 'ESC' 키: 시뮬레이션 종료
    if key == 'escape':
        print("Exiting simulation...")
        quit()

# --- Main Update Loop ---
def update():
    """ 매 프레임 호출되는 메인 업데이트 함수 """
    global scan_timer, scan_history # 전역 변수 사용

    # --- Agent Control ---
    # 에이전트 이동 및 회전 처리, 충돌 감지
    original_position = agent.position # 이동 전 위치 저장
    total_delta_x, total_delta_z = 0.0, 0.0 # 프레임 당 이동량 초기화
    speed_dt = AGENT_SPEED * time.dt # 프레임 시간 적용 속도
    # 키 입력에 따른 이동량 계산
    if held_keys['w']: total_delta_x += agent.forward.x * speed_dt; total_delta_z += agent.forward.z * speed_dt
    if held_keys['s']: total_delta_x -= agent.forward.x * speed_dt; total_delta_z -= agent.forward.z * speed_dt
    if held_keys['a']: total_delta_x -= agent.right.x * speed_dt; total_delta_z -= agent.right.z * speed_dt
    if held_keys['d']: total_delta_x += agent.right.x * speed_dt; total_delta_z += agent.right.z * speed_dt
    # X축 이동 및 충돌 처리
    agent.x += total_delta_x
    hit_info_x = agent.intersects(traverse_target=scene) # 씬 전체와 충돌 검사
    if hit_info_x.hit and hit_info_x.entity in obstacles: # 장애물과 충돌 시
        agent.x = original_position.x # X축 이동 취소
    # Z축 이동 및 충돌 처리
    agent.z += total_delta_z
    hit_info_z = agent.intersects(traverse_target=scene)
    if hit_info_z.hit and hit_info_z.entity in obstacles: # 장애물과 충돌 시
        agent.z = original_position.z # Z축 이동 취소
    # Y축 고정 (점프 등 방지)
    agent.y = AGENT_HEIGHT
    # 회전 처리
    if held_keys['q']: agent.rotation_y -= ROTATION_SPEED * time.dt
    if held_keys['e']: agent.rotation_y += ROTATION_SPEED * time.dt

    # --- Periodic LiDAR Scan for Mapping History ---
    scan_timer += time.dt # 타이머 증가
    if scan_timer >= SCAN_INTERVAL: # 설정된 간격 도달 시
        scan_timer -= SCAN_INTERVAL # 타이머 리셋 (정확히 간격 유지 위함)

        # 현재 에이전트의 실제 (Ground Truth) 자세 얻기
        current_pos_xz = agent.world_position.xz # x, z 좌표
        current_rot_rad = math.radians(agent.world_rotation_y) # y축 회전 (라디안)
        # (x, z, 회전각) 튜플로 저장
        current_pose = (current_pos_xz.x, current_pos_xz.y, current_rot_rad)
        # 현재 위치에서 라이다 스캔 수행
        relative_points = generate_relative_lidar_map() # NumPy array (N, 2) 반환

        # 스캔된 포인트가 있을 경우 기록
        if relative_points.shape[0] > 0:
             # scan_history 에는 리스트 형태로 저장 (Pickle 호환성 고려)
             scan_record = {'pose': current_pose, 'relative_points': relative_points.tolist()}
             scan_history.append(scan_record)

    # --- LiDAR Visualization Update (Ursina Lines) ---
    update_lidar_visualization() # 매 프레임 라이다 선 업데이트

    # --- UI Update ---
    # 화면 좌측 상단에 표시될 정보 업데이트
    pos_str = f"Pos: ({agent.x:.1f}, {agent.z:.1f})" # 현재 에이전트 위치
    rot_str = f"Rot (Y): {agent.rotation_y:.0f}°" # 현재 에이전트 회전 (도)
    # 키 설명 업데이트
    map_info = f"Scans: {len(scan_history)} | 'M': Map | 'N': GT | 'O': Detect | 'C': Clear | 'P': Save"
    # UI 텍스트 설정
    info_text.text = f"{pos_str}\n{rot_str}\n{map_info}"
    # UI 위치 조정 (창 크기 변경 대응)
    info_text.x = -0.5 * window.aspect_ratio + 0.02 # 왼쪽 끝으로 변경
    info_text.y = 0.48                          # 상단

    # --- Mouse state ---
    # 마우스 커서 보이도록 유지
    if mouse.locked: mouse.locked = False
    if not mouse.visible: mouse.visible = True

# --- Start Simulation ---
if __name__ == '__main__':
    # Ursina 애플리케이션 실행 (update 함수가 매 프레임 호출됨)
    app.run()