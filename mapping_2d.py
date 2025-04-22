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
    # 현재 시간 추가하여 오류 메시지 개선 (2025-04-18 기준)
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] Error importing Ursina or Scikit-learn/NumPy: {e}")
    print(f"[{t_now}] Make sure Ursina, Scikit-learn, NumPy, Pandas, Matplotlib are installed.")
    print(f"[{t_now}] Try: pip install ursina scikit-learn numpy pandas matplotlib")
    sys.exit(1)
except Exception as e:
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] An unexpected error occurred during import: {e}")
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
    title="LiDAR Mapping Analysis", # 제목 변경
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

# --- Helper Function to Generate Map Data ---
def generate_map_data(history):
    """ history 로부터 전역 맵 포인트와 궤적 데이터를 생성 """
    global_map_points_list = []
    agent_trajectory_x, agent_trajectory_z = [], []
    if not history: return np.empty((0, 2)), [], [] # 데이터 없으면 빈 값 반환

    for scan_record in history:
        pose = scan_record['pose']; relative_points = scan_record['relative_points']
        if not isinstance(relative_points, np.ndarray): relative_points = np.array(relative_points)
        if relative_points.shape[0] == 0: continue
        scan_pos_x, scan_pos_z, scan_rot_rad = pose
        agent_trajectory_x.append(scan_pos_x); agent_trajectory_z.append(scan_pos_z)
        cos_a, sin_a = math.cos(scan_rot_rad), math.sin(scan_rot_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        world_points = np.dot(relative_points, rot_matrix.T) + np.array([[scan_pos_x, scan_pos_z]])
        global_map_points_list.append(world_points)

    map_points_np = np.concatenate(global_map_points_list, axis=0) if global_map_points_list else np.empty((0, 2))
    return map_points_np, agent_trajectory_x, agent_trajectory_z


# --- *** COMBINED PLOTTING FUNCTION *** ---
def plot_all_maps(history, obs_df, gnd_info):
    """ 3개의 맵(LiDAR Points, Ground Truth, Detected Obstacles)을 하나의 Figure에 표시 """
    print("Generating combined map plots...")
    if not history: print("No scan history to plot."); return
    if obs_df is None: print("No obstacle data available for GT map."); return

    # 1. 데이터 준비: 전역 포인트 및 궤적 생성
    map_points_np, agent_trajectory_x, agent_trajectory_z = generate_map_data(history)
    print(f"Total global points: {len(map_points_np)}")
    # 궤적 데이터만 있고 포인트가 없는 경우 처리
    if len(map_points_np) == 0 and not agent_trajectory_x:
        print("No points generated and no trajectory data.")
        return
    elif len(map_points_np) == 0 and agent_trajectory_x:
        print("No points generated, plotting trajectory only.")
        # 궤적만 플롯하는 로직 (단일 플롯)
        fig_traj, ax_traj = plt.subplots(figsize=(6, 6))
        ax_traj.plot(agent_trajectory_x, agent_trajectory_z, marker='o', markersize=3, linestyle='-', color='red', label='Agent Trajectory (GT)')
        if agent_trajectory_x: ax_traj.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], s=50, c='magenta', marker='*', label='Last Scan Pose (GT)')
        ax_traj.set_title("Agent Trajectory Only")
        ax_traj.set_xlabel("World X"); ax_traj.set_ylabel("World Z")
        ax_traj.grid(True); ax_traj.set_aspect('equal'); ax_traj.legend()
        plt.show(block=True)
        return

    # 2. 서브플롯 생성 (1행 3열)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # 가로로 길게 Figure 크기 조정
    fig.suptitle('LiDAR Mapping Analysis', fontsize=16) # 전체 Figure 제목

    # --- Plot 1: Global LiDAR Map (Points & Trajectory) ---
    ax1 = axes[0]
    ax1.scatter(map_points_np[:, 0], map_points_np[:, 1], s=1, c='blue', alpha=0.5, label='Map Points') # 점 크기/투명도 조정
    if agent_trajectory_x:
        ax1.plot(agent_trajectory_x, agent_trajectory_z, marker='.', markersize=2, linestyle='-', color='red', alpha=0.7, label='Trajectory (GT)')
        ax1.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], s=60, c='magenta', marker='*', label='Last Pose (GT)', zorder=5)
    ax1.set_title("LiDAR Point Map & Trajectory")
    ax1.set_xlabel("World X (m)"); ax1.set_ylabel("World Z (m)")
    ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_aspect('equal', adjustable='box'); ax1.legend()
    # 축 범위 설정 (일관성을 위해 모든 플롯 동일하게)
    ax1.set_xlim(-AREA_SIZE / 2, AREA_SIZE / 2); ax1.set_ylim(-AREA_SIZE / 2, AREA_SIZE / 2)


    # --- Plot 2: Ground Truth Map ---
    ax2 = axes[1]
    gnd_size = gnd_info['size']
    ax2.add_patch(patches.Rectangle((-gnd_size/2, -gnd_size/2), gnd_size, gnd_size, edgecolor='gray', facecolor='none', linestyle='--', label='_nolegend_'))
    for index, row in obs_df.iterrows():
        x, z, w, d, rot = row['center_x'], row['center_z'], row['width'], row['depth'], row['rotation_y']
        bottom_left_x = x - w / 2; bottom_left_z = z - d / 2
        ax2.add_patch(patches.Rectangle((bottom_left_x, bottom_left_z), w, d, edgecolor='black', facecolor='darkgray', angle=rot, label='GT Obstacles' if index == 0 else '_nolegend_'))
    ax2.scatter(0, 0, s=100, c='red', marker='x', label='Start (0, 0)')
    ax2.set_title("Ground Truth Map")
    ax2.set_xlabel("World X (m)"); ax2.set_ylabel("World Z (m)")
    ax2.grid(True, linestyle='--', alpha=0.6); ax2.set_aspect('equal', adjustable='box'); ax2.legend()
    ax2.set_xlim(-AREA_SIZE / 2, AREA_SIZE / 2); ax2.set_ylim(-AREA_SIZE / 2, AREA_SIZE / 2)

    # --- Plot 3: Detected Obstacles (Clusters & Trajectory) ---
    ax3 = axes[2]
    detected_obstacle_count = 0
    # DBSCAN 수행
    if len(map_points_np) >= DBSCAN_MIN_SAMPLES:
        try:
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(map_points_np)
            labels = db.labels_
            unique_labels = set(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

            for k, col in zip(unique_labels, colors):
                if k == -1: continue # 노이즈 제외
                cluster_mask = (labels == k)
                cluster_points = map_points_np[cluster_mask]
                if len(cluster_points) > 0:
                    min_x, min_z = np.min(cluster_points, axis=0)
                    max_x, max_z = np.max(cluster_points, axis=0)
                    width = max_x - min_x; height = max_z - min_z
                    rect_label = 'Detected Obstacles' if detected_obstacle_count == 0 else '_nolegend_'
                    rect = patches.Rectangle((min_x, min_z), width, height, linewidth=1, edgecolor=col, facecolor=col, alpha=0.5, label=rect_label)
                    ax3.add_patch(rect)
                    detected_obstacle_count += 1
            print(f'Detected Obstacles Plot: Found {detected_obstacle_count} clusters.')
        except Exception as e:
            print(f"Error during DBSCAN for plotting: {e}")
            ax3.text(0.5, 0.5, 'DBSCAN Error', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    else:
        print("Detected Obstacles Plot: Not enough points for clustering.")
        ax3.text(0.5, 0.5, 'Not Enough Points', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    # 궤적 및 마지막 위치 플롯 (Plot 1과 동일하게)
    if agent_trajectory_x:
        ax3.plot(agent_trajectory_x, agent_trajectory_z, marker='.', markersize=2, linestyle='-', color='red', alpha=0.7, label='Trajectory (GT)')
        ax3.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], s=60, c='magenta', marker='*', label='Last Pose (GT)', zorder=5)

    ax3.set_title(f"Detected Obstacles ({detected_obstacle_count} clusters)")
    ax3.set_xlabel("World X (m)"); ax3.set_ylabel("World Z (m)")
    ax3.grid(True, linestyle='--', alpha=0.6); ax3.set_aspect('equal', adjustable='box'); ax3.legend()
    ax3.set_xlim(-AREA_SIZE / 2, AREA_SIZE / 2); ax3.set_ylim(-AREA_SIZE / 2, AREA_SIZE / 2)

    # 서브플롯 간 간격 조정 및 표시
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 상단 제목 공간 확보
    plt.show(block=True)


# --- UI Display ---
info_text = Text(origin=(-0.5, 0.5), # 기준점 (화면 왼쪽 상단)
                 scale=(0.8, 0.8),   # 텍스트 크기
                 x=-0.5 * window.aspect_ratio + 0.02, # x 위치 (왼쪽 끝)
                 y=0.48,            # y 위치 (상단)
                 text="Initializing...")

# --- Input Handling (Modified) ---
def input(key):
    """ 키보드 입력 처리 """
    global scan_history, obstacle_df # 전역 변수 사용 선언

    # --- 'M' 키: 통합 맵 플롯 호출 ---
    if key == 'm' or key == 'M':
        print("Plotting combined maps...")
        # 통합 플롯 함수 호출
        plot_all_maps(scan_history, obstacle_df, ground_info)

    # --- 'N' 키, 'O' 키 기능은 'M'으로 통합됨 ---

    # 'C' 키: LiDAR 스캔 기록 삭제
    if key == 'c' or key == 'C':
        print("Clearing LiDAR map history.")
        scan_history = []

    # 'P' 키: 데이터 저장
    if key == 'p' or key == 'P':
        print("Saving data...")
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S') # 현재 시간 사용
            if scan_history:
                import pickle
                filename_hist = f"scan_history_{timestamp}.pkl"
                with open(filename_hist, 'wb') as f: pickle.dump(scan_history, f)
                print(f"Scan history saved to {filename_hist}")
            else: print("No scan history to save.")
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
    global scan_timer, scan_history

    # --- Agent Control ---
    original_position = agent.position; total_delta_x, total_delta_z = 0.0, 0.0
    speed_dt = AGENT_SPEED * time.dt
    if held_keys['w']: total_delta_x += agent.forward.x * speed_dt; total_delta_z += agent.forward.z * speed_dt
    if held_keys['s']: total_delta_x -= agent.forward.x * speed_dt; total_delta_z -= agent.forward.z * speed_dt
    if held_keys['a']: total_delta_x -= agent.right.x * speed_dt; total_delta_z -= agent.right.z * speed_dt
    if held_keys['d']: total_delta_x += agent.right.x * speed_dt; total_delta_z += agent.right.z * speed_dt
    agent.x += total_delta_x
    hit_info_x = agent.intersects(traverse_target=scene)
    collided_obs_x = hit_info_x.hit and hasattr(hit_info_x.entity, 'tag') and hit_info_x.entity.tag == OBSTACLE_TAG
    if collided_obs_x: agent.x = original_position.x
    agent.z += total_delta_z
    hit_info_z = agent.intersects(traverse_target=scene)
    collided_obs_z = hit_info_z.hit and hasattr(hit_info_z.entity, 'tag') and hit_info_z.entity.tag == OBSTACLE_TAG
    if collided_obs_z: agent.z = original_position.z
    agent.y = AGENT_HEIGHT
    if held_keys['q']: agent.rotation_y -= ROTATION_SPEED * time.dt
    if held_keys['e']: agent.rotation_y += ROTATION_SPEED * time.dt

    # --- Periodic LiDAR Scan for Mapping History ---
    scan_timer += time.dt
    if scan_timer >= SCAN_INTERVAL:
        scan_timer -= SCAN_INTERVAL
        current_pos_xz = agent.world_position.xz
        current_rot_rad = math.radians(agent.world_rotation_y)
        current_pose = (current_pos_xz.x, current_pos_xz.y, current_rot_rad)
        relative_points = generate_relative_lidar_map() # NumPy array (N, 2) 반환
        if relative_points.shape[0] > 0:
             scan_record = {'pose': current_pose, 'relative_points': relative_points.tolist()}
             scan_history.append(scan_record)

    # --- LiDAR Visualization Update (Ursina Lines) ---
    update_lidar_visualization()

    # --- UI Update ---
    pos_str = f"Pos: ({agent.x:.1f}, {agent.z:.1f})"
    rot_str = f"Rot (Y): {agent.rotation_y:.0f}°"
    # --- 키 설명 수정 ---
    map_info = f"Scans: {len(scan_history)} | 'M': Show All Maps | 'C': Clear | 'P': Save"
    info_text.text = f"{pos_str}\n{rot_str}\n{map_info}"
    info_text.x = -0.5 * window.aspect_ratio + 0.02 # UI 위치 왼쪽 상단으로 변경
    info_text.y = 0.48

    # --- Mouse state ---
    if mouse.locked: mouse.locked = False
    if not mouse.visible: mouse.visible = True

# --- Start Simulation ---
if __name__ == '__main__':
    # Ursina 애플리케이션 실행 (update 함수가 매 프레임 호출됨)
    app.run()