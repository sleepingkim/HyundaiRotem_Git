# 필요한 라이브러리 임포트 (PyTorch 추가)
from ursina import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Matplotlib Interactive Mode Setup ---
plt.ion()
plot_fig = None
plot_ax = None

# --- Configuration ---
MAP_SIZE = 20
NUM_OBSTACLES = 100
SENSOR_RANGE = 8
SENSOR_FOV = 120
SENSOR_RAYS = 90
MATCH_THRESHOLD = 0.8
MATCH_THRESHOLD_SQ = MATCH_THRESHOLD**2
CIRCLE_SEGMENTS = 36
CAMERA_PAN_SPEED = 15
CAMERA_ZOOM_SPEED = 1
ROBOT_MOVE_SPEED = 5
ROBOT_TURN_SPEED = 90
GROUPING_THRESHOLD = 1.0 # 기존 그룹핑 파라미터 (아직 사용됨)
MIN_GROUP_SIZE = 3       # 기존 그룹핑 파라미터 (아직 사용됨)

# *** 딥러닝 모델 관련 설정 ***
POINTNET_FEATURE_DIM = 128 # PointNet이 추출할 특징 벡터의 차원
NUM_POINTS_FOR_MODEL = 128 # 모델이 입력으로 받을 포인트 수 (고정 크기 입력 가정)

# --- Global Variables ---
global_obstacles_entities = []
global_obstacles_positions = []
robot = None
estimated_pose = None
estimated_pose_marker = None
ground = None
fov_lines = []
sensor_range_visual = None
last_detected_local_points = [] # Vec2 리스트
last_clustered_centroids_relative = [] # Vec2 리스트 (기존 방식)
last_pointnet_features = None # 추출된 딥러닝 특징 벡터 저장용
matched_global_indices = []
last_localization_score = -1.0

# --- PointNet-like Model Definition (Simplified for 2D) ---
# 실제 PointNet은 더 복잡하지만, 핵심 아이디어(MLP + MaxPool)를 적용한 예시
class SimplePointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2, feature_dim=POINTNET_FEATURE_DIM):
        super().__init__()
        # 점별 특징 추출 MLP (64 -> 128)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
        # (실제 PointNet에는 T-Net 같은 변환 네트워크가 있지만 여기서는 생략)

    def forward(self, x):
        # x: 입력 포인트 클라우드 (Batch, Dims, NumPoints) - 예: [1, 2, 128]
        if x is None or x.shape[2] == 0: # 입력 포인트가 없을 경우
             # feature_dim 크기의 0 벡터 반환 또는 다른 처리
             return torch.zeros(x.shape[0], POINTNET_FEATURE_DIM, device=x.device)

        point_features = self.mlp1(x) # 각 포인트별로 특징 계산
        # Max Pooling을 통해 전체 포인트 클라우드를 대표하는 전역 특징 추출
        global_feature, _ = torch.max(point_features, dim=2) # (Batch, feature_dim)
        return global_feature

# *** 모델 인스턴스 생성 및 설정 ***
# 실제로는 여기서 미리 학습된 가중치를 로드해야 함
pointnet_model = SimplePointNetFeatureExtractor(input_dim=2, feature_dim=POINTNET_FEATURE_DIM)
pointnet_model.eval() # 추론 모드로 설정 (Dropout, BatchNorm 등 비활성화)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능하면 GPU로
# pointnet_model.to(device)
# print(f"Using device: {device}")
# print("PointNet model created (using random weights as no training data provided).")
# -- 참고: 위 device 설정은 GPU가 있을 경우 사용, 없을 경우 CPU 사용


# --- Helper Functions (Ursina related) ---
def generate_global_map():
    """기존 맵 요소를 지우고, 새 바닥과 무작위 장애물을 생성합니다."""
    global global_obstacles_entities, global_obstacles_positions, ground
    for obs in global_obstacles_entities: destroy(obs)
    global_obstacles_entities.clear(); global_obstacles_positions.clear()
    if ground: destroy(ground)
    ground = Entity(model='plane', scale=MAP_SIZE * 2, color=color.dark_gray, texture='white_cube', texture_scale=(MAP_SIZE, MAP_SIZE), collider='box', name='ground_plane')
    global_obstacles_positions = []
    for _ in range(NUM_OBSTACLES):
        pos = Vec3(random.uniform(-MAP_SIZE, MAP_SIZE), 0.5, random.uniform(-MAP_SIZE, MAP_SIZE))
        if pos.length() > 3: add_obstacle(pos)

def add_obstacle(position, scale_y=None):
    """주어진 위치에 단일 장애물 Entity를 생성하고 관련 리스트에 추가합니다."""
    global global_obstacles_entities, global_obstacles_positions, matched_global_indices, last_localization_score
    if scale_y is None: scale_y = random.uniform(1, 3)
    if robot and distance(position, robot.position) < 1.0:
         print("Cannot add obstacle too close to the robot."); return
    # print(f"Creating obstacle entity at {position}")
    obstacle = Entity(model='cube', position=position, color=color.gray, collider='box', scale_y=scale_y)
    global_obstacles_entities.append(obstacle)
    global_obstacles_positions.append(Vec3(position.x, 0, position.z))
    matched_global_indices = []; last_localization_score = -1.0
    if estimated_pose_marker: estimated_pose_marker.enabled = False

def simulate_lidar(robot_entity):
    """로봇 위치에서 가상 LiDAR 스캔을 수행하여 로봇 기준 상대 좌표 점들을 반환합니다."""
    global last_detected_local_points, SENSOR_FOV, SENSOR_RANGE, SENSOR_RAYS
    detected_points_relative = []
    origin = robot_entity.world_position + Vec3(0, 0.1, 0)
    robot_rotation_y_rad = math.radians(robot_entity.world_rotation_y)
    fov_rad = math.radians(SENSOR_FOV)
    start_angle = -fov_rad / 2; end_angle = fov_rad / 2
    angle_step = fov_rad / (SENSOR_RAYS - 1) if SENSOR_RAYS > 1 else 0

    ignore_list = [robot_entity] + fov_lines
    if sensor_range_visual: ignore_list.append(sensor_range_visual)

    for i in range(SENSOR_RAYS):
        current_angle_relative = start_angle + i * angle_step
        world_angle = robot_rotation_y_rad + current_angle_relative
        direction = Vec3(math.sin(world_angle), 0, math.cos(world_angle)).normalized()
        hit_info = raycast(origin=origin, direction=direction, distance=SENSOR_RANGE, ignore=ignore_list, debug=False)
        if hit_info.hit and hit_info.entity != ground:
             hit_point_world = hit_info.world_point
             relative_pos_world = hit_point_world - origin
             cos_a = math.cos(-robot_rotation_y_rad); sin_a = math.sin(-robot_rotation_y_rad)
             x_rel = relative_pos_world.x * cos_a - relative_pos_world.z * sin_a
             z_rel = relative_pos_world.x * sin_a + relative_pos_world.z * cos_a
             detected_points_relative.append(Vec2(x_rel, z_rel)) # Ursina Vec2 사용

    last_detected_local_points = detected_points_relative
    return detected_points_relative

# --- 기존 중심점 계산 함수 (아직 유사도 계산에 임시로 사용됨) ---
def find_centroids_simple(local_scan_points, threshold, min_size):
    if not local_scan_points: return []
    centroids = []
    remaining_points = local_scan_points[:]
    num_points = len(remaining_points)
    processed_indices = [False] * num_points
    for i in range(num_points):
        if processed_indices[i]: continue
        current_cluster = [remaining_points[i]]
        processed_indices[i] = True
        cluster_indices_to_process = [i]
        idx_in_queue = 0
        while idx_in_queue < len(cluster_indices_to_process):
            current_point_index = cluster_indices_to_process[idx_in_queue]
            p1 = remaining_points[current_point_index]
            idx_in_queue += 1
            for j in range(num_points):
                if not processed_indices[j]:
                    p2 = remaining_points[j]
                    dist_sq = (p1.x - p2.x)**2 + (p1.y - p2.y)**2 # Vec2의 y는 z좌표
                    if dist_sq < threshold**2:
                        current_cluster.append(p2)
                        processed_indices[j] = True
                        cluster_indices_to_process.append(j)
        if len(current_cluster) >= min_size:
            sum_x = sum(p.x for p in current_cluster)
            sum_z = sum(p.y for p in current_cluster) # Vec2.y는 상대 z
            count = len(current_cluster)
            centroid = Vec2(sum_x / count, sum_z / count) # Ursina Vec2 사용
            centroids.append(centroid)
    return centroids

# --- 기존 유사도 계산 함수 (중심점 기반) ---
# !!! 중요: 이 함수는 PointNet 특징을 사용하지 않습니다. !!!
# !!! PointNet 특징을 사용하려면 이 함수를 완전히 대체하거나 수정해야 합니다. !!!
def calculate_similarity(potential_pose, local_features_relative, global_map_points_xz):
    """가상 로봇 자세에서 로컬 특징점(중심점)들과 전역 장애물 위치 간의 유사도를 계산합니다."""
    global MATCH_THRESHOLD_SQ
    potential_pos = Vec3(potential_pose[0], 0, potential_pose[1]); potential_angle_rad = math.radians(potential_pose[2])
    total_score = 0.0; epsilon = 0.1; cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
    global_map_xz_tuples = [(p.x, p.z) for p in global_map_points_xz]
    if not local_features_relative: return 0.0

    # 각 로컬 특징점(중심점)에 대해 반복
    for feature_rel in local_features_relative: # feature_rel은 Vec2 타입
        x_rot = feature_rel.x * cos_a - feature_rel.y * sin_a
        z_rot = feature_rel.x * sin_a + feature_rel.y * cos_a
        world_pt_guess_x = potential_pos.x + x_rot
        world_pt_guess_z = potential_pos.z + z_rot
        min_dist_sq = float('inf')
        for gx, gz in global_map_xz_tuples:
            dist_sq = (world_pt_guess_x - gx)**2 + (world_pt_guess_z - gz)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq
        if min_dist_sq < MATCH_THRESHOLD_SQ: total_score += 1.0 / (math.sqrt(min_dist_sq) + epsilon)
    return total_score

# --- Localization Logic ---
def perform_localization():
    """LiDAR 스캔 -> PointNet 특징 추출 -> (임시) 중심점 계산 -> 위치 추정"""
    global robot, estimated_pose, estimated_pose_marker, global_obstacles_positions
    global last_localization_score, matched_global_indices, last_clustered_centroids_relative
    global GROUPING_THRESHOLD, MIN_GROUP_SIZE
    global pointnet_model, last_pointnet_features # PointNet 모델 및 결과 변수

    print("--- Starting Localization with PointNet Feature Extraction ---");
    if not robot: print("Robot not initialized."); return

    # 1. LiDAR 스캔 수행 (결과는 Vec2 리스트)
    local_scan_points = simulate_lidar(robot)
    print(f"Detected {len(local_scan_points)} raw points locally.")

    # 2. *** PointNet 특징 추출 ***
    if len(local_scan_points) > 0:
        # 2a. 데이터를 PyTorch Tensor로 변환 (Batch=1, Dims=2, NumPoints)
        # 모델이 고정된 수의 포인트를 입력받는다고 가정 (NUM_POINTS_FOR_MODEL)
        # 포인트 수가 부족하면 패딩, 많으면 샘플링/선택 필요
        num_detected = len(local_scan_points)
        if num_detected >= NUM_POINTS_FOR_MODEL:
            # 포인트 수가 충분하거나 많으면, 앞에서부터 NUM_POINTS_FOR_MODEL 개 선택
            selected_points = local_scan_points[:NUM_POINTS_FOR_MODEL]
        else:
            # 포인트 수가 부족하면, 마지막 포인트를 반복하여 패딩 (또는 0으로 패딩)
            selected_points = local_scan_points + [local_scan_points[-1]] * (NUM_POINTS_FOR_MODEL - num_detected)

        # Ursina Vec2 리스트를 numpy 배열 -> PyTorch 텐서로 변환
        points_np = np.array([[p.x, p.y] for p in selected_points]) # (NUM_POINTS_FOR_MODEL, 2)
        points_tensor = torch.tensor(points_np, dtype=torch.float32).unsqueeze(0) # (1, NUM_POINTS_FOR_MODEL, 2)
        points_tensor = points_tensor.permute(0, 2, 1) # (1, 2, NUM_POINTS_FOR_MODEL) 모델 입력 형식에 맞춤

        # 2b. PointNet 모델로 특징 추출 (학습된 가중치 필요!)
        with torch.no_grad(): # 그래디언트 계산 비활성화 (추론 시)
            last_pointnet_features = pointnet_model(points_tensor) # (1, POINTNET_FEATURE_DIM)
        print(f"Extracted PointNet features (shape: {last_pointnet_features.shape})")
        # print(f"Feature vector (first 10): {last_pointnet_features.squeeze()[:10].numpy()}")
    else:
        print("No points detected, skipping PointNet feature extraction.")
        last_pointnet_features = None

    # 3. *** (임시) 기존 방식의 중심점 계산 ***
    # !!! 중요: 아래 로직은 PointNet 특징을 직접 사용하지 않음 !!!
    # !!! 실제로는 추출된 'last_pointnet_features'를 사용하여 위치 추정을 해야 함 !!!
    # !!! 여기서는 데모를 위해 기존 로직을 임시로 사용 !!!
    last_clustered_centroids_relative = find_centroids_simple(local_scan_points, GROUPING_THRESHOLD, MIN_GROUP_SIZE)
    print(f"Found {len(last_clustered_centroids_relative)} centroids using simple grouping (for temporary matching).")

    # 4. 로컬라이제이션 가능 여부 확인 (임시: 중심점 기준)
    if not last_clustered_centroids_relative: # 임시 조건
    # if last_pointnet_features is None: # 실제로는 특징 벡터 유무로 판단해야 함
        print("No valid features/centroids found, cannot localize."); estimated_pose = None; last_localization_score = -1.0; matched_global_indices = []
        if estimated_pose_marker: estimated_pose_marker.enabled = False; return
    else:
        if estimated_pose_marker: estimated_pose_marker.enabled = True

    # 5. 최적 자세 탐색 (Grid Search - 임시: 중심점 유사도 기반)
    search_radius = 3.0; angle_search_range = 30; pos_step = 0.5; angle_step = 5
    best_score = -1.0
    if estimated_pose: start_pos = Vec3(estimated_pose[0], 0, estimated_pose[1]); start_angle = estimated_pose[2]
    else: start_pos = robot.position; start_angle = robot.rotation_y
    current_best_pose = (start_pos.x, start_pos.z, start_angle); search_count = 0

    for dx in np.arange(-search_radius, search_radius + pos_step, pos_step):
        for dz in np.arange(-search_radius, search_radius + pos_step, pos_step):
            for dangle in np.arange(-angle_search_range, angle_search_range + angle_step, angle_step):
                potential_x = start_pos.x + dx; potential_z = start_pos.z + dz; potential_angle = (start_angle + dangle) % 360
                potential_pose_tuple = (potential_x, potential_z, potential_angle); search_count += 1
                # *** 유사도 계산 (임시: 중심점 기반) ***
                # !!! 실제로는 PointNet 특징 벡터와 맵 특징을 비교하는 로직 필요 !!!
                score = calculate_similarity(potential_pose_tuple, last_clustered_centroids_relative, global_obstacles_positions)
                if score > best_score: best_score = score; current_best_pose = potential_pose_tuple

    # 6. 최적 결과 저장
    estimated_pose = current_best_pose; last_localization_score = best_score
    print(f"Search completed. Checked {search_count} poses.")
    print(f"Best Estimated Pose (based on temp. centroid matching): x={estimated_pose[0]:.2f}, z={estimated_pose[1]:.2f}, angle={estimated_pose[2]:.2f}")
    print(f"Actual Robot Pose:   x={robot.x:.2f}, z={robot.z:.2f}, angle={robot.rotation_y:.2f}")
    print(f"Best Score (Centroid Inverse Dist Sum): {last_localization_score:.4f}")

    # 7. 매칭된 전역 장애물 인덱스 찾기 (시각화용 - 임시: 중심점 기준)
    matched_indices_set = set()
    if estimated_pose and last_clustered_centroids_relative:
        est_x, est_z, est_angle_deg = estimated_pose
        potential_pos = Vec3(est_x, 0, est_z); potential_angle_rad = math.radians(est_angle_deg)
        cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
        match_threshold_sq_now = MATCH_THRESHOLD**2
        for centroid_rel in last_clustered_centroids_relative:
            x_rot = centroid_rel.x * cos_a - centroid_rel.y * sin_a; z_rot = centroid_rel.x * sin_a + centroid_rel.y * cos_a
            world_pt_guess_x = potential_pos.x + x_rot; world_pt_guess_z = potential_pos.z + z_rot
            min_dist_sq = float('inf'); best_match_idx = -1
            for idx, global_pt in enumerate(global_obstacles_positions):
                dist_sq = (world_pt_guess_x - global_pt.x)**2 + (world_pt_guess_z - global_pt.z)**2
                if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_match_idx = idx
            if best_match_idx != -1 and min_dist_sq < match_threshold_sq_now:
                matched_indices_set.add(best_match_idx)
    matched_global_indices = list(matched_indices_set); print(f"Indices of matched global obstacles: {matched_global_indices}")

    # 8. Ursina 추정 위치 마커 업데이트
    if estimated_pose_marker: estimated_pose_marker.position = Vec3(estimated_pose[0], 0.1, estimated_pose[1]); estimated_pose_marker.rotation_y = estimated_pose[2]
    else:
        estimated_pose_marker = Entity(model='arrow', color=color.lime, scale=1.5, position=Vec3(estimated_pose[0], 0.1, estimated_pose[1]), rotation_y = estimated_pose[2])
        Entity(model='sphere', scale=0.5, color=color.green, parent=estimated_pose_marker, y=-0.2)
    print("--- Localization with PointNet Feature Extraction Finished (Matching part is temporary) ---")


# --- Visualization Update Functions ---
# (update_fov_visualization, update_sensor_range_visualization - 변경 없음)
def update_fov_visualization():
    global fov_lines, SENSOR_FOV, SENSOR_RANGE;
    if not robot: return;
    for line in fov_lines: destroy(line);
    fov_lines.clear()
    origin = robot.world_position + Vec3(0, 0.05, 0); robot_rot_y_rad = math.radians(robot.world_rotation_y)
    fov_rad = math.radians(SENSOR_FOV); angle_left_rad = robot_rot_y_rad + fov_rad / 2; angle_right_rad = robot_rot_y_rad - fov_rad / 2
    dir_left = Vec3(math.sin(angle_left_rad), 0, math.cos(angle_left_rad)); dir_right = Vec3(math.sin(angle_right_rad), 0, math.cos(angle_right_rad))
    end_left = origin + dir_left * SENSOR_RANGE; end_right = origin + dir_right * SENSOR_RANGE
    line_color = color.cyan; line_thickness = 2
    fov_lines.append(Entity(model=Mesh(vertices=[origin, end_left], mode='line', thickness=line_thickness), color=line_color))
    fov_lines.append(Entity(model=Mesh(vertices=[origin, end_right], mode='line', thickness=line_thickness), color=line_color))

def update_sensor_range_visualization():
    global sensor_range_visual, SENSOR_RANGE, CIRCLE_SEGMENTS;
    if not robot: return;
    center_pos = robot.world_position + Vec3(0, 0.02, 0); vertices = []
    angle_step = 2 * math.pi / CIRCLE_SEGMENTS
    for i in range(CIRCLE_SEGMENTS + 1):
        angle = i * angle_step; x = center_pos.x + SENSOR_RANGE * math.cos(angle); z = center_pos.z + SENSOR_RANGE * math.sin(angle)
        vertices.append(Vec3(x, center_pos.y, z))
    if sensor_range_visual: sensor_range_visual.model.vertices = vertices; sensor_range_visual.model.generate()
    else: sensor_range_visual = Entity(model=Mesh(vertices=vertices, mode='line', thickness=2), color=color.yellow)


# --- Matplotlib Plotting Function ---
# (기존과 거의 동일, 제목에 PointNet 언급 추가 가능)
def update_plot_data_and_redraw():
    global plot_fig, plot_ax, global_obstacles_positions, robot, estimated_pose
    global matched_global_indices, last_localization_score, last_clustered_centroids_relative
    global last_pointnet_features # PointNet 특징 정보도 활용 가능 (예: 제목에 표시)

    global_obs_pos = global_obstacles_positions
    actual_pose = (robot.x, robot.z, robot.rotation_y) if robot else None

    if plot_fig is None: plot_fig, plot_ax = plt.subplots(figsize=(8.5, 8))
    plot_ax.clear()

    # 1. Plot Global Obstacles
    unmatched_obs_x, unmatched_obs_z = [], []; matched_obs_x, matched_obs_z = [], []
    if global_obs_pos:
        for idx, p in enumerate(global_obs_pos):
            if idx in matched_global_indices: matched_obs_x.append(p.x); matched_obs_z.append(p.z)
            else: unmatched_obs_x.append(p.x); unmatched_obs_z.append(p.z)
    plot_ax.scatter(unmatched_obs_x, unmatched_obs_z, c='grey', marker='s', s=100, label='Global Obstacles (Unmatched)')
    plot_ax.scatter(matched_obs_x, matched_obs_z, c='magenta', marker='s', s=120, label='Global Obstacles (Matched)', edgecolors='black')

    # 2. Plot Clustered Centroids (World - 임시)
    centroids_world_x, centroids_world_z = [], []
    if last_clustered_centroids_relative and actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose
        angle_rad = math.radians(robot_angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        for pt in last_clustered_centroids_relative:
            x_rot = pt.x * cos_a - pt.y * sin_a; z_rot = pt.x * sin_a + pt.y * cos_a
            centroids_world_x.append(robot_x + x_rot); centroids_world_z.append(robot_z + z_rot)
    plot_ax.scatter(centroids_world_x, centroids_world_z, c='red', marker='x', s=80, label='Detected Centers (World - Temp)')

    # 3. Plot Actual Robot Pose
    if actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose; plot_ax.scatter(robot_x, robot_z, c='blue', marker='o', s=150, label='Actual Pose')
        angle_rad = math.radians(robot_angle_deg); arrow_len = 1.5; plot_ax.arrow(robot_x, robot_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='blue', ec='blue')

    # 4. Plot Estimated Robot Pose
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose; plot_ax.scatter(est_x, est_z, c='lime', marker='o', s=150, label='Estimated Pose', alpha=0.7)
        angle_rad = math.radians(est_angle_deg); arrow_len = 1.5; plot_ax.arrow(est_x, est_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='lime', ec='lime', alpha=0.7)

    # Formatting & Score Text
    plot_ax.set_xlabel("X coordinate"); plot_ax.set_ylabel("Z coordinate")
    title_text = "2D Map & Localization Attempt"
    if last_pointnet_features is not None:
        title_text += "\n(Using PointNet Feat. Extraction - Matching is Temp.)" # PointNet 사용 명시
    else:
         title_text += "\n(Using Centroid Extraction)"

    if last_localization_score >= 0: title_text += f"\nSimilarity Score (Temp): {last_localization_score:.4f}"
    plot_ax.set_title(title_text); plot_ax.set_aspect('equal', adjustable='box'); limit = MAP_SIZE * 1.1; plot_ax.set_xlim(-limit, limit); plot_ax.set_ylim(-limit, limit); plot_ax.grid(True); plot_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    try: plot_fig.canvas.draw(); plt.pause(0.01); plt.tight_layout(rect=[0, 0, 0.85, 1])
    except Exception as e: print(f"Matplotlib plotting error: {e}")

# --- Ursina Application Setup ---
app = Ursina()
# (초기화 코드 변경 없음)
generate_global_map()
robot = Entity(model='sphere', color=color.blue, collider='sphere', position=(0, 0.2, 0))
robot_forward = Entity(model='cube', scale=(0.1, 0.1, 0.5), color=color.red, parent=robot, z=0.3)
ec = EditorCamera(rotation_speed=100, panning_speed=100, zoom_speed=CAMERA_ZOOM_SPEED)
camera.position = (0, 25, -25); camera.rotation_x = 45

instructions = Text(
    text="WASD/Mouse=Move/Rotate Camera, Arrow Keys=Pan Camera\n"
         "QE=Rotate Robot, L=Localize(PointNet Feat. + Temp Match) & Plot, Left Click=Add Obstacle", # 로컬라이제이션 방식 명시
    origin=(-0.5, 0.5), scale=1.5, position=window.top_left + Vec2(0.01, -0.01), background=True
)

# --- Initial Setup Calls ---
update_fov_visualization()
update_sensor_range_visualization()

# --- Input Handling ---
def input(key):
    global ground, robot
    if key == 'l':
        perform_localization() # PointNet 특징 추출 포함된 로컬라이제이션 호출
        update_plot_data_and_redraw()
    elif key == 'left mouse down':
        origin = camera.world_position
        if mouse.world_point:
            direction = (mouse.world_point - origin).normalized()
            hit_info = raycast(origin=origin, direction=direction, distance=inf,
                               ignore=[robot] + fov_lines + ([sensor_range_visual] if sensor_range_visual else []))
            if hit_info.hit and hit_info.entity == ground:
                click_pos = hit_info.world_point; add_obstacle(Vec3(click_pos.x, 0.5, click_pos.z))
            # elif hit_info.hit: print(f"Click hit {hit_info.entity.name if hasattr(hit_info.entity, 'name') else 'unnamed entity'}, not ground.")
            # else: print("Click ray did not hit the ground plane.")
        # else: print("Cannot determine mouse target in 3D space to cast ray.")

# --- Update Loop ---
# (변경 없음)
def update():
    global robot, sensor_range_visual
    if not robot: return
    move_speed = ROBOT_MOVE_SPEED * time.dt; turn_speed = ROBOT_TURN_SPEED * time.dt; moved_or_rotated = False
    # input_active = any(held_keys[k] for k in ['w', 'a', 's', 'd', 'q', 'e']) # Ursina 4.0+ 에서 필요 없을 수 있음

    if held_keys['w']: robot.position += robot.forward * move_speed; moved_or_rotated = True
    if held_keys['s']: robot.position -= robot.forward * move_speed; moved_or_rotated = True
    if held_keys['a']: robot.position -= robot.right * move_speed; moved_or_rotated = True
    if held_keys['d']: robot.position += robot.right * move_speed; moved_or_rotated = True
    if held_keys['q']: robot.rotation_y -= turn_speed; moved_or_rotated = True
    if held_keys['e']: robot.rotation_y += turn_speed; moved_or_rotated = True

    if robot.y < 0.1: robot.y = 0.1
    if moved_or_rotated:
        update_fov_visualization();
        update_sensor_range_visualization()

    pan_amount = CAMERA_PAN_SPEED * time.dt
    if hasattr(camera, 'parent') and camera.parent is not None:
        cam_pivot = camera.parent
        try:
            forward_xz = Vec3(camera.forward.x, 0, camera.forward.z).normalized()
            right_xz = Vec3(camera.right.x, 0, camera.right.z).normalized()
            if held_keys['up arrow']: cam_pivot.position += forward_xz * pan_amount
            if held_keys['down arrow']: cam_pivot.position -= forward_xz * pan_amount
            if held_keys['left arrow']: cam_pivot.position -= right_xz * pan_amount
            if held_keys['right arrow']: cam_pivot.position += right_xz * pan_amount
        except Exception as e: pass # 가끔 발생하는 에러 무시


# --- Start the Application ---
if __name__ == '__main__':
    # PyTorch 설치 확인
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("="*50)
        print("ERROR: PyTorch is not installed.")
        print("Please install PyTorch by following the instructions at https://pytorch.org/")
        print("Example: pip install torch torchvision torchaudio")
        print("="*50)
        exit()

    app.run()