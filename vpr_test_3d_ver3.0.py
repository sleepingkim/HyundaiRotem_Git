# 필요 라이브러리 설치 (pip install ursina numpy matplotlib scipy)
from ursina import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import platform
import time

# --- Matplotlib Setup ---
plt.ion()
plot_fig, plot_ax = plt.subplots(figsize=(10, 10))
plot_fig.canvas.manager.set_window_title('EKF-SLAM State Visualization')

# matplotlib 한글깨짐 방지 설정
if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Linux': plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# --- Configuration ---
# Simulation Environment
MAP_SIZE = 25
NUM_RANDOM_OBSTACLES = 100 # *** 초기 랜덤 장애물 개수 설정 ***

# Robot Control Parameters
ROBOT_MOVE_SPEED = 4.0
ROBOT_TURN_SPEED = 80.0

# Sensor Parameters
SENSOR_RANGE = 12.0
SENSOR_FOV = 180
SENSOR_RAYS = 180

# Feature Extraction Parameters
GROUPING_THRESHOLD = 1.5
MIN_GROUP_SIZE = 3

# --- EKF-SLAM Parameters ---
# Noise
MOTION_NOISE_STD = {'linear': 0.05, 'angular': np.radians(1.0)}
MEASUREMENT_NOISE_STD = {'range': 0.15, 'bearing': np.radians(2.0)}
# Data Association
MAHALANOBIS_GATE_THRESHOLD = 2.5
NEW_LANDMARK_MIN_DISTANCE = 2.0
# Initialization
INITIAL_POSE_UNCERTAINTY = np.diag([0.01**2, 0.01**2, np.radians(0.1)**2])
INITIAL_LANDMARK_UNCERTAINTY_DIAG = [0.3**2, 0.3**2]

# Sensor Range Visualization
CIRCLE_SEGMENTS = 36

# --- Global Variables ---
# Ursina Specific
ursina_obstacles = []
robot_entity = None
estimated_pose_marker = None
landmark_markers = {} # Key: Landmark ID, Value: Ursina Entity
ground = None
sensor_range_visual = None
# EKF-SLAM State
state_vector = None
covariance_matrix = None
landmark_map = {} # Key: Landmark ID, Value: index in state vector
next_landmark_id = 0
# Control Input & Timing
accumulated_distance = 0.0
accumulated_angle_rad = 0.0
last_update_time = time.time()
# Visualization Control
plot_needs_update = True

# --- Helper Functions ---
def angle_wrap(angle_rad):
    while angle_rad > np.pi: angle_rad -= 2 * np.pi
    while angle_rad < -np.pi: angle_rad += 2 * np.pi
    return angle_rad

def plot_covariance_ellipse(ax, mean, cov, n_std=2, **kwargs):
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-9)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width = 2 * n_std * np.sqrt(vals[0]); height = 2 * n_std * np.sqrt(vals[1])
        ell = plt.matplotlib.patches.Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
        ax.add_patch(ell); return ell
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix issue, cannot plot ellipse."); return None

# --- Ursina World Setup ---
def setup_world():
    """ Ursina 3D 환경 초기화, 로봇/마커 생성, 초기 장애물 배치 """
    global ground, robot_entity, estimated_pose_marker, last_update_time
    global ursina_obstacles, landmark_markers, sensor_range_visual

    # 기존 요소 제거
    for obs in ursina_obstacles: destroy(obs); ursina_obstacles.clear()
    for lm_marker in landmark_markers.values():
        if lm_marker: destroy(lm_marker)
    landmark_markers.clear()
    if ground: destroy(ground)
    if robot_entity: destroy(robot_entity)
    if estimated_pose_marker: destroy(estimated_pose_marker)
    if sensor_range_visual: destroy(sensor_range_visual); sensor_range_visual = None

    # 새 환경 요소 생성
    ground = Entity(model='plane', scale=MAP_SIZE * 2, color=color.dark_gray, texture='white_cube', texture_scale=(MAP_SIZE, MAP_SIZE), collider='box', name='ground_plane')
    robot_entity = Entity(model='sphere', color=color.blue, collider='sphere', position=(0, 0.2, 0), name='robot_actual')
    Entity(model='cube', scale=(0.2, 0.2, 0.7), color=color.red, parent=robot_entity, z=0.4, origin_y=-0.5, name='robot_front_indicator')
    estimated_pose_marker = Entity(model='arrow', color=color.lime, scale=1.5, origin_y=-0.5, enabled=False, name='robot_estimated')
    Entity(model='sphere', scale=0.5, color=color.green, parent=estimated_pose_marker, y=0, name='estimated_marker_base')

    # --- *** 초기 랜덤 장애물(Ground Truth) 배치 *** ---
    print(f"Adding {NUM_RANDOM_OBSTACLES} initial random ground truth obstacles.")
    placed_obstacles_positions = [] # 중복 방지용
    attempts = 0
    max_attempts = NUM_RANDOM_OBSTACLES * 5 # 최대 시도 횟수

    while len(placed_obstacles_positions) < NUM_RANDOM_OBSTACLES and attempts < max_attempts:
        attempts += 1
        # 맵 가장자리는 피하고, 로봇 초기 위치 근처도 피함
        pos_x = random.uniform(-MAP_SIZE * 0.8, MAP_SIZE * 0.8)
        pos_z = random.uniform(-MAP_SIZE * 0.8, MAP_SIZE * 0.8)
        position = Vec3(pos_x, 0.5, pos_z)

        # 너무 중앙(초기 위치)에 가깝거나 다른 장애물과 너무 가까우면 다시 시도
        if distance_xz(position, Vec3(0,0,0)) < 3.0: continue

        is_too_close = False
        for existing_pos in placed_obstacles_positions:
            if distance_xz(position, existing_pos) < 1.5: # 최소 간격
                is_too_close = True
                break
        if is_too_close: continue

        # 유효한 위치면 장애물 추가
        add_ground_truth_obstacle(position)
        placed_obstacles_positions.append(position) # 추가된 위치 기록

    if len(placed_obstacles_positions) < NUM_RANDOM_OBSTACLES:
        print(f"Warning: Could only place {len(placed_obstacles_positions)} obstacles due to proximity constraints.")
    # --- *** 초기 장애물 배치 끝 *** ---

    update_sensor_range_visualization() # 센서 범위 시각화 초기 생성
    last_update_time = time.time()

def add_ground_truth_obstacle(position, scale_y=None):
    global ursina_obstacles
    if scale_y is None: scale_y = random.uniform(1.5, 3.0)
    if robot_entity and distance_xz(position, robot_entity.position) < 1.0: return
    obstacle = Entity(model='cube', position=position, color=color.light_gray, collider='box', scale_y=scale_y, texture='brick', name=f'GT_Obstacle_{len(ursina_obstacles)}') # 색상 변경
    ursina_obstacles.append(obstacle)
    # print(f"Added Ground Truth Obstacle at {position.xz}") # 로그 너무 많으면 주석 처리

def distance_xz(vec1, vec2):
    return math.sqrt((vec1.x - vec2.x)**2 + (vec1.z - vec2.z)**2)

# --- Sensor Range Visualization ---
def update_sensor_range_visualization():
    global sensor_range_visual, robot_entity, SENSOR_RANGE, CIRCLE_SEGMENTS
    if not robot_entity: return
    center_pos = robot_entity.world_position + Vec3(0, 0.02, 0)
    vertices = []
    angle_step = 2 * math.pi / CIRCLE_SEGMENTS
    for i in range(CIRCLE_SEGMENTS + 1):
        angle = i * angle_step; x = center_pos.x + SENSOR_RANGE * math.cos(angle); z = center_pos.z + SENSOR_RANGE * math.sin(angle)
        vertices.append(Vec3(x, center_pos.y, z))
    if sensor_range_visual:
        try:
             if hasattr(sensor_range_visual.model, 'vertices'):
                  sensor_range_visual.model.vertices = vertices; sensor_range_visual.model.generate()
             else: sensor_range_visual.model = Mesh(vertices=vertices, mode='line', thickness=2)
        except Exception as e:
             print(f"Error updating sensor range visual: {e}"); destroy(sensor_range_visual)
             sensor_range_visual = Entity(model=Mesh(vertices=vertices, mode='line', thickness=2), color=color.yellow)
    else: sensor_range_visual = Entity(model=Mesh(vertices=vertices, mode='line', thickness=2), color=color.yellow)

# --- LiDAR Simulation and Feature Extraction ---
# (이전 코드와 동일)
def simulate_lidar(robot_actual_pose_entity):
    detected_points_relative = []
    origin = robot_actual_pose_entity.world_position + Vec3(0, 0.2, 0)
    robot_rotation_y_rad = math.radians(robot_actual_pose_entity.world_rotation_y)
    fov_rad = math.radians(SENSOR_FOV); start_angle = -fov_rad / 2
    angle_step = fov_rad / (SENSOR_RAYS - 1) if SENSOR_RAYS > 1 else 0
    ignore_list = [robot_actual_pose_entity]
    for i in range(SENSOR_RAYS):
        current_angle_relative = start_angle + i * angle_step
        world_angle = robot_rotation_y_rad + current_angle_relative
        direction = Vec3(math.sin(world_angle), 0, math.cos(world_angle)).normalized()
        hit_info = raycast(origin=origin, direction=direction, distance=SENSOR_RANGE, ignore=ignore_list, debug=False)
        if hit_info.hit and hit_info.entity != ground:
             hit_point_world = hit_info.world_point
             relative_pos_world = hit_point_world - origin
             cos_rot = math.cos(-robot_rotation_y_rad); sin_rot = math.sin(-robot_rotation_y_rad)
             x_rel_robot = relative_pos_world.x * cos_rot - relative_pos_world.z * sin_rot
             z_rel_robot = relative_pos_world.x * sin_rot + relative_pos_world.z * cos_rot
             dist = math.sqrt(x_rel_robot**2 + z_rel_robot**2)
             bearing = math.atan2(x_rel_robot, z_rel_robot)
             noisy_dist = dist + np.random.normal(0, MEASUREMENT_NOISE_STD['range'])
             noisy_bearing = bearing + np.random.normal(0, MEASUREMENT_NOISE_STD['bearing'])
             noisy_x_rel = noisy_dist * math.sin(noisy_bearing); noisy_z_rel = noisy_dist * math.cos(noisy_bearing)
             detected_points_relative.append(Vec2(noisy_x_rel, noisy_z_rel))
    return detected_points_relative

def find_centroids_from_lidar(local_scan_points, threshold, min_size):
    if not local_scan_points: return []
    centroids = []; points_np = np.array([(p.x, p.y) for p in local_scan_points]); num_points = len(points_np)
    if num_points < min_size: return []
    processed = np.zeros(num_points, dtype=bool); threshold_sq = threshold**2
    for i in range(num_points):
        if processed[i]: continue
        current_cluster_indices = [i]; processed[i] = True; queue = [i]; head = 0
        while head < len(queue):
            current_idx = queue[head]; head += 1; p1 = points_np[current_idx]
            for j in range(num_points):
                if not processed[j]:
                    p2 = points_np[j]; dist_sq = np.sum((p1 - p2)**2)
                    if dist_sq < threshold_sq: processed[j] = True; queue.append(j); current_cluster_indices.append(j)
        if len(current_cluster_indices) >= min_size:
            cluster_points = points_np[current_cluster_indices]; centroid = np.mean(cluster_points, axis=0)
            centroids.append(Vec2(centroid[0], centroid[1]))
    return centroids

# --- EKF-SLAM Core Logic ---
def initialize_slam():
    global state_vector, covariance_matrix, landmark_map, next_landmark_id
    state_vector = np.zeros((3, 1)); covariance_matrix = INITIAL_POSE_UNCERTAINTY.copy()
    landmark_map = {}; next_landmark_id = 0
    print("EKF-SLAM Initialized.")

def predict_step(total_dist_moved, total_delta_theta_rad):
    global state_vector, covariance_matrix
    if state_vector is None: return
    num_landmarks = len(landmark_map); state_dim = 3 + 2 * num_landmarks
    x, z, theta = state_vector[0:3].flatten(); theta_pred = angle_wrap(theta + total_delta_theta_rad)
    if abs(total_delta_theta_rad) < 1e-6:
        x_pred = x + total_dist_moved * np.cos(theta); z_pred = z + total_dist_moved * np.sin(theta)
    else:
        avg_theta = angle_wrap(theta + total_delta_theta_rad / 2.0)
        x_pred = x + total_dist_moved * np.cos(avg_theta); z_pred = z + total_dist_moved * np.sin(avg_theta)
    state_vector[0] = x_pred; state_vector[1] = z_pred; state_vector[2] = theta_pred
    Fx_robot = np.array([[1, 0, -total_dist_moved * np.sin(avg_theta if abs(total_delta_theta_rad) > 1e-6 else theta)],
                         [0, 1,  total_dist_moved * np.cos(avg_theta if abs(total_delta_theta_rad) > 1e-6 else theta)],
                         [0, 0, 1]])
    Fx = np.eye(state_dim); Fx[0:3, 0:3] = Fx_robot
    var_lin = (MOTION_NOISE_STD['linear'] * abs(total_dist_moved))**2 + 1e-6
    var_ang = (MOTION_NOISE_STD['angular'] * abs(total_delta_theta_rad))**2 + 1e-6
    Q_motion = np.diag([var_lin, var_lin, var_ang])
    Q_full = np.zeros((state_dim, state_dim)); Q_full[0:3, 0:3] = Q_motion
    covariance_matrix = Fx @ covariance_matrix @ Fx.T + Q_full

def update_step(measurements_relative):
    global state_vector, covariance_matrix, landmark_map, next_landmark_id, plot_needs_update
    num_landmarks_in_map = len(landmark_map); state_dim = 3 + 2 * num_landmarks_in_map
    if state_dim == 0 or state_vector is None: return
    robot_pose_est = state_vector[0:3]; x_r, z_r, theta_r = robot_pose_est.flatten()
    R = np.diag([MEASUREMENT_NOISE_STD['range']**2, MEASUREMENT_NOISE_STD['bearing']**2])
    potential_associations = []
    for lm_id, lm_idx_in_state in landmark_map.items():
        lm_state_start_idx = 3 + 2 * lm_idx_in_state
        if lm_state_start_idx + 1 >= state_vector.shape[0] or lm_state_start_idx + 1 >= covariance_matrix.shape[0]: continue
        lm_x, lm_z = state_vector[lm_state_start_idx : lm_state_start_idx + 2].flatten()
        delta_x = lm_x - x_r; delta_z = lm_z - z_r; q = delta_x**2 + delta_z**2; sqrt_q = np.sqrt(q)
        if sqrt_q < 0.1: continue
        expected_range = sqrt_q; cos_r, sin_r = np.cos(theta_r), np.sin(theta_r)
        rel_x_expected = delta_x * cos_r + delta_z * sin_r; rel_z_expected = -delta_x * sin_r + delta_z * cos_r
        expected_bearing = angle_wrap(np.arctan2(rel_x_expected, rel_z_expected))
        z_hat = np.array([[expected_range], [expected_bearing]])
        H = np.zeros((2, state_dim)); H[0, 0] = -delta_x / sqrt_q; H[0, 1] = -delta_z / sqrt_q; H[0, 2] = 0
        H[0, lm_state_start_idx] = delta_x / sqrt_q; H[0, lm_state_start_idx + 1] = delta_z / sqrt_q
        d_rel_x_dxr = -cos_r; d_rel_x_dzr = -sin_r; d_rel_x_dthr = rel_z_expected
        d_rel_z_dxr = sin_r;  d_rel_z_dzr = -cos_r; d_rel_z_dthr = -rel_x_expected
        d_rel_x_dlmx = cos_r; d_rel_x_dlmz = sin_r; d_rel_z_dlmx = -sin_r; d_rel_z_dlmz = cos_r
        if q > 1e-6:
            H[1, 0] = (rel_z_expected * d_rel_x_dxr - rel_x_expected * d_rel_z_dxr) / q
            H[1, 1] = (rel_z_expected * d_rel_x_dzr - rel_x_expected * d_rel_z_dzr) / q
            H[1, 2] = (rel_z_expected * d_rel_x_dthr - rel_x_expected * d_rel_z_dthr) / q
            H[1, lm_state_start_idx] = (rel_z_expected * d_rel_x_dlmx - rel_x_expected * d_rel_z_dlmx) / q
            H[1, lm_state_start_idx + 1] = (rel_z_expected * d_rel_x_dlmz - rel_x_expected * d_rel_z_dlmz) / q
        try: S = H @ covariance_matrix @ H.T + R; S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError: continue
        for meas_idx, measurement in enumerate(measurements_relative):
            meas_range = np.sqrt(measurement.x**2 + measurement.y**2); meas_bearing = angle_wrap(np.arctan2(measurement.x, measurement.y))
            z_measured = np.array([[meas_range], [meas_bearing]]); y = z_measured - z_hat; y[1] = angle_wrap(y[1])
            dist_sq = y.T @ S_inv @ y
            if dist_sq < MAHALANOBIS_GATE_THRESHOLD**2:
                potential_associations.append({'meas_idx': meas_idx, 'lm_id': lm_id, 'dist_sq': dist_sq, 'H': H.copy(), 'S_inv': S_inv.copy(), 'y': y.copy()})
    final_associations = {}; used_measurements = set(); used_landmarks = set()
    potential_associations.sort(key=lambda x: x['dist_sq'])
    for assoc in potential_associations:
        if assoc['meas_idx'] not in used_measurements and assoc['lm_id'] not in used_landmarks:
            final_associations[assoc['meas_idx']] = assoc; used_measurements.add(assoc['meas_idx']); used_landmarks.add(assoc['lm_id'])
    num_updates = 0
    if final_associations:
        for meas_idx, assoc_info in final_associations.items():
            H = assoc_info['H']; S_inv = assoc_info['S_inv']; y = assoc_info['y']
            K = covariance_matrix @ H.T @ S_inv; state_vector = state_vector + K @ y; state_vector[2] = angle_wrap(state_vector[2])
            I = np.eye(covariance_matrix.shape[0]); covariance_matrix = (I - K @ H) @ covariance_matrix @ (I - K @ H).T + K @ R @ K.T
            num_updates += 1
    num_new_landmarks = 0; current_num_landmarks = len(landmark_map)
    for meas_idx, measurement in enumerate(measurements_relative):
        if meas_idx not in used_measurements:
            x_rel, z_rel = measurement.x, measurement.y; theta_r = state_vector[2, 0]; cos_t, sin_t = np.cos(theta_r), np.sin(theta_r)
            new_lm_x_world = state_vector[0, 0] + (x_rel * cos_t - z_rel * sin_t); new_lm_z_world = state_vector[1, 0] + (x_rel * sin_t + z_rel * cos_t)
            is_close_to_existing = False
            for lm_id_existing, lm_idx_in_state_existing in landmark_map.items():
                lm_state_start_idx_existing = 3 + 2 * lm_idx_in_state_existing
                if lm_state_start_idx_existing + 1 < len(state_vector):
                    lx_existing, lz_existing = state_vector[lm_state_start_idx_existing : lm_state_start_idx_existing + 2].flatten()
                    dist = np.sqrt((new_lm_x_world - lx_existing)**2 + (new_lm_z_world - lz_existing)**2)
                    if dist < NEW_LANDMARK_MIN_DISTANCE: is_close_to_existing = True; break
            if not is_close_to_existing:
                 new_state_vector = np.vstack((state_vector, [[new_lm_x_world], [new_lm_z_world]]))
                 old_dim = state_vector.shape[0]; new_dim = new_state_vector.shape[0]
                 new_covariance_matrix = np.zeros((new_dim, new_dim))
                 new_covariance_matrix[:old_dim, :old_dim] = covariance_matrix
                 new_covariance_matrix[old_dim:, old_dim:] = np.diag(INITIAL_LANDMARK_UNCERTAINTY_DIAG)
                 if new_state_vector.shape[0] != new_covariance_matrix.shape[0]: print(f"Error: Dim mismatch before update!"); continue
                 state_vector = new_state_vector; covariance_matrix = new_covariance_matrix
                 landmark_map[next_landmark_id] = current_num_landmarks; current_num_landmarks += 1; next_landmark_id += 1; num_new_landmarks += 1
    if num_new_landmarks > 0:
         print(f"EKF Update: Added {num_new_landmarks} new landmarks. Total: {len(landmark_map)}")
         expected_dim = 3 + 2*len(landmark_map)
         if state_vector.shape[0] != covariance_matrix.shape[0] or state_vector.shape[0] != expected_dim:
              print(f"CRITICAL Error: Dimension mismatch AFTER adding landmarks!")
         plot_needs_update = True
    elif num_updates > 0: plot_needs_update = True


# --- Triggered SLAM Step ---
def run_slam_step_on_demand(total_dist, total_dtheta_rad):
    global plot_needs_update
    if state_vector is None: print("SLAM not initialized."); return
    print(f"\n--- Triggered SLAM Step ---")
    print(f"Accumulated Motion: dist={total_dist:.3f} m, angle={np.degrees(total_dtheta_rad):.2f} deg")
    predict_step(total_dist, total_dtheta_rad); print("EKF Prediction Complete.")
    lidar_points_relative = simulate_lidar(robot_entity); print(f"LiDAR Scan: {len(lidar_points_relative)} points detected.")
    features_relative = find_centroids_from_lidar(lidar_points_relative, GROUPING_THRESHOLD, MIN_GROUP_SIZE); print(f"Feature Extraction: {len(features_relative)} centroids found.")
    if features_relative: update_step(features_relative); print("EKF Update Complete.")
    else: print("No features found for update.")
    plot_needs_update = True

# --- Visualization ---
def apply_zx_reflection_transform(x, z, theta_rad=None):
    display_x = z; display_z = x; display_theta_rad = None
    if theta_rad is not None: display_theta_rad = angle_wrap(np.pi/2.0 - theta_rad)
    return display_x, display_z, display_theta_rad

def update_ursina_visuals():
    global estimated_pose_marker, landmark_markers, state_vector
    if state_vector is None: return
    est_x, est_z, est_theta_rad = state_vector[0:3].flatten()
    display_x, display_z, display_theta_rad = apply_zx_reflection_transform(est_x, est_z, est_theta_rad)
    estimated_pose_marker.enabled = True; estimated_pose_marker.position = Vec3(display_x, 0.15, display_z)
    estimated_pose_marker.rotation_y = np.degrees(display_theta_rad)
    current_landmark_ids_in_state = set(landmark_map.keys()); existing_marker_ids = list(landmark_markers.keys())
    for lm_id in existing_marker_ids:
        lm_marker_entity = landmark_markers.get(lm_id)
        if lm_id not in current_landmark_ids_in_state:
            if lm_marker_entity: destroy(lm_marker_entity); del landmark_markers[lm_id]
        elif lm_id in landmark_map:
             lm_idx_in_state = landmark_map[lm_id]; lm_state_start_idx = 3 + 2 * lm_idx_in_state
             if lm_state_start_idx + 1 < len(state_vector):
                 lx, lz = state_vector[lm_state_start_idx : lm_state_start_idx + 2].flatten()
                 display_lx, display_lz, _ = apply_zx_reflection_transform(lx, lz)
                 if lm_marker_entity: lm_marker_entity.position = Vec3(display_lx, 0.1, display_lz)
    for lm_id, lm_idx_in_state in landmark_map.items():
        if lm_id not in landmark_markers:
            lm_state_start_idx = 3 + 2 * lm_idx_in_state
            if lm_state_start_idx + 1 < len(state_vector):
                lx, lz = state_vector[lm_state_start_idx : lm_state_start_idx + 2].flatten()
                display_lx, display_lz, _ = apply_zx_reflection_transform(lx, lz)
                marker = Entity(model='sphere', color=color.orange, scale=0.4, position=Vec3(display_lx, 0.1, display_lz), name=f'LM_{lm_id}')
                landmark_markers[lm_id] = marker

def update_slam_plot():
    global plot_ax, state_vector, covariance_matrix, landmark_map, robot_entity, plot_needs_update
    if state_vector is None or covariance_matrix is None: return
    plot_ax.clear(); labels_added = set(); plot_title_note = " (Display Mirrored)"
    if robot_entity:
        gt_x, gt_z = robot_entity.x, robot_entity.z; gt_theta_rad = np.radians(robot_entity.rotation_y)
        plot_ax.plot(gt_x, gt_z, 'bo', markersize=8, label='실제 로봇 위치')
        plot_ax.arrow(gt_x, gt_z, 1.0 * np.cos(gt_theta_rad), 1.0 * np.sin(gt_theta_rad), head_width=0.3, head_length=0.5, fc='blue', ec='blue')
    est_x, est_z, est_theta_rad = state_vector[0:3].flatten()
    display_x, display_z, display_theta_rad = apply_zx_reflection_transform(est_x, est_z, est_theta_rad)
    plot_ax.plot(display_x, display_z, 'go', markersize=8, label='추정 로봇 위치 (EKF)')
    plot_ax.arrow(display_x, display_z, 1.0 * np.cos(display_theta_rad), 1.0 * np.sin(display_theta_rad), head_width=0.3, head_length=0.5, fc='lime', ec='lime', alpha=0.7)
    if covariance_matrix is not None and covariance_matrix.shape[0] >= 2:
        robot_cov = covariance_matrix[0:2, 0:2]; label = '로봇 불확실성' if 'robot_unc' not in labels_added else None
        plot_covariance_ellipse(plot_ax, (display_x, display_z), robot_cov, n_std=2, alpha=0.3, color='lime', label=label)
        if label: labels_added.add('robot_unc')
    lm_display_x_coords, lm_display_z_coords = [], []
    if landmark_map:
        for lm_id, lm_idx_in_state in landmark_map.items():
            lm_state_start_idx = 3 + 2 * lm_idx_in_state
            if lm_state_start_idx + 1 < state_vector.shape[0] and lm_state_start_idx + 1 < covariance_matrix.shape[0]:
                lx, lz = state_vector[lm_state_start_idx : lm_state_start_idx + 2].flatten()
                display_lx, display_lz, _ = apply_zx_reflection_transform(lx, lz)
                lm_display_x_coords.append(display_lx); lm_display_z_coords.append(display_lz)
                lm_cov = covariance_matrix[lm_state_start_idx:lm_state_start_idx+2, lm_state_start_idx:lm_state_start_idx+2]
                label = '랜드마크 불확실성' if 'lm_unc' not in labels_added else None
                plot_covariance_ellipse(plot_ax, (display_lx, display_lz), lm_cov, n_std=2, alpha=0.4, color='orange', label=label)
                if label: labels_added.add('lm_unc')
        label = '추정 랜드마크 (EKF)' if 'lm_pos' not in labels_added else None
        if lm_display_x_coords: plot_ax.scatter(lm_display_x_coords, lm_display_z_coords, c='orange', marker='s', s=60, label=label)
        if label: labels_added.add('lm_pos')
    gt_obs_x = [obs.x for obs in ursina_obstacles]; gt_obs_z = [obs.z for obs in ursina_obstacles]
    if gt_obs_x:
        label = '실제 장애물 (GT)' if 'gt' not in labels_added else None
        plot_ax.scatter(gt_obs_x, gt_obs_z, c='gray', marker='X', s=80, label=label, alpha=0.7)
        if label: labels_added.add('gt')
    plot_ax.set_xlabel("X 좌표 (m)"); plot_ax.set_ylabel("Z 좌표 (m)")
    plot_ax.set_title(f"EKF-SLAM 상태 (랜드마크: {len(landmark_map)})" + plot_title_note)
    plot_ax.set_aspect('equal', adjustable='box'); plot_ax.grid(True)
    plot_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plot_ax.set_xlim(-MAP_SIZE, MAP_SIZE); plot_ax.set_ylim(-MAP_SIZE, MAP_SIZE)
    try: plt.tight_layout(rect=[0, 0, 0.85, 1]); plot_fig.canvas.draw(); plot_fig.canvas.flush_events()
    except Exception as e: print(f"Matplotlib plotting error: {e}")
    plot_needs_update = False

# --- Ursina Application Setup ---
app = Ursina()
camera.orthographic = False; camera.position = (0, 35, -35); camera.rotation_x = 45; camera.fov = 60
setup_world() # 월드 생성 (내부에서 랜덤 장애물 배치 호출)
initialize_slam() # SLAM 초기화
instructions = Text(text="[WASD]: Move | [QE]: Rotate | [L]: Localize/Map Update | [LClick]: Add Obstacle (GT) | [P]: Plot Update | [ESC]: Quit\n(Display Mirrored)",
                    origin=(-0.5, 0.5), scale=1.5, position=window.top_left + Vec2(0.01, -0.01), background=True)

# --- Input Handling ---
def input(key):
    global ground, robot_entity, plot_needs_update
    global accumulated_distance, accumulated_angle_rad
    if key == 'l':
        run_slam_step_on_demand(accumulated_distance, accumulated_angle_rad)
        accumulated_distance = 0.0; accumulated_angle_rad = 0.0 # 누적값 리셋
        plot_needs_update = True
    elif key == 'left mouse down' and mouse.hovered_entity == ground:
        if mouse.world_point: add_ground_truth_obstacle(Vec3(mouse.world_point.x, 0.5, mouse.world_point.z))
    elif key == 'p': plot_needs_update = True
    elif key == 'escape': app.quit()

# --- Update Loop ---
def update():
    global last_update_time, plot_needs_update
    global accumulated_distance, accumulated_angle_rad

    current_time = time.time(); dt = current_time - last_update_time
    if dt <= 1e-3: return
    last_update_time = current_time

    linear_velocity = 0.0; angular_velocity_deg = 0.0; moved = False
    if held_keys['w']: linear_velocity = ROBOT_MOVE_SPEED; moved=True
    if held_keys['s']: linear_velocity = -ROBOT_MOVE_SPEED / 2; moved=True
    if held_keys['q']: angular_velocity_deg = -ROBOT_TURN_SPEED; moved=True
    if held_keys['e']: angular_velocity_deg = ROBOT_TURN_SPEED; moved=True

    if moved:
         delta_dist = linear_velocity * dt; delta_theta_deg = angular_velocity_deg * dt
         robot_entity.position += robot_entity.forward * delta_dist
         robot_entity.rotation_y += delta_theta_deg
         if robot_entity.y < 0.1: robot_entity.y = 0.1
         accumulated_distance += delta_dist # 이동량 누적
         accumulated_angle_rad += np.radians(delta_theta_deg) # 각도 변화량 누적 (라디안)
         accumulated_angle_rad = angle_wrap(accumulated_angle_rad)
         update_sensor_range_visualization() # 로봇 이동 시 센서 범위 업데이트

    # SLAM 스텝은 'L' 키 입력 시 input() 함수 내에서 호출됨

    update_ursina_visuals() # Ursina 시각화는 매 프레임 업데이트
    if plot_needs_update: update_slam_plot() # 플롯 업데이트 필요 시 실행


# --- Start the Application ---
update_slam_plot()
app.run()

# --- Cleanup ---
plt.ioff()
plt.close(plot_fig)
print("Application closed.")