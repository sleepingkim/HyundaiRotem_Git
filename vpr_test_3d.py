from ursina import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt # Matplotlib 사용

# --- Matplotlib Interactive Mode Setup ---
# Matplotlib 플롯을 실시간으로 업데이트하고 Ursina 창과 함께 표시하기 위한 설정입니다.
plt.ion() # 대화형 모드 활성화
plot_fig = None # Matplotlib Figure 객체를 저장할 변수 (처음엔 없음)
plot_ax = None  # Matplotlib Axes 객체를 저장할 변수 (처음엔 없음)

# --- Configuration ---
# 시뮬레이션 환경 및 알고리즘의 주요 파라미터(설정값)들입니다.
MAP_SIZE = 20           # 맵의 절반 크기 (맵은 -MAP_SIZE ~ +MAP_SIZE 범위)
NUM_OBSTACLES = 100      # 맵에 생성될 초기 장애물 개수
SENSOR_RANGE = 8        # 로봇 센서(LiDAR)의 최대 탐지 거리
SENSOR_FOV = 120       # 로봇 센서의 시야각 (도)
SENSOR_RAYS = 90        # 로봇 센서가 한 번에 쏘는 광선의 수
MATCH_THRESHOLD = 0.8   # 위치 추정 시, 로컬 특징점과 글로벌 장애물 위치가 '일치'한다고 판단하는 최대 거리 임계값
MATCH_THRESHOLD_SQ = MATCH_THRESHOLD**2 # 거리 비교 시 제곱근 계산을 피하기 위해 임계값의 제곱 사용
CIRCLE_SEGMENTS = 36    # 센서 범위 원 시각화 시 사용할 선분 개수
CAMERA_PAN_SPEED = 15   # 방향키로 카메라 이동 시 속도
CAMERA_ZOOM_SPEED = 1   # EditorCamera의 마우스 휠 줌 속도
ROBOT_MOVE_SPEED = 5    # WASD 키로 로봇 이동 시 속도
ROBOT_TURN_SPEED = 90   # QE 키로 로봇 회전 시 속도 (도/초)
# *** 단순 그룹핑 파라미터 추가 (튜닝 필요) ***
GROUPING_THRESHOLD = 1.0 # 같은 그룹으로 묶일 최대 거리 (월드 유닛)
MIN_GROUP_SIZE = 3       # 유효한 그룹(장애물)으로 인정할 최소 스캔 포인트 개수

# --- Global Variables ---
# 프로그램 전체에서 사용되는 변수들입니다.
global_obstacles_entities = [] # 맵 상의 장애물 Ursina Entity 객체 리스트
global_obstacles_positions = [] # 장애물의 중심 월드 좌표(Vec3(x, 0, z)) 리스트 (로컬라이제이션용)
robot = None                    # 로봇 Entity 객체
estimated_pose = None           # 추정된 로봇 자세 Tuple (x, z, angle)
estimated_pose_marker = None    # 추정된 자세 시각화용 녹색 화살표 Entity
# local_map_display_entities 제거됨 (오프셋 시각화 기능 삭제됨)
ground = None                   # 바닥 Entity 객체
fov_lines = []                  # FOV 시각화용 선 Entity 리스트
sensor_range_visual = None      # 센서 범위 원 시각화 Entity
last_detected_local_points = [] # 가장 최근 LiDAR 스캔 결과 (로봇 기준 상대 좌표 Vec2 리스트)
last_clustered_centroids_relative = [] # 스캔 포인트를 그룹핑한 후 계산된 중심점들의 리스트 (로봇 기준 상대 좌표 Vec2)
matched_global_indices = []     # 가장 최근 위치 추정 시, 추정된 로컬 중심점과 매칭된 전역 장애물의 인덱스 리스트
last_localization_score = -1.0  # 가장 최근 위치 추정 시의 최고 유사도 점수

# --- Helper Functions (Ursina related) ---
def generate_global_map():
    """기존 맵 요소를 지우고, 새 바닥과 무작위 장애물을 생성합니다."""
    global global_obstacles_entities, global_obstacles_positions, ground
    # 기존 Entity들 제거
    for obs in global_obstacles_entities: destroy(obs)
    global_obstacles_entities.clear(); global_obstacles_positions.clear()
    if ground: destroy(ground)
    # 새 바닥 생성
    ground = Entity(model='plane', scale=MAP_SIZE * 2, color=color.dark_gray, texture='white_cube', texture_scale=(MAP_SIZE, MAP_SIZE), collider='box', name='ground_plane')
    global_obstacles_positions = [] # 장애물 위치 리스트 초기화
    # 설정된 개수만큼 장애물 생성 시도
    for _ in range(NUM_OBSTACLES):
        # 맵 범위 내에서 무작위 위치 선정
        pos = Vec3(random.uniform(-MAP_SIZE, MAP_SIZE), 0.5, random.uniform(-MAP_SIZE, MAP_SIZE))
        # 너무 중앙에 가깝지 않으면 장애물 추가
        if pos.length() > 3: add_obstacle(pos)

def add_obstacle(position, scale_y=None):
    """주어진 위치에 단일 장애물 Entity를 생성하고 관련 리스트에 추가합니다."""
    global global_obstacles_entities, global_obstacles_positions, matched_global_indices, last_localization_score
    if scale_y is None: scale_y = random.uniform(1, 3) # 높이 랜덤 설정
    # 로봇과 너무 가까우면 추가하지 않음
    if robot and distance(position, robot.position) < 1.0:
         print("Cannot add obstacle too close to the robot."); return
    print(f"Creating obstacle entity at {position}") # 생성 위치 로그
    # Ursina Entity 생성
    obstacle = Entity(model='cube', position=position, color=color.gray, collider='box', scale_y=scale_y)
    global_obstacles_entities.append(obstacle) # Entity 리스트에 추가
    global_obstacles_positions.append(Vec3(position.x, 0, position.z)) # 로컬라이제이션용 XZ 좌표 리스트에 추가
    # 장애물이 추가되면 이전 로컬라이제이션 정보는 무효화됨
    matched_global_indices = []; last_localization_score = -1.0
    if estimated_pose_marker: estimated_pose_marker.enabled = False

def simulate_lidar(robot_entity):
    """로봇 위치에서 가상 LiDAR 스캔을 수행하여 로봇 기준 상대 좌표 점들을 반환합니다."""
    global last_detected_local_points, SENSOR_FOV, SENSOR_RANGE, SENSOR_RAYS
    detected_points_relative = []
    origin = robot_entity.world_position + Vec3(0, 0.1, 0) # 로봇 살짝 위에서 광선 발사
    robot_rotation_y_rad = math.radians(robot_entity.world_rotation_y) # 로봇 현재 각도 (라디안)
    fov_rad = math.radians(SENSOR_FOV) # 시야각 (라디안)
    start_angle = -fov_rad / 2; end_angle = fov_rad / 2 # 스캔 시작/끝 각도 (로봇 기준)
    angle_step = fov_rad / (SENSOR_RAYS - 1) if SENSOR_RAYS > 1 else 0 # 광선 간 각도 간격

    # 무시할 엔티티 목록 (로봇 자신, 시각화 요소 등)
    ignore_list = [robot_entity] + fov_lines
    if sensor_range_visual: ignore_list.append(sensor_range_visual)

    # 설정된 광선 수만큼 반복
    for i in range(SENSOR_RAYS):
        current_angle_relative = start_angle + i * angle_step # 로봇 기준 현재 광선 각도
        world_angle = robot_rotation_y_rad + current_angle_relative # 월드 기준 절대 각도
        direction = Vec3(math.sin(world_angle), 0, math.cos(world_angle)).normalized() # XZ 평면 방향 벡터 계산
        # 레이캐스트 수행
        hit_info = raycast(origin=origin, direction=direction, distance=SENSOR_RANGE, ignore=ignore_list, debug=False)
        # 충돌했고, 충돌 대상이 ground가 아니라면
        if hit_info.hit and hit_info.entity != ground:
             hit_point_world = hit_info.world_point # 충돌 지점 월드 좌표
             relative_pos_world = hit_point_world - origin # 로봇 중심 기준 상대 월드 벡터
             # 로봇의 회전을 없애는 변환 (로봇이 0도를 보고 있다고 가정했을 때의 상대 좌표 계산)
             cos_a = math.cos(-robot_rotation_y_rad); sin_a = math.sin(-robot_rotation_y_rad)
             x_rel = relative_pos_world.x * cos_a - relative_pos_world.z * sin_a
             z_rel = relative_pos_world.x * sin_a + relative_pos_world.z * cos_a
             detected_points_relative.append(Vec2(x_rel, z_rel)) # 상대 X, Z 좌표 저장 (Vec2)

    last_detected_local_points = detected_points_relative # 결과를 전역 변수에 저장
    return detected_points_relative

# generate_local_map_visualization 함수 제거됨

# --- 추가된 함수: 단순 거리 기반 그룹핑 및 평균점 계산 ---
def find_centroids_simple(local_scan_points, threshold, min_size):
    """
    주어진 거리 임계값(threshold) 내의 점들을 그룹화하고,
    최소 크기(min_size) 이상인 그룹의 평균점(centroid) 리스트를 반환합니다.
    """
    if not local_scan_points: return [] # 입력 점이 없으면 빈 리스트 반환

    centroids = [] # 계산된 평균점(중심점)들을 저장할 리스트
    remaining_points = local_scan_points[:] # 원본 리스트를 변경하지 않기 위해 복사
    num_points = len(remaining_points)
    processed_indices = [False] * num_points # 각 점이 처리되었는지(그룹에 속했는지) 추적하는 리스트

    # 모든 점을 순회
    for i in range(num_points):
        if processed_indices[i]: continue # 이미 처리된 점이면 건너뜀

        # 새 클러스터(그룹) 시작
        current_cluster = [remaining_points[i]] # 현재 점을 첫 멤버로 추가
        processed_indices[i] = True             # 처리됨으로 표시
        # 현재 클러스터에 새로 추가되어 이웃을 탐색해야 할 점들의 인덱스를 담는 큐(Queue) 역할 리스트
        cluster_indices_to_process = [i]
        idx_in_queue = 0 # 큐에서 처리할 인덱스

        # 큐에 처리할 점이 있는 동안 반복 (BFS와 유사)
        while idx_in_queue < len(cluster_indices_to_process):
            current_point_index = cluster_indices_to_process[idx_in_queue]
            p1 = remaining_points[current_point_index] # 기준점
            idx_in_queue += 1

            # 아직 처리되지 않은 다른 모든 점들과 거리 비교
            for j in range(num_points):
                if not processed_indices[j]: # 아직 그룹에 속하지 않은 점만 확인
                    p2 = remaining_points[j]
                    # 두 점(Vec2) 사이의 거리 제곱 계산 (제곱근 계산 생략하여 성능 향상)
                    dist_sq = (p1.x - p2.x)**2 + (p1.y - p2.y)**2 # Vec2의 y는 상대 z좌표임
                    # 거리가 임계값 제곱보다 작으면 같은 그룹으로 간주
                    if dist_sq < threshold**2:
                        current_cluster.append(p2)        # 그룹에 추가
                        processed_indices[j] = True       # 처리됨 표시
                        cluster_indices_to_process.append(j) # 이 점의 이웃도 나중에 탐색해야 함

        # 탐색 완료 후, 현재 그룹의 크기가 최소 크기 이상이면 평균점 계산
        if len(current_cluster) >= min_size:
            sum_x = sum(p.x for p in current_cluster)
            sum_z = sum(p.y for p in current_cluster) # Vec2.y는 상대 z
            count = len(current_cluster)
            centroid = Vec2(sum_x / count, sum_z / count) # 평균점(중심점) 계산
            centroids.append(centroid) # 결과 리스트에 추가

    return centroids # 계산된 모든 중심점 리스트 반환


# --- Localization Logic ---
def calculate_similarity(potential_pose, local_features_relative, global_map_points_xz):
    """가상 로봇 자세에서 로컬 특징점(중심점)들과 전역 장애물 위치 간의 유사도를 계산합니다."""
    global MATCH_THRESHOLD_SQ
    potential_pos = Vec3(potential_pose[0], 0, potential_pose[1]); potential_angle_rad = math.radians(potential_pose[2])
    total_score = 0.0; epsilon = 0.1; cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
    global_map_xz_tuples = [(p.x, p.z) for p in global_map_points_xz] # 비교 편의를 위해 튜플 리스트로 변환
    if not local_features_relative: return 0.0 # 특징점이 없으면 점수 0

    # 각 로컬 특징점(중심점)에 대해 반복
    for feature_rel in local_features_relative:
        # 특징점을 'potential_pose'에 따라 가상의 월드 좌표로 변환
        x_rot = feature_rel.x * cos_a - feature_rel.y * sin_a
        z_rot = feature_rel.x * sin_a + feature_rel.y * cos_a
        world_pt_guess_x = potential_pos.x + x_rot
        world_pt_guess_z = potential_pos.z + z_rot
        min_dist_sq = float('inf') # 가장 가까운 전역 장애물과의 최소 거리 제곱값 초기화
        # 모든 전역 장애물 위치와 거리 비교
        for gx, gz in global_map_xz_tuples:
            dist_sq = (world_pt_guess_x - gx)**2 + (world_pt_guess_z - gz)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq # 최소 거리 업데이트
        # 최소 거리가 임계값 이내이면 점수 추가 (역거리 가중)
        if min_dist_sq < MATCH_THRESHOLD_SQ: total_score += 1.0 / (math.sqrt(min_dist_sq) + epsilon)
    return total_score # 최종 유사도 점수 반환

def perform_localization():
    """단순 그룹핑으로 얻은 평균점을 이용해 위치를 추정합니다."""
    global robot, estimated_pose, estimated_pose_marker, global_obstacles_positions
    global last_localization_score, matched_global_indices, last_clustered_centroids_relative
    global GROUPING_THRESHOLD, MIN_GROUP_SIZE # 그룹핑 파라미터 사용

    print("--- Starting Simplified Centroid Localization ---");
    if not robot: print("Robot not initialized."); return

    # 1. LiDAR 스캔 수행
    local_scan_points = simulate_lidar(robot)
    print(f"Detected {len(local_scan_points)} raw points locally.")

    # 2. *** 스캔 포인트 그룹핑 및 평균점(중심점) 계산 ***
    last_clustered_centroids_relative = find_centroids_simple(local_scan_points, GROUPING_THRESHOLD, MIN_GROUP_SIZE)
    print(f"Found {len(last_clustered_centroids_relative)} centroids (groups >= {MIN_GROUP_SIZE} points).")

    # 3. 로컬라이제이션 가능 여부 확인 (유효한 중심점이 있는지)
    if not last_clustered_centroids_relative:
        print("No valid centroids found, cannot localize."); estimated_pose = None; last_localization_score = -1.0; matched_global_indices = []
        if estimated_pose_marker: estimated_pose_marker.enabled = False; return
    else:
        if estimated_pose_marker: estimated_pose_marker.enabled = True

    # 4. 최적 자세 탐색 (Grid Search)
    # 탐색 범위 및 간격 설정
    search_radius = 3.0; angle_search_range = 30; pos_step = 0.5; angle_step = 5
    best_score = -1.0 # 최고 점수 초기화
    # 탐색 시작점 설정 (이전 추정값 또는 현재 로봇 위치)
    if estimated_pose: start_pos = Vec3(estimated_pose[0], 0, estimated_pose[1]); start_angle = estimated_pose[2]
    else: start_pos = robot.position; start_angle = robot.rotation_y
    current_best_pose = (start_pos.x, start_pos.z, start_angle); search_count = 0

    # 설정된 범위 내 모든 후보 자세(potential_pose)에 대해 반복
    for dx in np.arange(-search_radius, search_radius + pos_step, pos_step):
        for dz in np.arange(-search_radius, search_radius + pos_step, pos_step):
            for dangle in np.arange(-angle_search_range, angle_search_range + angle_step, angle_step):
                # 후보 자세 생성
                potential_x = start_pos.x + dx; potential_z = start_pos.z + dz; potential_angle = (start_angle + dangle) % 360
                potential_pose_tuple = (potential_x, potential_z, potential_angle); search_count += 1
                # *** 유사도 계산 시, 원본 스캔 대신 '평균점(중심점)' 리스트 사용 ***
                score = calculate_similarity(potential_pose_tuple, last_clustered_centroids_relative, global_obstacles_positions)
                # 최고 점수 업데이트
                if score > best_score: best_score = score; current_best_pose = potential_pose_tuple

    # 5. 최적 결과 저장
    estimated_pose = current_best_pose; last_localization_score = best_score
    print(f"Search completed. Checked {search_count} poses.")
    print(f"Best Estimated Pose: x={estimated_pose[0]:.2f}, z={estimated_pose[1]:.2f}, angle={estimated_pose[2]:.2f}")
    print(f"Actual Robot Pose:   x={robot.x:.2f}, z={robot.z:.2f}, angle={robot.rotation_y:.2f}")
    print(f"Best Score (Centroid Inverse Dist Sum): {last_localization_score:.4f}") # 점수 의미 명시

    # 6. 매칭된 전역 장애물 인덱스 찾기 (시각화용) - 중심점 기준
    matched_indices_set = set()
    if estimated_pose and last_clustered_centroids_relative:
        est_x, est_z, est_angle_deg = estimated_pose
        potential_pos = Vec3(est_x, 0, est_z); potential_angle_rad = math.radians(est_angle_deg)
        cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
        match_threshold_sq_now = MATCH_THRESHOLD**2

        for centroid_rel in last_clustered_centroids_relative: # 각 중심점에 대해
            # 최적 추정 자세 기준으로 월드 좌표 계산
            x_rot = centroid_rel.x * cos_a - centroid_rel.y * sin_a; z_rot = centroid_rel.x * sin_a + centroid_rel.y * cos_a
            world_pt_guess_x = potential_pos.x + x_rot; world_pt_guess_z = potential_pos.z + z_rot
            min_dist_sq = float('inf'); best_match_idx = -1
            # 가장 가까운 전역 장애물 찾기
            for idx, global_pt in enumerate(global_obstacles_positions):
                dist_sq = (world_pt_guess_x - global_pt.x)**2 + (world_pt_guess_z - global_pt.z)**2
                if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_match_idx = idx
            # 임계값 내에 있으면 해당 전역 장애물 인덱스 저장
            if best_match_idx != -1 and min_dist_sq < match_threshold_sq_now:
                matched_indices_set.add(best_match_idx)
    matched_global_indices = list(matched_indices_set); print(f"Indices of matched global obstacles: {matched_global_indices}")

    # 7. Ursina 추정 위치 마커 업데이트
    if estimated_pose_marker: estimated_pose_marker.position = Vec3(estimated_pose[0], 0.1, estimated_pose[1]); estimated_pose_marker.rotation_y = estimated_pose[2]
    else:
        estimated_pose_marker = Entity(model='arrow', color=color.lime, scale=1.5, position=Vec3(estimated_pose[0], 0.1, estimated_pose[1]), rotation_y = estimated_pose[2])
        Entity(model='sphere', scale=0.5, color=color.green, parent=estimated_pose_marker, y=-0.2)
    print("--- Simplified Centroid Localization Finished ---")


# --- Visualization Update Functions ---
def update_fov_visualization():
    global fov_lines, SENSOR_FOV, SENSOR_RANGE;
    if not robot: return;
    # 이전 FOV 선 제거
    for line in fov_lines: destroy(line);
    fov_lines.clear()
    # 현재 로봇 위치/방향 기준으로 새로 그리기
    origin = robot.world_position + Vec3(0, 0.05, 0); robot_rot_y_rad = math.radians(robot.world_rotation_y)
    fov_rad = math.radians(SENSOR_FOV); angle_left_rad = robot_rot_y_rad + fov_rad / 2; angle_right_rad = robot_rot_y_rad - fov_rad / 2
    dir_left = Vec3(math.sin(angle_left_rad), 0, math.cos(angle_left_rad)); dir_right = Vec3(math.sin(angle_right_rad), 0, math.cos(angle_right_rad))
    end_left = origin + dir_left * SENSOR_RANGE; end_right = origin + dir_right * SENSOR_RANGE
    line_color = color.cyan; line_thickness = 2
    # 새로 생성하여 리스트에 추가
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

# --- Matplotlib Plotting Function (Centroid Version) ---
def update_plot_data_and_redraw():
    """Matplotlib 플롯 업데이트 (군집 중심점 표시)"""
    global plot_fig, plot_ax, global_obstacles_positions, robot, estimated_pose
    global matched_global_indices, last_localization_score, last_clustered_centroids_relative # 중심점 데이터 사용

    global_obs_pos = global_obstacles_positions
    actual_pose = (robot.x, robot.z, robot.rotation_y) if robot else None

    if plot_fig is None: plot_fig, plot_ax = plt.subplots(figsize=(8.5, 8))
    plot_ax.clear()

    # 1. Plot Global Obstacles (Matched vs Unmatched)
    unmatched_obs_x, unmatched_obs_z = [], []; matched_obs_x, matched_obs_z = [], []
    if global_obs_pos:
        for idx, p in enumerate(global_obs_pos):
            if idx in matched_global_indices: matched_obs_x.append(p.x); matched_obs_z.append(p.z)
            else: unmatched_obs_x.append(p.x); unmatched_obs_z.append(p.z)
    plot_ax.scatter(unmatched_obs_x, unmatched_obs_z, c='grey', marker='s', s=100, label='Global Obstacles (Unmatched)')
    plot_ax.scatter(matched_obs_x, matched_obs_z, c='magenta', marker='s', s=120, label='Global Obstacles (Matched)', edgecolors='black')

    # 2. *** Plot Clustered Centroids (Transformed to World) ***
    centroids_world_x, centroids_world_z = [], []
    if last_clustered_centroids_relative and actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose
        angle_rad = math.radians(robot_angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        for pt in last_clustered_centroids_relative: # 중심점 사용
            x_rot = pt.x * cos_a - pt.y * sin_a; z_rot = pt.x * sin_a + pt.y * cos_a
            centroids_world_x.append(robot_x + x_rot); centroids_world_z.append(robot_z + z_rot)
    # 빨간색 X 마커로 중심점 표시
    plot_ax.scatter(centroids_world_x, centroids_world_z, c='red', marker='x', s=80, label='Detected Centers (World)')

    # 3. Plot Actual Robot Pose
    if actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose; plot_ax.scatter(robot_x, robot_z, c='blue', marker='o', s=150, label='Actual Pose')
        angle_rad = math.radians(robot_angle_deg); arrow_len = 1.5; plot_ax.arrow(robot_x, robot_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='blue', ec='blue')
    # 4. Plot Estimated Robot Pose
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose; plot_ax.scatter(est_x, est_z, c='lime', marker='o', s=150, label='Estimated Pose', alpha=0.7)
        angle_rad = math.radians(est_angle_deg); arrow_len = 1.5; plot_ax.arrow(est_x, est_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='lime', ec='lime', alpha=0.7)

    # Formatting & Score Text
    plot_ax.set_xlabel("X coordinate"); plot_ax.set_ylabel("Z coordinate"); title_text = "2D Map and Centroid Localization" # 제목 변경
    if last_localization_score >= 0: title_text += f"\nSimilarity Score: {last_localization_score:.4f}"
    plot_ax.set_title(title_text); plot_ax.set_aspect('equal', adjustable='box'); limit = MAP_SIZE * 1.1; plot_ax.set_xlim(-limit, limit); plot_ax.set_ylim(-limit, limit); plot_ax.grid(True); plot_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    try: plot_fig.canvas.draw(); plt.pause(0.01); plt.tight_layout(rect=[0, 0, 0.85, 1])
    except Exception as e: print(f"Matplotlib plotting error: {e}")


# --- Ursina Application Setup ---
app = Ursina()
generate_global_map()
robot = Entity(model='sphere', color=color.blue, collider='sphere', position=(0, 0.2, 0))
robot_forward = Entity(model='cube', scale=(0.1, 0.1, 0.5), color=color.red, parent=robot, z=0.3)
ec = EditorCamera(rotation_speed=100, panning_speed=100, zoom_speed=CAMERA_ZOOM_SPEED)
camera.position = (0, 25, -25); camera.rotation_x = 45

instructions = Text(
    text="WASD/Mouse=Move/Rotate Camera, Arrow Keys=Pan Camera\n"
         "QE=Rotate Robot, L=Localize(Centroid) & Plot, Left Click=Add Obstacle", # 로컬라이제이션 방식 명시
    origin=(-0.5, 0.5), scale=1.5, position=window.top_left + Vec2(0.01, -0.01), background=True
)

# --- Initial Setup Calls ---
update_fov_visualization()
update_sensor_range_visualization()

# --- Input Handling ---
def input(key):
    global ground, robot
    if key == 'l':
        perform_localization() # 군집화(단순 그룹핑) 기반 로컬라이제이션 호출
        update_plot_data_and_redraw()
    elif key == 'left mouse down':
        # 장애물 추가 로직 (Raycast 사용)
        origin = camera.world_position
        if mouse.world_point:
            direction = (mouse.world_point - origin).normalized()
            hit_info = raycast(origin=origin, direction=direction, distance=inf,
                               ignore=[robot] + fov_lines + ([sensor_range_visual] if sensor_range_visual else []))
            if hit_info.hit and hit_info.entity == ground:
                click_pos = hit_info.world_point; add_obstacle(Vec3(click_pos.x, 0.5, click_pos.z))
            elif hit_info.hit: print(f"Click hit {hit_info.entity.name if hasattr(hit_info.entity, 'name') else 'unnamed entity'}, not ground.")
            else: print("Click ray did not hit the ground plane.")
        else: print("Cannot determine mouse target in 3D space to cast ray.")
    # Zoom은 EditorCamera가 처리

# --- Update Loop ---
def update():
    global robot, sensor_range_visual
    if not robot: return
    move_speed = ROBOT_MOVE_SPEED * time.dt; turn_speed = ROBOT_TURN_SPEED * time.dt; moved_or_rotated = False
    input_active = False

    if held_keys['w'] and not input_active: robot.position += robot.forward * move_speed; moved_or_rotated = True
    if held_keys['s'] and not input_active: robot.position -= robot.forward * move_speed; moved_or_rotated = True
    if held_keys['a'] and not input_active: robot.position -= robot.right * move_speed; moved_or_rotated = True
    if held_keys['d'] and not input_active: robot.position += robot.right * move_speed; moved_or_rotated = True
    if held_keys['q'] and not input_active: robot.rotation_y -= turn_speed; moved_or_rotated = True
    if held_keys['e'] and not input_active: robot.rotation_y += turn_speed; moved_or_rotated = True

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
        except: pass

# --- Start the Application ---
app.run()