from ursina import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt # Matplotlib 사용

# --- Matplotlib Interactive Mode Setup ---
plt.ion() # 대화형 모드 켜기
plot_fig = None # Figure 객체 저장용 전역 변수
plot_ax = None  # Axes 객체 저장용 전역 변수

# --- Configuration ---
MAP_SIZE = 20
NUM_OBSTACLES = 15
SENSOR_RANGE = 8
SENSOR_FOV = 180
SENSOR_RAYS = 90
# 유사도 계산 및 매칭 확인을 위한 임계값 (거리)
MATCH_THRESHOLD = 0.8
# 계산 효율성을 위해 임계값의 제곱을 미리 계산
MATCH_THRESHOLD_SQ = MATCH_THRESHOLD**2

# --- Global Variables ---
global_obstacles_entities = []
global_obstacles_positions = [] # Vec3(x, 0, z)
robot = None
estimated_pose = None # Tuple (x, z, angle)
estimated_pose_marker = None
local_map_display_entities = []
ground = None
fov_lines = []
last_detected_local_points = [] # Vec2(rel_x, rel_z)
matched_global_indices = [] # 추정 위치 계산에 사용된 전역 장애물 인덱스 리스트
last_localization_score = -1.0 # 마지막 위치 추정 시의 최고 유사도 점수

# --- Helper Functions (Ursina related) ---
def generate_global_map():
    global global_obstacles_entities, global_obstacles_positions, ground
    ground = Entity(model='plane', scale=MAP_SIZE * 2, color=color.dark_gray, texture='white_cube', texture_scale=(MAP_SIZE, MAP_SIZE), collider='box', name='ground_plane')
    global_obstacles_positions = []
    for _ in range(NUM_OBSTACLES):
        pos = Vec3(random.uniform(-MAP_SIZE, MAP_SIZE), 0.5, random.uniform(-MAP_SIZE, MAP_SIZE))
        if pos.length() > 3: add_obstacle(pos)

def add_obstacle(position, scale_y=None):
    global global_obstacles_entities, global_obstacles_positions, matched_global_indices, last_localization_score
    if scale_y is None: scale_y = random.uniform(1, 3)
    obstacle = Entity(model='cube', position=position, color=color.gray, collider='box', scale_y=scale_y)
    global_obstacles_entities.append(obstacle)
    global_obstacles_positions.append(Vec3(position.x, 0, position.z)) # Store XZ
    # Reset localization info when map changes
    matched_global_indices = []
    last_localization_score = -1.0
    if estimated_pose_marker: estimated_pose_marker.enabled = False
    print(f"Obstacle added at {position}. Localization info reset.")
    # Optional: update plot if open
    # if plot_fig: update_plot_data_and_redraw()

def simulate_lidar(robot_entity):
    global last_detected_local_points
    detected_points_relative = []
    origin = robot_entity.world_position + Vec3(0, 0.1, 0)
    robot_rotation_y_rad = math.radians(robot_entity.world_rotation_y)
    start_angle = math.radians(-SENSOR_FOV / 2); end_angle = math.radians(SENSOR_FOV / 2)
    angle_step = (end_angle - start_angle) / (SENSOR_RAYS - 1) if SENSOR_RAYS > 1 else 0
    ignore_list = [robot_entity] + fov_lines
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
             detected_points_relative.append(Vec2(x_rel, z_rel))
    last_detected_local_points = detected_points_relative
    return detected_points_relative

def generate_local_map_visualization(relative_points): # Ursina visualization
    global local_map_display_entities
    for entity in local_map_display_entities: destroy(entity)
    local_map_display_entities.clear()
    display_offset = Vec3(10, 0, -15)
    origin_marker = Entity(model='sphere', scale=0.3, position=display_offset, color=color.red)
    local_map_display_entities.append(origin_marker)
    for point in relative_points:
        display_pos = display_offset + Vec3(point.x, 0.1, point.y)
        point_entity = Entity(model='sphere', scale=0.15, position=display_pos, color=color.yellow)
        local_map_display_entities.append(point_entity)

# --- Localization Logic ---

# !!! 여기가 수정된 유사도 계산 함수 !!!
def calculate_similarity(potential_pose, local_map_points_relative, global_map_points_xz):
    """Calculates similarity score using inverse distance weighting for matched points."""
    potential_pos = Vec3(potential_pose[0], 0, potential_pose[1])
    potential_angle_rad = math.radians(potential_pose[2])
    total_score = 0.0
    # 0으로 나누는 것을 방지하고 매우 가까운 점에 높은 점수를 주기 위한 작은 값
    epsilon = 0.1
    # MATCH_THRESHOLD_SQ는 전역으로 정의됨

    cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
    # global_map_points_xz는 Vec3 리스트이므로, XZ 튜플로 변환
    global_map_xz_tuples = [(p.x, p.z) for p in global_map_points_xz]

    if not local_map_points_relative: # 지역 스캔 포인트가 없으면 0점 반환
        return 0.0

    for local_pt in local_map_points_relative: # local_pt는 Vec2
        # 로컬 포인트를 월드 좌표로 변환 (가상)
        x_rot = local_pt.x * cos_a - local_pt.y * sin_a
        z_rot = local_pt.x * sin_a + local_pt.y * cos_a
        world_pt_guess_x = potential_pos.x + x_rot
        world_pt_guess_z = potential_pos.z + z_rot

        min_dist_sq = float('inf')
        # 가장 가까운 전역 장애물 포인트까지의 거리(제곱) 찾기
        for gx, gz in global_map_xz_tuples:
            dist_sq = (world_pt_guess_x - gx)**2 + (world_pt_guess_z - gz)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

        # 임계값 안에 들어오면, 역거리에 기반한 점수 추가
        if min_dist_sq < MATCH_THRESHOLD_SQ:
            distance = math.sqrt(min_dist_sq)
            total_score += 1.0 / (distance + epsilon) # 가까울수록 높은 점수

    # 가중치 합계를 최종 점수로 반환
    return total_score


def perform_localization():
    global robot, estimated_pose, estimated_pose_marker, global_obstacles_positions
    global last_localization_score, matched_global_indices # 전역 변수 사용

    print("--- Starting Localization ---")
    if not robot: print("Robot not initialized."); return

    # 1. Simulate Scan & Visualize Locally
    local_map_points = simulate_lidar(robot) # last_detected_local_points 업데이트됨
    print(f"Detected {len(local_map_points)} points locally.")
    generate_local_map_visualization(local_map_points)

    # 2. Check if Localization Possible
    if not local_map_points:
        print("No points detected, cannot localize.")
        estimated_pose = None; last_localization_score = -1.0; matched_global_indices = []
        if estimated_pose_marker: estimated_pose_marker.enabled = False
        return
    else:
        if estimated_pose_marker: estimated_pose_marker.enabled = True

    # 3. Search for Best Pose
    search_radius = 3.0; angle_search_range = 30
    pos_step = 0.5; angle_step = 5 # 그리드 간격 (필요시 조절)
    best_score = -1.0 # 점수는 0 이상이므로 -1.0으로 초기화

    if estimated_pose: # 이전 추정값이 있으면 거기서 시작
        start_pos = Vec3(estimated_pose[0], 0, estimated_pose[1])
        start_angle = estimated_pose[2]
    else: # 없으면 로봇의 현재 위치에서 시작
        start_pos = robot.position
        start_angle = robot.rotation_y
    current_best_pose = (start_pos.x, start_pos.z, start_angle) # 현재까지 최적 포즈

    search_count = 0
    for dx in np.arange(-search_radius, search_radius + pos_step, pos_step):
        for dz in np.arange(-search_radius, search_radius + pos_step, pos_step):
            for dangle in np.arange(-angle_search_range, angle_search_range + angle_step, angle_step):
                potential_x = start_pos.x + dx
                potential_z = start_pos.z + dz
                potential_angle = (start_angle + dangle) % 360
                potential_pose_tuple = (potential_x, potential_z, potential_angle)
                search_count += 1
                # *** 개선된 유사도 함수 호출 ***
                score = calculate_similarity(potential_pose_tuple, local_map_points, global_obstacles_positions)
                if score > best_score:
                    best_score = score
                    current_best_pose = potential_pose_tuple

    # 4. Store Best Pose and Score
    estimated_pose = current_best_pose
    last_localization_score = best_score # 최고 점수 저장
    print(f"Search completed. Checked {search_count} poses.")
    print(f"Best Estimated Pose: x={estimated_pose[0]:.2f}, z={estimated_pose[1]:.2f}, angle={estimated_pose[2]:.2f}")
    print(f"Actual Robot Pose:   x={robot.x:.2f}, z={robot.z:.2f}, angle={robot.rotation_y:.2f}")
    print(f"Best Score (Inverse Dist Sum): {last_localization_score:.4f}") # 점수 의미 명시

    # 5. Identify Matched Global Obstacles for Visualization
    matched_indices_set = set()
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose
        potential_pos = Vec3(est_x, 0, est_z)
        potential_angle_rad = math.radians(est_angle_deg)
        cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)

        for local_pt in local_map_points:
            x_rot = local_pt.x * cos_a - local_pt.y * sin_a
            z_rot = local_pt.x * sin_a + local_pt.y * cos_a
            world_pt_guess_x = potential_pos.x + x_rot
            world_pt_guess_z = potential_pos.z + z_rot

            min_dist_sq = float('inf')
            best_match_idx = -1
            for idx, global_pt in enumerate(global_obstacles_positions):
                dist_sq = (world_pt_guess_x - global_pt.x)**2 + (world_pt_guess_z - global_pt.z)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_match_idx = idx

            if best_match_idx != -1 and min_dist_sq < MATCH_THRESHOLD_SQ:
                matched_indices_set.add(best_match_idx)

    matched_global_indices = list(matched_indices_set) # 전역 리스트 업데이트
    print(f"Indices of matched global obstacles: {matched_global_indices}")

    # 6. Update Ursina Marker
    if estimated_pose_marker:
        estimated_pose_marker.position = Vec3(estimated_pose[0], 0.1, estimated_pose[1])
        estimated_pose_marker.rotation_y = estimated_pose[2]
    else:
        estimated_pose_marker = Entity(model='arrow', color=color.lime, scale=1.5,
                                       position=Vec3(estimated_pose[0], 0.1, estimated_pose[1]),
                                       rotation_y = estimated_pose[2])
        Entity(model='sphere', scale=0.5, color=color.green, parent=estimated_pose_marker, y=-0.2)
    print("--- Localization Finished ---")

def update_fov_visualization():
    global fov_lines
    if not robot: return
    for line in fov_lines: destroy(line)
    fov_lines.clear()
    origin = robot.world_position + Vec3(0, 0.05, 0)
    robot_rot_y_rad = math.radians(robot.world_rotation_y)
    angle_left_rad = robot_rot_y_rad + math.radians(SENSOR_FOV / 2)
    angle_right_rad = robot_rot_y_rad + math.radians(-SENSOR_FOV / 2)
    dir_left = Vec3(math.sin(angle_left_rad), 0, math.cos(angle_left_rad))
    dir_right = Vec3(math.sin(angle_right_rad), 0, math.cos(angle_right_rad))
    end_left = origin + dir_left * SENSOR_RANGE
    end_right = origin + dir_right * SENSOR_RANGE
    line_color = color.cyan; line_thickness = 2
    fov_lines.append(Entity(model=Mesh(vertices=[origin, end_left], mode='line', thickness=line_thickness), color=line_color))
    fov_lines.append(Entity(model=Mesh(vertices=[origin, end_right], mode='line', thickness=line_thickness), color=line_color))

# --- Matplotlib Plotting Function ---
def update_plot_data_and_redraw():
    global plot_fig, plot_ax, global_obstacles_positions, last_detected_local_points
    global robot, estimated_pose, matched_global_indices, last_localization_score

    global_obs_pos = global_obstacles_positions
    local_scan_relative = last_detected_local_points
    actual_pose = (robot.x, robot.z, robot.rotation_y) if robot else None

    if plot_fig is None:
        plot_fig, plot_ax = plt.subplots(figsize=(8.5, 8))

    plot_ax.clear()

    # 1. Global Obstacles (Matched vs Unmatched)
    unmatched_obs_x, unmatched_obs_z = [], []
    matched_obs_x, matched_obs_z = [], []
    if global_obs_pos:
        for idx, p in enumerate(global_obs_pos):
            if idx in matched_global_indices:
                matched_obs_x.append(p.x); matched_obs_z.append(p.z)
            else:
                unmatched_obs_x.append(p.x); unmatched_obs_z.append(p.z)
    plot_ax.scatter(unmatched_obs_x, unmatched_obs_z, c='grey', marker='s', s=100, label='Global Obstacles (Unmatched)')
    plot_ax.scatter(matched_obs_x, matched_obs_z, c='magenta', marker='s', s=120, label='Global Obstacles (Matched)', edgecolors='black')

    # 2. Local Scan (World Coords)
    scan_world_x, scan_world_z = [], []
    if local_scan_relative and actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose
        angle_rad = math.radians(robot_angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        for pt in local_scan_relative:
            x_rot = pt.x * cos_a - pt.y * sin_a; z_rot = pt.x * sin_a + pt.y * cos_a
            scan_world_x.append(robot_x + x_rot); scan_world_z.append(robot_z + z_rot)
    if scan_world_x:
        plot_ax.scatter(scan_world_x, scan_world_z, c='yellow', marker='o', s=30, label='Detected Scan (World)')

    # 3. Actual Robot Pose
    if actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose
        plot_ax.scatter(robot_x, robot_z, c='blue', marker='o', s=150, label='Actual Pose')
        angle_rad = math.radians(robot_angle_deg); arrow_len = 1.5
        plot_ax.arrow(robot_x, robot_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='blue', ec='blue')

    # 4. Estimated Robot Pose
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose
        plot_ax.scatter(est_x, est_z, c='lime', marker='o', s=150, label='Estimated Pose', alpha=0.7)
        angle_rad = math.radians(est_angle_deg); arrow_len = 1.5
        plot_ax.arrow(est_x, est_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='lime', ec='lime', alpha=0.7)

    # Formatting & Score Text
    plot_ax.set_xlabel("X coordinate"); plot_ax.set_ylabel("Z coordinate")
    title_text = "2D Map and Localization"
    if last_localization_score >= 0:
        # 점수 형식을 지수 표기법(e) 대신 소수점 표기
        title_text += f"\nSimilarity Score: {last_localization_score:.4f}"
    plot_ax.set_title(title_text)
    plot_ax.set_aspect('equal', adjustable='box')
    limit = MAP_SIZE * 1.1
    plot_ax.set_xlim(-limit, limit); plot_ax.set_ylim(-limit, limit)
    plot_ax.grid(True)
    plot_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    try:
        plot_fig.canvas.draw()
        plt.pause(0.01)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    except Exception as e:
        print(f"Matplotlib plotting error: {e}")

# --- Ursina Application Setup ---
app = Ursina()
generate_global_map()
robot = Entity(model='sphere', color=color.blue, collider='sphere', position=(0, 0.2, 0))
robot_forward = Entity(model='cube', scale=(0.1, 0.1, 0.5), color=color.red, parent=robot, z=0.3)
EditorCamera()
camera.y = 25; camera.rotation_x = 70
instructions = Text(
    text="WASD=Move, QE=Rotate, L=Localize & Update Plot, Left Click=Add Obstacle",
    origin=(-0.5, -0.5), scale=1.5, position=window.bottom_left + Vec2(0.01, 0.01)
)
update_fov_visualization()

# --- Input Handling ---
def input(key):
    global ground
    if key == 'l':
        perform_localization() # Updates globals: estimated_pose, score, matched_indices
        update_plot_data_and_redraw() # Uses updated globals
    elif key == 'left mouse down':
        if mouse.hovered_entity == ground:
            click_pos = mouse.world_point
            new_obstacle_pos = Vec3(click_pos.x, 0.5, click_pos.z)
            add_obstacle(new_obstacle_pos) # Resets localization info
            # Optional: update plot immediately?
            # if plot_fig: update_plot_data_and_redraw()
        else:
            print("Click on the ground plane (dark grey area) to add an obstacle.")

# --- Update Loop ---
def update():
    global robot
    if not robot: return
    move_speed = 5 * time.dt; turn_speed = 90 * time.dt
    moved_or_rotated = False
    if held_keys['w']: robot.position += robot.forward * move_speed; moved_or_rotated = True
    if held_keys['s']: robot.position -= robot.forward * move_speed; moved_or_rotated = True
    if held_keys['q']: robot.rotation_y -= turn_speed; moved_or_rotated = True
    if held_keys['e']: robot.rotation_y += turn_speed; moved_or_rotated = True
    if robot.y < 0.1: robot.y = 0.1
    if moved_or_rotated:
        update_fov_visualization()
        # Optionally reset matched indices on move to force recalculation if desired
        # global matched_global_indices, last_localization_score
        # matched_global_indices = []
        # last_localization_score = -1.0
        # if plot_fig: update_plot_data_and_redraw()


# --- Start the Application ---
app.run()