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
NUM_OBSTACLES = 50
SENSOR_RANGE = 8
SENSOR_FOV = 180
SENSOR_RAYS = 90
MATCH_THRESHOLD = 0.8
MATCH_THRESHOLD_SQ = MATCH_THRESHOLD**2
CIRCLE_SEGMENTS = 36
# 카메라 설정값 추가/수정
CAMERA_PAN_SPEED = 15 # 방향키 이동 속도
CAMERA_DEFAULT_DISTANCE = 30 # Top-down 뷰 기본 높이
CAMERA_ZOOM_SPEED = 2 # 마우스 휠 줌 속도 (작을수록 민감)

# --- Global Variables ---
global_obstacles_entities = []
global_obstacles_positions = [] # Vec3(x, 0, z)
robot = None
estimated_pose = None # Tuple (x, z, angle)
estimated_pose_marker = None
local_map_display_entities = []
ground = None
fov_lines = []
sensor_range_visual = None
last_detected_local_points = [] # Vec2(rel_x, rel_z)
matched_global_indices = []
last_localization_score = -1.0

# --- Helper Functions (Ursina related - unchanged) ---
def generate_global_map():
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
    global global_obstacles_entities, global_obstacles_positions, matched_global_indices, last_localization_score
    if scale_y is None: scale_y = random.uniform(1, 3)
    obstacle = Entity(model='cube', position=position, color=color.gray, collider='box', scale_y=scale_y)
    global_obstacles_entities.append(obstacle)
    global_obstacles_positions.append(Vec3(position.x, 0, position.z)) # Store XZ
    matched_global_indices = []; last_localization_score = -1.0
    if estimated_pose_marker: estimated_pose_marker.enabled = False
    print(f"Obstacle added at {position}. Localization info reset.")

def simulate_lidar(robot_entity):
    global last_detected_local_points, SENSOR_FOV, SENSOR_RANGE, SENSOR_RAYS
    detected_points_relative = []; origin = robot_entity.world_position + Vec3(0, 0.1, 0)
    robot_rotation_y_rad = math.radians(robot_entity.world_rotation_y); fov_rad = math.radians(SENSOR_FOV)
    start_angle = -fov_rad / 2; end_angle = fov_rad / 2
    angle_step = fov_rad / (SENSOR_RAYS - 1) if SENSOR_RAYS > 1 else 0
    ignore_list = [robot_entity] + fov_lines
    if sensor_range_visual: ignore_list.append(sensor_range_visual)
    for i in range(SENSOR_RAYS):
        current_angle_relative = start_angle + i * angle_step; world_angle = robot_rotation_y_rad + current_angle_relative
        direction = Vec3(math.sin(world_angle), 0, math.cos(world_angle)).normalized()
        hit_info = raycast(origin=origin, direction=direction, distance=SENSOR_RANGE, ignore=ignore_list, debug=False)
        if hit_info.hit and hit_info.entity != ground:
             hit_point_world = hit_info.world_point; relative_pos_world = hit_point_world - origin
             cos_a = math.cos(-robot_rotation_y_rad); sin_a = math.sin(-robot_rotation_y_rad)
             x_rel = relative_pos_world.x * cos_a - relative_pos_world.z * sin_a; z_rel = relative_pos_world.x * sin_a + relative_pos_world.z * cos_a
             detected_points_relative.append(Vec2(x_rel, z_rel))
    last_detected_local_points = detected_points_relative
    return detected_points_relative

def generate_local_map_visualization(relative_points):
    global local_map_display_entities
    for entity in local_map_display_entities: destroy(entity);
    local_map_display_entities.clear()
    display_offset = Vec3(10, 0, -15); origin_marker = Entity(model='sphere', scale=0.3, position=display_offset, color=color.red)
    local_map_display_entities.append(origin_marker)
    for point in relative_points:
        display_pos = display_offset + Vec3(point.x, 0.1, point.y)
        point_entity = Entity(model='sphere', scale=0.15, position=display_pos, color=color.yellow)
        local_map_display_entities.append(point_entity)

# --- Localization Logic (calculate_similarity, perform_localization - unchanged) ---
def calculate_similarity(potential_pose, local_map_points_relative, global_map_points_xz):
    global MATCH_THRESHOLD_SQ
    potential_pos = Vec3(potential_pose[0], 0, potential_pose[1]); potential_angle_rad = math.radians(potential_pose[2])
    total_score = 0.0; epsilon = 0.1; cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
    global_map_xz_tuples = [(p.x, p.z) for p in global_map_points_xz]
    if not local_map_points_relative: return 0.0
    for local_pt in local_map_points_relative:
        x_rot = local_pt.x * cos_a - local_pt.y * sin_a; z_rot = local_pt.x * sin_a + local_pt.y * cos_a
        world_pt_guess_x = potential_pos.x + x_rot; world_pt_guess_z = potential_pos.z + z_rot
        min_dist_sq = float('inf')
        for gx, gz in global_map_xz_tuples:
            dist_sq = (world_pt_guess_x - gx)**2 + (world_pt_guess_z - gz)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq
        if min_dist_sq < MATCH_THRESHOLD_SQ: total_score += 1.0 / (math.sqrt(min_dist_sq) + epsilon)
    return total_score

def perform_localization():
    global robot, estimated_pose, estimated_pose_marker, global_obstacles_positions, last_localization_score, matched_global_indices
    print("--- Starting Localization ---");
    if not robot: print("Robot not initialized."); return
    local_map_points = simulate_lidar(robot); print(f"Detected {len(local_map_points)} points locally.")
    generate_local_map_visualization(local_map_points)
    if not local_map_points:
        print("No points detected, cannot localize."); estimated_pose = None; last_localization_score = -1.0; matched_global_indices = []
        if estimated_pose_marker: estimated_pose_marker.enabled = False; return
    else:
        if estimated_pose_marker: estimated_pose_marker.enabled = True
    search_radius = 3.0; angle_search_range = 30; pos_step = 0.5; angle_step = 5; best_score = -1.0
    if estimated_pose: start_pos = Vec3(estimated_pose[0], 0, estimated_pose[1]); start_angle = estimated_pose[2]
    else: start_pos = robot.position; start_angle = robot.rotation_y
    current_best_pose = (start_pos.x, start_pos.z, start_angle); search_count = 0
    for dx in np.arange(-search_radius, search_radius + pos_step, pos_step):
        for dz in np.arange(-search_radius, search_radius + pos_step, pos_step):
            for dangle in np.arange(-angle_search_range, angle_search_range + angle_step, angle_step):
                potential_x = start_pos.x + dx; potential_z = start_pos.z + dz; potential_angle = (start_angle + dangle) % 360
                potential_pose_tuple = (potential_x, potential_z, potential_angle); search_count += 1
                score = calculate_similarity(potential_pose_tuple, local_map_points, global_obstacles_positions)
                if score > best_score: best_score = score; current_best_pose = potential_pose_tuple
    estimated_pose = current_best_pose; last_localization_score = best_score
    # ... (rest of perform_localization including print statements and marker update) ...
    print(f"Search completed. Checked {search_count} poses.")
    print(f"Best Estimated Pose: x={estimated_pose[0]:.2f}, z={estimated_pose[1]:.2f}, angle={estimated_pose[2]:.2f}")
    print(f"Actual Robot Pose:   x={robot.x:.2f}, z={robot.z:.2f}, angle={robot.rotation_y:.2f}")
    print(f"Best Score (Inverse Dist Sum): {last_localization_score:.4f}")
    matched_indices_set = set()
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose; potential_pos = Vec3(est_x, 0, est_z); potential_angle_rad = math.radians(est_angle_deg)
        cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
        for local_pt in local_map_points:
            x_rot = local_pt.x * cos_a - local_pt.y * sin_a; z_rot = local_pt.x * sin_a + local_pt.y * cos_a
            world_pt_guess_x = potential_pos.x + x_rot; world_pt_guess_z = potential_pos.z + z_rot
            min_dist_sq = float('inf'); best_match_idx = -1
            for idx, global_pt in enumerate(global_obstacles_positions):
                dist_sq = (world_pt_guess_x - global_pt.x)**2 + (world_pt_guess_z - global_pt.z)**2
                if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_match_idx = idx
            if best_match_idx != -1 and min_dist_sq < MATCH_THRESHOLD_SQ: matched_indices_set.add(best_match_idx)
    matched_global_indices = list(matched_indices_set); print(f"Indices of matched global obstacles: {matched_global_indices}")
    if estimated_pose_marker: estimated_pose_marker.position = Vec3(estimated_pose[0], 0.1, estimated_pose[1]); estimated_pose_marker.rotation_y = estimated_pose[2]
    else:
        estimated_pose_marker = Entity(model='arrow', color=color.lime, scale=1.5, position=Vec3(estimated_pose[0], 0.1, estimated_pose[1]), rotation_y = estimated_pose[2])
        Entity(model='sphere', scale=0.5, color=color.green, parent=estimated_pose_marker, y=-0.2)
    print("--- Localization Finished ---")

# --- Visualization Update Functions ---
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

# --- Matplotlib Plotting Function (unchanged) ---
def update_plot_data_and_redraw():
    global plot_fig, plot_ax, global_obstacles_positions, last_detected_local_points, robot, estimated_pose, matched_global_indices, last_localization_score
    global_obs_pos = global_obstacles_positions; local_scan_relative = last_detected_local_points; actual_pose = (robot.x, robot.z, robot.rotation_y) if robot else None
    if plot_fig is None: plot_fig, plot_ax = plt.subplots(figsize=(8.5, 8))
    plot_ax.clear(); unmatched_obs_x, unmatched_obs_z = [], []; matched_obs_x, matched_obs_z = [], []
    if global_obs_pos:
        for idx, p in enumerate(global_obs_pos):
            if idx in matched_global_indices: matched_obs_x.append(p.x); matched_obs_z.append(p.z)
            else: unmatched_obs_x.append(p.x); unmatched_obs_z.append(p.z)
    plot_ax.scatter(unmatched_obs_x, unmatched_obs_z, c='grey', marker='s', s=100, label='Global Obstacles (Unmatched)')
    plot_ax.scatter(matched_obs_x, matched_obs_z, c='magenta', marker='s', s=120, label='Global Obstacles (Matched)', edgecolors='black')
    scan_world_x, scan_world_z = [], []
    if local_scan_relative and actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose; angle_rad = math.radians(robot_angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        for pt in local_scan_relative: x_rot = pt.x * cos_a - pt.y * sin_a; z_rot = pt.x * sin_a + pt.y * cos_a; scan_world_x.append(robot_x + x_rot); scan_world_z.append(robot_z + z_rot)
    if scan_world_x: plot_ax.scatter(scan_world_x, scan_world_z, c='yellow', marker='o', s=30, label='Detected Scan (World)')
    if actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose; plot_ax.scatter(robot_x, robot_z, c='blue', marker='o', s=150, label='Actual Pose')
        angle_rad = math.radians(robot_angle_deg); arrow_len = 1.5; plot_ax.arrow(robot_x, robot_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='blue', ec='blue')
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose; plot_ax.scatter(est_x, est_z, c='lime', marker='o', s=150, label='Estimated Pose', alpha=0.7)
        angle_rad = math.radians(est_angle_deg); arrow_len = 1.5; plot_ax.arrow(est_x, est_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad), head_width=0.5, head_length=0.7, fc='lime', ec='lime', alpha=0.7)
    plot_ax.set_xlabel("X coordinate"); plot_ax.set_ylabel("Z coordinate"); title_text = "2D Map and Localization"
    if last_localization_score >= 0: title_text += f"\nSimilarity Score: {last_localization_score:.4f}"
    plot_ax.set_title(title_text); plot_ax.set_aspect('equal', adjustable='box'); limit = MAP_SIZE * 1.1; plot_ax.set_xlim(-limit, limit); plot_ax.set_ylim(-limit, limit); plot_ax.grid(True); plot_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    try: plot_fig.canvas.draw(); plt.pause(0.01); plt.tight_layout(rect=[0, 0, 0.85, 1])
    except Exception as e: print(f"Matplotlib plotting error: {e}")


# --- Ursina Application Setup ---
app = Ursina()
generate_global_map()
robot = Entity(model='sphere', color=color.blue, collider='sphere', position=(0, 0.2, 0))
robot_forward = Entity(model='cube', scale=(0.1, 0.1, 0.5), color=color.red, parent=robot, z=0.3)

# *** 카메라 설정: Top-down 고정 뷰 ***
camera.orthographic = True
camera.rotation = (90, 0, 0) # 위에서 아래(-Y) 방향으로 설정
camera.position = (0, CAMERA_DEFAULT_DISTANCE, 0) # 맵 중앙 위쪽
camera.fov = MAP_SIZE * 2.5 # 맵 크기에 맞춰 FOV 조절 (Orthographic에서는 View 크기)

# EditorCamera 제거됨

# Instructions Text Update
instructions = Text(
    text="QE=Rotate Robot, Arrow Keys=Pan Camera (Top-Down View)\n"
         "L=Localize & Plot, Left Click=Add Obstacle, Mouse Wheel=Zoom",
    origin=(-0.5, 0.5), scale=1.5, position=window.top_left + Vec2(0.01, -0.01), background=True
)

# --- Initial Setup Calls ---
update_fov_visualization()
update_sensor_range_visualization()


# --- Input Handling ---
def input(key):
    global ground, CAMERA_ZOOM_SPEED # Zoom speed 사용
    if key == 'l':
        perform_localization()
        update_plot_data_and_redraw()
    elif key == 'left mouse down':
        # Top-down 뷰에서만 장애물 추가
        if mouse.world_point:
            # Intersect plane to ensure click is on ground level
            plane_intersection = mouse.intersect_plane(plane_normal=Vec3(0,1,0), plane_origin=Vec3(0,0,0))
            if plane_intersection:
                 print(f"Adding obstacle near: {plane_intersection}")
                 add_obstacle(Vec3(plane_intersection.x, 0.5, plane_intersection.z))
            else:
                 print("Click on the ground plane (dark grey area) to add an obstacle.")
    # *** 추가: 마우스 휠 줌 기능 ***
    elif key == 'scroll up':
        # Orthographic zoom in = decrease fov
        camera.fov = max(1, camera.fov - CAMERA_ZOOM_SPEED)
        print(f"Camera FOV (Ortho Size): {camera.fov:.2f}")
    elif key == 'scroll down':
        # Orthographic zoom out = increase fov
        camera.fov += CAMERA_ZOOM_SPEED
        print(f"Camera FOV (Ortho Size): {camera.fov:.2f}")


# --- Update Loop ---
def update():
    global robot, sensor_range_visual
    if not robot: return

    # Robot Movement
    move_speed = 5 * time.dt; turn_speed = 90 * time.dt
    moved_or_rotated = False
    input_active = False # No input fields currently

    if held_keys['w'] and not input_active: robot.position += robot.forward * move_speed; moved_or_rotated = True
    if held_keys['s'] and not input_active: robot.position -= robot.forward * move_speed; moved_or_rotated = True
    if held_keys['q'] and not input_active: robot.rotation_y -= turn_speed; moved_or_rotated = True
    if held_keys['e'] and not input_active: robot.rotation_y += turn_speed; moved_or_rotated = True

    if robot.y < 0.1: robot.y = 0.1
    if moved_or_rotated:
        update_fov_visualization()
        update_sensor_range_visualization()

    # *** 수정된 카메라 패닝 로직 (Top-Down View) ***
    pan_amount = CAMERA_PAN_SPEED * time.dt
    # Top-down (ZX) view: Up/Down -> Z axis, Left/Right -> X axis
    vertical_vec = Vec3(0, 0, 1)
    horizontal_vec = Vec3(1, 0, 0)

    # Apply movement directly to camera position
    if held_keys['up arrow']: camera.position += vertical_vec * pan_amount
    if held_keys['down arrow']: camera.position -= vertical_vec * pan_amount
    if held_keys['left arrow']: camera.position -= horizontal_vec * pan_amount
    if held_keys['right arrow']: camera.position += horizontal_vec * pan_amount


# --- Start the Application ---
app.run()