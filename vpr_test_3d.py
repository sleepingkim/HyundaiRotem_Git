from ursina import *
import random
import math
import numpy as np
# Matplotlib 추가
import matplotlib.pyplot as plt

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

# --- Helper Functions (Ursina related - unchanged from previous version) ---
def generate_global_map():
    global global_obstacles_entities, global_obstacles_positions, ground
    ground = Entity(model='plane', scale=MAP_SIZE * 2, color=color.dark_gray, texture='white_cube', texture_scale=(MAP_SIZE, MAP_SIZE), collider='box', name='ground_plane')
    global_obstacles_positions = []
    for _ in range(NUM_OBSTACLES):
        pos = Vec3(random.uniform(-MAP_SIZE, MAP_SIZE), 0.5, random.uniform(-MAP_SIZE, MAP_SIZE))
        if pos.length() > 3: add_obstacle(pos)

def add_obstacle(position, scale_y=None):
    global global_obstacles_entities, global_obstacles_positions
    if scale_y is None: scale_y = random.uniform(1, 3)
    obstacle = Entity(model='cube', position=position, color=color.gray, collider='box', scale_y=scale_y)
    global_obstacles_entities.append(obstacle)
    global_obstacles_positions.append(Vec3(position.x, 0, position.z)) # Store XZ
    print(f"Obstacle added at {position}")
    # Obstacle added, maybe update plot if it's open? (Optional enhancement)
    # if plot_fig: update_plot_data_and_redraw() # Need a function like this

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

def calculate_similarity(potential_pose, local_map_points_relative, global_map_points_xz):
    potential_pos = Vec3(potential_pose[0], 0, potential_pose[1])
    potential_angle_rad = math.radians(potential_pose[2])
    score = 0; match_threshold_sq = 0.8**2
    cos_a = math.cos(potential_angle_rad); sin_a = math.sin(potential_angle_rad)
    global_map_xz_tuples = [(p.x, p.z) for p in global_map_points_xz]
    for local_pt in local_map_points_relative:
        x_rot = local_pt.x * cos_a - local_pt.y * sin_a
        z_rot = local_pt.x * sin_a + local_pt.y * cos_a
        world_pt_guess_x = potential_pos.x + x_rot
        world_pt_guess_z = potential_pos.z + z_rot
        min_dist_sq = float('inf')
        for gx, gz in global_map_xz_tuples:
            dist_sq = (world_pt_guess_x - gx)**2 + (world_pt_guess_z - gz)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq
        if min_dist_sq < match_threshold_sq: score += 1
    if local_map_points_relative: score /= len(local_map_points_relative)
    return score

def perform_localization():
    global robot, estimated_pose, estimated_pose_marker, global_obstacles_positions
    # ... (Localization logic remains the same as before) ...
    print("--- Starting Localization ---")
    if not robot: print("Robot not initialized."); return

    local_map_points = simulate_lidar(robot) # last_detected_local_points 업데이트됨
    print(f"Detected {len(local_map_points)} points locally.")
    generate_local_map_visualization(local_map_points)

    if not local_map_points:
        print("No points detected, cannot localize.")
        estimated_pose = None # 추정값 없음
        if estimated_pose_marker: estimated_pose_marker.enabled = False
        return
    else:
        if estimated_pose_marker: estimated_pose_marker.enabled = True

    search_radius = 3.0; angle_search_range = 30
    pos_step = 0.5; angle_step = 5
    best_score = -1

    if estimated_pose:
        start_pos = Vec3(estimated_pose[0], 0, estimated_pose[1])
        start_angle = estimated_pose[2]
    else:
        start_pos = robot.position
        start_angle = robot.rotation_y
    current_best_pose = (start_pos.x, start_pos.z, start_angle)

    search_count = 0
    for dx in np.arange(-search_radius, search_radius + pos_step, pos_step):
        for dz in np.arange(-search_radius, search_radius + pos_step, pos_step):
            for dangle in np.arange(-angle_search_range, angle_search_range + angle_step, angle_step):
                potential_x = start_pos.x + dx
                potential_z = start_pos.z + dz
                potential_angle = (start_angle + dangle) % 360
                potential_pose_tuple = (potential_x, potential_z, potential_angle)
                search_count += 1
                score = calculate_similarity(potential_pose_tuple, local_map_points, global_obstacles_positions)
                if score > best_score:
                    best_score = score
                    current_best_pose = potential_pose_tuple

    estimated_pose = current_best_pose
    print(f"Search completed. Checked {search_count} poses.")
    print(f"Best Estimated Pose: x={estimated_pose[0]:.2f}, z={estimated_pose[1]:.2f}, angle={estimated_pose[2]:.2f}")
    print(f"Actual Robot Pose:   x={robot.x:.2f}, z={robot.z:.2f}, angle={robot.rotation_y:.2f}")
    print(f"Best Score: {best_score:.4f}")

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
     # ... (FOV visualization logic remains the same) ...
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


# --- Matplotlib Plotting Function (Now Updates Existing Plot) ---
def update_plot_data_and_redraw():
    """Updates the Matplotlib plot with current data without blocking."""
    global plot_fig, plot_ax, global_obstacles_positions, last_detected_local_points, robot, estimated_pose

    # --- Get Current Data ---
    global_obs_pos = global_obstacles_positions
    local_scan_relative = last_detected_local_points
    if robot:
        actual_pose = (robot.x, robot.z, robot.rotation_y)
    else:
        actual_pose = None # Robot not ready
    # estimated_pose is already updated globally by perform_localization

    # --- Initialize Plot if it doesn't exist ---
    if plot_fig is None:
        plot_fig, plot_ax = plt.subplots(figsize=(8, 8))
        # Make sure the window appears (might be needed depending on backend)
        # plot_fig.show() # Sometimes needed, sometimes plt.pause is enough

    # --- Clear Previous Plot Elements ---
    plot_ax.clear()

    # --- Redraw Elements ---
    # 1. Global Obstacles
    if global_obs_pos:
        obs_x = [p.x for p in global_obs_pos]
        obs_z = [p.z for p in global_obs_pos]
        plot_ax.scatter(obs_x, obs_z, c='grey', marker='s', s=100, label='Global Obstacles')

    # 2. Local Scan (World Coords)
    scan_world_x = []
    scan_world_z = []
    if local_scan_relative and actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose
        angle_rad = math.radians(robot_angle_deg); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        for pt in local_scan_relative:
            x_rot = pt.x * cos_a - pt.y * sin_a
            z_rot = pt.x * sin_a + pt.y * cos_a
            scan_world_x.append(robot_x + x_rot)
            scan_world_z.append(robot_z + z_rot)
    if scan_world_x:
        plot_ax.scatter(scan_world_x, scan_world_z, c='yellow', marker='o', s=30, label='Detected Scan (World)')

    # 3. Actual Robot Pose
    if actual_pose:
        robot_x, robot_z, robot_angle_deg = actual_pose
        plot_ax.scatter(robot_x, robot_z, c='blue', marker='o', s=150, label='Actual Pose')
        angle_rad = math.radians(robot_angle_deg); arrow_len = 1.5
        plot_ax.arrow(robot_x, robot_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad),
                      head_width=0.5, head_length=0.7, fc='blue', ec='blue')

    # 4. Estimated Robot Pose
    if estimated_pose:
        est_x, est_z, est_angle_deg = estimated_pose
        plot_ax.scatter(est_x, est_z, c='lime', marker='o', s=150, label='Estimated Pose', alpha=0.7)
        angle_rad = math.radians(est_angle_deg); arrow_len = 1.5
        plot_ax.arrow(est_x, est_z, arrow_len * math.sin(angle_rad), arrow_len * math.cos(angle_rad),
                      head_width=0.5, head_length=0.7, fc='lime', ec='lime', alpha=0.7)

    # --- Re-apply Formatting ---
    plot_ax.set_xlabel("X coordinate")
    plot_ax.set_ylabel("Z coordinate")
    plot_ax.set_title("2D Map and Localization (Live Update)")
    plot_ax.set_aspect('equal', adjustable='box')
    limit = MAP_SIZE * 1.1
    plot_ax.set_xlim(-limit, limit)
    plot_ax.set_ylim(-limit, limit)
    plot_ax.grid(True)
    plot_ax.legend()

    # --- Draw and Process Events ---
    try:
        plot_fig.canvas.draw() # Redraw the canvas
        # Give matplotlib event loop a chance to process (very short pause)
        plt.pause(0.01) # Adjust if needed, smaller values are less blocking
    except Exception as e:
        print(f"Matplotlib plotting error: {e}")
        # Handle cases where the plot window might have been closed manually
        # Reset plot_fig and plot_ax so it gets recreated next time
        # plot_fig = None
        # plot_ax = None

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
        # 1. Perform localization (updates estimated_pose and last_detected_local_points)
        perform_localization()
        # 2. Update the plot (non-blocking)
        update_plot_data_and_redraw()

    elif key == 'left mouse down':
        if mouse.hovered_entity == ground:
            click_pos = mouse.world_point
            new_obstacle_pos = Vec3(click_pos.x, 0.5, click_pos.z)
            add_obstacle(new_obstacle_pos)
             # Optional: Update plot immediately when obstacle is added
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
    if moved_or_rotated: update_fov_visualization()

    # --- Optional: Auto-update plot while moving ---
    # If you want the plot to update more frequently (e.g., whenever robot moves)
    # Be cautious, this might impact performance more.
    # if moved_or_rotated and plot_fig:
    #     update_plot_data_and_redraw() # Call update here instead of just on 'L'


# --- Start the Application ---
app.run()

# --- Cleanup Matplotlib (Optional) ---
# plt.ioff() # Turn off interactive mode if needed after app closes
# plt.close(plot_fig) # Close the figure if it exists