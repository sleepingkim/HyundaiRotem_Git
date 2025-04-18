import sys
import os

# (Ursina 경로 설정 코드는 필요시 유지)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# ursina_path = os.path.join(current_dir, '..')
# sys.path.append(ursina_path)

try:
    from ursina import *
except ImportError as e:
    print(f"Error importing Ursina: {e}")
    print("Make sure Ursina is installed (`pip install ursina`) and accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during Ursina import: {e}")
    sys.exit(1)


# --- Required Libraries ---
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For drawing obstacles
import pandas as pd # For DataFrame storage
# import threading # 스레딩 제거

# --- Simulation Parameters ---
NUM_OBSTACLES = 100
AREA_SIZE = 60
AGENT_SPEED = 5
ROTATION_SPEED = 100
AGENT_HEIGHT = 0.5
LIDAR_RANGE = 15
LIDAR_FOV = 150
NUM_LIDAR_RAYS = 90
LIDAR_VIS_HEIGHT = 0.6
LIDAR_COLOR = color.cyan
OBSTACLE_TAG = "obstacle"
SCAN_INTERVAL = 1.0 # 초 단위 스캔 간격 (맵핑 데이터 저장용)

# --- Application Setup ---
app = Ursina(
    title="LiDAR Mapping & Ground Truth (Blocking Plot)",
    borderless=False,
)

# --- Mouse Setup ---
mouse.visible = True
mouse.locked = False

# --- Global Variables ---
scan_timer = 0.0
scan_history = []           # Stores {'pose': (x, z, rot_rad), 'relative_points': [(rx, ry), ...]}
obstacle_df = None          # DataFrame to store obstacle ground truth
ground_info = {'size': AREA_SIZE, 'center_x': 0, 'center_z': 0} # Ground info

# --- Environment Setup (Ground, Sky, Obstacles) ---
ground = Entity(model='plane', scale=(AREA_SIZE, 1, AREA_SIZE), color=color.light_gray,
                texture='white_cube', texture_scale=(AREA_SIZE/2, AREA_SIZE/2), collider='box')
sky = Sky()
obstacles_list_for_df = [] # Temp list to build DataFrame
obstacles = []             # List of Ursina entities
min_obstacle_distance_from_spawn = 3.0

for i in range(NUM_OBSTACLES):
    while True:
        pos_x = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5)
        pos_z = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5)
        distance_from_spawn = math.sqrt(pos_x**2 + pos_z**2)
        if distance_from_spawn >= min_obstacle_distance_from_spawn:
            break
    scale_x = random.uniform(0.5, 3)
    scale_y = random.uniform(1, 4)
    scale_z = random.uniform(0.5, 3)
    pos_y = scale_y / 2
    obstacle_color = color.random_color()

    # Create Ursina entity
    obs_entity = Entity(model='cube', position=(pos_x, pos_y, pos_z), scale=(scale_x, scale_y, scale_z),
                        color=obstacle_color, collider='box', tag=OBSTACLE_TAG, name=f"obstacle_{i}")
    obstacles.append(obs_entity)

    # Store info for DataFrame
    obstacles_list_for_df.append({
        'name': f"obstacle_{i}",
        'center_x': pos_x,
        'center_z': pos_z,
        'width': scale_x,
        'depth': scale_z,
        'height': scale_y,
        'rotation_y': 0 # Assuming initial cubes are axis-aligned
    })

# --- Create Obstacle DataFrame ---
obstacle_df = pd.DataFrame(obstacles_list_for_df)
print(f"Created DataFrame with {len(obstacle_df)} obstacles.")

# --- Agent Setup ---
agent = Entity(model='sphere', color=color.blue, position=(0, AGENT_HEIGHT, 0),
               collider='sphere', scale=1)

# --- Camera Setup ---
camera.parent = agent
camera.position = (0, 10, -12)
camera.rotation_x = 45
camera.rotation_y = 0
camera.fov = 75

# --- LiDAR Visualization (Lines in Ursina) ---
lidar_lines = []
def update_lidar_visualization():
    # (Function remains the same)
    global lidar_lines
    for line in lidar_lines: destroy(line)
    lidar_lines.clear()
    start_angle = agent.world_rotation_y - LIDAR_FOV / 2
    angle_step = LIDAR_FOV / (NUM_LIDAR_RAYS - 1) if NUM_LIDAR_RAYS > 1 else 0
    origin = agent.world_position + Vec3(0, LIDAR_VIS_HEIGHT - AGENT_HEIGHT, 0)
    for i in range(NUM_LIDAR_RAYS):
        current_angle_deg = start_angle + i * angle_step
        current_angle_rad = math.radians(current_angle_deg)
        direction = Vec3(math.sin(current_angle_rad), 0, math.cos(current_angle_rad)).normalized()
        hit_info = raycast(origin, direction, distance=LIDAR_RANGE, ignore=[agent,], debug=False)
        if hit_info.hit:
            end_point = hit_info.world_point; line_color = color.red if hit_info.entity != ground else LIDAR_COLOR
        else:
            end_point = origin + direction * LIDAR_RANGE; line_color = LIDAR_COLOR
        if distance(origin, end_point) > 0.01:
            line = Entity(model=Mesh(vertices=[origin, end_point], mode='line', thickness=2), color=line_color)
            lidar_lines.append(line)


# --- Function to Perform Scan and Generate Relative Map Data ---
def generate_relative_lidar_map():
    # (Function remains the same)
    relative_points = []
    agent_pos_world = agent.world_position
    agent_rot_y_rad = math.radians(agent.world_rotation_y)
    scan_origin = agent_pos_world + Vec3(0, LIDAR_VIS_HEIGHT - AGENT_HEIGHT, 0)
    angle_step_rad = math.radians(LIDAR_FOV / (NUM_LIDAR_RAYS - 1)) if NUM_LIDAR_RAYS > 1 else 0
    start_angle_world_rad = math.radians(agent.world_rotation_y - LIDAR_FOV / 2)
    for i in range(NUM_LIDAR_RAYS):
        ray_angle_world_rad = start_angle_world_rad + i * angle_step_rad
        ray_direction_world = Vec3(math.sin(ray_angle_world_rad), 0, math.cos(ray_angle_world_rad)).normalized()
        hit_info = raycast(scan_origin, ray_direction_world, distance=LIDAR_RANGE, ignore=[agent,], debug=False)
        if hit_info.hit and hit_info.entity != ground:
            hit_point_world = hit_info.world_point
            world_vec = hit_point_world - agent_pos_world
            world_vec_xz = Vec2(world_vec.x, world_vec.z)
            cos_a = math.cos(-agent_rot_y_rad); sin_a = math.sin(-agent_rot_y_rad)
            relative_x = world_vec_xz.x * cos_a - world_vec_xz.y * sin_a
            relative_y = world_vec_xz.x * sin_a + world_vec_xz.y * cos_a
            relative_points.append((relative_x, relative_y))
    return relative_points

# --- Function to Build and Visualize GLOBAL LiDAR Map ---
def plot_global_lidar_map(history):
    """Builds and displays the accumulated LiDAR map (using blocking plt.show())."""
    # (Function remains the same as before)
    print(f"Building global LiDAR map from {len(history)} scans...")
    global_map_points_x, global_map_points_z = [], []
    agent_trajectory_x, agent_trajectory_z = [], []
    if not history: print("No scan history to plot."); return
    for scan_record in history:
        pose = scan_record['pose']
        relative_points = scan_record['relative_points']
        scan_pos_x, scan_pos_z, scan_rot_rad = pose
        agent_trajectory_x.append(scan_pos_x); agent_trajectory_z.append(scan_pos_z)
        cos_a, sin_a = math.cos(scan_rot_rad), math.sin(scan_rot_rad)
        for rel_x, rel_y in relative_points:
            world_x = rel_x * cos_a - rel_y * sin_a + scan_pos_x
            world_z = rel_x * sin_a + rel_y * cos_a + scan_pos_z
            global_map_points_x.append(world_x); global_map_points_z.append(world_z)
    print(f"Total global points accumulated: {len(global_map_points_x)}")
    fig_lidar, ax_lidar = plt.subplots(figsize=(10, 10))
    if global_map_points_x: ax_lidar.scatter(global_map_points_x, global_map_points_z, s=2, c='blue', label='Map Points (World)')
    if agent_trajectory_x:
        ax_lidar.plot(agent_trajectory_x, agent_trajectory_z, marker='o', markersize=3, linestyle='-', color='red', label='Agent Trajectory')
        ax_lidar.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], s=50, c='magenta', marker='*', label='Last Scan Pose')
    ax_lidar.set_title("Accumulated LiDAR Map (World Coordinates)"); ax_lidar.set_xlabel("World X"); ax_lidar.set_ylabel("World Z")
    ax_lidar.grid(True, linestyle='--', alpha=0.6); ax_lidar.set_aspect('equal', adjustable='box'); ax_lidar.legend()
    plt.show() # Blocking call

# --- Function to Visualize FULL Ground Truth Map ---
def plot_full_map(obs_df, gnd_info):
    """Displays the ground truth map (obstacles) using Matplotlib (blocking plt.show())."""
    print("Building ground truth map...")
    if obs_df is None or obs_df.empty: print("No obstacle data available."); return

    fig_full, ax_full = plt.subplots(figsize=(10, 10))
    gnd_size = gnd_info['size']
    ax_full.add_patch(patches.Rectangle((-gnd_size/2, -gnd_size/2), gnd_size, gnd_size,
                                        edgecolor='gray', facecolor='none', linestyle='--', label='Ground Area'))
    for index, row in obs_df.iterrows():
        x, z, w, d, rot = row['center_x'], row['center_z'], row['width'], row['depth'], row['rotation_y']
        bottom_left_x = x - w / 2; bottom_left_z = z - d / 2
        # Simple axis-aligned rectangle (add rotation logic if needed)
        ax_full.add_patch(patches.Rectangle((bottom_left_x, bottom_left_z), w, d,
                                            edgecolor='black', facecolor='darkgray', angle=rot)) # Added angle
    ax_full.scatter(0, 0, s=100, c='red', marker='x', label='Agent Start (0, 0)')
    ax_full.set_title("Ground Truth Map (Obstacle Layout)"); ax_full.set_xlabel("World X"); ax_full.set_ylabel("World Z")
    ax_full.grid(True, linestyle='--', alpha=0.6); ax_full.set_aspect('equal', adjustable='box'); ax_full.legend()
    plt.show() # Blocking call

# --- UI Display ---
info_text = Text(origin=(0.5, 0.5), scale=(0.8, 0.8), x=0.5 * window.aspect_ratio - 0.02, y=0.48,
                 text="Initializing...")

# --- Input Handling ---
def input(key):
    """Handles keyboard input."""
    global scan_history, obstacle_df # Make sure df is accessible if needed

    # 'M' key for blocking LiDAR map plot
    if key == 'm' or key == 'M':
        print("Plotting global LiDAR map...")
        # Call the blocking plot function directly
        plot_global_lidar_map(scan_history) # Pass the accumulated history

    # 'N' key for blocking Full map plot
    if key == 'n' or key == 'N':
        print("Plotting full ground truth map...")
        # Call the blocking plot function directly
        if obstacle_df is not None:
             plot_full_map(obstacle_df, ground_info)
        else:
             print("Obstacle DataFrame not ready.")

    # 'C' key to clear LiDAR map history
    if key == 'c' or key == 'C':
        print("Clearing LiDAR map history.")
        scan_history = []

    # 'P' key to save data (Optional)
    if key == 'p' or key == 'P':
        print("Saving data...")
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # Save scan history (Pickle recommended for list structure)
            if scan_history:
                history_df_to_save = pd.DataFrame(scan_history) # Convert just before saving
                filename_hist = f"scan_history_{timestamp}.pkl"
                history_df_to_save.to_pickle(filename_hist)
                print(f"Scan history saved to {filename_hist}")
            else:
                print("No scan history to save.")
            # Save obstacle ground truth
            if obstacle_df is not None:
                filename_obs = f"obstacles_{timestamp}.csv"
                obstacle_df.to_csv(filename_obs, index=False)
                print(f"Obstacle data saved to {filename_obs}")
            else:
                 print("No obstacle data to save.")
        except Exception as e:
             print(f"Error saving data: {e}")


    if key == 'escape':
        print("Exiting simulation...")
        # Optional: Add automatic saving on exit here as well
        quit()

# --- Main Update Loop ---
def update():
    global scan_timer, scan_history

    # --- Agent Control ---
    # (Movement and collision logic remains the same)
    original_position = agent.position
    total_delta_x, total_delta_z = 0.0, 0.0
    speed_dt = AGENT_SPEED * time.dt
    if held_keys['w']: total_delta_x += agent.forward.x * speed_dt; total_delta_z += agent.forward.z * speed_dt
    if held_keys['s']: total_delta_x -= agent.forward.x * speed_dt; total_delta_z -= agent.forward.z * speed_dt
    if held_keys['a']: total_delta_x -= agent.right.x * speed_dt; total_delta_z -= agent.right.z * speed_dt
    if held_keys['d']: total_delta_x += agent.right.x * speed_dt; total_delta_z += agent.right.z * speed_dt
    agent.x += total_delta_x
    for obs in obstacles:
        if agent.intersects(obs).hit: agent.x = original_position.x; break
    agent.z += total_delta_z
    for obs in obstacles:
        if agent.intersects(obs).hit: agent.z = original_position.z; break
    agent.y = AGENT_HEIGHT
    if held_keys['q']: agent.rotation_y -= ROTATION_SPEED * time.dt
    if held_keys['e']: agent.rotation_y += ROTATION_SPEED * time.dt

    # --- Periodic LiDAR Scan for Mapping ---
    scan_timer += time.dt
    if scan_timer >= SCAN_INTERVAL:
        scan_timer -= SCAN_INTERVAL

        current_pos_xz = agent.world_position.xz
        current_rot_rad = math.radians(agent.world_rotation_y)
        current_pose = (current_pos_xz.x, current_pos_xz.y, current_rot_rad)
        relative_points = generate_relative_lidar_map()

        if relative_points:
             scan_record = {'pose': current_pose, 'relative_points': relative_points}
             scan_history.append(scan_record)

    # --- LiDAR Visualization Update (Ursina Lines) ---
    update_lidar_visualization()

    # --- UI Update ---
    pos_str = f"Pos: ({agent.x:.1f}, {agent.z:.1f})"
    rot_str = f"Rot (Y): {agent.rotation_y:.0f}°"
    # Updated UI text for keys M, N, C, P
    map_info = f"Scans: {len(scan_history)} | 'M': LiDAR Map | 'N': Full Map | 'C': Clear | 'P': Save"
    info_text.text = f"{pos_str}\n{rot_str}\n{map_info}"
    info_text.x = 0.5 * window.aspect_ratio - 0.02
    info_text.y = 0.48

    # --- Mouse state ---
    if mouse.locked: mouse.locked = False
    if not mouse.visible: mouse.visible = True

# --- Start Simulation ---
if __name__ == '__main__':
    app.run()