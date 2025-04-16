from ursina import *
import random
import math
import numpy as np # Using numpy for easier vector math and rotations

# --- Configuration ---
MAP_SIZE = 20       # Size of the map plane
NUM_OBSTACLES = 15  # Number of obstacles on the map
SENSOR_RANGE = 8    # How far the robot's sensor can "see"
SENSOR_FOV = 180    # Field of view for the sensor (degrees)
SENSOR_RAYS = 90    # Number of rays to cast for sensing

# --- Global Variables ---
global_obstacles_entities = [] # List to hold Ursina entities for obstacles
global_obstacles_positions = [] # List to hold just the Vec3 positions
robot = None
estimated_pose_marker = None
local_map_display_entities = [] # Entities to visualize the local map

# --- Helper Functions ---
def generate_global_map():
    """Creates the ground plane and random obstacles."""
    global global_obstacles_entities, global_obstacles_positions
    # Ground
    ground = Entity(model='plane', scale=MAP_SIZE * 2, color=color.dark_gray, texture='white_cube', texture_scale=(MAP_SIZE, MAP_SIZE), collider='box')

    # Obstacles
    global_obstacles_positions = []
    for _ in range(NUM_OBSTACLES):
        pos = Vec3(random.uniform(-MAP_SIZE, MAP_SIZE),
                   0.5, # Place obstacle base on the ground
                   random.uniform(-MAP_SIZE, MAP_SIZE))
        # Ensure obstacles aren't too close to the center spawn
        if pos.length() > 3:
            obstacle = Entity(model='cube',
                              position=pos,
                              color=color.gray,
                              collider='box',
                              scale_y=random.uniform(1, 3)) # Vary height
            global_obstacles_entities.append(obstacle)
            global_obstacles_positions.append(pos) # Store position

def simulate_lidar(robot_entity):
    """
    Simulates sensor readings by raycasting from the robot.
    Returns a list of detected points relative to the robot's position and orientation.
    """
    detected_points_relative = []
    origin = robot_entity.world_position + Vec3(0, 0.1, 0) # Raycast slightly above ground
    robot_rotation_y_rad = math.radians(robot_entity.world_rotation_y)
    start_angle = math.radians(-SENSOR_FOV / 2)
    end_angle = math.radians(SENSOR_FOV / 2)
    angle_step = (end_angle - start_angle) / (SENSOR_RAYS - 1) if SENSOR_RAYS > 1 else 0

    for i in range(SENSOR_RAYS):
        current_angle_relative = start_angle + i * angle_step
        # Combine relative angle with robot's world rotation
        world_angle = robot_rotation_y_rad + current_angle_relative

        # Calculate direction vector in world space (X-Z plane)
        direction = Vec3(math.sin(world_angle), 0, math.cos(world_angle)).normalized()

        # Raycast
        hit_info = raycast(origin=origin,
                           direction=direction,
                           distance=SENSOR_RANGE,
                           ignore=[robot_entity], # Ignore self
                           debug=False) # Set to True to see rays

        if hit_info.hit:
            # Calculate hit point relative to robot's origin
            hit_point_world = hit_info.world_point
            relative_pos_world = hit_point_world - origin

            # Rotate the relative vector to align with robot's local frame (0 rotation)
            # Inverse rotation: rotate by -robot_rotation_y
            cos_a = math.cos(-robot_rotation_y_rad)
            sin_a = math.sin(-robot_rotation_y_rad)
            x_rel = relative_pos_world.x * cos_a - relative_pos_world.z * sin_a
            z_rel = relative_pos_world.x * sin_a + relative_pos_world.z * cos_a

            detected_points_relative.append(Vec2(x_rel, z_rel)) # Store as 2D relative points

    return detected_points_relative

def generate_local_map_visualization(relative_points):
    """Updates the visualization of the local map."""
    global local_map_display_entities
    # Clear previous local map visualization
    for entity in local_map_display_entities:
        destroy(entity)
    local_map_display_entities.clear()

    # Create new entities for the current local map (offset for visibility)
    display_offset = Vec3(10, 0, -15) # Position the local map display area
    origin_marker = Entity(model='sphere', scale=0.3, position=display_offset, color=color.red) # Mark local map origin
    local_map_display_entities.append(origin_marker)

    for point in relative_points:
        # Display points relative to the display_offset origin
        display_pos = display_offset + Vec3(point.x, 0.1, point.y) # Use y from Vec2 as z
        point_entity = Entity(model='sphere', scale=0.15, position=display_pos, color=color.yellow)
        local_map_display_entities.append(point_entity)


def calculate_similarity(potential_pose, local_map_points_relative, global_map_points):
    """
    Calculates a similarity score between the local map (placed at potential_pose)
    and the global map. Higher score means better match.
    This is a simplified scoring method.
    """
    potential_pos = Vec3(potential_pose[0], 0, potential_pose[1]) # x, z from pose
    potential_angle_rad = math.radians(potential_pose[2]) # angle from pose

    score = 0
    match_threshold = 0.8 # How close points need to be to count as a match

    # Transform local points to potential world coordinates
    cos_a = math.cos(potential_angle_rad)
    sin_a = math.sin(potential_angle_rad)

    for local_pt in local_map_points_relative:
        # Rotate local point by potential angle
        x_rot = local_pt.x * cos_a - local_pt.y * sin_a
        z_rot = local_pt.x * sin_a + local_pt.y * cos_a

        # Translate to potential position
        world_pt_guess = potential_pos + Vec3(x_rot, 0, z_rot)

        # Find the closest global obstacle point to this transformed local point
        min_dist_sq = float('inf')
        for global_pt in global_map_points:
            dist_sq = (world_pt_guess.x - global_pt.x)**2 + (world_pt_guess.z - global_pt.z)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

        # Add to score based on proximity (closer is better)
        # Simple scoring: count points within a threshold
        if math.sqrt(min_dist_sq) < match_threshold:
            score += 1
            # More sophisticated: score = 1 / (min_dist + epsilon)
            # score += 1.0 / (math.sqrt(min_dist_sq) + 0.1)


    # Normalize score by number of points (optional, helps if number of local points varies)
    if local_map_points_relative:
         score /= len(local_map_points_relative)

    return score


def perform_localization():
    """
    Attempts to estimate the robot's pose (x, z, angle) by matching
    the local sensor map to the global map.
    """
    global robot, estimated_pose_marker, global_obstacles_positions

    print("--- Starting Localization ---")
    if not robot:
        print("Robot not initialized.")
        return

    # 1. Get sensor data (points relative to robot)
    local_map_points = simulate_lidar(robot)
    print(f"Detected {len(local_map_points)} points locally.")

    # 2. Visualize the generated local map
    generate_local_map_visualization(local_map_points)

    if not local_map_points:
        print("No points detected, cannot localize.")
        return

    # 3. Define Search Space (Simplified Grid Search)
    # For a real system, this would be more sophisticated (e.g., particle filter)
    # Here, we search around the robot's *current* position as a simple demo.
    search_radius = 3.0 # Search +/- this distance from current pos
    angle_search_range = 30 # Search +/- this angle (degrees)
    pos_step = 1.0      # Grid step for position search
    angle_step = 10     # Step for angle search (degrees)

    best_score = -1
    best_pose = (robot.x, robot.z, robot.rotation_y) # Default to current pose

    current_pos = robot.position
    current_angle = robot.rotation_y

    search_count = 0

    # Iterate through potential positions (x, z)
    for dx in np.arange(-search_radius, search_radius + pos_step, pos_step):
        for dz in np.arange(-search_radius, search_radius + pos_step, pos_step):
            # Iterate through potential angles (yaw)
            for dangle in np.arange(-angle_search_range, angle_search_range + angle_step, angle_step):
                potential_x = current_pos.x + dx
                potential_z = current_pos.z + dz
                potential_angle = (current_angle + dangle) % 360 # Wrap angle
                potential_pose = (potential_x, potential_z, potential_angle)
                search_count += 1

                # 4. Calculate Similarity Score for this pose
                score = calculate_similarity(potential_pose, local_map_points, global_obstacles_positions)

                # 5. Update Best Estimate
                if score > best_score:
                    best_score = score
                    best_pose = potential_pose
                    # print(f"New best: {best_pose}, Score: {score:.4f}") # Debugging

    print(f"Search completed. Checked {search_count} poses.")
    print(f"Best Estimated Pose: x={best_pose[0]:.2f}, z={best_pose[1]:.2f}, angle={best_pose[2]:.2f}")
    print(f"Actual Robot Pose:   x={robot.x:.2f}, z={robot.z:.2f}, angle={robot.rotation_y:.2f}")
    print(f"Best Score: {best_score:.4f}")


    # 6. Visualize the Estimated Pose
    if estimated_pose_marker:
        estimated_pose_marker.position = Vec3(best_pose[0], 0.1, best_pose[1])
        estimated_pose_marker.rotation_y = best_pose[2]
    else:
        # Create marker if it doesn't exist
        estimated_pose_marker = Entity(model='arrow',
                                       color=color.lime,
                                       scale=1.5,
                                       position=Vec3(best_pose[0], 0.1, best_pose[1]),
                                       rotation_y = best_pose[2])
        # Add a base to the marker for better visibility
        base = Entity(model='sphere', scale=0.5, color=color.green, parent=estimated_pose_marker)
        base.y = -0.2 # Position base slightly below arrow origin

    print("--- Localization Finished ---")


# --- Ursina Application Setup ---
app = Ursina()

# Generate the environment
generate_global_map()

# Create the robot
robot = Entity(model='sphere', color=color.blue, collider='sphere', position=(0, 0.2, 0))
# Add a forward direction indicator
robot_forward = Entity(model='cube', scale=(0.1, 0.1, 0.5), color=color.red, parent=robot)
robot_forward.z = 0.3 # Position it in front of the robot center

# Camera setup
EditorCamera()
camera.y = 25 # Start with a higher view
camera.rotation_x = 70 # Look downwards

# Instructions Text
instructions = Text(
    text="WASD = Move Robot, Q/E = Rotate Robot, L = Perform Localization",
    origin=(-0.5, -0.5),
    scale=1.5,
    position = (-0.85, -0.45)
)

# --- Input Handling ---
def input(key):
    if key == 'l':
        perform_localization()
    # Add more key bindings if needed

# --- Update Loop ---
def update():
    global robot
    move_speed = 5 * time.dt
    turn_speed = 90 * time.dt

    # Robot movement (relative to its own orientation)
    if held_keys['w']:
        robot.position += robot.forward * move_speed
    if held_keys['s']:
        robot.position -= robot.forward * move_speed
    if held_keys['a']:
        robot.position -= robot.right * move_speed
    if held_keys['d']:
        robot.position += robot.right * move_speed

    # Robot rotation
    if held_keys['q']:
        robot.rotation_y -= turn_speed
    if held_keys['e']:
        robot.rotation_y += turn_speed

    # Keep robot roughly on the ground (simple collision avoidance)
    if robot.y < 0.1:
       robot.y = 0.1

    # --- Optional: Real-time local map update (can be slow) ---
    # if held_keys['space']: # Update local map continuously if space is held
    #     local_points = simulate_lidar(robot)
    #     generate_local_map_visualization(local_points)

app.run()