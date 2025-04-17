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

import random
import math

# --- Simulation Parameters ---
NUM_OBSTACLES = 100
AREA_SIZE = 60
AGENT_SPEED = 5      # Movement speed
ROTATION_SPEED = 100 # Rotation speed (degrees/sec)
AGENT_HEIGHT = 0.5   # Agent's height or y-position on the plane
LIDAR_RANGE = 15
LIDAR_FOV = 150
NUM_LIDAR_RAYS = 90
LIDAR_VIS_HEIGHT = 0.6 # Height at which LiDAR rays originate/visualized
LIDAR_COLOR = color.cyan
OBSTACLE_TAG = "obstacle"

# --- Application Setup ---
app = Ursina(
    title="SLAM Simulation Environment (Keyboard Control)",
    borderless=False, # Ensure window has borders and controls
    # vsync=False # Disable vsync might make movement smoother sometimes, but can cause tearing
)

# --- Mouse Setup ---
mouse.visible = True
mouse.locked = False # Ensure mouse is not locked to the window center

# --- Environment Setup ---
ground = Entity(model='plane',
                scale=(AREA_SIZE, 1, AREA_SIZE),
                color=color.light_gray,
                texture='white_cube',
                texture_scale=(AREA_SIZE/2, AREA_SIZE/2),
                collider='box')

sky = Sky() # Optional nice background
# --- Obstacle Generation ---
obstacles = []
min_obstacle_distance_from_spawn = 3.0 # Minimum distance from (0,0)

for i in range(NUM_OBSTACLES):
    while True: # Keep trying until a valid position is found
        pos_x = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5)
        pos_z = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5)

        # Check distance from agent spawn point (0, 0)
        distance_from_spawn = math.sqrt(pos_x**2 + pos_z**2)

        if distance_from_spawn >= min_obstacle_distance_from_spawn:
            # Position is valid (far enough from spawn)
            break # Exit the while loop

        # If too close, the while loop continues and generates new pos_x, pos_z

    # --- Position is now valid, proceed with scaling and creation ---
    scale_x = random.uniform(0.5, 3)
    scale_y = random.uniform(1, 4)
    scale_z = random.uniform(0.5, 3)
    pos_y = scale_y / 2 # Place base on the ground
    obstacle_color = color.random_color()

    obs = Entity(model='cube',
                 position=(pos_x, pos_y, pos_z),
                 scale=(scale_x, scale_y, scale_z),
                 color=obstacle_color,
                 collider='box',
                 tag=OBSTACLE_TAG,
                 name=f"obstacle_{i}")
    obstacles.append(obs)

# --- Agent (Object) Setup ---
# Use a simple Entity instead of FirstPersonController
agent = Entity(
    model='sphere',             # 모델을 'sphere'로 변경
    color=color.blue,
    position=(0, AGENT_HEIGHT, 0), # AGENT_HEIGHT가 구의 반지름이 되어 바닥에 닿도록 함 (보통 scale=1일 때 반지름 0.5)
    collider='sphere',          # 콜라이더도 'sphere'로 변경 (더 정확한 충돌 감지)
    scale=1                     # 구의 기본 크기 (지름 1)
)

# --- Camera Setup ---
# Make the camera follow the agent from a 3rd person perspective
camera.parent = agent
camera.position = (0, 5, -8) # Offset from agent (x=0, y=up, z=behind)
camera.rotation_x = 30       # Angle downwards
camera.rotation_y = 0        # Follows agent's y rotation automatically
camera.fov = 75              # Field of view

# --- LiDAR Visualization ---
lidar_lines = []

def update_lidar_visualization():
    global lidar_lines
    for line in lidar_lines:
        destroy(line)
    lidar_lines.clear()

    start_angle = agent.world_rotation_y - LIDAR_FOV / 2
    angle_step = LIDAR_FOV / (NUM_LIDAR_RAYS - 1) if NUM_LIDAR_RAYS > 1 else 0
    origin = agent.world_position + Vec3(0, LIDAR_VIS_HEIGHT - AGENT_HEIGHT, 0)
    detected_points_count = 0

    for i in range(NUM_LIDAR_RAYS):
        current_angle_deg = start_angle + i * angle_step
        current_angle_rad = math.radians(current_angle_deg)
        direction = Vec3(math.sin(current_angle_rad), 0, math.cos(current_angle_rad)).normalized()

        # --- 수정된 부분 ---
        # traverse_targets 인수를 제거합니다.
        hit_info = raycast(origin, direction, distance=LIDAR_RANGE, ignore=[agent,], debug=False)
        # ------------------

        if hit_info.hit:
            end_point = hit_info.world_point
            line_color = color.red
            # Optional: Check specifically if it hit an obstacle vs the ground if needed
            if hit_info.entity != ground and hasattr(hit_info.entity, 'tags') and OBSTACLE_TAG in hit_info.entity.tags:
                 detected_points_count += 1
        else:
            end_point = origin + direction * LIDAR_RANGE
            line_color = LIDAR_COLOR

        if distance(origin, end_point) > 0.01:
            line = Entity(model=Mesh(vertices=[origin, end_point], mode='line', thickness=2), color=line_color)
            lidar_lines.append(line)

    return detected_points_count
# --- UI Display ---
info_text = Text(
    origin=(0.5, 0.5),
    scale=(0.8, 0.8),
    x=0.5 * window.aspect_ratio - 0.02,
    y=0.48,
    text="Initializing..."
)

# --- Main Update Loop ---
def update():
    # --- Agent Control ---
    original_position = agent.position # 충돌 시 복원을 위해 현재 위치 저장

    # --- 이동량 계산 (객체 기준 상대 좌표) ---
    total_delta_x = 0.0
    total_delta_z = 0.0
    speed_dt = AGENT_SPEED * time.dt

    if held_keys['w']:
        total_delta_x += agent.forward.x * speed_dt
        total_delta_z += agent.forward.z * speed_dt
    if held_keys['s']:
        total_delta_x -= agent.forward.x * speed_dt
        total_delta_z -= agent.forward.z * speed_dt
    if held_keys['a']:
        total_delta_x -= agent.right.x * speed_dt
        total_delta_z -= agent.right.z * speed_dt
    if held_keys['d']:
        total_delta_x += agent.right.x * speed_dt
        total_delta_z += agent.right.z * speed_dt

    # --- 충돌 감지 및 이동 적용 (개선된 축 별 확인) ---

    # 1. X축 이동 시도
    agent.x += total_delta_x
    # X축 이동 후 충돌 확인
    for obs in obstacles:
        if agent.intersects(obs).hit:
            agent.x = original_position.x # 충돌 시 X 위치만 원복
            break # X 충돌 확인 중지

    # 2. Z축 이동 시도 (X축 이동이 반영된 상태에서)
    agent.z += total_delta_z
    # Z축 이동 후 충돌 확인
    for obs in obstacles:
        if agent.intersects(obs).hit:
            # Z 위치를 원복할 때는 현재 X 위치(충돌로 원복되었을 수도 있음)는 유지
            agent.z = original_position.z
            break # Z 충돌 확인 중지

    # 3. Y축 위치 고정
    agent.y = AGENT_HEIGHT

    # --- 회전 (QE) ---
    if held_keys['q']:
        agent.rotation_y -= ROTATION_SPEED * time.dt
    if held_keys['e']:
        agent.rotation_y += ROTATION_SPEED * time.dt

    # --- LiDAR 업데이트 ---
    detected_count = update_lidar_visualization()

    # --- UI 업데이트 ---
    pos_str = f"Pos: ({agent.x:.1f}, {agent.z:.1f})"
    rot_str = f"Rot (Y): {agent.rotation_y:.0f}°"
    cam_str = "Cam: OK (Visual)"
    lidar_str = f"LiDAR Hits: {detected_count}/{NUM_LIDAR_RAYS}"
    info_text.text = f"{pos_str}\n{rot_str}\n{cam_str}\n{lidar_str}"
    info_text.x = 0.5 * window.aspect_ratio - 0.02
    info_text.y = 0.48

    # --- 마우스 상태 확인 ---
    if mouse.locked:
        mouse.locked = False
    if not mouse.visible:
        mouse.visible = True
app.run()