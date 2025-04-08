import pygame
import numpy as np
import math
<<<<<<< HEAD
import heapq # A* 구현 시 사용
=======
import heapq
import time
>>>>>>> shin

# --- 초기 설정 ---
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
<<<<<<< HEAD
pygame.display.set_caption("2D Autonomous Driving Simulation")
clock = pygame.time.Clock()
=======
pygame.display.set_caption("2D Autonomous Driving Simulation with A* & Manual Override")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 24)
BOLD_FONT = pygame.font.SysFont(None, 28, bold=True)
>>>>>>> shin

# --- 색상 정의 ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
<<<<<<< HEAD

# --- 맵 데이터 (간단한 예시: 벽) ---
# 실제로는 이미지 파일을 로드하거나, 더 복잡한 구조 사용 가능
# 여기서는 Rect 객체 리스트로 벽을 표현
walls = [
    pygame.Rect(100, 100, 20, 400),
    pygame.Rect(680, 100, 20, 400),
    pygame.Rect(100, 100, 580, 20),
    pygame.Rect(100, 480, 580, 20),
    pygame.Rect(300, 200, 200, 20) # 내부 장애물
]

# --- 차량 클래스 ---
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, x, y, angle=0):
        super().__init__()
        self.image = pygame.Surface([30, 20], pygame.SRCALPHA) # 투명 배경
        pygame.draw.rect(self.image, BLUE, (0, 0, 30, 20)) # 파란색 사각형으로 차량 표현
        pygame.draw.circle(self.image, RED, (25, 10), 5) # 앞쪽 표시
=======
DARK_GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)

# --- 맵 데이터 (벽) ---
walls = [
    pygame.Rect(100, 100, 20, 400), # Left wall
    pygame.Rect(680, 100, 20, 400), # Right wall
    pygame.Rect(100, 100, 580, 20), # Top wall
    pygame.Rect(100, 480, 580, 20), # Bottom wall
    pygame.Rect(300, 200, 200, 20), # Internal obstacle 1
    pygame.Rect(300, 300, 50, 100)  # Internal obstacle 2
]

# --- A* 관련 설정 ---
GRID_SIZE = 20
VEHICLE_RADIUS_BUFFER = 15

# --- 차량 클래스 (max_steering 증가 및 가속도/속도 복원) ---
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, x, y, angle=0):
        super().__init__()
        self.width = 30
        self.height = 20
        self.image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
        pygame.draw.rect(self.image, BLUE, (0, 0, self.width, self.height))
        pygame.draw.circle(self.image, RED, (self.width - 5, self.height // 2), 5)
>>>>>>> shin
        self.original_image = self.image
        self.rect = self.image.get_rect(center=(x, y))
        self.x = float(x)
        self.y = float(y)
<<<<<<< HEAD
        self.angle = math.radians(angle) # 각도는 라디안 사용
        self.speed = 0.0
        self.max_speed = 3.0
        self.acceleration = 0.1
        self.steering_angle = 0.0 # 조향각 (라디안)
        self.max_steering = math.radians(40) # 최대 조향각

    def update(self, control_signal):
        # control_signal: {'throttle': float, 'steering': float} (-1 to 1)
=======
        self.angle = math.radians(angle)
        self.speed = 0.0
        # 속도/가속도 이전 값으로 복원 또는 적절히 조절
        self.max_speed = 3.0 # << 수정됨 (2.0 -> 3.0)
        self.acceleration = 0.1 # << 수정됨 (0.05 -> 0.1)
        self.steering_angle = 0.0
        # 최대 조향각 증가
        # 정지 시 회전
        self.max_steering = math.radians(270) # << 수정됨 (40 -> 45)
        self.last_valid_position = (self.x, self.y)

    def update(self, control_signal, dt):
        collided = False
        self.last_valid_position = (self.x, self.y)
>>>>>>> shin

        # 1. 조향각 업데이트
        self.steering_angle = control_signal['steering'] * self.max_steering

<<<<<<< HEAD
        # 2. 속도 업데이트 (간단한 가속/감속)
        if control_signal['throttle'] > 0:
            self.speed += self.acceleration
        elif control_signal['throttle'] < 0:
            self.speed -= self.acceleration * 2 # 브레이크는 더 강하게
        else:
            # 자연 감속 (마찰)
            self.speed *= 0.98
        self.speed = max(-self.max_speed / 2, min(self.max_speed, self.speed)) # 후진 속도 제한

        # 3. 차량 각도 업데이트 (자전거 모델 기반)
        # L = 차량 길이 (축간 거리, 여기서는 단순화)
        L = 25 # 예시 값
        if abs(self.steering_angle) > 1e-4: # 조향각이 있을 때만 회전
            turn_radius = L / math.tan(self.steering_angle)
            angular_velocity = self.speed / turn_radius
            self.angle += angular_velocity * (1/60.0) # dt = 1/fps (60fps 기준)

        # 4. 위치 업데이트
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle) # Pygame 좌표계는 y가 아래로 증가

        # 5. 이미지 회전 및 위치 업데이트
        self.image = pygame.transform.rotate(self.original_image, -math.degrees(self.angle)) # Pygame 각도는 degree, 반시계방향
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

        # 충돌 처리 (여기서는 간단히 벽과 충돌 시 멈춤)
        for wall in walls:
            if self.rect.colliderect(wall):
                # 충돌 시 이전 위치로 복귀 (간단한 처리)
                self.x -= self.speed * math.cos(self.angle)
                self.y -= self.speed * math.sin(self.angle)
                self.speed = 0
                self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))
                break
=======
        # 2. 속도 업데이트
        # 수동 조작 시에는 직접적인 스로틀 값을 사용할 수 있도록 수정 가능 (현재는 동일 로직 사용)
        if control_signal['throttle'] > 0:
            self.speed += self.acceleration * abs(control_signal['throttle'])
        elif control_signal['throttle'] < 0:
            # 후진 시 감속 계수 원래대로 (2배)
            self.speed -= self.acceleration * 2 * abs(control_signal['throttle']) # << 수정됨 (1 -> 2)
        else:
            self.speed *= 0.9
        self.speed = max(-self.max_speed / 2, min(self.max_speed, self.speed))

        # 3. 차량 각도 업데이트 (weight 제거!)
        L = self.width * 0.8
        if abs(self.speed) > 0.1 and abs(self.steering_angle) > 1e-4 :
             turn_radius = L / math.tan(self.steering_angle)
             angular_velocity = self.speed / turn_radius
             # 각속도 제한 (선택적 안정화)
             max_angular_velocity = math.radians(180)
             angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity))
             # weight 제거!
             weight = 5
             self.angle += weight*angular_velocity * dt # << 수정됨

        elif abs(self.speed) < 0.1 and abs(control_signal['steering']) > 0:
             manual_turn_rate = math.radians(90) # 제자리 회전 속도 증가
             # 제자리 회전 방향 수정 (이전 코드 오류 수정)
             self.angle += manual_turn_rate * control_signal['steering'] * dt # << 수정됨 (-) 제거

        # 4. 위치 업데이트
        self.x += self.speed * math.cos(self.angle) * dt * 60
        self.y += self.speed * math.sin(self.angle) * dt * 60

        # 5. 이미지 회전 및 위치 업데이트
        self.image = pygame.transform.rotate(self.original_image, -math.degrees(self.angle))
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

        # 충돌 처리
        for wall in walls:
            if self.rect.colliderect(wall):
                collided = True; break
        if collided:
            self.x, self.y = self.last_valid_position
            self.speed = 0
            self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))
            # 충돌 시 각도 유지 로직 제거 또는 수정 가능 (현재는 제거)

        return collided # 충돌 여부 반환
>>>>>>> shin

    def draw(self, surface):
        surface.blit(self.image, self.rect)

<<<<<<< HEAD
# --- A* 경로 계획 함수 ---
# (구현 필요 - 간단히 설명)
# 입력: 맵(grid), 시작점(tuple), 목표점(tuple)
# 출력: 경로(waypoints list)
def a_star_pathfinding(grid_map, start_node, goal_node):
    # 1. 그리드 맵 생성: walls 정보를 바탕으로 이동 가능한 셀과 불가능한 셀 구분
    # 2. Open 리스트(힙큐), Closed 리스트(셋) 초기화
    # 3. 시작 노드를 Open 리스트에 추가
    # 4. Open 리스트가 빌 때까지 반복:
    #    a. f_cost가 가장 낮은 노드(current)를 Open 리스트에서 꺼내 Closed 리스트에 추가
    #    b. current가 목표 노드면 경로 역추적 후 반환
    #    c. current의 이웃 노드들에 대해:
    #       i. 이웃이 벽이 아니고 Closed 리스트에 없으면:
    #       ii. g_cost, h_cost, f_cost 계산
    #       iii. 이웃이 Open 리스트에 없거나 더 좋은 경로면 정보 업데이트 후 Open 리스트에 추가
    # 5. 경로를 찾지 못하면 None 반환
    print(f"A* Pathfinding from {start_node} to {goal_node}")
    # 여기서는 임시로 직선 경로 반환 (실제 A* 구현 필요)
    path = []
    steps = 10 # 경로점 개수
    if start_node and goal_node:
        for i in range(steps + 1):
            x = start_node[0] + (goal_node[0] - start_node[0]) * (i / steps)
            y = start_node[1] + (goal_node[1] - start_node[1]) * (i / steps)
            path.append((int(x), int(y)))
    return path


# --- 제어 로직 함수 ---
# (구현 필요 - 간단히 설명)
# 입력: 차량 상태(vehicle), 경로(path)
# 출력: 제어 신호 {'throttle': float, 'steering': float}
def simple_controller(vehicle, path):
    if not path or len(path) < 2:
        return {'throttle': 0.0, 'steering': 0.0} # 경로 없으면 정지

    # 가장 가까운 경로점 찾기 (또는 다음 목표점 설정)
    target_point = None
    current_segment_index = 0 # 실제로는 현재 위치 기반으로 찾아야 함
    if len(path) > 1 :
        target_point = path[1] # 일단 다음 점을 목표로

    if target_point is None:
         return {'throttle': 0.0, 'steering': 0.0}

    # 목표점까지의 벡터 계산
=======
# --- A* Node 클래스 (이전과 동일) ---
class Node:
    def __init__(self, position, parent=None):
        self.position = position; self.parent = parent
        self.g = 0; self.h = 0; self.f = 0
    def __lt__(self, other): return self.f < other.f
    def __eq__(self, other): return self.position == other.position
    def __hash__(self): return hash(self.position)

# --- 좌표 변환 함수 ---
def to_grid_coords(world_x, world_y, grid_size):
    return int(world_x // grid_size), int(world_y // grid_size)
def to_world_coords(grid_x, grid_y, grid_size):
    return grid_x * grid_size + grid_size / 2, grid_y * grid_size + grid_size / 2

# --- Grid Map 생성 함수 ---
def create_grid_map(walls, width, height, grid_size, buffer):
    grid_width = width // grid_size; grid_height = height // grid_size
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for r in range(grid_height):
        for c in range(grid_width):
            world_x, world_y = to_world_coords(c, r, grid_size)
            cell_rect = pygame.Rect(world_x - grid_size/2, world_y - grid_size/2, grid_size, grid_size)
            inflated_cell_rect = cell_rect.inflate(buffer * 2, buffer * 2)
            is_obstacle = False
            for wall in walls:
                if inflated_cell_rect.colliderect(wall): is_obstacle = True; break
            if not (0 <= cell_rect.left and cell_rect.right <= width and \
                    0 <= cell_rect.top and cell_rect.bottom <= height): is_obstacle = True
            if is_obstacle: grid_map[r, c] = 1
    return grid_map

# --- 휴리스틱 함수 ---
def heuristic(a, b): # 유클리드 거리 사용 권장
    return math.sqrt((a.position[0] - b.position[0])**2 + (a.position[1] - b.position[1])**2)
    # return abs(a.position[0] - b.position[0]) + abs(a.position[1] - b.position[1]) # 맨해튼

# --- A* 경로 탐색 함수 ---
def a_star_pathfinding(grid_map, start_pos_grid, goal_pos_grid):
    start_time = time.time()
    start_node = Node(start_pos_grid); goal_node = Node(goal_pos_grid)
    map_h, map_w = grid_map.shape
    if not (0 <= start_pos_grid[0] < map_w and 0 <= start_pos_grid[1] < map_h) or \
       not (0 <= goal_pos_grid[0] < map_w and 0 <= goal_pos_grid[1] < map_h):
        print("Warning: Start or goal position is out of map bounds.")
        return []
    if grid_map[start_node.position[1]][start_node.position[0]] == 1 or \
       grid_map[goal_node.position[1]][goal_node.position[0]] == 1:
        print("시작 또는 목표 지점이 장애물 위에 있습니다.")
        return []
    open_list = []; closed_list = set()
    heapq.heappush(open_list, start_node)
    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position in closed_list: continue
        closed_list.add(current_node.position)
        if current_node == goal_node:
            path = []; temp = current_node
            while temp is not None: path.append(temp.position); temp = temp.parent
            elapsed_time = time.time() - start_time
            print(f"Path found! Time: {elapsed_time:.4f}s, Length: {len(path)}")
            return path[::-1]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            if not (0 <= neighbor_pos[0] < map_w and 0 <= neighbor_pos[1] < map_h): continue
            if grid_map[neighbor_pos[1]][neighbor_pos[0]] == 1: continue
            if neighbor_pos in closed_list: continue
            neighbor_node = Node(neighbor_pos, current_node)
            move_cost = 1.414 if abs(dx) == 1 and abs(dy) == 1 else 1.0
            neighbor_node.g = current_node.g + move_cost
            neighbor_node.h = heuristic(neighbor_node, goal_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            heapq.heappush(open_list, neighbor_node)
    elapsed_time = time.time() - start_time
    print(f"Path not found. Time: {elapsed_time:.4f}s")
    return []

# --- 제어 로직 함수 (개선된 버전 적용) ---
def simple_controller(vehicle, path_world, current_path_index):
    if not path_world or current_path_index >= len(path_world):
        return {'throttle': 0.0, 'steering': 0.0}, current_path_index

    target_point = path_world[current_path_index]
>>>>>>> shin
    target_dx = target_point[0] - vehicle.x
    target_dy = target_point[1] - vehicle.y
    distance_to_target = math.sqrt(target_dx**2 + target_dy**2)

<<<<<<< HEAD
    # 목표점 방향 각도 계산
    target_angle = math.atan2(target_dy, target_dx)

    # 차량 현재 각도와 목표 각도 차이 계산
    angle_error = target_angle - vehicle.angle
    # 각도 차이 정규화 (-pi ~ pi)
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    # P 제어기 (비례 제어) - 조향
    Kp_steering = 0.8 # 제어 게인 (튜닝 필요)
    steering_signal = Kp_steering * angle_error
    steering_signal = max(-1.0, min(1.0, steering_signal)) # -1 ~ 1 제한

    # P 제어기 - 스로틀 (목표점에 가까워지면 감속)
    Kp_throttle = 0.5
    target_speed = vehicle.max_speed * (distance_to_target / 100.0) # 거리에 비례 (튜닝 필요)
    target_speed = min(vehicle.max_speed, target_speed)
    throttle_signal = Kp_throttle * (target_speed - vehicle.speed) # 목표 속도와 현재 속도 차이
    throttle_signal = max(-1.0, min(1.0, throttle_signal))

    # 목표점에 매우 가까우면 정지 또는 다음 경로점 설정 로직 필요
    if distance_to_target < 15: # 임계값
        # path.pop(0) # 다음 경로점으로 (주의: 이 방식은 문제 발생 가능성 있음)
        throttle_signal = -0.5 # 도착 시 감속/정지

    return {'throttle': throttle_signal, 'steering': steering_signal}


# --- 메인 루프 ---
vehicle = Vehicle(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100)
goal_pos = None
current_path = []

running = True
while running:
=======
    if distance_to_target < 30 and current_path_index < len(path_world) - 1: # 마지막 점 아닐 때만 인덱스 증가
        current_path_index += 1
    elif distance_to_target < 15 and current_path_index == len(path_world) - 1: # 마지막 점 도착 시
        # 정지 또는 완료 상태 처리
         return {'throttle': -0.5, 'steering': 0.0}, current_path_index + 1 # 완료 표시 위해 인덱스 증가


    target_angle = math.atan2(target_dy, target_dx)
    angle_error = target_angle - vehicle.angle
    # *** 각도 정규화 오류 수정 ***
    while angle_error > math.pi: angle_error -= 2 * math.pi # << 부호 수정
    while angle_error < -math.pi: angle_error += 2 * math.pi # << 부호 수정

    Kp_steering = 1.4 # << 조향 게인 증가
    steering_signal = Kp_steering * angle_error
    steering_signal = max(-1.0, min(1.0, steering_signal))

    # --- 스로틀 계산 (회전 시 속도 유지 강화) ---
    target_speed = vehicle.max_speed
    if dist_to_final < 120: # 최종점 가까워지면 감속
         target_speed *= max(0.15, min(1.0, dist_to_final / 120.0))

    Kp_throttle = 0.9 # << 스로틀 게인 증가
    throttle_signal = Kp_throttle * (target_speed - vehicle.speed)
    throttle_signal = max(-1.0, min(1.0, throttle_signal))

    return {'throttle': throttle_signal, 'steering': steering_signal}, current_path_index


# --- 메인 루프 ---
vehicle = Vehicle(150, 450, angle=0)
goal_pos_world = None
current_path_world = []
current_path_index = 0
grid_map = create_grid_map(walls, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, VEHICLE_RADIUS_BUFFER)
manual_control_active = False
last_path_failed = False

running = True
while running:
    dt = clock.tick(60) / 1000.0
    if dt > 0.1: dt = 0.1
    keys = pygame.key.get_pressed()

>>>>>>> shin
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
<<<<<<< HEAD
            goal_pos = event.pos # 마우스 클릭 위치를 목표로 설정
            # A* 경로 계획 수행 (시작점은 차량 현재 위치)
            start_node = (int(vehicle.x), int(vehicle.y))
            # TODO: A* 알고리즘을 위한 그리드 맵 생성 및 노드 변환 필요
            current_path = a_star_pathfinding(None, start_node, goal_pos) # grid_map 전달 필요

    # 1. 제어 입력 생성 (Controller)
    control_signal = simple_controller(vehicle, current_path)

    # 2. 차량 상태 업데이트 (Vehicle Dynamics)
    vehicle.update(control_signal)

    # 3. 화면 그리기 (Rendering)
    screen.fill(WHITE)

    # 맵 그리기
    for wall in walls:
        pygame.draw.rect(screen, GRAY, wall)

    # 차량 그리기
    vehicle.draw(screen)

    # 목표 지점 그리기
    if goal_pos:
        pygame.draw.circle(screen, GREEN, goal_pos, 10)

    # 경로 그리기
    if len(current_path) > 1:
        pygame.draw.lines(screen, RED, False, current_path, 2)

    pygame.display.flip()
    clock.tick(60) # 60 FPS
=======
            goal_pos_world = event.pos
            start_pos_world = (vehicle.x, vehicle.y)
            start_pos_grid = to_grid_coords(start_pos_world[0], start_pos_world[1], GRID_SIZE)
            goal_pos_grid = to_grid_coords(goal_pos_world[0], goal_pos_world[1], GRID_SIZE)
            print(f"New goal set. Start: {start_pos_grid}, Goal: {goal_pos_grid}")
            grid_path = a_star_pathfinding(grid_map, start_pos_grid, goal_pos_grid)
            if grid_path:
                current_path_world = []
                for grid_node_pos in grid_path:
                    world_x, world_y = to_world_coords(grid_node_pos[0], grid_node_pos[1], GRID_SIZE)
                    current_path_world.append((world_x, world_y))
                current_path_index = 0; manual_control_active = False; last_path_failed = False
                print(f"Path found with {len(current_path_world)} waypoints. Switching to AUTO mode.")
            else: # 경로 탐색 실패
                current_path_world = []
                if not manual_control_active:
                     print("Switching to MANUAL mode due to pathfinding failure.")
                     manual_control_active = True
                last_path_failed = True

    # --- 제어 신호 결정 ---
    control_signal = {'throttle': 0.0, 'steering': 0.0}
    if manual_control_active:
        throttle_input = 0.0; steer_input = 0.0
        if keys[pygame.K_UP]:    throttle_input = 0.8
        elif keys[pygame.K_DOWN]: throttle_input = -0.6
        # 전후진 시 수동 조작 steering 값 수정 (-1.0 ~ 1.0) 
        # # 2.9~3.0 사이에서 결정해야할듯 
        if keys[pygame.K_LEFT]:  steer_input = -2.96 # << 수정됨 # 여긴 반드시 '-'
        elif keys[pygame.K_RIGHT]: steer_input = 2.96 # << 수정됨 # 여긴 반드시 '+'
        control_signal['throttle'] = throttle_input
        control_signal['steering'] = steer_input
    else: # 자동 조작
        if current_path_world:
            control_signal, current_path_index_new = simple_controller(vehicle, current_path_world, current_path_index)
            current_path_index = current_path_index_new
            if current_path_index >= len(current_path_world): # 경로 완료
                if not manual_control_active:
                    print("Path completed. Switching to MANUAL mode.")
                    manual_control_active = True
                current_path_world = []
        else: # 자동 모드인데 경로 없음
             if not manual_control_active: manual_control_active = True

    # --- 차량 상태 업데이트 ---
    collided = vehicle.update(control_signal, dt)
    if collided:
        if not manual_control_active: # 자동 모드에서 충돌 시
            print("Collision detected! Switching to MANUAL mode.")
            manual_control_active = True
            current_path_world = []

    # --- 화면 그리기 ---
    screen.fill(WHITE)
    for wall in walls: pygame.draw.rect(screen, DARK_GRAY, wall)
    vehicle.draw(screen)
    if goal_pos_world:
        pygame.draw.circle(screen, GREEN, goal_pos_world, 10)
        pygame.draw.circle(screen, BLACK, goal_pos_world, 10, 1)
    if len(current_path_world) > 1 and not manual_control_active:
        pygame.draw.lines(screen, MAGENTA, False, current_path_world, 3)
        # Lookahead target 시각화 (디버깅용)
        # if 'target_point' in locals() and target_point is not None:
        #      pygame.draw.circle(screen, YELLOW, target_point, 5)

    # 상태 정보
    speed_text = FONT.render(f"Speed: {vehicle.speed:.1f}", True, BLACK)
    angle_text = FONT.render(f"Angle: {math.degrees(vehicle.angle):.1f}", True, BLACK)
    screen.blit(speed_text, (10, 10)); screen.blit(angle_text, (10, 30))
    # 모드 표시
    mode_text_str = "Mode: MANUAL (Use Arrow Keys)" if manual_control_active else "Mode: AUTO"
    mode_color = CYAN if manual_control_active else GREEN
    mode_text = BOLD_FONT.render(mode_text_str, True, mode_color)
    screen.blit(mode_text, (SCREEN_WIDTH - mode_text.get_width() - 10, 10))
    # 실패 메시지
    if last_path_failed and manual_control_active:
         fail_text = FONT.render("Pathfinding Failed!", True, RED)
         screen.blit(fail_text, (SCREEN_WIDTH - fail_text.get_width() - 10, 40))


    pygame.display.flip()

pygame.quit()