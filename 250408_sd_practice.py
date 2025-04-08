import pygame
import numpy as np
import math
import heapq
import time

# --- 초기 설정 ---
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Autonomous Driving Simulation with A* & Manual Override")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 24)
BOLD_FONT = pygame.font.SysFont(None, 28, bold=True)

# --- 색상 정의 ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255) # 수동 조작 모드 표시용

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

# --- 차량 클래스 (update 메서드에서 충돌 여부 반환하도록 수정) ---
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, x, y, angle=0):
        super().__init__()
        self.width = 30
        self.height = 20
        self.image = pygame.Surface([self.width, self.height], pygame.SRCALPHA)
        pygame.draw.rect(self.image, BLUE, (0, 0, self.width, self.height))
        pygame.draw.circle(self.image, RED, (self.width - 5, self.height // 2), 5)
        self.original_image = self.image
        self.rect = self.image.get_rect(center=(x, y))
        self.x = float(x)
        self.y = float(y)
        self.angle = math.radians(angle)
        self.speed = 0.0
        self.max_speed = 3.0
        self.acceleration = 0.08
        self.steering_angle = 0.0
        self.max_steering = math.radians(40)
        self.last_valid_position = (self.x, self.y)

    def update(self, control_signal, dt):
        collided = False # 충돌 상태 초기화
        self.last_valid_position = (self.x, self.y)

        # 1. 조향각 업데이트
        self.steering_angle = control_signal['steering'] * self.max_steering

        # 2. 속도 업데이트
        # 수동 조작 시에는 직접적인 스로틀 값을 사용할 수 있도록 수정 가능 (현재는 동일 로직 사용)
        if control_signal['throttle'] > 0:
            self.speed += self.acceleration * abs(control_signal['throttle']) # 스로틀 강도 반영
        elif control_signal['throttle'] < 0:
            self.speed -= self.acceleration * 2 * abs(control_signal['throttle'])
        else:
            self.speed *= 0.98 # 자연 감속
        self.speed = max(-self.max_speed / 2, min(self.max_speed, self.speed))

        # 3. 차량 각도 업데이트
        L = self.width * 0.8
        # 속도가 매우 낮을 때는 회전 효과 줄임 (제자리 회전 방지)
        weight = 50 # 회전 가중치
        if abs(self.speed) > 0.1 and abs(self.steering_angle) > 1e-4 :
            
             turn_radius = L / math.tan(self.steering_angle)
             angular_velocity = self.speed / turn_radius
             self.angle += angular_velocity * dt * weight # 회전 가중치 추가
             
             
        # 속도가 0에 가까울 때 수동으로 좌우키 누르면 제자리에서 약간 회전하도록 추가 (선택 사항)
        elif abs(self.speed) < 0.1 and abs(control_signal['steering']) > 0:
             manual_turn_rate = math.radians(60) # 초당 60도 회전
             self.angle += manual_turn_rate * control_signal['steering'] * dt


        # 4. 위치 업데이트
        self.x += self.speed * math.cos(self.angle) * dt * 60
        self.y += self.speed * math.sin(self.angle) * dt * 60

        # 5. 이미지 회전 및 위치 업데이트
        self.image = pygame.transform.rotate(self.original_image, -math.degrees(self.angle))
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

        # 충돌 처리
        for wall in walls:
            if self.rect.colliderect(wall):
                collided = True
                break
        if collided:
            self.x, self.y = self.last_valid_position
            self.speed = 0
            self.angle = math.atan2(self.last_valid_position[1] - self.y, self.last_valid_position[0] - self.x) # 이전 방향 유지 시도
            self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

        return collided # 충돌 여부 반환

    def draw(self, surface):
        surface.blit(self.image, self.rect)

# --- A* Node 클래스 (이전과 동일) ---
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    def __lt__(self, other): return self.f < other.f
    def __eq__(self, other): return self.position == other.position
    def __hash__(self): return hash(self.position)

# --- 좌표 변환 함수 (이전과 동일) ---
def to_grid_coords(world_x, world_y, grid_size):
    return int(world_x // grid_size), int(world_y // grid_size)
def to_world_coords(grid_x, grid_y, grid_size):
    return grid_x * grid_size + grid_size / 2, grid_y * grid_size + grid_size / 2

# --- Grid Map 생성 함수 (이전과 동일) ---
def create_grid_map(walls, width, height, grid_size, buffer):
    grid_width = width // grid_size
    grid_height = height // grid_size
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for r in range(grid_height):
        for c in range(grid_width):
            world_x, world_y = to_world_coords(c, r, grid_size)
            cell_rect = pygame.Rect(world_x - grid_size/2, world_y - grid_size/2, grid_size, grid_size)
            inflated_cell_rect = cell_rect.inflate(buffer * 2, buffer * 2)
            is_obstacle = False
            for wall in walls:
                if inflated_cell_rect.colliderect(wall):
                    is_obstacle = True
                    break
            if not (0 <= cell_rect.left and cell_rect.right <= width and \
                    0 <= cell_rect.top and cell_rect.bottom <= height):
                 is_obstacle = True

            if is_obstacle:
                 grid_map[r, c] = 1
    return grid_map

# --- 휴리스틱 함수 (이전과 동일) ---
def heuristic(a, b):
    return abs(a.position[0] - b.position[0]) + abs(a.position[1] - b.position[1])

# --- A* 경로 탐색 함수 (이전과 동일) ---
def a_star_pathfinding(grid_map, start_pos_grid, goal_pos_grid):
    start_time = time.time()
    start_node = Node(start_pos_grid)
    goal_node = Node(goal_pos_grid)
    if not (0 <= start_pos_grid[0] < grid_map.shape[1] and 0 <= start_pos_grid[1] < grid_map.shape[0]) or \
       not (0 <= goal_pos_grid[0] < grid_map.shape[1] and 0 <= goal_pos_grid[1] < grid_map.shape[0]):
        print("시작 또는 목표 지점이 맵 범위를 벗어났습니다.")
        return []
    if grid_map[start_node.position[1]][start_node.position[0]] == 1 or \
       grid_map[goal_node.position[1]][goal_node.position[0]] == 1:
        print("시작 또는 목표 지점이 장애물 위에 있습니다.")
        return []
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, start_node)
    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position in closed_list: # 이미 처리된 노드면 건너뜀 (heapq 특성상 중복 가능성)
             continue
        closed_list.add(current_node.position)
        if current_node == goal_node:
            path = []
            temp = current_node
            while temp is not None: path.append(temp.position); temp = temp.parent
            elapsed_time = time.time() - start_time
            print(f"Path found! Time: {elapsed_time:.4f}s, Length: {len(path)}")
            return path[::-1]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            if not (0 <= neighbor_pos[0] < grid_map.shape[1] and 0 <= neighbor_pos[1] < grid_map.shape[0]): continue
            if grid_map[neighbor_pos[1]][neighbor_pos[0]] == 1: continue
            if neighbor_pos in closed_list: continue
            neighbor_node = Node(neighbor_pos, current_node)
            move_cost = 1.414 if abs(dx) == 1 and abs(dy) == 1 else 1.0
            neighbor_node.g = current_node.g + move_cost
            neighbor_node.h = heuristic(neighbor_node, goal_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            # open_list 내 중복 체크 및 더 나은 경로 업데이트 (간략화: 그냥 추가)
            heapq.heappush(open_list, neighbor_node)
    elapsed_time = time.time() - start_time
    print(f"Path not found. Time: {elapsed_time:.4f}s")
    return []

# --- 제어 로직 함수 (이전과 동일) ---
def simple_controller(vehicle, path_world, current_path_index):
    if not path_world or current_path_index >= len(path_world):
        return {'throttle': 0.0, 'steering': 0.0}, current_path_index

    target_point = path_world[current_path_index]
    target_dx = target_point[0] - vehicle.x
    target_dy = target_point[1] - vehicle.y
    distance_to_target = math.sqrt(target_dx**2 + target_dy**2)

    if distance_to_target < 30 and current_path_index < len(path_world) - 1: # 마지막 점 아닐 때만 인덱스 증가
        current_path_index += 1
    elif distance_to_target < 15 and current_path_index == len(path_world) - 1: # 마지막 점 도착 시
        # 정지 또는 완료 상태 처리
         return {'throttle': -0.5, 'steering': 0.0}, current_path_index + 1 # 완료 표시 위해 인덱스 증가


    target_angle = math.atan2(target_dy, target_dx)
    angle_error = target_angle - vehicle.angle
    while angle_error > math.pi: angle_error -= 2 * math.pi
    while angle_error < -math.pi: angle_error += 2 * math.pi

    Kp_steering = 0.9
    steering_signal = Kp_steering * angle_error
    steering_signal = max(-1.0, min(1.0, steering_signal))

    angle_diff_abs = abs(math.degrees(angle_error))
    if angle_diff_abs > 30: target_speed_ratio = 0.3
    elif angle_diff_abs < 10: target_speed_ratio = 1.0
    else: target_speed_ratio = 0.7

    # 남은 경로가 짧을 경우 감속 (마지막 경로점 접근 시 부드럽게)
    if current_path_index == len(path_world) - 1:
        target_speed_ratio *= max(0.1, min(1.0, distance_to_target / 50.0)) # 거리에 비례하여 감속


    target_speed = vehicle.max_speed * target_speed_ratio
    throttle_signal = 0.5 * (target_speed - vehicle.speed)
    throttle_signal = max(-1.0, min(1.0, throttle_signal))

    return {'throttle': throttle_signal, 'steering': steering_signal}, current_path_index


# --- 메인 루프 ---
# 시작 위치를 벽 내부로 변경 (예: 좌측 하단 근처)
vehicle = Vehicle(150, 450, angle=0)
goal_pos_world = None
current_path_world = []
current_path_index = 0
grid_map = create_grid_map(walls, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, VEHICLE_RADIUS_BUFFER)
manual_control_active = False # 수동 조작 모드 플래그
last_path_failed = False # 경로 탐색 실패 여부 플래그

running = True
while running:
    dt = clock.tick(60) / 1000.0
    keys = pygame.key.get_pressed() # 키 입력 상태 확인

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            goal_pos_world = event.pos
            start_pos_world = (vehicle.x, vehicle.y)
            start_pos_grid = to_grid_coords(start_pos_world[0], start_pos_world[1], GRID_SIZE)
            goal_pos_grid = to_grid_coords(goal_pos_world[0], goal_pos_world[1], GRID_SIZE)

            grid_path = a_star_pathfinding(grid_map, start_pos_grid, goal_pos_grid)

            if grid_path:
                current_path_world = []
                for grid_node_pos in grid_path:
                    world_x, world_y = to_world_coords(grid_node_pos[0], grid_node_pos[1], GRID_SIZE)
                    current_path_world.append((world_x, world_y))
                current_path_index = 0
                manual_control_active = False # 새 경로 찾으면 자동 모드로 전환
                last_path_failed = False
            else:
                print("Pathfinding failed. Switching to manual control.")
                current_path_world = []
                manual_control_active = True # 경로 못 찾으면 수동 모드로
                last_path_failed = True

    # --- 제어 신호 결정 ---
    control_signal = {'throttle': 0.0, 'steering': 0.0}

    if manual_control_active:
        # 수동 조작 로직
        if keys[pygame.K_UP]:
            control_signal['throttle'] = 0.8 # 전진 강도
        elif keys[pygame.K_DOWN]:
            control_signal['throttle'] = -0.6 # 후진 강도

        if keys[pygame.K_LEFT]:
            control_signal['steering'] = 4.0 # 좌회전 강도
        elif keys[pygame.K_RIGHT]:
            control_signal['steering'] = 4.0 # 우회전 강도

    else:
        # 자동 조작 로직 (경로 있을 때만)
        if current_path_world:
            control_signal, current_path_index_new = simple_controller(vehicle, current_path_world, current_path_index)
            current_path_index = current_path_index_new
            # 경로 완료 시 수동 모드로 전환
            if current_path_index >= len(current_path_world):
                print("Path completed. Switching to manual control.")
                manual_control_active = True
                current_path_world = [] # 완료된 경로는 비움
        else:
             # 경로 없으면 (초기 상태 등) 수동 모드 유지 또는 활성화
             manual_control_active = True


    # --- 차량 상태 업데이트 ---
    collided = vehicle.update(control_signal, dt)
    if collided:
        print("Collision detected! Switching to manual control.")
        manual_control_active = True # 충돌 시 수동 모드로 전환
        current_path_world = [] # 충돌 시 현재 경로 무효화


    # --- 화면 그리기 ---
    screen.fill(WHITE)
    # 그리드 시각화 (선택)
    # for r in range(grid_map.shape[0]):
    #     for c in range(grid_map.shape[1]):
    #         color = DARK_GRAY if grid_map[r,c] == 1 else GRAY
    #         pygame.draw.rect(screen, color, (c*GRID_SIZE, r*GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)

    for wall in walls: pygame.draw.rect(screen, DARK_GRAY, wall)
    vehicle.draw(screen)
    if goal_pos_world:
        pygame.draw.circle(screen, GREEN, goal_pos_world, 10)
        pygame.draw.circle(screen, BLACK, goal_pos_world, 10, 1)
    if len(current_path_world) > 1 and not manual_control_active: # 자동 모드일 때만 경로 표시
        pygame.draw.lines(screen, MAGENTA, False, current_path_world, 3)
        if current_path_index < len(current_path_world):
             pygame.draw.circle(screen, YELLOW, current_path_world[current_path_index], 5)

    # 상태 정보 표시
    speed_text = FONT.render(f"Speed: {vehicle.speed:.1f}", True, BLACK)
    angle_text = FONT.render(f"Angle: {math.degrees(vehicle.angle):.1f}", True, BLACK)
    screen.blit(speed_text, (10, 10))
    screen.blit(angle_text, (10, 30))

    # 제어 모드 표시
    mode_text_str = "Mode: MANUAL (Use Arrow Keys)" if manual_control_active else "Mode: AUTO"
    mode_color = CYAN if manual_control_active else GREEN
    mode_text = BOLD_FONT.render(mode_text_str, True, mode_color)
    screen.blit(mode_text, (SCREEN_WIDTH - mode_text.get_width() - 10, 10))

    # 경로 탐색 실패 메시지 표시좌
    if last_path_failed and manual_control_active:
         fail_text = FONT.render("Pathfinding Failed!", True, RED)
         screen.blit(fail_text, (SCREEN_WIDTH - fail_text.get_width() - 10, 40))


    pygame.display.flip()

pygame.quit()