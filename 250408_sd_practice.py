import pygame
import numpy as np
import math
import heapq # A* 구현 시 사용

# --- 초기 설정 ---
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Autonomous Driving Simulation")
clock = pygame.time.Clock()

# --- 색상 정의 ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

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
        self.original_image = self.image
        self.rect = self.image.get_rect(center=(x, y))
        self.x = float(x)
        self.y = float(y)
        self.angle = math.radians(angle) # 각도는 라디안 사용
        self.speed = 0.0
        self.max_speed = 3.0
        self.acceleration = 0.1
        self.steering_angle = 0.0 # 조향각 (라디안)
        self.max_steering = math.radians(40) # 최대 조향각

    def update(self, control_signal):
        # control_signal: {'throttle': float, 'steering': float} (-1 to 1)

        # 1. 조향각 업데이트
        self.steering_angle = control_signal['steering'] * self.max_steering

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

    def draw(self, surface):
        surface.blit(self.image, self.rect)

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
    target_dx = target_point[0] - vehicle.x
    target_dy = target_point[1] - vehicle.y
    distance_to_target = math.sqrt(target_dx**2 + target_dy**2)

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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
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

pygame.quit()