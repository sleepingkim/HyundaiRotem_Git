# -*- coding: utf-8 -*-
# 필요한 라이브러리들을 임포트합니다.
import sys
import os
import time

# (주석 처리됨) Ursina 라이브러리 경로를 수동으로 설정해야 할 경우 사용합니다.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# ursina_path = os.path.join(current_dir, '..')
# sys.path.append(ursina_path)

try:
    # Ursina: 3D 게임 엔진 및 시뮬레이션 환경 제공
    from ursina import *
    # Scikit-learn: DBSCAN 클러스터링 알고리즘 사용
    from sklearn.cluster import DBSCAN
    # NumPy: 다차원 배열 및 수학 연산 지원 (필수)
    import numpy as np
    # Matplotlib: 데이터 시각화 (특히 3D 플롯)
    from mpl_toolkits.mplot3d import Axes3D # 3D 플롯 위해 추가
except ImportError as e:
    # 필수 라이브러리가 설치되지 않았을 경우 오류 메시지 출력 및 종료
    t_now = time.strftime("%Y-%m-%d %H:%M:%S") # 현재 시간 포함
    print(f"[{t_now}] 오류: 필수 라이브러리 임포트 실패: {e}")
    print(f"[{t_now}] Ursina, Scikit-learn, NumPy, Pandas, Matplotlib 라이브러리가 설치되어 있는지 확인하세요.")
    print(f"[{t_now}] 설치 명령어 예시: pip install ursina scikit-learn numpy pandas matplotlib")
    sys.exit(1) # 프로그램 종료
except Exception as e:
    # 기타 예기치 않은 임포트 오류 발생 시 처리
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] 임포트 중 예기치 않은 오류 발생: {e}")
    sys.exit(1)

# --- 추가 필수 라이브러리 ---
import random # 무작위 값 생성 (장애물 배치 등)
import math   # 수학 함수 사용 (삼각 함수, 각도 변환 등)
import matplotlib.pyplot as plt # 2D 및 3D 그래프 생성
import matplotlib.patches as patches # 2D 도형(사각형 등) 그리기 (Ground Truth 맵)
import pandas as pd # 데이터프레임 사용 (장애물 정보 저장)
import pickle # 파이썬 객체 직렬화/역직렬화 (scan_history 저장/로드)

# --- 시뮬레이션 환경 설정값 ---
NUM_OBSTACLES = 80      # 생성할 장애물 개수
AREA_SIZE = 50          # 시뮬레이션 영역의 크기 (정사각형 영역)

AGENT_SPEED = 5         # 에이전트(로봇)의 이동 속도
ROTATION_SPEED = 100    # 에이전트의 회전 속도 (Q, E 키)
AGENT_HEIGHT = 0.5      # 에이전트의 높이 (Y 좌표)

LIDAR_RANGE = 15        # 라이다 센서의 최대 감지 거리
LIDAR_FOV = 120         # 라이다의 수평 시야각 (도)
NUM_LIDAR_RAYS = 24     # 라이다의 수평 해상도 (수평 방향 스캔 라인 수)

# --- 3D 라이다 추가 설정값 ---
LIDAR_VERTICAL_FOV = 30 # 라이다의 수직 시야각 (도)
NUM_LIDAR_CHANNELS = 8  # 라이다의 수직 채널 수 (수직 방향 스캔 라인 수)
# -------------------------
LIDAR_VIS_HEIGHT = 0.6  # 라이다 센서의 시각화 및 스캔 시작 높이 (에이전트 바닥 기준 오프셋)
LIDAR_COLOR = color.cyan # 라이다 광선 기본 색상 (미충돌 시)
OBSTACLE_TAG = "obstacle" # 장애물 엔티티를 식별하기 위한 태그
SCAN_INTERVAL = 1.0     # 라이다 스캔 실행 간격 (초)

# --- 클러스터링 알고리즘 (DBSCAN) 설정값 ---
DBSCAN_EPS = 0.7        # DBSCAN: 클러스터 내 점들 간의 최대 거리 (epsilon)
                        # 3D 공간이므로 2D보다 약간 크게 설정될 수 있음
DBSCAN_MIN_SAMPLES = 10 # DBSCAN: 클러스터를 형성하기 위한 최소 점 개수 (MinPts)
                        # 점 밀도에 따라 조정 필요

# --- Ursina 애플리케이션 설정 ---
app = Ursina(
    title="3D LiDAR Simulation & 3D BBox Estimation", # 창 제목 설정
    borderless=False, # 창 테두리 표시
)

# --- 마우스 설정 ---
# 이 설정은 마우스 커서를 보이게 하고 잠금을 해제하여,
# 마우스로 카메라 시점을 제어하는 기능을 비활성화합니다.
mouse.visible = True # 마우스 커서 보이기
mouse.locked = False # 마우스 커서 잠금 해제

# --- 전역 변수 초기화 ---
scan_timer = 0.0 # 다음 스캔까지 남은 시간을 추적하는 타이머
# scan_history: 각 스캔 시점의 에이전트 자세와 감지된 상대 좌표점들을 저장하는 리스트
# 각 요소는 {'pose': (x, z, yaw_rad), 'relative_points': [[lx, ly, lz], ...]} 형태의 딕셔너리
scan_history = []
obstacle_df = None # 실제 장애물 정보를 저장할 Pandas DataFrame (Ground Truth용)
# 시뮬레이션 영역(바닥) 정보를 저장하는 딕셔너리
ground_info = {'size': AREA_SIZE, 'center_x': 0, 'center_z': 0}

# --- 환경 요소 생성 (바닥, 하늘, 장애물) ---
# 바닥 생성: 평면 모델, 지정된 크기, 밝은 회색, 텍스처 적용, 충돌체 설정
ground = Entity(model='plane', scale=(AREA_SIZE, 1, AREA_SIZE), color=color.light_gray, texture='white_cube', texture_scale=(AREA_SIZE/2, AREA_SIZE/2), collider='box')
# 하늘 생성: 기본적인 하늘 배경
sky = Sky()
# 실제 장애물 정보(Ground Truth)를 DataFrame으로 만들기 위한 리스트
obstacles_list_for_df = []
# Ursina 환경 내 장애물 엔티티들을 저장할 리스트
obstacles = []
# 에이전트 시작 위치에서 장애물까지의 최소 거리 (시작 시 충돌 방지)
min_obstacle_distance_from_spawn = 3.0
# 지정된 개수(NUM_OBSTACLES)만큼 장애물 생성
for i in range(NUM_OBSTACLES):
    # 장애물이 시작 위치와 너무 가깝지 않도록 위치를 무작위로 선정
    while True:
        pos_x = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5) # X 좌표 무작위 설정
        pos_z = random.uniform(-AREA_SIZE / 2.5, AREA_SIZE / 2.5) # Z 좌표 무작위 설정
        # 시작점(0, 0)과의 거리 계산
        distance_from_spawn = math.sqrt(pos_x**2 + pos_z**2)
        # 최소 거리 이상이면 위치 확정
        if distance_from_spawn >= min_obstacle_distance_from_spawn: break
    # 장애물의 크기(너비, 높이, 깊이)를 무작위로 설정
    scale_x = random.uniform(0.5, 3)
    scale_y = random.uniform(1, 5) # 높이는 더 다양하게
    scale_z = random.uniform(0.5, 3)
    # 장애물의 Y 좌표는 높이의 절반으로 설정 (바닥에 놓이도록)
    pos_y = scale_y / 2
    # 장애물 색상 무작위 설정
    obstacle_color = color.random_color()
    # Ursina Entity로 장애물 생성
    obs_entity = Entity(
        model='cube',                 # 모델 형태: 큐브
        position=(pos_x, pos_y, pos_z), # 위치 설정
        scale=(scale_x, scale_y, scale_z), # 크기 설정
        color=obstacle_color,         # 색상 설정
        collider='box',               # 충돌체 형태: 박스
        tag=OBSTACLE_TAG,             # 태그 설정 (라이다 감지용)
        name=f"obstacle_{i}"          # 이름 설정 (디버깅용)
    )
    obstacles.append(obs_entity) # 생성된 엔티티를 리스트에 추가
    # Ground Truth 정보를 DataFrame용 리스트에 추가
    obstacles_list_for_df.append({
        'name': f"obstacle_{i}",
        'center_x': pos_x, 'center_z': pos_z, # 중심 좌표 (XZ)
        'width': scale_x, 'depth': scale_z, 'height': scale_y, # 크기 정보
        'rotation_y': 0 # 현재는 회전 없음 (필요시 추가 가능)
    })
# Ground Truth 장애물 정보로 Pandas DataFrame 생성
obstacle_df = pd.DataFrame(obstacles_list_for_df)
print(f"실제 장애물 정보 DataFrame 생성 완료 ({len(obstacle_df)}개).")

# --- 에이전트(로봇) 및 카메라 설정 ---
# 에이전트 생성: 구체 모델, 파란색, 초기 위치(0, AGENT_HEIGHT, 0), 충돌체 설정
agent = Entity(model='sphere', color=color.blue, position=(0, AGENT_HEIGHT, 0), collider='sphere', scale=1)
# 카메라 설정:
camera.parent = agent        # 카메라를 에이전트의 자식으로 설정 (에이전트를 따라다님)
camera.position = (0, 15, -8) # 에이전트 기준 카메라의 상대적 위치 (뒤쪽 위)
camera.rotation_x = 45        # 카메라의 초기 상하 각도 (아래를 보도록)
camera.rotation_y = 0         # 카메라의 초기 좌우 각도 (정면)
camera.fov = 75               # 카메라의 시야각 (Field of View)

# --- 유틸리티 함수 ---
def normalize_angle(rad):
    """ 라디안 각도를 -pi 에서 +pi 사이로 정규화합니다. """
    while rad > math.pi: rad -= 2 * math.pi
    while rad < -math.pi: rad += 2 * math.pi
    return rad

# --- 3D 라이다 시각화 함수 ---
lidar_lines = [] # Ursina 씬에 그려진 라이다 선들을 저장하는 리스트
def update_lidar_visualization():
    """ 현재 라이다 스캔 결과를 Ursina 씬에 실시간으로 그립니다. """
    global lidar_lines
    # 이전 프레임에서 그린 라인들을 제거
    for line in lidar_lines:
        destroy(line)
    lidar_lines.clear() # 리스트 비우기

    # 현재 에이전트의 월드 좌표 및 Y축 회전 각도(도) 가져오기
    agent_pos = agent.world_position
    agent_rot_y_deg = agent.world_rotation_y
    # 라이다 센서의 월드 좌표 계산 (에이전트 위치 + 지정된 높이 오프셋)
    # 주의: agent.up을 사용해야 에이전트 기울어짐에도 정확한 높이 오프셋 적용 가능
    # lidar_origin_world = agent_pos + agent.up * LIDAR_VIS_HEIGHT
    # 현재 코드에서는 간략화된 버전 사용 (에이전트가 기울지 않는다고 가정)
    lidar_origin_world = agent_pos + Vec3(0, LIDAR_VIS_HEIGHT, 0)

    # 수평/수직 각도 단계 계산
    h_angle_step = LIDAR_FOV / (NUM_LIDAR_RAYS - 1) if NUM_LIDAR_RAYS > 1 else 0
    v_angle_step = LIDAR_VERTICAL_FOV / (NUM_LIDAR_CHANNELS - 1) if NUM_LIDAR_CHANNELS > 1 else 0
    # 수직 스캔 시작 각도 (중심 아래)
    v_angle_start = -LIDAR_VERTICAL_FOV / 2

    # 모든 수직 채널(라인)에 대해 반복
    for ch in range(NUM_LIDAR_CHANNELS):
        v_angle_deg = v_angle_start + ch * v_angle_step # 현재 채널의 수직 각도 (도)
        v_angle_rad = math.radians(v_angle_deg) # 라디안으로 변환
        cos_v = math.cos(v_angle_rad); sin_v = math.sin(v_angle_rad)
        # 현재 채널의 수평 스캔 시작 각도 (월드 기준, 에이전트 회전 고려)
        h_angle_start_world_deg = agent_rot_y_deg - LIDAR_FOV / 2

        # 모든 수평 광선(Ray)에 대해 반복
        for i in range(NUM_LIDAR_RAYS):
            h_angle_world_deg = h_angle_start_world_deg + i * h_angle_step # 현재 광선의 수평 각도 (월드 기준)
            h_angle_world_rad = math.radians(h_angle_world_deg) # 라디안으로 변환
            cos_h = math.cos(h_angle_world_rad); sin_h = math.sin(h_angle_world_rad)

            # 3D 광선 방향 벡터 계산 (월드 좌표계 기준)
            # Y축이 위쪽인 좌표계 기준: (cos(수직각)*sin(수평각), sin(수직각), cos(수직각)*cos(수평각))
            # 참고: 이 계산은 에이전트의 Yaw(Y축 회전)만 고려한 간략화된 방식입니다.
            #       에이전트가 Pitch나 Roll을 할 경우 정확한 방향 계산을 위해 쿼터니언 사용 필요.
            direction = Vec3(cos_v * sin_h, sin_v, cos_v * cos_h).normalized()

            # 레이캐스팅(Raycasting) 수행: 센서 원점에서 계산된 방향으로 광선 발사
            hit_info = raycast(
                origin=lidar_origin_world, # 광선 시작점
                direction=direction,       # 광선 방향
                distance=LIDAR_RANGE,      # 최대 탐지 거리
                ignore=[agent,],           # 무시할 엔티티 (자기 자신)
                debug=False,               # 디버그용 시각화 비활성화
                traverse_target=scene      # 검색 대상 (전체 씬)
            )

            # 레이캐스팅 결과 처리
            if hit_info.hit: # 광선이 어딘가에 충돌했을 경우
                end_point = hit_info.world_point # 충돌 지점 월드 좌표
                # 충돌한 엔티티가 장애물인지 확인 (태그 비교)
                is_obstacle = hasattr(hit_info.entity, 'tag') and hit_info.entity.tag == OBSTACLE_TAG
                # 충돌한 엔티티가 바닥인지 확인
                is_ground = hit_info.entity == ground

                # 충돌 대상에 따라 선 색상 결정
                if is_obstacle: line_color = color.red     # 장애물: 빨간색
                elif is_ground: line_color = color.gray    # 바닥: 회색
                else: line_color = color.orange  # 기타: 주황색 (디버깅용)
            else: # 광선이 아무것에도 충돌하지 않고 최대 거리에 도달한 경우
                end_point = lidar_origin_world + direction * LIDAR_RANGE # 최대 거리 지점
                line_color = LIDAR_COLOR     # 미충돌: 청록색 (기본 라이다 색상)

            # 매우 짧은 선은 그리지 않음 (시각적 노이즈 방지)
            if distance(lidar_origin_world, end_point) > 0.01:
                # Ursina의 Line 모델을 사용하여 라이다 광선 그리기
                line = Entity(model=Mesh(vertices=[lidar_origin_world, end_point], mode='line', thickness=1), color=line_color, alpha=0.7)
                lidar_lines.append(line) # 그려진 선을 리스트에 추가 (다음 프레임에서 지우기 위해)


# --- 3D 라이다 스캔 및 상대 좌표 데이터 생성 함수 ---
def generate_relative_lidar_map_3d():
    """ 에이전트 현재 위치/자세 기준으로, 감지된 장애물 포인트들의 상대 3D 좌표를 생성합니다. """
    relative_points_3d = [] # 상대 좌표 [rx, ry, rz]를 저장할 리스트

    # 에이전트의 현재 월드 위치 및 Y축 회전 각도(라디안)
    agent_pos_world = agent.world_position
    agent_rot_y_rad = math.radians(agent.world_rotation_y)
    # 라이다 스캔 원점 (월드 좌표계)
    scan_origin_world = agent_pos_world + Vec3(0, LIDAR_VIS_HEIGHT, 0)

    # 수평/수직 각도 단계 계산
    h_angle_step = LIDAR_FOV / (NUM_LIDAR_RAYS - 1) if NUM_LIDAR_RAYS > 1 else 0
    v_angle_step = LIDAR_VERTICAL_FOV / (NUM_LIDAR_CHANNELS - 1) if NUM_LIDAR_CHANNELS > 1 else 0
    v_angle_start = -LIDAR_VERTICAL_FOV / 2

    # 모든 수직/수평 각도 조합에 대해 레이캐스팅 수행
    for ch in range(NUM_LIDAR_CHANNELS):
        v_angle_deg = v_angle_start + ch * v_angle_step
        v_angle_rad = math.radians(v_angle_deg)
        cos_v = math.cos(v_angle_rad); sin_v = math.sin(v_angle_rad)
        h_angle_start_world_deg = agent.world_rotation_y - LIDAR_FOV / 2

        for i in range(NUM_LIDAR_RAYS):
            h_angle_world_deg = h_angle_start_world_deg + i * h_angle_step
            h_angle_world_rad = math.radians(h_angle_world_deg)
            cos_h = math.cos(h_angle_world_rad); sin_h = math.sin(h_angle_world_rad)

            # 3D 광선 방향 벡터 계산 (월드 좌표계 기준, Yaw만 고려한 간략화)
            direction = Vec3(cos_v * sin_h, sin_v, cos_v * cos_h).normalized()
            # 레이캐스팅 수행
            hit_info = raycast(scan_origin_world, direction, distance=LIDAR_RANGE, ignore=[agent,], debug=False, traverse_target=scene)

            # **장애물에 충돌한 경우에만** 상대 좌표 계산
            if hit_info.hit and hasattr(hit_info.entity, 'tag') and hit_info.entity.tag == OBSTACLE_TAG:
                hit_point_world = hit_info.world_point # 충돌 지점 월드 좌표
                # 에이전트 중심에서 충돌 지점까지의 벡터 (월드 좌표계 기준)
                world_vec = hit_point_world - agent_pos_world

                # 월드 벡터를 에이전트 로컬 좌표계로 변환 (Y축 회전만 고려)
                # 에이전트의 현재 Y축 회전(agent_rot_y_rad)의 반대 방향(-agent_rot_y_rad)으로 회전시켜 로컬 좌표를 얻음
                cos_a = math.cos(-agent_rot_y_rad); sin_a = math.sin(-agent_rot_y_rad)
                # 2D 회전 변환 적용 (XZ 평면)
                relative_x = world_vec.x * cos_a - world_vec.z * sin_a
                relative_y = world_vec.y # Y 값은 회전에 영향받지 않는다고 가정 (에이전트 기울기 없음 가정)
                relative_z = world_vec.x * sin_a + world_vec.z * cos_a
                # 계산된 상대 좌표 [rx, ry, rz]를 리스트에 추가
                relative_points_3d.append([relative_x, relative_y, relative_z])

    # 상대 좌표 리스트를 NumPy 배열로 변환하여 반환 (데이터가 없으면 빈 배열 반환)
    return np.array(relative_points_3d) if relative_points_3d else np.empty((0, 3))

# --- 누적된 스캔 데이터로부터 전역 3D 맵 데이터 생성 함수 ---
def generate_map_data_3d(history):
    """ 스캔 기록(history)으로부터 전역 3D 맵 포인트 클라우드와 에이전트 궤적을 생성합니다. """
    global_map_points_list = [] # 모든 스캔의 전역 좌표점들을 모을 리스트
    agent_trajectory_x, agent_trajectory_z = [], [] # 에이전트의 X, Z 궤적 저장 리스트
    # 기록이 없으면 빈 데이터 반환
    if not history: return np.empty((0, 3)), [], []

    # 각 스캔 기록에 대해 반복 처리
    for scan_record in history:
        # 스캔 당시의 에이전트 자세(위치 x, z, 회전 yaw) 가져오기
        pose = scan_record['pose'] # (x, z, rot_rad)
        # 스캔 당시 감지된 상대 좌표점들 가져오기
        relative_points_list = scan_record['relative_points'] # [[rx, ry, rz], ...]
        relative_points = np.array(relative_points_list) # NumPy 배열로 변환
        # 해당 스캔에서 감지된 포인트가 없으면 다음 기록으로 넘어감
        if relative_points.shape[0] == 0: continue

        # 스캔 당시 에이전트의 월드 위치 (X, Z) 및 Y축 회전 각도(라디안)
        scan_pos_x, scan_pos_z, scan_rot_rad = pose
        # 스캔 당시 에이전트의 월드 위치 (Vec3 형태, Y 좌표 포함)
        agent_pos_world = Vec3(scan_pos_x, AGENT_HEIGHT, scan_pos_z)
        # 에이전트 궤적 기록
        agent_trajectory_x.append(scan_pos_x); agent_trajectory_z.append(scan_pos_z)

        # 상대 좌표점들을 월드 좌표로 변환
        # 스캔 당시 에이전트의 Y축 회전(scan_rot_rad)을 사용하여 변환
        cos_a, sin_a = math.cos(scan_rot_rad), math.sin(scan_rot_rad)
        # 2D 회전 행렬 생성 (Y축 기준 회전)
        rot_matrix_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # 상대 좌표에서 X, Z 성분만 추출 (N, 2) 형태
        relative_xz = relative_points[:, [0, 2]] # rx, rz
        # 2D 회전 적용: relative_xz (N, 2) @ rot_matrix_2d.T (2, 2) -> rotated_xz (N, 2)
        rotated_xz = np.dot(relative_xz, rot_matrix_2d.T)
        # 월드 좌표 계산: 회전된 XZ 좌표에 스캔 당시 에이전트의 XZ 위치 더하기
        world_x = rotated_xz[:, 0] + scan_pos_x
        world_z = rotated_xz[:, 1] + scan_pos_z
        # 월드 Y 좌표 계산: 상대 Y 좌표에 스캔 당시 에이전트의 월드 Y 위치 더하기
        # (generate_relative_lidar_map_3d에서 relative_y = world_vec.y = hit_point_world.y - agent_pos_world.y 로 계산되었음)
        # 따라서 world_y = relative_y + agent_pos_world.y 는 hit_point_world.y 와 동일함.
        world_y = relative_points[:, 1] + agent_pos_world.y

        # 계산된 월드 좌표들을 (N, 3) 형태의 배열 [X, Y, Z]로 결합
        world_points = np.stack((world_x, world_y, world_z), axis=-1)

        # 현재 스캔의 월드 좌표점들을 전체 리스트에 추가
        global_map_points_list.append(world_points)

    # 모든 스캔의 월드 좌표점들을 하나의 큰 NumPy 배열로 합침
    map_points_np = np.concatenate(global_map_points_list, axis=0) if global_map_points_list else np.empty((0, 3))
    # 최종 전역 맵 포인트 클라우드와 에이전트 궤적 반환
    return map_points_np, agent_trajectory_x, agent_trajectory_z

# --- 3D 직육면체(Bounding Box) 그리기 헬퍼 함수 ---
def draw_cuboid(ax, min_coords, max_coords, color='r', alpha=0.1, linewidth=1, label=None):
    """ Matplotlib 3D 축(ax)에 주어진 최소/최대 좌표로 정의되는 직육면체를 그립니다. """
    # min_coords: (min_x, min_y, min_z) - 클러스터 포인트들의 최소 월드 좌표
    # max_coords: (max_x, max_y, max_z) - 클러스터 포인트들의 최대 월드 좌표
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords

    # 직육면체의 8개 꼭지점(vertex) 좌표 정의
    # Matplotlib 3D 플롯은 기본적으로 (X, Y, Z) 순서로 축을 다루지만,
    # 이 코드의 다른 부분들(특히 포인트 클라우드 플롯)과의 일관성을 위해
    # 데이터를 (World X, World Z, World Y(높이)) 순서로 넣어 플로팅합니다.
    vertices = [
        (min_x, min_z, min_y), # 0: 바닥면 좌측 앞
        (max_x, min_z, min_y), # 1: 바닥면 우측 앞
        (max_x, max_z, min_y), # 2: 바닥면 우측 뒤
        (min_x, max_z, min_y), # 3: 바닥면 좌측 뒤
        (min_x, min_z, max_y), # 4: 윗면 좌측 앞
        (max_x, min_z, max_y), # 5: 윗면 우측 앞
        (max_x, max_z, max_y), # 6: 윗면 우측 뒤
        (min_x, max_z, max_y)  # 7: 윗면 좌측 뒤
    ]
    vertices = np.array(vertices) # NumPy 배열로 변환

    # 직육면체의 12개 모서리(edge)를 정의하는 꼭지점 인덱스 쌍 리스트
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 바닥면 모서리 4개
        (4, 5), (5, 6), (6, 7), (7, 4),  # 윗면 모서리 4개
        (0, 4), (1, 5), (2, 6), (3, 7)   # 바닥면과 윗면을 잇는 수직 모서리 4개
    ]

    # 각 모서리를 선으로 그리기
    plotted_label = False # 라벨 중복 방지 플래그
    for i, edge in enumerate(edges):
        p1_idx, p2_idx = edge # 모서리를 구성하는 두 꼭지점의 인덱스
        # 두 꼭지점의 X, Z, Y 좌표를 각각 추출하여 plot 함수 인자로 전달
        xs = vertices[[p1_idx, p2_idx], 0] # X 좌표들
        zs = vertices[[p1_idx, p2_idx], 1] # Z 좌표들 (Matplotlib의 Y축에 해당)
        ys = vertices[[p1_idx, p2_idx], 2] # Y(높이) 좌표들 (Matplotlib의 Z축에 해당)
        # 첫 번째 모서리에만 라벨(label)을 지정하여 범례에 한 번만 표시되도록 함
        current_label = label if not plotted_label else None
        # 선 그리기 (alpha 값은 약간 진하게 설정)
        ax.plot(xs, zs, ys, color=color, alpha=alpha*2, linewidth=linewidth, label=current_label)
        if current_label: plotted_label = True # 라벨이 그려졌음을 표시

    # (주석 처리됨) 면 채우기 옵션:
    # 필요시 6개의 면을 정의하여 Poly3DCollection으로 채울 수 있으나,
    # 시각적으로 복잡해지고 성능 저하를 유발할 수 있어 여기서는 생략합니다.
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # faces = np.array([
    #     [vertices[0], vertices[1], vertices[5], vertices[4]], # 앞면
    #     [vertices[1], vertices[2], vertices[6], vertices[5]], # 오른쪽면
    #     # ... 나머지 면들 정의 ...
    # ])
    # ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors=color, alpha=alpha))


# --- 통합 맵 플로팅 함수 (3D Bounding Box 포함 버전) ---
def plot_all_maps(history, obs_df, gnd_info):
    """ 3개의 맵(3D 라이다 포인트, Ground Truth, 감지된 3D BBox)을 하나의 Figure에 표시합니다. """
    print("통합 맵 생성 중 (3D Bounding Box 포함)...")
    # 데이터 유효성 검사
    if not history: print("플롯할 스캔 기록이 없습니다."); return
    if obs_df is None: print("Ground Truth 맵을 위한 장애물 데이터가 없습니다."); return

    # 1. 데이터 준비: 전역 3D 포인트 클라우드 및 에이전트 궤적 생성
    map_points_np_3d, agent_trajectory_x, agent_trajectory_z = generate_map_data_3d(history)
    print(f"생성된 총 전역 3D 포인트 개수: {len(map_points_np_3d)}")

    # 포인트는 없지만 궤적만 있는 경우 처리
    if len(map_points_np_3d) == 0 and not agent_trajectory_x:
        print("생성된 포인트와 궤적 데이터가 모두 없습니다.")
        return
    elif len(map_points_np_3d) == 0 and agent_trajectory_x:
        print("생성된 포인트는 없으나 궤적 데이터는 있습니다. 궤적만 플롯합니다.")
        fig_traj, ax_traj = plt.subplots(figsize=(6, 6)) # 2D 플롯 생성
        ax_traj.plot(agent_trajectory_x, agent_trajectory_z, marker='o', markersize=3, linestyle='-', color='red', label='Agent Trajectory (GT)')
        if agent_trajectory_x: # 마지막 위치 표시
             ax_traj.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], s=50, c='magenta', marker='*', label='Last Scan Pose (GT)')
        ax_traj.set_title("Agent Trajectory Only (No 3D Points)")
        ax_traj.set_xlabel("World X"); ax_traj.set_ylabel("World Z")
        ax_traj.grid(True); ax_traj.set_aspect('equal'); ax_traj.legend()
        plt.show(block=True) # 플롯 표시
        return

    # 2. Matplotlib Figure 및 서브플롯 생성 (1행 3열)
    fig = plt.figure(figsize=(24, 8)) # Figure 전체 크기 설정
    fig.suptitle('3D LiDAR Mapping Analysis with 3D Bounding Box Estimation', fontsize=16) # 전체 제목

    # --- 서브플롯 1: 3D 라이다 포인트 클라우드 ---
    ax1 = fig.add_subplot(131, projection='3d') # 1행 3열 중 첫 번째, 3D 투영 설정
    if len(map_points_np_3d) > 0:
        # scatter 플롯: X좌표, Z좌표, Y좌표(높이) 순서로 인자 전달
        # 점 색상은 Y(높이) 값에 따라 viridis 컬러맵으로 지정, alpha는 투명도
        ax1.scatter(map_points_np_3d[:, 0], map_points_np_3d[:, 2], map_points_np_3d[:, 1],
                    s=1, c=map_points_np_3d[:, 1], cmap='viridis', alpha=0.3)

    # 에이전트 궤적을 XZ 평면(높이 0)에 플롯
    if agent_trajectory_x:
        # plot 함수에 zs=0, zdir='z' 옵션을 주어 XZ 평면에 그리도록 함
        ax1.plot(agent_trajectory_x, agent_trajectory_z, zs=0, zdir='z', marker='.', markersize=2, linestyle='-', color='red', alpha=0.7, label='Trajectory (GT)')
        # 마지막 스캔 위치를 별표로 표시
        ax1.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], zs=0, zdir='z', s=60, c='magenta', marker='*', label='Last Pose (GT)', depthshade=False)

    ax1.set_title("3D LiDAR Point Cloud") # 서브플롯 제목
    ax1.set_xlabel("World X (m)") # X축 라벨
    ax1.set_ylabel("World Z (m)") # Y축 라벨 (Matplotlib 3D 기준)
    ax1.set_zlabel("World Y (m)") # Z축 라벨 (Matplotlib 3D 기준, 높이)
    limit = AREA_SIZE / 1.8 # 플롯 축 범위 설정용 변수
    ax1.set_xlim(-limit, limit); ax1.set_ylim(-limit, limit); ax1.set_zlim(0, 10) # X, Z, Y 축 범위 설정
    ax1.view_init(elev=30., azim=-60) # 초기 3D 뷰 각도 설정 (elev: 고도, azim: 방위각)
    # ax1.legend() # 포인트 수가 많으면 범례가 복잡해지므로 생략 가능

    # --- 서브플롯 2: Ground Truth 맵 (2D) ---
    ax2 = fig.add_subplot(132) # 1행 3열 중 두 번째, 기본 2D 플롯
    gnd_size = gnd_info['size'] # 바닥 크기
    # 바닥 영역 테두리 그리기 (회색 점선)
    ax2.add_patch(patches.Rectangle((-gnd_size/2, -gnd_size/2), gnd_size, gnd_size, edgecolor='gray', facecolor='none', linestyle='--', label='_nolegend_'))
    # 저장된 실제 장애물 정보(obstacle_df)를 기반으로 각 장애물 그리기
    for index, row in obs_df.iterrows():
        x, z, w, d, rot = row['center_x'], row['center_z'], row['width'], row['depth'], row['rotation_y']
        # 장애물 중심 좌표와 크기로부터 좌측 하단 좌표 계산
        bottom_left_x = x - w / 2; bottom_left_z = z - d / 2
        # 사각형 패치 생성 및 추가 (어두운 회색 채움, 검은색 테두리)
        # 첫 번째 장애물에만 라벨 부여
        ax2.add_patch(patches.Rectangle((bottom_left_x, bottom_left_z), w, d, edgecolor='black', facecolor='darkgray', angle=rot, label='GT Obstacles' if index == 0 else '_nolegend_'))
    # 시작점(0, 0)을 빨간색 X 마커로 표시
    ax2.scatter(0, 0, s=100, c='red', marker='x', label='Start (0, 0)')
    ax2.set_title("Ground Truth Map (2D)") # 서브플롯 제목
    ax2.set_xlabel("World X (m)"); ax2.set_ylabel("World Z (m)") # 축 라벨
    ax2.grid(True, linestyle='--', alpha=0.6) # 그리드 표시
    ax2.set_aspect('equal', adjustable='box') # X, Y 축 비율 동일하게 설정
    ax2.legend(fontsize='small') # 범례 표시 (작은 글씨)
    ax2.set_xlim(-limit, limit); ax2.set_ylim(-limit, limit) # 축 범위 설정

    # --- 서브플롯 3: 감지된 장애물 (3D Bounding Boxes) ---
    ax3 = fig.add_subplot(133, projection='3d') # 1행 3열 중 세 번째, 3D 투영 설정
    detected_obstacle_count = 0 # 감지된 장애물(클러스터) 개수 카운터
    # 클러스터링을 위한 최소 포인트 개수 충족 시 DBSCAN 실행
    if len(map_points_np_3d) >= DBSCAN_MIN_SAMPLES:
        try:
            print("DBSCAN 클러스터링 실행 중...")
            # DBSCAN 객체 생성 및 학습 (포인트 클라우드 데이터 사용)
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(map_points_np_3d)
            labels = db.labels_ # 각 포인트에 할당된 클러스터 레이블 (-1은 노이즈)
            unique_labels = set(labels) # 고유한 레이블 집합
            # 클러스터별 색상 지정을 위한 컬러맵 생성
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            print(f"DBSCAN 완료. {num_clusters}개의 클러스터 발견.")

            cluster_label_printed = False # 범례 라벨 중복 방지 플래그
            # 고유한 클러스터 레이블 각각에 대해 반복
            for k, col in zip(unique_labels, colors):
                if k == -1: continue # 노이즈 포인트(-1 레이블)는 건너뜀

                # 현재 클러스터(k)에 속하는 포인트들만 선택
                cluster_mask = (labels == k)
                cluster_points_3d = map_points_np_3d[cluster_mask]

                # 클러스터에 포인트가 존재하는 경우
                if len(cluster_points_3d) > 0:
                    # *** 클러스터의 3D Axis-Aligned Bounding Box (AABB) 계산 ***
                    # 각 축(X, Y, Z) 방향으로 최소/최대 좌표 계산
                    min_coords = np.min(cluster_points_3d, axis=0) # (min_x, min_y, min_z)
                    max_coords = np.max(cluster_points_3d, axis=0) # (max_x, max_y, max_z)

                    # *** 계산된 3D Bounding Box 그리기 ***
                    # 범례에 'Detected 3D BBox' 라벨을 한 번만 표시
                    box_label = 'Detected 3D BBox' if not cluster_label_printed else None
                    # draw_cuboid 헬퍼 함수 호출하여 직육면체 그리기
                    draw_cuboid(ax3, min_coords, max_coords, color=col, alpha=0.1, linewidth=1.5, label=box_label)
                    if not cluster_label_printed: cluster_label_printed = True # 라벨 그려짐 표시

                    detected_obstacle_count += 1 # 감지된 장애물 개수 증가

            print(f'감지된 장애물 플롯: {detected_obstacle_count}개의 3D 경계 상자 그림.')

        except Exception as e: # DBSCAN 또는 플로팅 중 오류 발생 시
            print(f"DBSCAN 또는 3D Box 플로팅 중 오류 발생: {e}")
            # 오류 메시지를 플롯 중앙에 텍스트로 표시 (위치 (0,0,0)은 임시)
            ax3.text(0, 0, 0, 'DBSCAN/Plotting Error', color='red')

    else: # 클러스터링 위한 최소 포인트 개수 미달 시
        print("DBSCAN을 위한 포인트 개수가 부족합니다.")
        # 메시지를 플롯 중앙에 텍스트로 표시
        ax3.text(0, 0, 0, 'Not Enough Points for DBSCAN', color='gray')

    # 에이전트 궤적을 3D 플롯의 XZ 평면(높이 0)에 표시
    if agent_trajectory_x:
        ax3.plot(agent_trajectory_x, agent_trajectory_z, zs=0, zdir='z', marker='.', markersize=2, linestyle='-', color='red', alpha=0.7, label='Trajectory (GT)')
        ax3.scatter(agent_trajectory_x[-1], agent_trajectory_z[-1], zs=0, zdir='z', s=60, c='magenta', marker='*', label='Last Pose (GT)', depthshade=False)

    ax3.set_title(f"Detected Obstacles ({detected_obstacle_count} 3D BBoxes)") # 서브플롯 제목
    ax3.set_xlabel("World X (m)") # X축 라벨
    ax3.set_ylabel("World Z (m)") # Y축 라벨 (Matplotlib 3D 기준)
    ax3.set_zlabel("World Y (m)") # Z축 라벨 (Matplotlib 3D 기준, 높이)
    ax3.set_xlim(-limit, limit); ax3.set_ylim(-limit, limit); ax3.set_zlim(0, 10) # 축 범위 설정
    ax3.view_init(elev=30., azim=-60) # 초기 3D 뷰 각도 설정 (ax1과 동일하게)
    # 감지된 박스나 궤적이 있을 경우 범례 표시
    if detected_obstacle_count > 0 or agent_trajectory_x:
        ax3.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 서브플롯 간 간격 및 전체 제목 위치 자동 조정
    plt.show(block=True) # Matplotlib 창을 표시하고 닫힐 때까지 코드 실행을 멈춤


# --- UI 텍스트 디스플레이 설정 ---
# 화면 좌측 상단에 표시될 정보 텍스트 객체 생성
info_text = Text(
    origin=(-0.5, 0.5), # 텍스트 기준점 (좌측 상단)
    scale=(0.8, 0.8),   # 텍스트 크기 스케일
    x=-0.5 * window.aspect_ratio + 0.02, # X 위치 (창 비율에 따라 동적 조절)
    y=0.48,             # Y 위치 (고정)
    text="Initializing..." # 초기 텍스트
)

# --- 키보드 입력 처리 함수 ---
def input(key):
    """ 키보드 입력을 받아 특정 동작을 수행합니다. """
    global scan_history, obstacle_df # 전역 변수 사용 선언

    # 'M' 키: 통합 맵 플롯 함수 호출
    if key == 'm' or key == 'M':
        print("통합 맵 플롯 요청 (3D BBox 포함)...")
        plot_all_maps(scan_history, obstacle_df, ground_info) # 수정된 플롯 함수 호출

    # 'C' 키: 누적된 라이다 스캔 기록 삭제
    if key == 'c' or key == 'C':
        print("라이다 맵 기록 삭제 중...")
        scan_history = [] # scan_history 리스트 비우기

    # 'P' 키: 현재 스캔 기록 및 Ground Truth 데이터 저장
    if key == 'p' or key == 'P':
        print("데이터 저장 중...")
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S') # 현재 시간으로 파일명 생성
            # 스캔 기록 저장 (pickle 포맷)
            if scan_history:
                filename_hist = f"scan_history_3d_{timestamp}.pkl"
                with open(filename_hist, 'wb') as f:
                    pickle.dump(scan_history, f) # 객체를 파일에 저장
                print(f"스캔 기록 저장 완료: {filename_hist}")
            else: print("저장할 스캔 기록이 없습니다.")
            # Ground Truth 장애물 정보 저장 (CSV 포맷)
            if obstacle_df is not None:
                filename_obs = f"gt_obstacles_{timestamp}.csv"
                obstacle_df.to_csv(filename_obs, index=False) # DataFrame을 CSV로 저장
                print(f"Ground Truth 장애물 정보 저장 완료: {filename_obs}")
            else: print("저장할 Ground Truth 장애물 데이터가 없습니다.")
        except Exception as e: # 저장 중 오류 발생 시
            print(f"데이터 저장 중 오류 발생: {e}")

    # 'ESC' 키: 시뮬레이션 종료
    if key == 'escape':
        print("시뮬레이션 종료 중...");
        quit() # 프로그램 종료

# --- 메인 업데이트 루프 함수 ---
def update():
    """ Ursina 엔진에 의해 매 프레임 호출되는 함수. 시뮬레이션 로직 업데이트 담당. """
    global scan_timer, scan_history # 전역 변수 사용 선언

    # --- 에이전트 이동 및 회전 제어 ---
    original_position = agent.position # 충돌 시 복원을 위한 현재 위치 저장
    total_delta_x, total_delta_z = 0.0, 0.0 # 이번 프레임의 X, Z 이동량
    speed_dt = AGENT_SPEED * time.dt # 프레임 시간(time.dt)을 고려한 이동 속도

    # W, S, A, D 키 입력에 따른 이동량 계산 (에이전트의 앞/뒤/좌/우 방향 기준)
    if held_keys['w']: # 전진
        total_delta_x += agent.forward.x * speed_dt
        total_delta_z += agent.forward.z * speed_dt
    if held_keys['s']: # 후진
        total_delta_x -= agent.forward.x * speed_dt
        total_delta_z -= agent.forward.z * speed_dt
    if held_keys['a']: # 좌측 이동
        total_delta_x -= agent.right.x * speed_dt
        total_delta_z -= agent.right.z * speed_dt
    if held_keys['d']: # 우측 이동
        total_delta_x += agent.right.x * speed_dt
        total_delta_z += agent.right.z * speed_dt

    # X축 이동 적용 및 충돌 검사
    agent.x += total_delta_x
    hit_info_x = agent.intersects(traverse_target=scene) # 이동 후 충돌 검사
    # 충돌했고, 충돌 대상이 장애물 태그를 가지고 있다면
    collided_obs_x = hit_info_x.hit and hasattr(hit_info_x.entity, 'tag') and hit_info_x.entity.tag == OBSTACLE_TAG
    if collided_obs_x:
        agent.x = original_position.x # X축 위치 원상 복구

    # Z축 이동 적용 및 충돌 검사
    agent.z += total_delta_z
    hit_info_z = agent.intersects(traverse_target=scene) # 이동 후 충돌 검사
    collided_obs_z = hit_info_z.hit and hasattr(hit_info_z.entity, 'tag') and hit_info_z.entity.tag == OBSTACLE_TAG
    if collided_obs_z:
        agent.z = original_position.z # Z축 위치 원상 복구

    # 에이전트 높이 고정
    agent.y = AGENT_HEIGHT

    # Q, E 키 입력에 따른 Y축 회전 (Yaw)
    if held_keys['q']: agent.rotation_y -= ROTATION_SPEED * time.dt
    if held_keys['e']: agent.rotation_y += ROTATION_SPEED * time.dt

    # --- 주기적인 3D 라이다 스캔 및 데이터 기록 ---
    scan_timer += time.dt # 타이머 증가
    # 스캔 간격(SCAN_INTERVAL)이 지났으면 스캔 실행
    if scan_timer >= SCAN_INTERVAL:
        scan_timer -= SCAN_INTERVAL # 타이머 리셋 (다음 간격까지 시간 차감)
        # 현재 에이전트의 자세(XZ 위치, Y 회전 라디안) 기록
        current_pos_xz = agent.world_position.xz
        current_rot_rad = math.radians(agent.world_rotation_y)
        current_pose = (current_pos_xz.x, current_pos_xz.y, current_rot_rad)
        # 라이다 스캔 함수 호출하여 상대 좌표 포인트 얻기
        relative_points_3d = generate_relative_lidar_map_3d()

        # 감지된 포인트가 있으면 기록에 추가
        if relative_points_3d.shape[0] > 0:
             # 자세 정보와 상대 포인트 리스트를 딕셔너리로 묶어 scan_history에 추가
             scan_record = {'pose': current_pose, 'relative_points': relative_points_3d.tolist()}
             scan_history.append(scan_record)

    # --- 3D 라이다 시각화 업데이트 ---
    update_lidar_visualization() # 매 프레임 라이다 가시선 업데이트

    # --- UI 텍스트 업데이트 ---
    pos_str = f"Pos: ({agent.x:.1f}, {agent.z:.1f})" # 현재 위치 문자열
    rot_str = f"Rot (Y): {agent.rotation_y:.0f}°"    # 현재 회전 각도 문자열
    map_info = f"Scans: {len(scan_history)} | 'M': Show 3D Maps | 'C': Clear | 'P': Save" # 스캔 및 제어 정보
    info_text.text = f"{pos_str}\n{rot_str}\n{map_info}" # 최종 UI 텍스트 설정
    # UI 텍스트 위치 동적 조절 (창 크기 변경 대응)
    info_text.x = -0.5 * window.aspect_ratio + 0.02
    info_text.y = 0.48 # Y 위치는 고정

    # --- 마우스 상태 강제 설정 (마우스 잠금 방지) ---
    # 아래 두 줄은 마우스 커서를 항상 보이게 하고 잠금을 해제하여,
    # 사용자가 마우스로 카메라 시점을 조종할 수 없도록 합니다.
    # 마우스 제어를 원하면 이 부분을 주석 처리하거나 수정해야 합니다.
    if mouse.locked: mouse.locked = False # 잠겨있으면 해제
    if not mouse.visible: mouse.visible = True # 보이지 않으면 보이게

# --- 시뮬레이션 시작 ---
if __name__ == '__main__':
    # Ursina 애플리케이션 실행
    # 이 함수는 내부적으로 무한 루프를 돌며 매 프레임마다 update() 함수를 호출합니다.
    app.run()