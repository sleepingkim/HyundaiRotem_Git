# -*- coding: utf-8 -*-
import pygame
import numpy as np
import cv2
import math
import time
import os # 파일 경로 처리용

# --- 설정 ---
# 창 크기
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
# 맵 이미지 파일 경로 (스크립트와 같은 위치에 있다고 가정)
MAP_FILENAME = "map.png"
# VPR 관련 설정
VPR_INTERVAL = 1.0  # VPR 실행 간격 (초)
VPR_VIEW_WIDTH = 240 # VPR에 사용할 카메라 뷰 이미지 너비
VPR_VIEW_HEIGHT = 180 # VPR에 사용할 카메라 뷰 이미지 높이
VPR_FOV_APPROX = 70   # 카메라 화각 근사치 (숫자가 작을수록 줌인 효과) -> 자르는 영역 크기 결정
# 탱크 설정
TANK_MOVE_SPEED = 80.0 # 초당 픽셀 이동 속도
TANK_TURN_SPEED = 120.0 # 초당 각도 회전 속도
TURRET_TURN_SPEED = 150.0
BARREL_PITCH_SPEED = 90.0
TANK_COLOR_ACTUAL = (0, 0, 255) # 실제 탱크 위치 색상 (파랑)
TANK_COLOR_ESTIMATED = (0, 255, 0) # 추정 탱크 위치 색상 (녹색)
# UI 설정
INFO_PANEL_WIDTH = 300 # 우측 정보 패널 너비
UI_FONT_SIZE = 18
UI_LINE_HEIGHT = 22
# 색상
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREY = (150, 150, 150)
COLOR_RED = (255, 50, 50)
COLOR_LIDAR = (0, 200, 200, 150) # LiDAR 색상 (알파 포함)

# --- Pygame 초기화 ---
pygame.init()
pygame.font.init() # 폰트 모듈 초기화
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D VPR Simulation (Python + Pygame + OpenCV)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, UI_FONT_SIZE) # 기본 시스템 폰트 사용

# --- 맵 로딩 ---
map_image_original = None
script_dir = os.path.dirname(__file__) # 현재 스크립트 경로
map_path = os.path.join(script_dir, MAP_FILENAME)
try:
    map_image_original = pygame.image.load(map_path).convert() # Pygame 형식으로 로드
    print(f"Map loaded successfully: {map_path}")
except pygame.error as e:
    print(f"Error loading map image '{MAP_FILENAME}': {e}")
    print("Please ensure 'map.png' exists in the same directory as the script.")
    pygame.quit()
    exit()

MAP_WIDTH, MAP_HEIGHT = map_image_original.get_size()
# 게임 월드 영역 설정 (맵 이미지 크기 사용)
world_rect = pygame.Rect(0, 0, MAP_WIDTH, MAP_HEIGHT)
# 화면에 표시될 맵 영역 (정보 패널 제외)
map_display_rect = pygame.Rect(0, 0, WINDOW_WIDTH - INFO_PANEL_WIDTH, WINDOW_HEIGHT)

# --- VPR 처리 클래스 ---
class VPRProcessor:
    """OpenCV를 사용하여 VPR 수행"""
    def __init__(self):
        self.feature_extractor = None
        self.matcher = None
        try:
            self.feature_extractor = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE) # 특징점 수 줄이고 속도 개선 시도
            # BFMatcher는 crossCheck=True를 사용하면 양방향 매칭으로 안정성 증가
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            print("ORB (VPR) Initialized Successfully.")
        except Exception as e:
            print(f"Error initializing OpenCV ORB/Matcher: {e}. VPR will not function.")

        self.map_kp = None
        self.map_des = None
        self.last_map_img_shape = None
        self.min_match_count = 8 # 매칭 인정 최소 개수 (조절 가능)
        # 전체 맵 특징점 미리 계산
        self.map_image_cv = pygame.surfarray.array3d(map_image_original).swapaxes(0, 1) # Pygame Surface -> NumPy array (RGB)
        self.map_image_cv = cv2.cvtColor(self.map_image_cv, cv2.COLOR_RGB2GRAY) # Gray 로 변환 (성능/안정성 위해)
        self.update_map_features(self.map_image_cv)


    def update_map_features(self, map_img_cv):
        """맵 이미지의 특징점을 계산 (필요시)."""
        if self.feature_extractor is None or map_img_cv is None: return
        # 이미 계산했으면 다시 하지 않음 (맵이 정적이므로)
        if self.map_des is None:
            print("Calculating map features...")
            self.map_kp, self.map_des = self.feature_extractor.detectAndCompute(map_img_cv, None)
            self.last_map_img_shape = map_img_cv.shape
            if self.map_des is not None:
                print(f"Map features calculated: {len(self.map_kp)} keypoints.")
            else:
                self.map_kp = []
                print("Warning: No features found in the map image.")

    def estimate_pose(self, camera_view_img_cv):
        """카메라 뷰 이미지와 맵 특징점을 비교하여 위치 추정."""
        if self.feature_extractor is None or self.matcher is None: return None
        if camera_view_img_cv is None or self.map_des is None: return None

        # 카메라 뷰에서 특징점 찾기
        cam_kp, cam_des = self.feature_extractor.detectAndCompute(camera_view_img_cv, None)

        if cam_des is None or len(cam_kp) < self.min_match_count:
            #print("VPR: Not enough features in camera view.")
            return None

        # 특징점 매칭
        try:
            matches = self.matcher.match(cam_des, self.map_des)
            # 좋은 매칭 선별 (거리가 짧을수록 좋음)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:max(self.min_match_count, int(len(matches)*0.2))] # 상위 20% 또는 최소 개수
        except Exception as e:
            print(f"VPR Error during matching: {e}")
            return None

        # 위치 추정: 매칭된 맵 특징점들의 평균 위치
        if len(good_matches) >= self.min_match_count:
            map_match_points_px = np.float32([self.map_kp[m.trainIdx].pt for m in good_matches])
            center_px = np.mean(map_match_points_px, axis=0) # (x, y) 픽셀 좌표
            # print(f"VPR Estimate: Pixel({center_px[0]:.1f}, {center_px[1]:.1f})")
            return tuple(center_px.astype(int)) # 정수 튜플로 반환
        else:
            #print(f"VPR: Not enough good matches ({len(good_matches)}).")
            return None

# --- Tank 클래스 ---
class Tank:
    """탱크의 상태와 동작을 관리."""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.body_heading = 0.0  # 0도 = 오른쪽 (East), 90도 = 위쪽 (North)
        self.turret_heading_rel = 0.0 # 본체 기준 터렛 각도
        self.barrel_pitch = 0.0    # 포신 상하 각도 (2D 시뮬레이션에서는 시각적 효과 외 의미 적음)
        self.speed = 0.0         # 현재 속도 (pixels/sec)
        self.size = 15           # 화면 표시 크기
        self.estimated_pos = None # VPR로 추정된 위치 (x, y) 튜플

    def update(self, dt, forward, turn, turret_turn, barrel_pitch_change):
        """입력에 따라 탱크 상태 업데이트."""
        # 회전
        self.body_heading = (self.body_heading + turn * TANK_TURN_SPEED * dt) % 360
        self.turret_heading_rel = (self.turret_heading_rel + turret_turn * TURRET_TURN_SPEED * dt) % 360
        self.barrel_pitch = np.clip(self.barrel_pitch + barrel_pitch_change * BARREL_PITCH_SPEED * dt, -10, 20)

        # 이동
        rad = math.radians(self.body_heading)
        move_dist = forward * TANK_MOVE_SPEED * dt
        self.x += move_dist * math.cos(rad)
        self.y -= move_dist * math.sin(rad) # Pygame 좌표계 (Y 아래로 증가)

        # 맵 경계 처리
        self.x = np.clip(self.x, 0, MAP_WIDTH - 1)
        self.y = np.clip(self.y, 0, MAP_HEIGHT - 1)

        self.speed = abs(forward * TANK_MOVE_SPEED)

    def get_total_heading(self):
        """본체 각도와 터렛 각도를 합산한 절대 각도 반환."""
        return (self.body_heading + self.turret_heading_rel) % 360

    def get_camera_view(self, map_surf, view_width, view_height, fov_degrees):
        """
        현재 위치/방향에서 맵 이미지 일부를 잘라내고 회전시켜 카메라 뷰 생성.
        """
        # 1. FOV와 뷰 크기를 기반으로 원본 맵에서 잘라낼 영역 크기 계산 (근사)
        # 화각이 좁을수록 더 넓은 영역을 잘라내서 확대 효과
        crop_scale_factor = math.tan(math.radians(90 - fov_degrees / 2)) if fov_degrees < 180 else 1.0
        # 회전 시 잘림 방지를 위해 더 크게 자름 (피타고라스 정리)
        diagonal = math.sqrt(view_width**2 + view_height**2)
        crop_size = int(max(view_width, view_height) * crop_scale_factor * 1.6) # 여유분 1.6배

        # 2. 탱크 위치 중심의 사각형 영역 계산 (맵 좌표 기준)
        cx, cy = int(self.x), int(self.y)
        half_crop = crop_size // 2
        crop_rect = pygame.Rect(cx - half_crop, cy - half_crop, crop_size, crop_size)

        # 3. 맵 경계에 맞게 영역 클리핑
        clipped_rect = crop_rect.clip(world_rect)
        if clipped_rect.width <= 0 or clipped_rect.height <= 0:
            return None # 맵 밖에 있거나 영역이 없음

        # 4. 해당 영역 잘라내기
        try:
            sub_surf = map_surf.subsurface(clipped_rect)
        except ValueError as e:
             # subsurface 생성 실패 (좌표 문제 등) - 검은 이미지 반환 시도
             print(f"Subsurface error: {e}, rect: {clipped_rect}")
             # return pygame.Surface((view_width, view_height)).convert() # 검은색 반환?
             return None

        # 잘라낸 이미지의 중심점 계산 (sub_surf 기준)
        local_center_x = cx - clipped_rect.left
        local_center_y = cy - clipped_rect.top

        # 5. Pygame Surface -> OpenCV Mat 변환 (회전을 위해)
        view = pygame.surfarray.array3d(sub_surf).swapaxes(0, 1) # RGB
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR) # BGR for OpenCV

        # 6. 카메라 방향(절대 각도)의 *반대* 방향으로 회전
        # Pygame 좌표계 (-Y가 위) 고려, 0도가 오른쪽
        # 예: 탱크가 0도(오른쪽)를 보면 이미지는 왼쪽이 위로 오도록 회전 (-90도)
        # 예: 탱크가 90도(위쪽)를 보면 이미지는 아래쪽이 위로 오도록 회전 (-180도)
        total_heading = self.get_total_heading()
        # OpenCV 회전각은 반시계가 + 이므로, 시계방향 회전 필요 -> -(total_heading)
        # Pygame Y축과 각도 방향 보정: (오른쪽 0도 기준 위로 90도 -> -90도 회전 필요)
        # => 실제 회전각 = -(total_heading)
        rotate_angle = -total_heading
        rot_mat = cv2.getRotationMatrix2D((local_center_x, local_center_y), rotate_angle, 1.0)

        # 회전 시 이미지 경계 확장 및 중앙 정렬
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((view.shape[1] * cos) + (view.shape[0] * sin))
        new_h = int((view.shape[1] * sin) + (view.shape[0] * cos))
        rot_mat[0, 2] += (new_w / 2) - local_center_x
        rot_mat[1, 2] += (new_h / 2) - local_center_y

        # 이미지 회전 (경계 밖은 검은색으로 채움)
        rotated_view = cv2.warpAffine(view, rot_mat, (new_w, new_h), borderValue=(0,0,0))

        # 7. 회전된 이미지의 중심에서 최종 뷰 크기만큼 잘라내기
        final_center_x, final_center_y = new_w // 2, new_h // 2
        half_view_w, half_view_h = view_width // 2, view_height // 2

        y1 = max(0, final_center_y - half_view_h)
        y2 = min(new_h, final_center_y + half_view_h + (view_height % 2)) # 홀수 높이 처리
        x1 = max(0, final_center_x - half_view_w)
        x2 = min(new_w, final_center_x + half_view_w + (view_width % 2)) # 홀수 너비 처리

        final_view_cv = rotated_view[y1:y2, x1:x2]

        # 최종 크기가 정확히 맞지 않으면 리사이즈 (검은색 패딩이 있을 수 있음)
        if final_view_cv.shape[1] != view_width or final_view_cv.shape[0] != view_height:
            # 크기가 0인 경우 방지
            if final_view_cv.shape[1] == 0 or final_view_cv.shape[0] == 0:
                return np.zeros((view_height, view_width, 3), dtype=np.uint8) # 검은 이미지 반환
            final_view_cv = cv2.resize(final_view_cv, (view_width, view_height), interpolation=cv2.INTER_AREA)

        return final_view_cv

    def draw(self, surface, camera_offset_x, camera_offset_y):
        """탱크를 화면에 그리기 (실제 위치)"""
        screen_x = int(self.x - camera_offset_x)
        screen_y = int(self.y - camera_offset_y)
        # 화면 밖이면 그리지 않음
        if screen_x < -self.size or screen_x > surface.get_width() + self.size or \
           screen_y < -self.size or screen_y > surface.get_height() + self.size:
            return

        # 탱크 몸체 (간단한 원으로 표시)
        pygame.draw.circle(surface, TANK_COLOR_ACTUAL, (screen_x, screen_y), self.size)

        # 탱크 방향선
        rad = math.radians(self.body_heading)
        end_x = screen_x + self.size * 1.2 * math.cos(rad)
        end_y = screen_y - self.size * 1.2 * math.sin(rad)
        pygame.draw.line(surface, COLOR_WHITE, (screen_x, screen_y), (int(end_x), int(end_y)), 2)

        # 터렛 방향선
        turret_rad = math.radians(self.get_total_heading())
        turret_end_x = screen_x + self.size * 1.0 * math.cos(turret_rad)
        turret_end_y = screen_y - self.size * 1.0 * math.sin(turret_rad)
        pygame.draw.line(surface, COLOR_GREY, (screen_x, screen_y), (int(turret_end_x), int(turret_end_y)), 3)

    def draw_estimated(self, surface, camera_offset_x, camera_offset_y):
        """VPR 추정 위치를 화면에 그리기"""
        if self.estimated_pos:
            screen_x = int(self.estimated_pos[0] - camera_offset_x)
            screen_y = int(self.estimated_pos[1] - camera_offset_y)
            # 화면 밖이면 그리지 않음
            if screen_x < -self.size or screen_x > surface.get_width() + self.size or \
               screen_y < -self.size or screen_y > surface.get_height() + self.size:
                return
            # 반투명 원으로 표시
            s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*TANK_COLOR_ESTIMATED, 128), (self.size, self.size), self.size)
            surface.blit(s, (screen_x - self.size, screen_y - self.size))
            # 실제 위치와 선으로 연결
            actual_screen_x = int(self.x - camera_offset_x)
            actual_screen_y = int(self.y - camera_offset_y)
            pygame.draw.line(surface, COLOR_RED, (actual_screen_x, actual_screen_y), (screen_x, screen_y), 1)


# --- LiDAR 시뮬레이션 함수 (Placeholder) ---
def simulate_lidar(tank_x, tank_y, num_rays=60, max_dist=200):
    """간단한 LiDAR 데이터 시뮬레이션 (고정된 원형 패턴)"""
    points = []
    for i in range(num_rays):
        angle = math.radians(i * (360 / num_rays))
        dist = max_dist * (0.8 + 0.2 * math.sin(angle * 5 + time.time() * 2)) # 약간의 변동 추가
        hit_x = tank_x + dist * math.cos(angle)
        hit_y = tank_y - dist * math.sin(angle) # Pygame Y축 고려
        points.append((hit_x, hit_y))
    return points

def draw_lidar(surface, points, tank_sx, tank_sy, cam_ox, cam_oy):
    """LiDAR 데이터를 화면에 그리기"""
    if not points: return
    # 반투명 효과를 위한 별도 Surface 생성 시도 (선택적)
    # lidar_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for px, py in points:
        screen_x = int(px - cam_ox)
        screen_y = int(py - cam_oy)
        # 화면 내에 있을 때만 그리기
        if 0 <= screen_x < surface.get_width() and 0 <= screen_y < surface.get_height():
             # pygame.draw.line(lidar_surf, COLOR_LIDAR, (tank_sx, tank_sy), (screen_x, screen_y), 1)
             pygame.draw.circle(surface, COLOR_LIDAR[:3], (screen_x, screen_y), 2) # 원으로 표시 (알파 미적용)
    # surface.blit(lidar_surf, (0,0))


# --- 카메라 스크롤 계산 ---
def get_camera_offset(target_x, target_y, map_w, map_h, screen_w, screen_h):
    """탱크가 화면 중앙에 오도록 카메라 오프셋 계산"""
    # 화면 중앙 좌표
    center_x = screen_w / 2
    center_y = screen_h / 2

    # 이상적인 카메라 좌상단 좌표
    ideal_cam_x = target_x - center_x
    ideal_cam_y = target_y - center_y

    # 카메라가 맵 밖으로 나가지 않도록 제한
    cam_x = max(0, min(ideal_cam_x, map_w - screen_w))
    cam_y = max(0, min(ideal_cam_y, map_h - screen_h))

    # 맵이 화면보다 작을 경우 중앙 정렬
    if map_w < screen_w:
        cam_x = (map_w - screen_w) / 2
    if map_h < screen_h:
        cam_y = (map_h - screen_h) / 2

    return int(cam_x), int(cam_y)


# --- 메인 루프 ---
running = True
tank = Tank(MAP_WIDTH / 2, MAP_HEIGHT / 2) # 맵 중앙에서 시작
vpr_processor = VPRProcessor() # VPR 객체 생성
last_vpr_time = time.time()
last_vpr_result_text = "Idle"
camera_offset_x, camera_offset_y = 0, 0 # 화면 스크롤 오프셋
show_debug_view = False # VPR 입력 이미지 표시 여부
vpr_input_view_cv = None # VPR 입력 이미지 저장용

while running:
    dt = clock.tick(60) / 1000.0 # 초 단위 시간 변화량, 최대 60 FPS

    # --- 이벤트 처리 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
             if event.key == pygame.K_v: # 'V' 키로 디버그 뷰 토글
                  show_debug_view = not show_debug_view

    # --- 입력 처리 ---
    keys = pygame.key.get_pressed()
    forward_input = 0.0
    turn_input = 0.0
    turret_turn_input = 0.0
    barrel_pitch_input = 0.0
    if keys[pygame.K_w] or keys[pygame.K_UP]: forward_input = 1.0
    if keys[pygame.K_s] or keys[pygame.K_DOWN]: forward_input = -1.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]: turn_input = 1.0 # 왼쪽 회전 (+)
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: turn_input = -1.0 # 오른쪽 회전 (-)
    if keys[pygame.K_q]: turret_turn_input = 1.0 # 터렛 왼쪽 (+)
    if keys[pygame.K_e]: turret_turn_input = -1.0 # 터렛 오른쪽 (-)
    if keys[pygame.K_r]: barrel_pitch_input = 1.0 # 포신 위 (+)
    if keys[pygame.K_f]: barrel_pitch_input = -1.0 # 포신 아래 (-)

    # --- 게임 로직 업데이트 ---
    tank.update(dt, forward_input, turn_input, turret_turn_input, barrel_pitch_input)

    # VPR 실행 (주기적)
    current_time = time.time()
    if current_time - last_vpr_time >= VPR_INTERVAL:
        last_vpr_time = current_time
        # 1. 카메라 뷰 생성
        vpr_input_view_cv = tank.get_camera_view(map_image_original, VPR_VIEW_WIDTH, VPR_VIEW_HEIGHT, VPR_FOV_APPROX)
        # 2. VPR 처리
        if vpr_input_view_cv is not None:
             # VPR 처리를 위해 그레이스케일 변환
             vpr_input_gray = cv2.cvtColor(vpr_input_view_cv, cv2.COLOR_BGR2GRAY)
             estimated_map_pos = vpr_processor.estimate_pose(vpr_input_gray)
             tank.estimated_pos = estimated_map_pos # 추정 위치 업데이트
             if estimated_map_pos:
                  last_vpr_result_text = f"Success: Est({estimated_map_pos[0]}, {estimated_map_pos[1]})"
             else:
                  last_vpr_result_text = "Failed / No Match"
        else:
             last_vpr_result_text = "Capture Failed"
             tank.estimated_pos = None

    # LiDAR 데이터 시뮬레이션
    lidar_points = simulate_lidar(tank.x, tank.y)

    # 카메라 스크롤 계산
    camera_offset_x, camera_offset_y = get_camera_offset(
        tank.x, tank.y, MAP_WIDTH, MAP_HEIGHT, map_display_rect.width, map_display_rect.height
    )

    # --- 그리기 ---
    screen.fill(COLOR_BLACK) # 전체 화면 검은색으로 지우기

    # 맵 그리기 (카메라 오프셋 적용)
    screen.blit(map_image_original, map_display_rect.topleft,
                (camera_offset_x, camera_offset_y, map_display_rect.width, map_display_rect.height))

    # 탱크 그리기 (실제 위치, 추정 위치) - 화면 좌표 기준으로 그림
    tank.draw(screen, camera_offset_x, camera_offset_y)
    tank.draw_estimated(screen, camera_offset_x, camera_offset_y)

    # LiDAR 그리기
    tank_screen_x = int(tank.x - camera_offset_x)
    tank_screen_y = int(tank.y - camera_offset_y)
    draw_lidar(screen, lidar_points, tank_screen_x, tank_screen_y, camera_offset_x, camera_offset_y)

    # --- 정보 패널 그리기 ---
    panel_x = WINDOW_WIDTH - INFO_PANEL_WIDTH
    pygame.draw.rect(screen, COLOR_GREY, (panel_x, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT), 1) # 패널 테두리

    # 텍스트 렌더링 및 표시
    info_texts = [
        f"FPS: {clock.get_fps():.1f}",
        "== Tank Info ==",
        f"Speed: {tank.speed:.1f} px/s", # 가상 단위 km/h 환산은 복잡하므로 px/s 표시
        f"Pos (X,Y): ({tank.x:.1f}, {tank.y:.1f})",
        f"Alt: N/A (2D)", # 고도 정보는 2D에서 의미 없음
        f"Body Hdg: {tank.body_heading:.1f}°",
        f"Turret Rel: {tank.turret_heading_rel:.1f}°",
        f"Turret Abs: {tank.get_total_heading():.1f}°",
        f"Barrel Pitch: {tank.barrel_pitch:.1f}°",
        "== VPR Info ==",
        f"Status: {last_vpr_result_text}",
        f"Est Pos: {tank.estimated_pos if tank.estimated_pos else 'N/A'}",
        "== LiDAR Info ==",
        f"Rays: {len(lidar_points)}",
        "== Controls ==",
        "WASD/Arrows: Move/Turn",
        "QE: Turret Rotate",
        "RF: Barrel Pitch",
        "V: Toggle VPR View"
        # Distance는 다른 탱크가 없으므로 N/A
    ]

    y_offset = 10
    for text in info_texts:
        text_surface = font.render(text, True, COLOR_WHITE)
        screen.blit(text_surface, (panel_x + 10, y_offset))
        y_offset += UI_LINE_HEIGHT

    # VPR 입력 뷰 표시 (토글)
    if show_debug_view and vpr_input_view_cv is not None:
        try:
            # OpenCV(BGR) -> Pygame Surface (RGB)
            debug_surf = pygame.surfarray.make_surface(cv2.cvtColor(vpr_input_view_cv, cv2.COLOR_BGR2RGB).swapaxes(0,1))
            debug_rect = debug_surf.get_rect(bottomleft=(panel_x + 10, WINDOW_HEIGHT - 10))
            screen.blit(debug_surf, debug_rect)
            pygame.draw.rect(screen, COLOR_WHITE, debug_rect, 1) # 테두리
        except Exception as e:
            print(f"Error displaying debug view: {e}")


    # --- 화면 업데이트 ---
    pygame.display.flip()


# --- 종료 ---
pygame.quit()
print("Simulation ended.")