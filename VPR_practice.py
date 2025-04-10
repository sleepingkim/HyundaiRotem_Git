# -*- coding: utf-8 -*-
from ursina import *
from ursina import camera # Camera 클래스 명시적 임포트
from ursina.prefabs.first_person_controller import FirstPersonController
import cv2
import numpy as np
import math
import time
import os
import random

# --- 설정 ---
# VPR 관련
VPR_INTERVAL = 1.0
VPR_VIEW_WIDTH = 240
VPR_VIEW_HEIGHT = 180
MAP_VIEW_WIDTH = 256
MAP_VIEW_HEIGHT = 256
VPR_FOV_APPROX = 70    # ***** 정의 추가 ***** 탱크 카메라 화각 (Degrees)
# GPS 시뮬레이션
GPS_ORIGIN_LAT = 37.5665
GPS_ORIGIN_LON = 126.9780
URSINA_UNIT_TO_METERS = 1.0
METERS_PER_DEG_LAT = 111132.954
METERS_PER_DEG_LON = METERS_PER_DEG_LAT * math.cos(math.radians(GPS_ORIGIN_LAT))
# LiDAR 시뮬레이션
LIDAR_NUM_RAYS = 90
LIDAR_MAX_DIST = 50.0
LIDAR_UPDATE_INTERVAL = 0.3
# 탱크 설정
TANK_MOVE_SPEED = 5.0
TANK_TURN_SPEED = 100.0
TURRET_TURN_SPEED = 120.0
BARREL_PITCH_SPEED = 60.0
# 기타
OBSTACLE_COUNT = 150

# --- VPR 처리 클래스 (이전과 동일) ---
class VPRProcessor:
    """OpenCV를 사용하여 VPR 수행"""
    def __init__(self):
        self.feature_extractor = None
        self.matcher = None
        try:
            self.feature_extractor = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            print("ORB (VPR) Initialized Successfully.")
        except Exception as e:
            print(f"Error initializing OpenCV ORB/Matcher: {e}. VPR will not function.")

        self.map_kp = None
        self.map_des = None
        self.last_map_hash = None
        self.min_match_count = 10

    def _compute_hash(self, image):
        if image is None: return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
            avg = np.mean(resized)
            hash_val = (resized >= avg).flatten().astype(np.uint8)
            return hash_val.tobytes()
        except Exception as e:
            print(f"Warning: Could not compute image hash: {e}")
            return None

    def update_map_features(self, map_img_cv):
        if self.feature_extractor is None or map_img_cv is None: return
        current_hash = self._compute_hash(map_img_cv)
        if current_hash is None or current_hash != self.last_map_hash or self.map_des is None:
            print("VPR: Updating map features...")
            map_gray = cv2.cvtColor(map_img_cv, cv2.COLOR_BGR2GRAY) if len(map_img_cv.shape) == 3 else map_img_cv
            self.map_kp, self.map_des = self.feature_extractor.detectAndCompute(map_gray, None)
            self.last_map_hash = current_hash
            if self.map_des is not None:
                print(f"VPR: Map features updated ({len(self.map_kp)} keypoints).")
            else:
                self.map_kp = []
                print("VPR: Warning - No features found in the map image.")

    def estimate_pose(self, camera_view_img_cv, map_view_img_cv):
        if self.feature_extractor is None or self.matcher is None: return None
        if camera_view_img_cv is None or map_view_img_cv is None: return None

        self.update_map_features(map_view_img_cv)
        cam_gray = cv2.cvtColor(camera_view_img_cv, cv2.COLOR_BGR2GRAY) if len(camera_view_img_cv.shape) == 3 else camera_view_img_cv
        cam_kp, cam_des = self.feature_extractor.detectAndCompute(cam_gray, None)

        if cam_des is None or len(cam_kp) < self.min_match_count or self.map_des is None or len(self.map_kp) < self.min_match_count:
            return None

        try:
            matches = self.matcher.match(cam_des, self.map_des)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:max(self.min_match_count, int(len(matches) * 0.20))]
        except Exception as e:
            print(f"VPR Error during matching: {e}")
            return None

        if len(good_matches) >= self.min_match_count:
            map_match_points_px = np.float32([self.map_kp[m.trainIdx].pt for m in good_matches])
            center_px = np.mean(map_match_points_px, axis=0)
            return tuple(center_px.astype(int))
        else:
            return None

# --- GPS 변환 함수 ---
def ursina_to_gps(x, z):
    delta_east_meters = x * URSINA_UNIT_TO_METERS
    delta_north_meters = z * URSINA_UNIT_TO_METERS
    delta_lon = delta_east_meters / METERS_PER_DEG_LON
    delta_lat = delta_north_meters / METERS_PER_DEG_LAT
    current_lon = GPS_ORIGIN_LON + delta_lon
    current_lat = GPS_ORIGIN_LAT + delta_lat
    return current_lat, current_lon

# --- 탱크 클래스 ---
class Tank(Entity):
    def __init__(self, position=(0, 0.4, 0), **kwargs):
        super().__init__(model=None, collider=None, **kwargs)
        self.current_speed = 0.0
        self.turret_rotation_speed = TURRET_TURN_SPEED
        self.barrel_pitch_speed = BARREL_PITCH_SPEED
        self.barrel_pitch = 0.0
        self.min_barrel_pitch = -10.0
        self.max_barrel_pitch = 20.0

        self.body = Entity(model='cube', color=color.dark_gray, scale=(2, 0.8, 3.5), parent=self, collider='box')
        self.turret = Entity(model='cube', color=color.gray, scale=(1.2/2, 0.6/0.8, 1.2/3.5), parent=self.body, y=0.5 + (0.6/0.8)*0.5, z=0.1, origin_y=-0.5)
        self.barrel = Entity(model='cube', color=color.black, scale=(0.2/1.2, 0.2/0.6, 2.0/1.2), parent=self.turret, y=0.5, z=0.5 + (2.0/1.2)*0.5, origin_z=-0.5)

        self.estimated_marker = None
        self.estimated_world_pos = None

    def update(self):
        friction = 0.5
        self.current_speed *= (1 - friction * time.dt)
        if abs(self.current_speed) < 0.1:
             self.current_speed = 0

        original_position = self.position
        self.position += self.forward * self.current_speed * time.dt

        if self.y < 0.4:
            self.y = 0.4
            self.current_speed = 0

        hit_info = self.intersects(ignore=(self,), traverse_target=scene)
        if hit_info.hit:
            self.position = original_position
            self.current_speed = 0

        self.body_angles = (self.rotation_x, self.rotation_y, self.rotation_z)
        self.turret_angles = (self.turret.rotation_y, self.barrel.rotation_x)

        if self.estimated_marker:
             if self.estimated_world_pos:
                  self.estimated_marker.world_position = self.estimated_world_pos
                  self.estimated_marker.visible = True
             else:
                  self.estimated_marker.visible = False

    def input(self, key):
        if key == 'w' or key == 'up arrow':
            self.current_speed = TANK_MOVE_SPEED
        elif key == 's' or key == 'down arrow':
            self.current_speed = -TANK_MOVE_SPEED * 0.7

        turn_amount = TANK_TURN_SPEED * time.dt
        if key == 'a' or key == 'left arrow':
            self.rotation_y -= turn_amount
        elif key == 'd' or key == 'right arrow':
            self.rotation_y += turn_amount

        if key == 'q':
            self.turret.rotation_y -= self.turret_rotation_speed * time.dt
        elif key == 'e':
            self.turret.rotation_y += self.turret_rotation_speed * time.dt

        if key == 'r':
            self.barrel_pitch += self.barrel_pitch_speed * time.dt
        elif key == 'f':
            self.barrel_pitch -= self.barrel_pitch_speed * time.dt

        self.barrel_pitch = clamp(self.barrel_pitch, self.min_barrel_pitch, self.max_barrel_pitch)
        self.barrel.rotation_x = self.barrel_pitch

# --- 메인 애플리케이션 ---
app = Ursina(title="Ursina VPR Simulation", borderless=False, fullscreen=False, vsync=True)

# --- 환경 생성 ---
ground = Entity(model='plane', scale=(200, 1, 200), color=color.rgb(60, 60, 60), texture='white_cube', texture_scale=(100, 100), collider='box')

obstacles = []
for i in range(OBSTACLE_COUNT):
    obs = Entity(model='cube',
                 name=f'obstacle_{i}',
                 color=color.random_color(),
                 collider='box',
                 position=(random.uniform(-90, 90), 0.5 * random.uniform(1, 3), random.uniform(-90, 90)),
                 scale=(random.uniform(1, 5), random.uniform(1, 3), random.uniform(1, 5)),
                 tag='obstacle')
    obstacles.append(obs)

# --- 탱크 생성 ---
tank = Tank(position=(0, 0.4, -10))

# --- VPR 프로세서 생성 ---
vpr_processor = VPRProcessor()
last_vpr_time = time.time()
estimated_pixel_pos = None

# --- 카메라 설정 ---
# EditorCamera()
main_camera = camera
main_camera.position = (0, 30, -30)
main_camera.rotation_x = 30
main_camera.fov = 75

# 탱크 및 미니맵 카메라 생성 시 VPR_FOV_APPROX 사용
tank_camera = camera(parent=tank.barrel, position=(0, 0.2, 1.0), rotation_y=0, fov=VPR_FOV_APPROX, enabled=False)
minimap_camera = camera(orthographic=True, fov=MAP_WORLD_SIZE,
                        rotation_x=90, y=100,
                        enabled=False)

# --- UI 설정 ---
info_texts = {}
ui_y = 0.45
ui_x = -window.aspect_ratio * 0.5 + 0.05 # UI 좌표계 기준 왼쪽으로 이동
info_labels = ["FPS", "Speed", "Pos", "Alt", "GPS_Lat", "GPS_Lon", "Body_Angles", "Turret_Angles", "VPR_Status", "Est_Map_Pos"]
for label in info_labels:
    info_texts[label] = Text(text=f"{label}: N/A", origin=(-0.5, 0.5), scale=0.8, x=ui_x, y=ui_y)
    ui_y -= 0.035

tank_view_ui_size = (0.25, 0.2)
tank_view_ui = Panel(scale=tank_view_ui_size, origin=(0.5, -0.5),
                     position=(window.aspect_ratio * 0.5 - 0.01, -0.5 + 0.01), # 오른쪽 아래
                     texture=Texture(None),
                     color=color.dark_gray)
# Text 위치 조정
Text("Tank View (VPR Input)", origin=(0.5, 0.5), scale=0.7, x=tank_view_ui.x, y=tank_view_ui.y - 0.01)


minimap_view_ui_size = tank_view_ui_size
minimap_view_ui = Panel(scale=minimap_view_ui_size, origin=(0.5, -0.5),
                        position=(tank_view_ui.x, tank_view_ui.y + tank_view_ui_size[1] + 0.02), # 탱크뷰 위
                        texture=Texture(None),
                        color=color.dark_gray)
# Text 위치 조정
Text("Map View (VPR Map)", origin=(0.5, 0.5), scale=0.7, x=minimap_view_ui.x, y=minimap_view_ui.y - 0.01)


lidar_dots = []
for _ in range(LIDAR_NUM_RAYS):
    dot = Entity(model='sphere', scale=0.15, color=color.cyan, enabled=False)
    lidar_dots.append(dot)
last_lidar_time = time.time()

tank.estimated_marker = Entity(model='cube', scale=(tank.body.scale_x, 0.2, tank.body.scale_z),
                               color=color.rgba(0, 255, 0, 150), origin_y=-0.5, visible=False)

# --- VPR 이미지 캡처 관련 변수 ---
temp_tank_view_file = "_temp_tank_view.png"
temp_map_view_file = "_temp_map_view.png"
is_capturing = False

# --- 메인 업데이트 함수 ---
def update():
    global last_vpr_time, estimated_pixel_pos, last_lidar_time, is_capturing

    if is_capturing: return

    # 메인 카메라 컨트롤
    follow_distance = 20.0
    follow_height = 15.0
    target_pos = tank.world_position + tank.back * follow_distance + Vec3(0, follow_height, 0)
    main_camera.world_position = lerp(main_camera.world_position, target_pos, time.dt * 5)
    main_camera.look_at(tank.world_position + Vec3(0, 1, 0))

    # --- 정보 업데이트 ---
    info_texts["FPS"].text = f"FPS: {int(performance_counter.fps)}"
    info_texts["Speed"].text = f"Speed: {tank.current_speed:.1f} u/s"
    info_texts["Pos"].text = f"Pos (X,Z): ({tank.x:.1f}, {tank.z:.1f})"
    info_texts["Alt"].text = f"Alt (Y): {tank.y:.1f} m"
    lat, lon = ursina_to_gps(tank.x, tank.z)
    info_texts["GPS_Lat"].text = f"GPS Lat: {lat:.6f}"
    info_texts["GPS_Lon"].text = f"GPS Lon: {lon:.6f}"
    info_texts["Body_Angles"].text = f"Body Y: {tank.rotation_y:.1f}"
    info_texts["Turret_Angles"].text = f"Turret/Barrel: ({tank.turret.rotation_y:.1f}, {tank.barrel.rotation_x:.1f})"

    if estimated_pixel_pos is not None and not is_capturing:
         map_cam_fov_width = minimap_camera.fov * minimap_camera.aspect_ratio
         map_cam_fov_height = minimap_camera.fov
         map_img_w, map_img_h = MAP_VIEW_WIDTH, MAP_VIEW_HEIGHT

         ratio_x = estimated_pixel_pos[0] / map_img_w
         ratio_z = 1.0 - (estimated_pixel_pos[1] / map_img_h)

         est_world_x = minimap_camera.x + (ratio_x - 0.5) * map_cam_fov_width
         est_world_z = minimap_camera.z + (ratio_z - 0.5) * map_cam_fov_height
         tank.estimated_world_pos = Vec3(est_world_x, tank.y, est_world_z)
         info_texts["Est_Map_Pos"].text = f"Est Map Px: ({estimated_pixel_pos[0]}, {estimated_pixel_pos[1]})"

    elif not is_capturing:
         info_texts["Est_Map_Pos"].text = f"Est Map Px: N/A"
         tank.estimated_world_pos = None
         if "VPR: Error" not in info_texts["VPR_Status"].text and \
            "VPR: Capture" not in info_texts["VPR_Status"].text and \
            "VPR: Processing" not in info_texts["VPR_Status"].text:
              if tank.estimated_world_pos is None and estimated_pixel_pos is None:
                  info_texts["VPR_Status"].text = "VPR: Idle / No Match"


    # --- VPR 실행 (주기적) ---
    current_time = time.time()
    if not is_capturing and current_time - last_vpr_time >= VPR_INTERVAL:
        last_vpr_time = current_time
        is_capturing = True
        info_texts["VPR_Status"].text = "VPR: Capturing..."
        invoke(capture_and_process_vpr, delay=0.05)

    # --- LiDAR 시뮬레이션 (주기적) ---
    if not is_capturing and current_time - last_lidar_time >= LIDAR_UPDATE_INTERVAL:
         last_lidar_time = current_time
         for dot in lidar_dots: dot.enabled = False

         origin = tank.turret.world_position + Vec3(0, 0.3, 0)
         idx = 0
         step = 360 / LIDAR_NUM_RAYS
         for i in range(LIDAR_NUM_RAYS):
              angle_rad = math.radians(tank.turret.world_rotation_y + (i * step))
              direction = Vec3(math.sin(angle_rad), 0, math.cos(angle_rad)).normalized()
              ignore_list = [tank, tank.body, tank.turret, tank.barrel]
              hit_info = raycast(origin, direction, distance=LIDAR_MAX_DIST, ignore=ignore_list, debug=False)

              if hit_info.hit and idx < len(lidar_dots):
                   dot = lidar_dots[idx]
                   dot.world_position = hit_info.world_point
                   dot.enabled = True
                   idx += 1

# --- VPR 캡처 및 처리 함수 ---
def capture_and_process_vpr():
    """스크린샷을 찍고 VPR을 처리하는 함수 (invoke로 호출됨)"""
    global estimated_pixel_pos, is_capturing

    # --- 스크린샷 및 크롭 방식 ---
    # 1. 탱크 뷰 캡처 준비
    original_cam_parent = camera.parent
    original_cam_pos = camera.position
    original_cam_rot = camera.rotation
    original_fov = camera.fov
    original_ortho = camera.orthographic

    camera.parent = tank_camera
    camera.position = (0,0,0)
    camera.rotation = (0,0,0)
    camera.fov = tank_camera.fov
    camera.orthographic = False # 탱크 카메라는 Perspective

    # 2. 스크린샷 저장 (탱크 뷰) - 약간의 지연 후 캡처 시도
    def cap_tank_view():
        try:
            screenshot(name_prefix=temp_tank_view_file.replace('.png',''), default_filename=0, overwrite=True)
            print("VPR: Captured tank view (screenshot).")
            invoke(cap_map_view, delay=0.05)
        except Exception as e:
            print(f"Error capturing tank view: {e}")
            info_texts["VPR_Status"].text = "VPR: Tank Cap Error"
            reset_camera_and_flag(original_cam_parent, original_cam_pos, original_cam_rot, original_fov, original_ortho)

    # 3. 미니맵 뷰 캡처 준비 및 실행 (cap_tank_view에서 호출됨)
    def cap_map_view():
        global is_capturing
        camera.parent = None
        camera.world_position = minimap_camera.world_position
        camera.world_rotation = minimap_camera.world_rotation
        camera.orthographic = True
        camera.fov = minimap_camera.fov

        try:
            screenshot(name_prefix=temp_map_view_file.replace('.png',''), default_filename=0, overwrite=True)
            print("VPR: Captured map view (screenshot).")
            invoke(process_screenshots, delay=0.05)
        except Exception as e:
            print(f"Error capturing map view: {e}")
            info_texts["VPR_Status"].text = "VPR: Map Cap Error"
            reset_camera_and_flag(original_cam_parent, original_cam_pos, original_cam_rot, original_fov, original_ortho)


    # 4. 스크린샷 처리 함수 (cap_map_view에서 호출됨)
    def process_screenshots():
        global estimated_pixel_pos, is_capturing
        try:
            if os.path.exists(temp_tank_view_file) and os.path.exists(temp_map_view_file):
                tank_view_img = cv2.imread(temp_tank_view_file)
                map_view_img = cv2.imread(temp_map_view_file)

                if tank_view_img is not None and map_view_img is not None:
                    tank_view_resized = cv2.resize(tank_view_img, (VPR_VIEW_WIDTH, VPR_VIEW_HEIGHT))
                    map_view_resized = cv2.resize(map_view_img, (MAP_VIEW_WIDTH, MAP_VIEW_HEIGHT))

                    info_texts["VPR_Status"].text = "VPR: Processing..."
                    estimated_pixel_pos = vpr_processor.estimate_pose(tank_view_resized, map_view_resized)

                    # UI 업데이트
                    try:
                        if isinstance(tank_view_ui.texture, Texture): destroy(tank_view_ui.texture)
                        if isinstance(minimap_view_ui.texture, Texture): destroy(minimap_view_ui.texture)

                        tank_view_ui.texture = load_texture(temp_tank_view_file.replace('.png', ''), path=os.getcwd())
                        minimap_view_ui.texture = load_texture(temp_map_view_file.replace('.png', ''), path=os.getcwd())

                        if tank_view_ui.texture is None or minimap_view_ui.texture is None:
                             raise ValueError("Failed to load texture from file.")

                    except Exception as tex_e:
                        print(f"Error loading texture for UI from file: {tex_e}. Trying fallback.")
                        try:
                            tex_tank = Texture(cv2.cvtColor(tank_view_resized, cv2.COLOR_BGR2RGBA).tobytes(), size=(VPR_VIEW_WIDTH, VPR_VIEW_HEIGHT), mode='rgba')
                            tex_map = Texture(cv2.cvtColor(map_view_resized, cv2.COLOR_BGR2RGBA).tobytes(), size=(MAP_VIEW_WIDTH, MAP_VIEW_HEIGHT), mode='rgba')
                            tank_view_ui.texture = tex_tank
                            minimap_view_ui.texture = tex_map
                        except Exception as cvt_e:
                            print(f"Error converting CV image to Texture: {cvt_e}")
                            tank_view_ui.texture = None; tank_view_ui.color = color.gray
                            minimap_view_ui.texture = None; minimap_view_ui.color = color.gray

                    if estimated_pixel_pos is None:
                        info_texts["VPR_Status"].text = "VPR: Failed / No Match"
                        tank.estimated_world_pos = None
                    else:
                        # 실제 상태 업데이트는 메인 update 루프에서 처리
                        pass

                else:
                    info_texts["VPR_Status"].text = "VPR: Read Capture Fail"
            else:
                info_texts["VPR_Status"].text = "VPR: Capture File Missing"

        except Exception as e:
            print(f"VPR Processing Error: {e}")
            import traceback
            traceback.print_exc()
            info_texts["VPR_Status"].text = f"VPR: Error"
        finally:
             reset_camera_and_flag(original_cam_parent, original_cam_pos, original_cam_rot, original_fov, original_ortho)

    # 5. 카메라 복귀 및 플래그 해제 함수
    def reset_camera_and_flag(parent, pos, rot, fov, ortho):
        global is_capturing
        # 메인 카메라 원래 상태로 복귀
        # 주의: parent가 None일 경우 처리 필요
        if parent is not None:
             camera.parent = parent
        else: # 원래 부모가 없었다면 None으로 설정
             camera.parent = None
        camera.position = pos
        camera.rotation = rot
        camera.fov = fov
        camera.orthographic = ortho
        is_capturing = False # 캡처/처리 완료 플래그 해제
        print("VPR cycle finished.")

    # --- VPR 사이클 시작 ---
    invoke(cap_tank_view, delay=0.1) # 첫 캡처 시작 예약 (조금 더 긴 딜레이)

# --- 전역 입력 처리 ---
def input(key):
    if key == 'escape':
        quit()

# --- 애플리케이션 실행 ---
if __name__ == '__main__':
    app.run()

    # --- 종료 시 임시 파일 정리 ---
    print("Cleaning up temporary files...")
    if os.path.exists(temp_tank_view_file):
        try: os.remove(temp_tank_view_file)
        except Exception as e: print(f"Could not remove temp file: {e}")
    if os.path.exists(temp_map_view_file):
        try: os.remove(temp_map_view_file)
        except Exception as e: print(f"Could not remove temp file: {e}")
    print("Cleanup complete.")