# 필요한 Ursina 구성 요소들을 명시적으로 import 합니다. (time 포함)
from ursina import (
    Ursina, Entity, scene, camera, held_keys, color, Vec3, distance, time, # 'time' 다시 포함
    Sky, DirectionalLight, Text, Panel, Circle, raycast, # boxcast 제거
    clamp, distance_xz, destroy, load_texture, Sequence, Func, Wait, # Sequence 등 추가
    random # random은 별도 import
)
# import random # 위에서 처리
import math

# --- 전역 변수 및 설정 ---
UPDATE_FREQUENCY = 10
VIRTUAL_KM_SCALE = 1000
VIRTUAL_METER_SCALE = 10
GPS_ORIGIN_LAT = 37.5665
GPS_ORIGIN_LON = 126.9780
GPS_SCALE = 0.0001

# --- Tank 클래스 정의 ---
class Tank(Entity):
    def __init__(self, position=(0, 1, 0), rotation=(0, 0, 0), **kwargs):
        super().__init__(position=position, rotation=rotation, **kwargs)

        # 1. 탱크 구성 요소 생성 (원래 색상으로)
        self.body = Entity(parent=self, model='cube', scale=(2, 0.5, 3), color=color.dark_gray, collider='box', name="TankBody")
        self.turret = Entity(parent=self.body, model='cube', scale=(1.2, 0.4, 1.2), y=0.45, z=0.2, color=color.gray, name="TankTurret")
        self.gun = Entity(parent=self.turret, model='cube', scale=(0.2, 0.2, 1.5), y=0.1, z=0.6, color=color.light_gray, name="TankGun")

        # 2. 탱크 상태 변수 (속도 계산 변수 복원)
        self.speed = 0
        self.velocity = Vec3(0, 0, 0)
        self.last_position = self.world_position
        self.last_update_time = time.time() # Ursina의 time 객체 사용

        # --- 위치 추정 관련 변수 ---
        self.estimated_position = self.position # 초기 추정 위치는 실제 위치
        self.estimated_rotation_y = self.rotation_y # 초기 추정 각도
        self.localization_certainty = 1.0 # 위치 추정 확실성 (1.0 = 완벽, 작을수록 불확실)
        self.is_localizing = False # 현재 위치 추정 중인지 여부 플래그

        # 3. 조작 관련 설정 (원래 속도 값 사용)
        self.move_speed = 5 # time.dt를 사용하므로 원래 값 복원
        self.rotation_speed = 50
        self.turret_rotation_speed = 80
        self.gun_rotation_speed = 40
        self.max_gun_angle = 20
        self.min_gun_angle = -5

    def update(self):
        # --- 이동 및 회전 로직 (time.dt 사용, 이동 후 충돌 검사) ---
        dt = time.dt # 프레임 시간 사용

        # 이동 처리 (앞/뒤) - '이동 후 검사' 방식 및 dt 적용
        if not self.is_localizing and (held_keys['w'] or held_keys['s']): # 위치 추정 중 아닐 때만 이동
            direction = 1 if held_keys['w'] else -1
            move_amount = direction * self.move_speed * dt # dt 적용
            original_position = self.position # 이동 전 위치 저장

            # 일단 이동
            self.position += self.forward * move_amount

            # 이동 후 다른 물체와 충돌(겹침)하는지 검사
            hit_info = self.intersects(ignore=(self, self.body, self.turret, self.gun))
            if hit_info.hit:
                # print(f"Collision with {hit_info.entity}, moving back!") # 디버깅용
                self.position = original_position # 충돌 시 원위치
            else:
                # 이동 성공 시 위치 추정 불확실성 약간 증가 (움직였으므로)
                self.localization_certainty = max(0.5, self.localization_certainty * 0.995)


        # 몸체 회전 (좌/우)
        if not self.is_localizing and held_keys['d']:
            self.rotation_y += self.rotation_speed * dt
            self.localization_certainty = max(0.5, self.localization_certainty * 0.99) # 회전 시 불확실성 더 증가
        if not self.is_localizing and held_keys['a']:
            self.rotation_y -= self.rotation_speed * dt
            self.localization_certainty = max(0.5, self.localization_certainty * 0.99)

        # 포탑/포신 회전은 위치 추정과 직접 관련 없으므로 항상 가능
        if held_keys['right arrow']:
            self.turret.rotation_y += self.turret_rotation_speed * dt
        if held_keys['left arrow']:
            self.turret.rotation_y -= self.turret_rotation_speed * dt

        target_gun_rot_x = self.gun.rotation_x
        if held_keys['up arrow']:
            target_gun_rot_x -= self.gun_rotation_speed * dt
        if held_keys['down arrow']:
            target_gun_rot_x += self.gun_rotation_speed * dt
        self.gun.rotation_x = clamp(target_gun_rot_x, self.min_gun_angle, self.max_gun_angle)


        # --- 상태 업데이트 (속도 등) ---
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0 / UPDATE_FREQUENCY:
            distance_moved = distance(self.world_position, self.last_position)
            time_elapsed = current_time - self.last_update_time
            if time_elapsed > 0:
                self.speed = (distance_moved / time_elapsed) * 3.6
            else:
                self.speed = 0
            self.last_position = self.world_position
            self.last_update_time = current_time

        # --- 추정 위치 업데이트 (불확실성 반영) ---
        # 실제 위치 주변에서 불확실성만큼 랜덤하게 흔들리는 것처럼 보이게 함
        uncertainty_offset = Vec3(random.uniform(-1, 1), 0, random.uniform(-1, 1)) * (1.0 - self.localization_certainty) * 2.0 # 불확실할수록 크게 흔들림
        self.estimated_position = self.position + uncertainty_offset
        # 각도도 약간 흔들리게 (옵션)
        uncertainty_rot = random.uniform(-10, 10) * (1.0 - self.localization_certainty)
        self.estimated_rotation_y = self.rotation_y + uncertainty_rot


    def start_localization(self):
        """ 위치 추정 프로세스 시작 (시뮬레이션) """
        if self.is_localizing:
            return # 이미 추정 중이면 반환

        print("Starting localization simulation...")
        self.is_localizing = True
        # 위치 추정 시퀀스 실행 (예: 2초간 주변 스캔 후 위치 보정)
        s = Sequence(
            Func(self.show_localization_effect, True), # 스캔 중이라는 시각 효과 (옵션)
            Wait(2.0), # 2초 동안 '스캔' (실제로는 대기)
            Func(self.perform_localization_correction), # 위치 보정 함수 호출
            Func(self.show_localization_effect, False), # 시각 효과 제거
            Func(setattr, self, 'is_localizing', False), # 플래그 리셋
            Func(print, "Localization finished.")
        )
        s.start()

    def perform_localization_correction(self):
        """ 위치 추정 및 보정 (시뮬레이션) """
        # 여기서는 간단히 확실성을 최대로 높여 추정 위치를 실제 위치에 가깝게 만듦
        # 실제로는 여기서 복잡한 매칭/필터링 알고리즘이 결과를 내놓음
        print("Performing localization correction...")
        self.localization_certainty = 1.0 # 위치 찾았다고 가정하고 확실성 최대로
        # 실제 위치로 바로 맞추거나 약간의 오차를 남길 수 있음
        self.estimated_position = self.position
        self.estimated_rotation_y = self.rotation_y

    def show_localization_effect(self, active):
         """ 위치 추정 중 시각 효과 (예: 포탑/포신 고정 또는 색상 변경) """
         if active:
             # 예: 포탑 색상을 잠시 변경
             self.turret.original_color = self.turret.color
             self.turret.color = color.blue
         else:
             # 원래 색상으로 복구
             if hasattr(self.turret, 'original_color'):
                 self.turret.color = self.turret.original_color


    def get_gps_coordinates(self, use_estimated=False):
        """ GPS 좌표 반환 (추정 위치 사용 옵션) """
        pos_to_use = self.estimated_position if use_estimated else self.position
        lat = GPS_ORIGIN_LAT + (pos_to_use.z - 0) * GPS_SCALE
        lon = GPS_ORIGIN_LON + (pos_to_use.x - 0) * GPS_SCALE
        return lat, lon

# --- Ursina 앱 설정 ---
app = Ursina()

# --- 환경 생성 (원래 색상) ---
ground = Entity(model='plane', scale=100, color=color.lime, texture='white_cube', collider='box')
sky = Sky() # 기본 하늘색
light = DirectionalLight(y=1, z=1, shadows=True)
light.look_at(Vec3(0,-1,-1))

# --- 장애물 생성 ---
obstacles = []
for i in range(50):
    obs_scale = random.uniform(1, 4)
    obs_pos = Vec3(random.uniform(-45, 45), 0.5 * obs_scale, random.uniform(-45, 45))
    if distance_xz(obs_pos, Vec3(0,0,0)) < 10:
        continue
    obs = Entity(model='cube', scale=obs_scale, position=obs_pos, color=color.random_color(), collider='box', name=f'obstacle_{i}')
    obstacles.append(obs)

# --- 탱크 인스턴스 생성 ---
player_tank = Tank(position=(0, 0.5, 0))

# --- 카메라 설정 ---
camera.position = (0, 15, -25)
camera.rotation_x = 30

# --- HUD 텍스트 요소 (추정 위치 추가) ---
hud_texts = {
    'Distance': Text(text='Distance: N/A', origin=(-0.5, 0.5), position=(-0.95, 0.45), scale=1.2),
    'Speed': Text(text='Speed: 0.0 Km/h', origin=(-0.5, 0.5), position=(-0.95, 0.40), scale=1.2),
    'Pos': Text(text='Pos: (0.0, 0.0)', origin=(-0.5, 0.5), position=(-0.95, 0.35), scale=1.2),
    'Est Pos': Text(text='Est Pos: (0.0, 0.0)', origin=(-0.5, 0.5), position=(-0.95, 0.30), scale=1.2, color=color.yellow), # 추정 위치
    'Alt': Text(text='Alt: 0.0 m', origin=(-0.5, 0.5), position=(-0.95, 0.25), scale=1.2),
    'Certainty': Text(text='Certainty: 1.0', origin=(-0.5, 0.5), position=(-0.95, 0.20), scale=1.2, color=color.cyan), # 확실성
    'Turret': Text(text='Turret: (0.0, 0.0)', origin=(-0.5, 0.5), position=(-0.95, 0.15), scale=1.2),
    'Body': Text(text='Body: (0.0, 0.0)', origin=(-0.5, 0.5), position=(-0.95, 0.10), scale=1.2),
    'GPS': Text(text='GPS: (Lat: 0.0, Lon: 0.0)', origin=(-0.5, 0.5), position=(-0.95, 0.05), scale=1.2),
    'Est GPS': Text(text='Est GPS: (Lat: 0.0, Lon: 0.0)', origin=(-0.5, 0.5), position=(-0.95, 0.00), scale=1.2, color=color.yellow), # 추정 GPS
    'LiDAR': Text(text='LiDAR', origin=(0.5, -0.5), position=(0.85, -0.85), scale=1.2),
    'Localize': Text(text='[L]ocalize', origin=(-1, -0.5), position=(-0.95, -0.95), scale=1.2), # 위치 추정 키 안내
}

# --- LiDAR 시뮬레이션 ---
lidar_display = Panel(
    model=Circle(resolution=36, mode='line'), scale=0.15, origin=(0, 0),
    position=(0.85, -0.9), color=color.green, thickness=2
)
LIDAR_HIT_COLOR = color.red
LIDAR_NO_HIT_COLOR = color.green

# --- HUD 업데이트 함수 ---
hud_update_counter = 0
def update_hud():
    global hud_update_counter
    hud_update_counter += 1
    if hud_update_counter % (60 // UPDATE_FREQUENCY) == 0: # 원래 빈도 조절 방식 복원
        hud_texts['Speed'].text = f'Speed: {player_tank.speed:.1f} Km/h'
        hud_texts['Pos'].text = f'Pos: ({player_tank.world_x:.1f}, {player_tank.world_z:.1f})'
        # 추정 위치 표시
        hud_texts['Est Pos'].text = f'Est Pos: ({player_tank.estimated_position.x:.1f}, {player_tank.estimated_position.z:.1f})'
        hud_texts['Alt'].text = f'Alt: {player_tank.world_y * VIRTUAL_METER_SCALE:.1f} m'
        # 확실성 표시
        hud_texts['Certainty'].text = f'Certainty: {player_tank.localization_certainty:.2f}'
        turret_yaw = (player_tank.turret.rotation_y + player_tank.rotation_y) % 360
        gun_pitch = -player_tank.gun.rotation_x
        hud_texts['Turret'].text = f'Turret: Yaw={turret_yaw:.1f}, Pitch={gun_pitch:.1f}'
        # 추정 각도 표시 (옵션)
        # body_yaw = player_tank.rotation_y % 360
        est_body_yaw = player_tank.estimated_rotation_y % 360
        hud_texts['Body'].text = f'Body: Yaw={est_body_yaw:.1f}, Pitch={player_tank.rotation_x:.1f}' # Yaw는 추정치 사용
        # GPS (실제/추정)
        lat, lon = player_tank.get_gps_coordinates(use_estimated=False)
        hud_texts['GPS'].text = f'GPS: (Lat: {lat:.6f}, Lon: {lon:.6f})'
        est_lat, est_lon = player_tank.get_gps_coordinates(use_estimated=True)
        hud_texts['Est GPS'].text = f'Est GPS: (Lat: {est_lat:.6f}, Lon: {est_lon:.6f})'


# LiDAR 업데이트 빈도 조절용 카운터
lidar_update_cycle_counter = 0
LIDAR_UPDATE_INTERVAL = 10

# --- 키 입력 처리 함수 ---
def input(key):
    if key == 'l': # L 키를 누르면 위치 추정 시뮬레이션 시작
        player_tank.start_localization()

# --- 메인 업데이트 루프 ---
def update():
    global lidar_update_cycle_counter

    player_tank.update() # 탱크 이동, 회전, 속도 계산, 추정 위치 '흔들림' 업데이트 등
    # player_tank.update_localization() # update 함수 내에 통합하거나 여기서 별도 호출
    update_hud()

    # --- LiDAR 시뮬레이션 업데이트 ---
    lidar_update_cycle_counter += 1
    if lidar_update_cycle_counter >= LIDAR_UPDATE_INTERVAL:
        lidar_update_cycle_counter = 0
        hit_detected_in_sector = False
        # ... (LiDAR 로직은 이전과 동일)
        for angle in range(0, 360, 10):
            world_angle_rad = math.radians(player_tank.world_rotation_y + player_tank.turret.rotation_y + angle)
            direction = Vec3(math.sin(world_angle_rad), 0, math.cos(world_angle_rad)).normalized()
            origin = player_tank.turret.world_position + Vec3(0, 0.1, 0)
            hit_info = raycast(origin, direction, distance=20, ignore=(player_tank, player_tank.body, player_tank.turret, player_tank.gun), debug=False)

            if hit_info.hit:
                hit_detected_in_sector = True
                break
        lidar_display.color = LIDAR_HIT_COLOR if hit_detected_in_sector else LIDAR_NO_HIT_COLOR


# --- 앱 실행 ---
app.run()