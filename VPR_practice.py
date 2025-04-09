import cv2
import numpy as np
import math
# import matplotlib.pyplot as plt # 시각화 필요시

# --- 설정 ---
MAP_IMAGE_PATH = './minimap.png' # 미니맵 이미지 경로
FEATURE_EXTRACTOR = cv2.ORB_create(nfeatures=2000) # 특징점 추출기 (SIFT, SURF 등 다른 것 사용 가능)
# FEATURE_EXTRACTOR = cv2.SIFT_create() # SIFT 사용 시 (특허 문제 확인 필요)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # 특징점 매칭기 (ORB는 NORM_HAMMING 사용)
# MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # SIFT/SURF는 NORM_L2 사용

# --- 미니맵 로드 및 전처리 ---
try:
    map_image_full = cv2.imread(MAP_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if map_image_full is None:
        raise FileNotFoundError(f"맵 이미지를 로드할 수 없습니다: {MAP_IMAGE_PATH}")
    map_h, map_w = map_image_full.shape
    print(f"미니맵 로드 완료: {map_w}x{map_h}")

    # 미니맵 전체에 대한 특징점 미리 계산 (성능 향상 위해)
    map_kp_full, map_des_full = FEATURE_EXTRACTOR.detectAndCompute(map_image_full, None)
    if map_des_full is None:
        print("경고: 미니맵 전체에서 특징점을 찾지 못했습니다.")
        map_kp_full, map_des_full = [], None # 특징점 없는 경우 처리

except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"오류 발생: {e}")
    exit()

# --- 가상 전차 클래스 ---
class Tank:
    def __init__(self, x=map_w // 2, y=map_h // 2, heading=0.0, speed=0.0, turret_angle_h=0.0, turret_angle_v=0.0):
        # 실제 값 (시뮬레이션용)
        self.x = float(x) # X 좌표 (픽셀 기준)
        self.y = float(y) # Y 좌표 (픽셀 기준)
        self.heading = float(heading) # 본체 방향 (도, 0=동쪽, 90=북쪽)
        self.speed = float(speed) # 현재 속도 (가상 단위)
        self.turret_angle_h = float(turret_angle_h) # 터렛 수평 각도 (본체 기준)
        self.turret_angle_v = float(turret_angle_v) # 포신 상하 각도

        # VPR 추정 값
        self.estimated_x = None
        self.estimated_y = None
        self.estimated_heading = None

    def move(self, distance):
        """전차를 현재 방향으로 'distance'만큼 이동"""
        rad = math.radians(self.heading)
        self.x += distance * math.cos(rad)
        self.y -= distance * math.sin(rad) # 이미지 좌표계는 y가 아래로 증가
        # 맵 경계 처리 (예시)
        self.x = max(0, min(map_w - 1, self.x))
        self.y = max(0, min(map_h - 1, self.y))

    def rotate_body(self, angle):
        """본체를 'angle'만큼 회전 (양수: 반시계)"""
        self.heading = (self.heading + angle) % 360

    def rotate_turret_h(self, angle):
        """터렛을 수평으로 'angle'만큼 회전 (양수: 반시계)"""
        self.turret_angle_h = (self.turret_angle_h + angle) % 360

    def get_camera_view(self, map_img, view_width=128, view_height=128, fov=60):
        """
        현재 위치/방향에서 카메라 뷰(이미지) 시뮬레이션.
        단순화: 맵의 일부를 회전/크롭하여 생성.
        """
        # 카메라 방향 = 본체 방향 + 터렛 수평 방향
        camera_heading_rad = math.radians(self.heading + self.turret_angle_h)

        # --- 방법 1: 간단히 현재 위치 주변 크롭 (회전 미반영) ---
        # x_int, y_int = int(self.x), int(self.y)
        # half_w, half_h = view_width // 2, view_height // 2
        # top, bottom = max(0, y_int - half_h), min(map_h, y_int + half_h)
        # left, right = max(0, x_int - half_w), min(map_w, x_int + half_w)
        # cropped = map_img[top:bottom, left:right]
        # # 크기가 다를 수 있으므로 리사이즈
        # view = cv2.resize(cropped, (view_width, view_height), interpolation=cv2.INTER_AREA)
        # return view

        # --- 방법 2: 회전을 고려한 크롭 (더 정확하지만 복잡) ---
        # 1. 현재 위치를 중심으로 충분히 큰 영역을 자른다.
        center_x, center_y = int(self.x), int(self.y)
        crop_size = int(max(view_width, view_height) * 1.5) # 회전 시 잘리지 않도록 여유
        half_crop = crop_size // 2
        x1, y1 = max(0, center_x - half_crop), max(0, center_y - half_crop)
        x2, y2 = min(map_w, center_x + half_crop), min(map_h, center_y + half_crop)
        large_crop = map_img[y1:y2, x1:x2]
        
        # 원본 이미지 내 좌표를 자른 이미지 내 좌표로 변환
        local_center_x = center_x - x1
        local_center_y = center_y - y1

        # 2. 카메라 방향의 반대 방향으로 이미지를 회전시킨다.
        # (카메라가 북쪽을 보면, 이미지는 남쪽이 위로 오도록 회전)
        rotate_angle = -(self.heading + self.turret_angle_h - 90) # OpenCV 회전 기준 (반시계) 및 좌표계 보정
        M = cv2.getRotationMatrix2D((local_center_x, local_center_y), rotate_angle, 1.0)
        
        # 회전 시 이미지 크기가 변경될 수 있으므로 경계 계산
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((large_crop.shape[1] * cos) + (large_crop.shape[0] * sin))
        new_h = int((large_crop.shape[1] * sin) + (large_crop.shape[0] * cos))
        
        # 회전 중심 이동 반영
        M[0, 2] += (new_w / 2) - local_center_x
        M[1, 2] += (new_h / 2) - local_center_y
        
        rotated_large = cv2.warpAffine(large_crop, M, (new_w, new_h))

        # 3. 회전된 이미지의 중심에서 원하는 크기(view_width, view_height)만큼 최종 크롭.
        final_center_x, final_center_y = new_w // 2, new_h // 2
        half_view_w, half_view_h = view_width // 2, view_height // 2
        fx1 = max(0, final_center_x - half_view_w)
        fy1 = max(0, final_center_y - half_view_h)
        fx2 = min(new_w, final_center_x + half_view_w)
        fy2 = min(new_h, final_center_y + half_view_h)
        
        final_view = rotated_large[fy1:fy2, fx1:fx2]

        # 최종 크기가 정확히 맞지 않을 수 있으므로 리사이즈 (필요시)
        if final_view.shape[1] != view_width or final_view.shape[0] != view_height:
             # 크기가 0인 경우 방지
            if final_view.shape[1] == 0 or final_view.shape[0] == 0:
                return np.zeros((view_height, view_width), dtype=np.uint8) # 검은 이미지 반환
            final_view = cv2.resize(final_view, (view_width, view_height), interpolation=cv2.INTER_AREA)
            
        return final_view

    def display_info(self):
        """요청된 정보들을 출력"""
        print("--- 전차 상태 ---")
        print(f"  실제 Pos (X,Y): ({self.x:.2f}, {self.y:.2f})")
        print(f"  추정 Pos (X,Y): ({self.estimated_x}, {self.estimated_y})") # 추후 VPR 결과 반영
        print(f"  실제 Heading: {self.heading:.2f} 도")
        print(f"  Speed: {self.speed:.2f} km/h (가상)")
        # print(f"  Alt: ... ") # 고도 정보는 2D 맵에서 직접 얻기 어려움
        print(f"  Body Angle (X,Y): ({self.heading:.2f}, 0.0)") # 2D 예시
        print(f"  Turret Angle (H,V): ({self.turret_angle_h:.2f}, {self.turret_angle_v:.2f})")
        # Distance, Mini Map, LiDAR는 별도 처리 필요

# --- VPR 함수 ---
def perform_vpr(camera_view_img, map_img, map_kp, map_des):
    """카메라 뷰와 미니맵을 비교하여 위치 추정"""
    if map_des is None or len(map_kp) == 0:
        print("맵 특징점이 없어 VPR 수행 불가")
        return None, None, None # x, y, heading

    # 1. 카메라 뷰에서 특징점 추출
    cam_kp, cam_des = FEATURE_EXTRACTOR.detectAndCompute(camera_view_img, None)
    if cam_des is None or len(cam_kp) < 10: # 매칭에 필요한 최소 특징점 수
        print("카메라 뷰에서 충분한 특징점을 찾지 못함")
        return None, None, None

    # 2. 특징점 매칭 (카메라 뷰 vs 전체 맵)
    matches = MATCHER.match(cam_des, map_des)

    # 3. 좋은 매칭 결과 선별 (예: 거리 기준)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50] # 상위 50개 매칭 사용 (조절 가능)

    if len(good_matches) < 10: # 위치 추정에 필요한 최소 매칭 수
        print("충분한 매칭 결과를 얻지 못함")
        return None, None, None

    # 4. 위치 추정: 매칭된 맵 특징점들의 평균 위치 계산
    map_match_points = np.float32([map_kp[m.trainIdx].pt for m in good_matches])
    estimated_center = np.mean(map_match_points, axis=0)
    estimated_x, estimated_y = estimated_center[0], estimated_center[1]

    # 5. 방향 추정 (선택 사항, 더 복잡)
    # - 매칭된 특징점들의 상대적 위치 변화 (Homography) 분석 등으로 추정 가능
    # - 여기서는 단순화를 위해 위치만 추정
    estimated_heading = None # 추후 구현

    # (디버깅) 매칭 결과 시각화
    # img_matches = cv2.drawMatches(camera_view_img, cam_kp, map_img, map_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.figure(figsize=(15, 5))
    # plt.imshow(img_matches)
    # plt.title('Feature Matches')
    # plt.show()


    return estimated_x, estimated_y, estimated_heading

# --- 메인 시뮬레이션 루프 (예시) ---
tank = Tank(x=map_w * 0.2, y=map_h * 0.7, heading=45.0) # 초기 위치 및 방향 설정

for i in range(10): # 10 스텝 시뮬레이션
    print(f"\n--- 스텝 {i+1} ---")
    # 1. 전차 이동/회전 (예시)
    tank.move(distance=20)
    tank.rotate_body(angle=5)
    tank.rotate_turret_h(angle=-10)
    tank.speed = 20 # 속도 설정 (가상)

    # 2. 현재 위치에서 카메라 뷰 얻기
    current_view = tank.get_camera_view(map_image_full, view_width=200, view_height=150)
    if current_view is None or current_view.size == 0:
        print("유효한 카메라 뷰를 생성할 수 없습니다.")
        continue

    # (선택) 현재 뷰 보여주기
    # cv2.imshow('Current Camera View', current_view)
    # cv2.waitKey(100) # 잠깐 보여주기

    # 3. VPR 수행
    est_x, est_y, est_h = perform_vpr(current_view, map_image_full, map_kp_full, map_des_full)

    # 4. 결과 업데이트 및 출력
    if est_x is not None and est_y is not None:
        tank.estimated_x = est_x
        tank.estimated_y = est_y
        # tank.estimated_heading = est_h # 방향 추정 구현 시
        print(f"VPR 결과: 추정 위치 ({est_x:.2f}, {est_y:.2f})")
    else:
        print("VPR 실패: 위치를 추정할 수 없음")
        # 이전 추정값 유지 또는 다른 방법 사용

    tank.display_info()

    # (선택) 결과 시각화: 미니맵에 실제 위치와 추정 위치 표시
    map_display = cv2.cvtColor(map_image_full, cv2.COLOR_GRAY2BGR)
    # 실제 위치 (파란색 원)
    cv2.circle(map_display, (int(tank.x), int(tank.y)), 15, (255, 0, 0), 2)
    # 추정 위치 (녹색 원)
    if tank.estimated_x is not None:
        cv2.circle(map_display, (int(tank.estimated_x), int(tank.estimated_y)), 15, (0, 255, 0), 2)
        # 추정 위치와 실제 위치 연결선 (빨간색)
        cv2.line(map_display, (int(tank.x), int(tank.y)), (int(tank.estimated_x), int(tank.estimated_y)), (0, 0, 255), 1)

    # 리사이즈 해서 보기 좋게
    display_scale = 0.5
    small_map_display = cv2.resize(map_display, (int(map_w * display_scale), int(map_h * display_scale)))
    cv2.imshow('Map with Tank Positions (Blue: Actual, Green: Estimated)', small_map_display)

    if cv2.waitKey(500) & 0xFF == ord('q'): # 0.5초 대기, q 누르면 종료
        break

cv2.destroyAllWindows()