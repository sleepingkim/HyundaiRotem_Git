# 필요 라이브러리 설치 (아직 설치되지 않았다면)
# pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
import time
 # matplotlib 한글깨짐 방지
import platform
if platform.system() == 'Darwin': #맥
        plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': #윈도우
        plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux': #리눅스 (구글 콜랩)
        #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
        #!mv malgun.ttf /usr/share/fonts/truetype/
        #import matplotlib.font_manager as fm 
        #fm._rebuild() 
        plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False #한글 폰트 사용시 마이너스 폰트 깨짐 해결
#matplotlib 패키지 한글 깨짐 처리 끝
# --- 환경 설정 ---

# 맵 상의 장애물(랜드마크) 위치 정의
landmarks = {
    'Tree1': np.array([2.0, 8.0]),
    'BuildingA': np.array([7.0, 2.0]),
    'Statue': np.array([10.0, 10.0]),
    'Bench': np.array([1.0, 1.0]),
    'LampPost': np.array([8.0, 7.0]),
    'Fountain': np.array([5.0, 12.0]),
    'Sign': np.array([-2.0, 5.0]),
    'Corner': np.array([13.0, 1.0]) # 추가 랜드마크
}

# --- 시뮬레이션 파라미터 ---
initial_true_position = np.array([1.0, 1.5])  # 시작 위치
target_position = np.array([12.0, 9.0]) # 목표 지점 좌표
sensor_range = 5.0
measurement_noise_std = 0.25

# --- 이동 및 종료 파라미터 ---
max_cycles = 500       # 최대 반복 횟수 (무한 루프 방지)
step_distance = 0.4    # 한 사이클당 이동 거리
heading_noise_std = 0.15 # 이동 방향의 무작위성 (라디안)
reach_threshold = 0.8  # 목표 도달로 간주하는 거리 (추정 위치 기준)
steering_gain = 0.3    # 목표 방향으로 향하는 정도 (0: 랜덤, 1: 완전 목표 지향)

# --- 시뮬레이션 함수 (이전과 동일) ---

def simulate_obstacle_detection(current_pos, known_landmarks, range_limit, noise_std):
    """주변 장애물 탐지 및 상대 위치 파악 (노이즈 포함)"""
    detected_landmarks = {}
    for name, landmark_abs_pos in known_landmarks.items():
        distance = np.linalg.norm(landmark_abs_pos - current_pos)
        if distance <= range_limit:
            true_relative_pos = landmark_abs_pos - current_pos
            noise = np.random.normal(0, noise_std, size=2)
            measured_relative_pos = true_relative_pos + noise
            detected_landmarks[name] = measured_relative_pos
    return detected_landmarks

def estimate_my_position(detected_info, known_landmarks_map):
    """감지 정보와 맵을 이용해 현재 위치 추정 (평균 방식)"""
    position_estimates = []
    if not detected_info:
        return None

    for name, measured_relative_pos in detected_info.items():
        if name in known_landmarks_map:
            landmark_abs_pos = known_landmarks_map[name]
            estimated_pos = landmark_abs_pos - measured_relative_pos
            position_estimates.append(estimated_pos)

    if not position_estimates:
         return None

    final_estimated_position = np.mean(position_estimates, axis=0)
    return final_estimated_position

# --- 메인 시뮬레이션 루프 ---

current_true_pos = initial_true_position.copy()
# 초기 방향: 목표 지점을 향하도록 설정
current_heading = np.arctan2(target_position[1] - current_true_pos[1],
                           target_position[0] - current_true_pos[0])

true_path = []
estimated_path = []
estimation_successful = [] # 추정 성공 여부 기록

# 가장 최근의 유효한 추정 위치 (초기값은 실제 시작 위치로 설정)
last_valid_estimate = current_true_pos.copy()
# 목표 지점까지의 거리 (추정 위치 기준)
distance_to_target = np.linalg.norm(last_valid_estimate - target_position)

cycle = 0

print(f"시뮬레이션 시작: 목표 지점 {target_position} 도달 시도 (최대 {max_cycles} 사이클)")
print(f"초기 실제 위치: {current_true_pos}, 목표 도달 기준 거리: {reach_threshold}")

# 루프 조건: 목표 거리보다 멀고, 최대 사이클을 넘지 않으면 계속 진행
while distance_to_target > reach_threshold and cycle < max_cycles:
    # 1. 현재 실제 위치 저장
    true_path.append(current_true_pos.copy())

    # 2. 현재 위치에서 장애물 탐지
    detected_obstacles_info = simulate_obstacle_detection(
        current_true_pos, landmarks, sensor_range, measurement_noise_std
    )

    # 3. 위치 추정
    current_estimated_pos = estimate_my_position(
        detected_obstacles_info, landmarks
    )

    # 4. 추정 결과 저장 및 목표 도달 여부 업데이트
    if current_estimated_pos is not None:
        estimated_path.append(current_estimated_pos.copy())
        estimation_successful.append(True)
        last_valid_estimate = current_estimated_pos.copy() # 유효 추정치 업데이트
        # 새로운 유효 추정치 기준으로 목표까지 거리 다시 계산
        distance_to_target = np.linalg.norm(last_valid_estimate - target_position)
    else:
        # 추정 실패 시 NaN 저장
        estimated_path.append(np.array([np.nan, np.nan]))
        estimation_successful.append(False)
        # distance_to_target은 이전 값 유지 (추정 실패했으므로 업데이트 불가)

    # 5. 다음 이동 계산 (목표 방향 + 약간의 랜덤성)
    # 목표 지점 방향 벡터 계산
    vector_to_target = target_position - current_true_pos
    target_heading = np.arctan2(vector_to_target[1], vector_to_target[0])

    # 현재 방향과 목표 방향 사이의 각도 차이 계산 (범위: -pi ~ pi)
    heading_error = target_heading - current_heading
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # 새로운 이동 방향 결정 (목표 방향 성분 + 랜덤 노이즈)
    # steering_gain 만큼 목표 방향으로 향하고, 나머지는 랜덤 노이즈 추가
    current_heading += steering_gain * heading_error + np.random.normal(0, heading_noise_std)
    # 헤딩 각도를 -pi ~ pi 범위로 유지 (선택적이지만 권장)
    current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi

    # 실제 위치 업데이트
    dx = step_distance * np.cos(current_heading)
    dy = step_distance * np.sin(current_heading)
    current_true_pos += np.array([dx, dy])

    # 사이클 카운터 증가
    cycle += 1

    # 진행 상황 출력 (선택적)
    if cycle % 20 == 0:
        status = "성공" if current_estimated_pos is not None else "실패"
        print(f"Cycle {cycle}: Est. Pos {last_valid_estimate.round(2)}, Dist to Target: {distance_to_target:.2f}, Last Est. Status: {status}")


# --- 루프 종료 후 처리 ---
# 마지막 실제 위치 추가 (루프 조건 검사 후 이동했으므로)
true_path.append(current_true_pos.copy())

print("\n시뮬레이션 종료.")
if distance_to_target <= reach_threshold:
    print(f"성공: {cycle} 사이클 만에 목표 지점 근처 도달! (최종 추정 위치 기준 거리: {distance_to_target:.3f})")
else:
    print(f"실패: 최대 {max_cycles} 사이클 내에 목표 지점에 도달하지 못했습니다. (최종 추정 위치 기준 거리: {distance_to_target:.3f})")

# 경로 데이터를 NumPy 배열로 변환
true_path = np.array(true_path)
estimated_path = np.array(estimated_path)
estimation_successful = np.array(estimation_successful) # bool 배열

# --- 최종 경로 시각화 ---
print("\n--- 최종 경로 시각화 ---")
fig, ax = plt.subplots(figsize=(12, 10))

# 맵 랜드마크 표시
landmark_coords = np.array(list(landmarks.values()))
ax.scatter(landmark_coords[:, 0], landmark_coords[:, 1], c='blue', marker='x', s=100, label='랜드마크 (맵)')
for name, pos in landmarks.items():
    ax.text(pos[0] + 0.1, pos[1] + 0.1, name, fontsize=9)

# 목표 지점 표시 (자주색 별)
ax.scatter(target_position[0], target_position[1], c='magenta', marker='*', s=350, label='목표 지점', zorder=7)
# 목표 도달 기준 원 표시 (자주색 점선 원)
target_circle = plt.Circle(target_position, reach_threshold, color='magenta', fill=False, linestyle=':', linewidth=2, label='목표 도달 기준')
ax.add_patch(target_circle)

# 실제 경로 표시 (녹색 실선)
if len(true_path) > 0:
    ax.plot(true_path[:, 0], true_path[:, 1], 'g-', linewidth=2, alpha=0.8, label='실제 경로')
    ax.scatter(true_path[0, 0], true_path[0, 1], c='lime', marker='o', s=120, label='시작 (실제)', zorder=5)
    ax.scatter(true_path[-1, 0], true_path[-1, 1], c='darkgreen', marker='s', s=120, label='끝 (실제)', zorder=5)

# 추정 경로 표시 (빨간색 점선) - NaN은 끊어져 표시됨
if len(estimated_path) > 0:
    ax.plot(estimated_path[:, 0], estimated_path[:, 1], 'r--', linewidth=2, alpha=0.8, label='추정 경로')
    valid_indices = np.where(estimation_successful)[0]
    if len(valid_indices) > 0:
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]
        ax.scatter(estimated_path[first_valid_idx, 0], estimated_path[first_valid_idx, 1],
                   c='orange', marker='o', s=120, label='시작 (추정)', zorder=6)
        ax.scatter(estimated_path[last_valid_idx, 0], estimated_path[last_valid_idx, 1],
                   c='darkred', marker='s', s=120, label='끝 (추정)', zorder=6)

# 평균 오차 계산 (성공한 추정에 대해서만)
valid_indices = np.where(estimation_successful)[0] # 추정 성공한 사이클 인덱스
if len(valid_indices) > 0:
    # 해당 사이클의 추정 위치와 실제 위치 간의 오차 계산
    errors = np.linalg.norm(estimated_path[valid_indices] - true_path[valid_indices], axis=1)
    avg_error = np.mean(errors)
    error_std = np.std(errors)
    print(f"\n평균 위치 추정 오차 (총 {len(errors)}번 성공): {avg_error:.3f} (표준편차: {error_std:.3f})")
    outcome = "성공" if distance_to_target <= reach_threshold else "실패"
    plot_title = f"VPR 시뮬레이션: 목표 {target_position} 도달 ({outcome}, {cycle} 사이클)\n평균 오차: {avg_error:.3f}"
else:
    print("\n유효한 위치 추정이 없어 평균 오차를 계산할 수 없습니다.")
    plot_title = f"VPR 시뮬레이션: 목표 {target_position} 도달 시도 ({cycle} 사이클, 추정 없음)"

# 그래프 설정
ax.set_xlabel("X 좌표")
ax.set_ylabel("Y 좌표")
ax.set_title(plot_title)
ax.legend(loc='best')
ax.grid(True)
ax.axis('equal')

# 축 범위 동적 설정 (모든 요소 포함)
all_x = list(landmark_coords[:, 0]) + list(true_path[:, 0]) + list(estimated_path[valid_indices, 0]) + [target_position[0]]
all_y = list(landmark_coords[:, 1]) + list(true_path[:, 1]) + list(estimated_path[valid_indices, 1]) + [target_position[1]]
if all_x and all_y:
    ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
    ax.set_ylim(min(all_y) - 2, max(all_y) + 2)
else:
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)

plt.show()