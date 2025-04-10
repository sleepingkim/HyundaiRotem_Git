# 필요 라이브러리 설치 (아직 설치되지 않았다면)
# pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt



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

# 1. & 3. 맵 상의 장애물(랜드마크) 위치 정의
# 미리 알고 있는 지도 정보라고 가정합니다. 각 랜드마크의 이름과 (x, y) 좌표.
landmarks = {
    'Tree1': np.array([2.0, 8.0]),
    'BuildingA': np.array([7.0, 2.0]),
    'Statue': np.array([10.0, 10.0]),
    'Bench': np.array([1.0, 1.0]),
    'LampPost': np.array([8.0, 7.0])
}

# 사용자의 실제 위치 (시뮬레이션을 위해 가정)
true_position = np.array([5.0, 6.0])

# 센서(카메라 등) 관련 파라미터
sensor_range = 5.0  # 감지 가능 최대 거리
measurement_noise_std = 0.3 # 측정 오차의 표준편차 (현실성 추가)

# --- 시뮬레이션 함수 ---

def simulate_obstacle_detection(current_pos, known_landmarks, range_limit, noise_std):
    """
    1. 주변 장애물 탐지 및 2. 상대 위치 파악 (시뮬레이션)
    현재 위치로부터 일정 거리 내에 있는 랜드마크를 찾고,
    측정 노이즈를 포함한 '상대적' 위치를 반환합니다.
    """
    detected_landmarks = {}
    print("\n--- 1 & 2. 주변 장애물 탐지 및 상대 위치 파악 ---")
    for name, landmark_abs_pos in known_landmarks.items():
        # 현재 위치와 랜드마크 사이의 실제 거리 계산
        distance = np.linalg.norm(landmark_abs_pos - current_pos)

        if distance <= range_limit:
            # 실제 상대 위치 벡터 (랜드마크 위치 - 현재 위치)
            true_relative_pos = landmark_abs_pos - current_pos

            # 측정 노이즈 생성 (평균 0, 표준편차 noise_std인 정규분포 따름)
            noise = np.random.normal(0, noise_std, size=2)

            # 노이즈가 추가된 측정된 상대 위치
            measured_relative_pos = true_relative_pos + noise

            # 감지된 랜드마크 정보 저장 (랜드마크 이름과 측정된 상대 위치)
            detected_landmarks[name] = measured_relative_pos
            print(f"- {name}: 실제 거리={distance:.2f}, 측정된 상대위치={measured_relative_pos}")
        else:
             # print(f"- {name}: 감지 범위 밖 (거리={distance:.2f})") # 필요시 주석 해제
             pass

    if not detected_landmarks:
        print("- 감지된 장애물 없음")
    return detected_landmarks

def estimate_my_position(detected_info, known_landmarks_map):
    """
    4. 나의 위치 추정
    감지된 랜드마크 정보(측정된 상대 위치)와 맵 정보(랜드마크 절대 위치)를 이용해
    현재 위치를 추정합니다. (간단한 평균 방식 사용)
    """
    position_estimates = []
    print("\n--- 4. 나의 위치 추정 ---")

    if not detected_info:
        print("경고: 감지된 랜드마크가 없어 위치를 추정할 수 없습니다.")
        return None # 위치 추정 불가

    for name, measured_relative_pos in detected_info.items():
        if name in known_landmarks_map:
            # 맵에서 해당 랜드마크의 절대 위치 가져오기
            landmark_abs_pos = known_landmarks_map[name]

            # 해당 랜드마크 관측을 기반으로 한 나의 위치 추정
            # 나의 위치 = 랜드마크 절대 위치 - 측정된 상대 위치
            estimated_pos = landmark_abs_pos - measured_relative_pos
            position_estimates.append(estimated_pos)
            print(f"- {name} 기반 추정 위치: {estimated_pos}")
        else:
            # 이론적으로는 발생하지 않아야 함 (감지된 것은 맵에 있어야 함)
            print(f"경고: 감지된 랜드마크 '{name}'가 맵에 없습니다.")

    if not position_estimates:
         print("경고: 유효한 랜드마크 정보로 위치를 추정할 수 없습니다.")
         return None

    # 모든 개별 추정 위치의 평균을 최종 추정 위치로 사용
    final_estimated_position = np.mean(position_estimates, axis=0)
    print(f"최종 추정 위치 (평균): {final_estimated_position}")
    return final_estimated_position

# --- 메인 실행 로직 ---

print(f"시뮬레이션 시작: 실제 위치 = {true_position}")

# 1. & 2. 주변 장애물 탐지 및 상대 위치 파악 시뮬레이션
detected_obstacles_info = simulate_obstacle_detection(true_position, landmarks, sensor_range, measurement_noise_std)

# 3. 맵 상의 위치 특정 (이미 'landmarks' 변수에 정의됨)
# 이 단계는 별도 함수보다는 'estimate_my_position' 함수 내에서 맵 정보를 활용하는 것으로 구현됩니다.

# 4. 나의 위치 추정
estimated_position = estimate_my_position(detected_obstacles_info, landmarks)

# --- 결과 시각화 ---
print("\n--- 결과 시각화 ---")
fig, ax = plt.subplots(figsize=(10, 10))

# 맵 랜드마크 표시 (파란색 X)
landmark_coords = np.array(list(landmarks.values()))
ax.scatter(landmark_coords[:, 0], landmark_coords[:, 1], c='blue', marker='x', s=100, label='랜드마크 (맵)')
for name, pos in landmarks.items():
    ax.text(pos[0] + 0.1, pos[1] + 0.1, name, fontsize=9) # 랜드마크 이름 표시

# 실제 위치 표시 (녹색 원)
ax.scatter(true_position[0], true_position[1], c='green', marker='o', s=150, label='실제 위치', alpha=0.8, zorder=5) # zorder로 위에 표시

# 센서 범위 표시 (녹색 점선 원)
sensor_circle = plt.Circle(true_position, sensor_range, color='green', fill=False, linestyle='--', linewidth=1.5, label='센서 범위')
ax.add_patch(sensor_circle)

# 감지된 랜드마크 강조 (주황색 사각형 테두리)
detected_names = list(detected_obstacles_info.keys())
if detected_names:
    detected_coords = np.array([landmarks[name] for name in detected_names])
    ax.scatter(detected_coords[:, 0], detected_coords[:, 1], s=180, facecolors='none', edgecolors='orange', linewidth=2, label='감지된 랜드마크', zorder=4)

# 추정된 위치 표시 (빨간색 +)
if estimated_position is not None:
    ax.scatter(estimated_position[0], estimated_position[1], c='red', marker='+', s=250, linewidth=3, label='추정된 위치', zorder=6)
    # 오차 계산 및 출력
    error = np.linalg.norm(true_position - estimated_position)
    print(f"\n최종 추정 오차: {error:.3f}")


# 그래프 설정
ax.set_xlabel("X 좌표")
ax.set_ylabel("Y 좌표")
ax.set_title("간단한 2D VPR 시뮬레이션 (장애물 기반 위치 추정)")
ax.legend(loc='upper left')
ax.grid(True)
ax.axis('equal') # X, Y 축 스케일을 동일하게 맞춰 거리 왜곡 방지

# 축 범위 자동 조절 (모든 점이 보이도록)
all_x = list(landmark_coords[:, 0]) + [true_position[0]] + ([estimated_position[0]] if estimated_position is not None else [])
all_y = list(landmark_coords[:, 1]) + [true_position[1]] + ([estimated_position[1]] if estimated_position is not None else [])
if all_x and all_y:
    ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
    ax.set_ylim(min(all_y) - 2, max(all_y) + 2)
else: # 점이 하나도 없을 경우 기본 범위 설정
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)

plt.show()