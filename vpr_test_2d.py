# 필요 라이브러리 설치 (아직 설치되지 않았다면)
# pip install numpy matplotlib pandas

import numpy as np # 수치 계산, 배열(벡터, 행렬) 처리에 필수적인 라이브러리
import matplotlib.pyplot as plt # 그래프 그리기, 데이터 시각화를 위한 라이브러리
from matplotlib.animation import FuncAnimation # Matplotlib에서 애니메이션을 만들기 위한 클래스
import pandas as pd # 데이터 분석 및 처리를 위한 라이브러리 (여기서는 로그 기록용 DataFrame 사용)
import time # 시간 관련 기능 (현재 코드에서는 직접 사용되지 않지만 유용할 수 있음)
import platform # 실행 중인 운영체제(OS) 정보를 얻기 위한 라이브러리

# matplotlib 한글깨짐 방지 설정 시작 ====================================
# 운영체제 종류 확인
if platform.system() == 'Darwin': # 맥OS 경우
        plt.rc('font', family='AppleGothic') # AppleGothic 폰트 사용
elif platform.system() == 'Windows': # 윈도우 경우
        plt.rc('font', family='Malgun Gothic') # Malgun Gothic 폰트 사용
elif platform.system() == 'Linux': # 리눅스 경우 (구글 코랩 환경 등)
        # 코랩 같은 환경에서는 폰트 파일을 직접 다운로드하고 설정해야 할 수 있음
        #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
        #!mv malgun.ttf /usr/share/fonts/truetype/
        #import matplotlib.font_manager as fm
        #fm._rebuild() # 폰트 캐시 재빌드
        plt.rc('font', family='Malgun Gothic') # Malgun Gothic 폰트 사용
# 한글 폰트 사용 시 축의 마이너스 부호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False
# matplotlib 패키지 한글 깨짐 처리 끝 =====================================

# --- 환경 설정 및 파라미터 ---

# initial_landmarks: 시뮬레이션 시작 시 기본 장애물(랜드마크) 설정
# 딕셔너리 형태: { '이름': np.array([x좌표, y좌표]), ... }
initial_landmarks = {
    'Tree1': np.array([2.0, 8.0]), 'BuildingA': np.array([7.0, 2.0]),
    'Statue': np.array([10.0, 10.0]), 'Bench': np.array([1.0, 1.0]),
    'LampPost': np.array([8.0, 7.0]), 'Fountain': np.array([5.0, 12.0]),
    'Sign': np.array([-2.0, 5.0]), 'Corner': np.array([13.0, 1.0])
}
# landmarks: 현재 시뮬레이션 환경의 장애물 정보. 우클릭으로 여기에 새 장애물이 추가됨 (전역 변수)
landmarks = initial_landmarks.copy() # 초기값 복사하여 사용 시작

# initial_true_position: 로봇(사용자)의 고정된 시작 위치
initial_true_position = np.array([1.0, 1.5])
# initial_target_position: 프로그램 시작 시의 초기 목표 지점
initial_target_position = np.array([12.0, 9.0])
# target_position: 현재 로봇이 도달해야 할 목표 지점. 좌클릭으로 변경됨 (전역 변수)
target_position = initial_target_position.copy()

# 시뮬레이션 파라미터 상세 설정
sensor_range = 5.0             # 로봇 센서가 장애물을 감지할 수 있는 최대 거리
measurement_noise_std = 0.25   # 센서 측정값에 포함될 오차(노이즈)의 표준편차 (값이 클수록 오차 커짐)
max_cycles = 300               # 한 번의 시뮬레이션 실행에서 최대 반복할 횟수 (애니메이션 최대 프레임 수)
step_distance = 0.4            # 로봇이 한 사이클(프레임)당 이동하는 기본 거리
heading_noise_std = 0.15       # 로봇 이동 방향에 추가될 무작위 변화량의 표준편차 (라디안 단위, 클수록 방향 불규칙)
reach_threshold = 0.8          # 로봇의 추정 위치가 목표 지점으로부터 이 거리 안에 들어오면 도달로 간주
steering_gain = 0.3            # 로봇이 목표 지점을 향해 방향을 얼마나 적극적으로 틀지 결정하는 값 (0:영향없음, 1:직진)

# 사용자 인터랙션 키 설정
START_KEY = 's'  # 장애물 추가 후 시뮬레이션 시작/재시작 키
REPEAT_KEY = 'r' # 현재 설정(목표, 장애물)으로 시뮬레이션 재시작 키
QUIT_KEY = 'q'   # 프로그램 종료 키

# --- 전역 변수 선언 ---
# 프로그램 전체에서 상태를 공유하고 업데이트해야 하는 변수들

# Matplotlib 관련 객체
fig = None # 전체 그림(Figure) 객체. 창 하나에 해당.
ax = None  # 그림 안의 실제 플롯 영역(Axes) 객체. 그래프가 그려지는 곳.
ani = None # 현재 실행 중인 애니메이션(FuncAnimation) 객체. 애니메이션 제어에 필요.

# 시뮬레이션 상태 관련
obstacle_counter = 0 # 우클릭으로 추가된 장애물 이름에 붙일 번호 (Obs1, Obs2...)

# 플롯 요소 핸들 (그래프 위의 개별 요소들을 가리키는 참조)
# 이 핸들을 통해 애니메이션 중 요소를 업데이트하거나, 인터랙션으로 상태 변경
landmark_scatter = None       # 모든 랜드마크 마커(점)들을 한 번에 관리하는 객체
landmark_text_handles = []    # 각 랜드마크 이름 텍스트 객체들을 저장하는 리스트 (텍스트 수정/삭제용)
target_scatter = None         # 목표 지점 마커(별) 객체
target_circle_patch = None    # 목표 도달 기준 원(Circle) 객체
line_true = None              # 실제 이동 경로를 그리는 선(Line2D) 객체
line_est = None               # 추정 이동 경로를 그리는 선(Line2D) 객체
point_true = None             # 현재 실제 위치를 나타내는 점(Marker) 객체
point_est = None              # 현재 추정 위치를 나타내는 점(Marker) 객체
cycle_text = None             # 현재 사이클(프레임) 번호를 표시하는 텍스트(Text) 객체
legend_handle = None          # 그래프의 범례(Legend) 객체

# 이벤트 로그 관련
# 사용자의 인터랙션(클릭) 기록을 저장할 pandas DataFrame
event_log_df = pd.DataFrame(columns=['Timestamp', 'EventType', 'X', 'Y', 'Details', 'PrevAvgError'])
# 가장 최근에 완료된 시뮬레이션의 평균 오차 값을 저장할 변수
last_avg_error = np.nan # 초기값은 Not a Number (계산된 적 없음)


# --- 시뮬레이션 핵심 함수들 ---

def simulate_obstacle_detection(current_pos, known_landmarks, range_limit, noise_std):
    """
    주어진 현재 위치(current_pos)에서, 알려진 랜드마크(known_landmarks) 중
    감지 범위(range_limit) 내에 있는 것들을 찾아, 측정 노이즈(noise_std)가
    포함된 상대 위치를 계산하여 딕셔너리로 반환합니다.
    """
    # known_landmarks 인자는 호출 시점에 전역 변수 'landmarks'의 현재 값을 받음
    detected_landmarks = {} # 이번 탐지 사이클에서 발견된 랜드마크 저장용
    for name, landmark_abs_pos in known_landmarks.items(): # 현재 환경의 모든 랜드마크 확인
        distance = np.linalg.norm(landmark_abs_pos - current_pos) # 로봇-랜드마크 거리
        if distance <= range_limit: # 센서 범위 안인가?
            true_relative_pos = landmark_abs_pos - current_pos # 실제 상대 벡터
            noise = np.random.normal(0, noise_std, size=2) # 2D 노이즈 생성
            measured_relative_pos = true_relative_pos + noise # 노이즈 추가된 측정값
            detected_landmarks[name] = measured_relative_pos # 결과 저장
    return detected_landmarks

def estimate_my_position(detected_info, known_landmarks_map):
    """
    탐지된 랜드마크 정보(detected_info)와 전체 랜드마크 맵(known_landmarks_map)을
    이용하여 현재 로봇의 위치를 추정합니다. (간단한 평균법 사용)
    """
    # known_landmarks_map 인자는 호출 시점에 전역 변수 'landmarks'의 현재 값을 받음
    position_estimates = [] # 개별 랜드마크 기반 추정치 저장 리스트
    if not detected_info: return None # 탐지된게 없으면 추정 불가

    for name, measured_relative_pos in detected_info.items(): # 탐지된 랜드마크별로 계산
        if name in known_landmarks_map: # 해당 랜드마크가 맵에 있는지 확인
            landmark_abs_pos = known_landmarks_map[name] # 맵에서 절대 좌표 가져오기
            # 로봇 추정 위치 = 랜드마크 절대 위치 - 측정된 로봇->랜드마크 상대 위치 벡터
            estimated_pos = landmark_abs_pos - measured_relative_pos
            position_estimates.append(estimated_pos)

    if not position_estimates: return None # 유효한 추정치가 없으면 불가

    # 모든 유효한 추정치들의 산술 평균을 최종 추정 위치로 결정
    final_estimated_position = np.mean(position_estimates, axis=0)
    return final_estimated_position

def run_simulation():
    """
    현재 설정된 전역 'target_position'과 'landmarks'를 사용하여
    시뮬레이션을 1회 실행하고, 그 결과를 반환합니다.
    반환값: 실제 경로, 추정 경로, 총 프레임 수, 이번 실행의 평균 경로 오차
    """
    global target_position, landmarks # 전역 변수 사용 명시
    print(f"--- 시뮬레이션 실행 (목표: {target_position.round(2)}, 랜드마크 {len(landmarks)}개) ---")

    # 시뮬레이션 상태 초기화
    current_true_pos = initial_true_position.copy() # 항상 고정된 시작 위치에서 출발
    current_heading = np.arctan2(target_position[1] - current_true_pos[1],
                               target_position[0] - current_true_pos[0]) # 초기 방향은 현재 목표점
    true_path_list = []          # 실제 경로 저장용
    estimated_path_list = []     # 추정 경로 저장용
    last_valid_estimate = current_true_pos.copy() # 가장 최근 유효 추정치 (초기값은 시작점)
    distance_to_target = np.linalg.norm(last_valid_estimate - target_position) # 현재 목표까지 거리
    cycle = 0                    # 사이클 카운터

    # 메인 루프: 목표 도달 또는 최대 사이클까지 반복
    while distance_to_target > reach_threshold and cycle < max_cycles:
        # 1. 현재 실제 위치 기록
        true_path_list.append(current_true_pos.copy())

        # 2. 장애물 탐지 (현재 landmarks 사용)
        detected_info = simulate_obstacle_detection(current_true_pos, landmarks, sensor_range, measurement_noise_std)

        # 3. 위치 추정 (현재 landmarks 사용)
        current_estimated_pos = estimate_my_position(detected_info, landmarks)

        # 4. 추정 결과 처리
        if current_estimated_pos is not None: # 추정 성공
            estimated_path_list.append(current_estimated_pos.copy())
            last_valid_estimate = current_estimated_pos.copy()
            # 추정 위치 기준으로 목표 거리 업데이트
            distance_to_target = np.linalg.norm(last_valid_estimate - target_position)
        else: # 추정 실패
            estimated_path_list.append(np.array([np.nan, np.nan])) # 경로 끊김 표시용 NaN

        # 5. 로봇 이동 (목표 방향 유도 + 랜덤성)
        vector_to_target = target_position - current_true_pos # 목표 방향 벡터
        target_heading = np.arctan2(vector_to_target[1], vector_to_target[0]) # 목표 각도
        heading_error = target_heading - current_heading # 현재 방향과의 오차
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi # 오차 각도 정규화 (-pi ~ pi)
        # 새 방향 = 현재방향 + (목표유도성분) + (랜덤성분)
        current_heading += steering_gain * heading_error + np.random.normal(0, heading_noise_std)
        current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi # 방향 각도 정규화
        # 이동 및 위치 업데이트
        dx = step_distance * np.cos(current_heading)
        dy = step_distance * np.sin(current_heading)
        current_true_pos += np.array([dx, dy])

        cycle += 1 # 사이클 증가

    # 루프 종료 후 마지막 실제 위치 추가
    true_path_list.append(current_true_pos.copy())
    # 리스트들을 NumPy 배열로 변환
    true_path = np.array(true_path_list)
    estimated_path = np.array(estimated_path_list)
    num_frames = cycle # 총 프레임 수 = 실행된 사이클 수

    # *** 이번 실행의 평균 오차 계산 ***
    avg_error_this_run = np.nan # 기본값
    valid_indices = np.where(~np.isnan(estimated_path[:, 0]))[0] # 추정 성공한 프레임 인덱스
    if len(valid_indices) > 0: # 성공한 추정이 있으면
        # 해당 프레임들의 실제 위치와 추정 위치 사이의 거리(오차) 계산
        errors = np.linalg.norm(estimated_path[valid_indices] - true_path[valid_indices], axis=1)
        avg_error_this_run = np.mean(errors) # 평균 오차 계산
        print(f"--- 이번 실행 평균 오차: {avg_error_this_run:.3f} ---")
    else:
        print("--- 이번 실행에서 유효한 추정값이 없어 평균 오차 계산 불가 ---")

    # 결과 요약 출력
    print(f"--- 데이터 생성 완료 ({num_frames} 프레임). ---")
    if distance_to_target <= reach_threshold: print(f"--- 결과: 목표 도달 성공! (거리: {distance_to_target:.3f}) ---")
    else: print(f"--- 결과: 최대 사이클({max_cycles}) 도달. (최종 거리: {distance_to_target:.3f}) ---")

    # 최종 결과 반환
    return true_path, estimated_path, num_frames, avg_error_this_run

def start_new_simulation_and_animation():
    """
    현재 전역 상태(landmarks, target_position)를 기반으로 새 시뮬레이션을 실행하고,
    플롯을 업데이트하며, 새 애니메이션을 시작/재시작합니다.
    이 함수는 프로그램 시작 시, 또는 's', 'r', 좌클릭 이벤트 발생 시 호출됩니다.
    """
    # 전역 변수 사용 선언
    global ani, ax, fig, target_position, landmarks, landmark_text_handles, last_avg_error
    global landmark_scatter, target_scatter, target_circle_patch
    global line_true, line_est, point_true, point_est, cycle_text, legend_handle

    # --- 1. 새 시뮬레이션 실행 및 결과 받기 ---
    true_path, estimated_path, num_frames, avg_error_this_run = run_simulation()
    # 이번 실행의 평균 오차를 전역 변수에 저장 (다음번 로그 기록 시 사용됨)
    last_avg_error = avg_error_this_run

    # --- 2. 플롯 설정 또는 업데이트 ---
    # 현재 landmarks 딕셔너리에서 좌표값들만 추출 (NumPy 배열로)
    landmark_coords = np.array(list(landmarks.values())) if landmarks else np.empty((0, 2))

    # 최초 실행 시 필요한 설정 수행
    if ax is None:
        print("최초 플롯 설정...")
        fig, ax = plt.subplots(figsize=(11, 9)) # Figure, Axes 객체 생성
        # 이벤트 핸들러 연결 (키보드, 마우스 클릭) - 딱 한 번만 연결하면 됨
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('button_press_event', on_click)

        # 플롯 요소들(선, 점, 텍스트 등) 생성 및 핸들 저장 (초기 데이터는 비어있음)
        landmark_scatter = ax.scatter([], [], c='blue', marker='x', s=80, label='랜드마크')
        target_scatter = ax.scatter([], [], c='magenta', marker='*', s=300, label='목표 지점', zorder=7)
        target_circle_patch = plt.Circle([np.nan, np.nan], reach_threshold, color='magenta', fill=False, linestyle=':', lw=1.5, label='목표 기준')
        ax.add_patch(target_circle_patch) # Axes에 원 추가
        line_true, = ax.plot([], [], 'g-', lw=2, alpha=0.7, label='실제 경로')
        line_est, = ax.plot([], [], 'r--', lw=2, alpha=0.7, label='추정 경로')
        point_true, = ax.plot([], [], 'go', markersize=9, label='현재 실제 위치')
        point_est, = ax.plot([], [], 'ro', markersize=9, markerfacecolor='orange', label='현재 추정 위치')
        cycle_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        legend_handle = ax.legend(loc='upper left', fontsize='small') # 범례 생성
        # 축 라벨, 그리드, 축 비율 등 기본 설정
        ax.set_xlabel("X 좌표"); ax.set_ylabel("Y 좌표")
        ax.grid(True); ax.axis('equal')

    # --- 3. (최초 및 반복 공통) 플롯 요소 데이터 업데이트 ---
    # 현재 landmarks 데이터로 랜드마크 마커들 위치 업데이트
    landmark_scatter.set_offsets(landmark_coords)
    # 현재 target_position으로 목표 마커 위치 및 원 중심 업데이트
    target_scatter.set_offsets(target_position)
    target_circle_patch.center = target_position
    # 원 객체가 Axes에 있는지 확인하고 없으면 추가 (보통은 불필요)
    if target_circle_patch not in ax.patches: ax.add_patch(target_circle_patch)

    # *** 랜드마크 텍스트 라벨 업데이트 ***
    # 중요: 이전에 그려진 모든 텍스트 라벨 제거 (메모리 누수 방지 및 정확성)
    for txt in landmark_text_handles:
        txt.remove()
    landmark_text_handles.clear() # 핸들 리스트 비우기
    # 현재 'landmarks' 딕셔너리 기준으로 텍스트 라벨 다시 그림
    for name, pos in landmarks.items():
        txt = ax.text(pos[0] + 0.1, pos[1] + 0.1, name, fontsize=9) # 이름 텍스트 생성
        landmark_text_handles.append(txt) # 새로 생성된 텍스트 핸들 저장

    # 애니메이션 시작 전 동적 요소들(선, 점) 초기 상태로 리셋
    line_true.set_data([], []); line_est.set_data([], [])
    point_true.set_data([], []); point_est.set_data([], [])
    point_est.set_visible(False); cycle_text.set_text('')

    # --- 4. 축 범위 재설정 ---
    # 현재 플롯에 있는 모든 요소(랜드마크, 경로, 목표, 시작점)가 보이도록 축 범위 동적 조절
    valid_estimated_x = estimated_path[~np.isnan(estimated_path[:, 0]), 0]
    valid_estimated_y = estimated_path[~np.isnan(estimated_path[:, 1]), 1]
    all_x_parts = [true_path[:, 0], valid_estimated_x, [target_position[0]], [initial_true_position[0]]]
    all_y_parts = [true_path[:, 1], valid_estimated_y, [target_position[1]], [initial_true_position[1]]]
    if landmark_coords.size > 0: # 랜드마크가 있을 때만 좌표 추가
        all_x_parts.insert(0, landmark_coords[:, 0])
        all_y_parts.insert(0, landmark_coords[:, 1])
    all_x = np.concatenate(all_x_parts) # 모든 x좌표 합치기
    all_y = np.concatenate(all_y_parts) # 모든 y좌표 합치기
    if len(all_x) > 0 : # 좌표 데이터가 있을 경우 범위 계산
        ax.set_xlim(np.nanmin(all_x) - 1.5, np.nanmax(all_x) + 1.5) # 최소/최대값 기준으로 여백두고 설정
        ax.set_ylim(np.nanmin(all_y) - 1.5, np.nanmax(all_y) + 1.5)
    else: # 데이터 없으면 기본 범위
        ax.set_xlim(-5, 15); ax.set_ylim(-5, 15)

    # 플롯 제목 업데이트 (현재 상호작용 방법 포함)
    ax.set_title(f"VPR 시뮬레이션 (좌클릭: 목표 / 우클릭: 장애물 / '{START_KEY}': 시작 / '{REPEAT_KEY}': 재시작 / '{QUIT_KEY}': 종료)")

    # --- 5. 새 애니메이션 함수 정의 ---
    # 이 함수들은 바로 위에서 생성된 true_path, estimated_path 등을 사용 (클로저)
    def init_animation():
        """애니메이션 시작 프레임 설정"""
        # 모든 동적 요소 초기화
        line_true.set_data([], []); line_est.set_data([], [])
        point_true.set_data([], []); point_est.set_data([], [])
        point_est.set_visible(False); cycle_text.set_text('')
        # 업데이트 대상 핸들 반환 (blit=True 사용시)
        return line_true, line_est, point_true, point_est, cycle_text

    def update_animation(frame):
        """애니메이션 각 프레임 업데이트 로직"""
        # 현재 프레임까지의 경로 데이터 설정
        line_true.set_data(true_path[:frame+1, 0], true_path[:frame+1, 1])
        est_segment = estimated_path[:frame+1] # 추정 경로는 NaN 포함 가능
        line_est.set_data(est_segment[:, 0], est_segment[:, 1]) # NaN은 자동으로 끊김
        # 현재 프레임의 실제 위치 마커 이동
        point_true.set_data([true_path[frame, 0]], [true_path[frame, 1]])
        # 현재 프레임의 추정 위치 마커 이동/숨김
        if frame < len(estimated_path) and not np.isnan(estimated_path[frame, 0]): # 추정 성공 시
            point_est.set_data([estimated_path[frame, 0]], [estimated_path[frame, 1]])
            point_est.set_visible(True)
        else: # 추정 실패 시
            point_est.set_visible(False)
        # 사이클 텍스트 업데이트
        cycle_text.set_text(f'Cycle: {frame + 1} / {num_frames}')
        # 업데이트된 핸들 반환
        return line_true, line_est, point_true, point_est, cycle_text

    # --- 6. 새 애니메이션 생성 및 시작 ---
    if num_frames > 0: # 시뮬레이션 결과 프레임이 있으면 애니메이션 생성
        print(f"애니메이션 생성 ({num_frames} 프레임)...")
        # FuncAnimation 객체를 생성하고 전역 변수 'ani'에 저장 (유지 목적)
        ani = FuncAnimation(fig,                  # 애니메이션을 표시할 Figure
                            update_animation,     # 프레임 업데이트 함수
                            frames=num_frames,    # 전체 프레임 수
                            init_func=init_animation, # 초기화 함수
                            interval=75,          # 프레임 간 지연(ms) - 속도
                            blit=True,            # 최적화된 그리기 (권장)
                            repeat=False)         # 반복 안 함
        fig.canvas.draw_idle() # Figure 업데이트 요청 (애니메이션 시작)
    else: # 시뮬레이션 결과 프레임이 없으면
        print("시뮬레이션 0 프레임. 애니메이션 없음.")
        init_animation() # 동적 요소 초기화 (화면 정리)
        fig.canvas.draw_idle()

# --- 이벤트 핸들러 함수들 ---

def on_key_press(event):
    """키보드 입력 이벤트 처리"""
    global ani, target_position, landmarks # 전역 변수 사용
    print(f"입력 키: {event.key}") # 눌린 키 출력
    if event.key == START_KEY: # 's' 키
        print(f"'{START_KEY}' 키 감지! 현재 장애물(총 {len(landmarks)}개) 및 목표({target_position.round(2)})로 시뮬레이션 시작...")
        start_new_simulation_and_animation() # 현재 상태로 시뮬/애니메이션 시작
    elif event.key == REPEAT_KEY: # 'r' 키
        print(f"'{REPEAT_KEY}' 키 감지! 현재 목표({target_position.round(2)}) 및 장애물로 시뮬레이션 재시작...")
        start_new_simulation_and_animation() # 's'와 동일하게 현재 상태로 재시작
    elif event.key == QUIT_KEY: # 'q' 키
        print(f"'{QUIT_KEY}' 키 감지. 종료합니다.")
        plt.close(fig) # Matplotlib 창 닫기 (프로그램 종료로 이어짐)

def on_click(event):
    """마우스 클릭 이벤트 처리 (좌: 목표 설정, 우: 장애물 추가)"""
    # 전역 변수 사용/수정 선언
    global target_position, target_scatter, target_circle_patch, ax
    global landmarks, obstacle_counter, landmark_scatter, landmark_text_handles
    global event_log_df, last_avg_error

    # 클릭이 플롯 영역(Axes) 안에서 발생했는지 확인
    if event.inaxes != ax: return

    # 이벤트 발생 시각 기록
    timestamp = pd.Timestamp.now()

    if event.button == 1: # 마우스 좌클릭 버튼
        # 클릭된 데이터 좌표 (x, y) 얻기
        new_target = np.array([event.xdata, event.ydata])
        print(f"--- 마우스 좌클릭! 새 목표 지점 설정: {new_target.round(2)} ---")
        target_position = new_target # 전역 목표 위치 업데이트

        # 이벤트 로그 기록 (DataFrame에 추가)
        new_log = {'Timestamp': timestamp, 'EventType': 'New Target',
                   'X': new_target[0], 'Y': new_target[1], 'Details': '',
                   'PrevAvgError': last_avg_error} # 직전 실행의 평균 오차 포함
        event_log_df.loc[len(event_log_df)] = new_log
        print(f">> 이벤트 로그 기록됨 (직전 평균 오차: {last_avg_error:.3f}).")

        # 시각적 피드백: 목표 마커, 원 위치 즉시 업데이트
        if target_scatter is not None: target_scatter.set_offsets(target_position)
        if target_circle_patch is not None: target_circle_patch.center = target_position
        # *** 좌클릭 시에는 바로 시뮬레이션/애니메이션 재시작 ***
        start_new_simulation_and_animation()

    elif event.button == 3: # 마우스 우클릭 버튼
        # 클릭된 데이터 좌표 얻기
        new_obstacle_pos = np.array([event.xdata, event.ydata])
        obstacle_counter += 1 # 새 장애물 번호 증가
        new_obstacle_name = f"Obs{obstacle_counter}" # 새 장애물 이름 생성 (Obs1, Obs2...)
        print(f"--- 마우스 우클릭! 새 장애물 '{new_obstacle_name}' 추가: {new_obstacle_pos.round(2)} ---")

        # 전역 landmarks 딕셔너리에 새 장애물 정보 추가
        landmarks[new_obstacle_name] = new_obstacle_pos

        # 이벤트 로그 기록 (DataFrame에 추가)
        new_log = {'Timestamp': timestamp, 'EventType': 'Add Obstacle',
                   'X': new_obstacle_pos[0], 'Y': new_obstacle_pos[1],
                   'Details': new_obstacle_name, # 장애물 이름 포함
                   'PrevAvgError': last_avg_error} # 직전 실행 평균 오차 포함
        event_log_df.loc[len(event_log_df)] = new_log
        print(f">> 이벤트 로그 기록됨 (직전 평균 오차: {last_avg_error:.3f}).")

        # *** 즉시 시각적 피드백 (애니메이션 재시작 없음) ***
        # 1. 모든 랜드마크 마커 위치 업데이트 (새 장애물 포함)
        if landmark_scatter is not None:
            landmark_scatter.set_offsets(np.array(list(landmarks.values())))
        # 2. 새 장애물 텍스트 라벨 즉시 추가 및 핸들 저장
        txt = ax.text(new_obstacle_pos[0] + 0.1, new_obstacle_pos[1] + 0.1, new_obstacle_name, fontsize=9)
        landmark_text_handles.append(txt)
        # 3. 변경 사항 화면에 반영 요청
        fig.canvas.draw_idle()
        # 사용자 안내 메시지
        print(f"'{new_obstacle_name}' 추가됨. 원하면 더 추가하고 '{START_KEY}' 키를 눌러 시뮬레이션을 시작하세요.")

    else: # 그 외 마우스 버튼 (휠 클릭 등) 무시
        print(f"다른 마우스 버튼({event.button}) 클릭은 무시합니다.")


# --- 메인 실행 부분 ---
# 프로그램 시작 시, fig 객체가 없으면(=최초 실행) 시뮬/애니메이션 시작 함수 호출
if fig is None:
     # 이 함수 내부에서 fig, ax 객체 생성 및 이벤트 핸들러 연결이 한 번 이루어짐
     start_new_simulation_and_animation()

# 사용자에게 인터랙션 방법 안내
print(f"\n애니메이션 실행 중. 플롯 창에서 좌클릭: 새 목표 / 우클릭: 장애물 추가 / '{START_KEY}': 시작/재시작 / '{QUIT_KEY}': 종료")

# Matplotlib 이벤트 루프 시작: 창을 보여주고 사용자 입력 대기. 창이 닫힐 때까지 여기서 멈춤.
plt.show()

# --- 프로그램 종료 후 로그 처리 ---
# plt.show()가 끝나면 (창이 닫히면) 이 부분이 실행됨
print("\n--- 최종 상호작용 이벤트 로그 ---")
if not event_log_df.empty: # 로그 DataFrame에 내용이 있으면
    # 전체 내용을 생략 없이 콘솔에 출력
    print(event_log_df.to_string())
    # CSV 파일로 저장 시도
    try:
        # index=False: DataFrame 인덱스 불포함, encoding='utf-8-sig': Excel 호환성 위한 인코딩
        event_log_df.to_csv("vpr_interaction_log.csv", index=False, encoding='utf-8-sig', float_format='%.3f')
        print("\n로그가 'vpr_interaction_log.csv' 파일로 저장되었습니다.")
    except Exception as e: # 파일 저장 중 오류 발생 시
        print(f"\nCSV 파일 저장 실패: {e}")
else: # 로그가 없으면
    print("기록된 상호작용 이벤트가 없습니다.")

print("플롯 창이 닫혔습니다.") # 최종 종료 메시지