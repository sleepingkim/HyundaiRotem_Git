# 필요 라이브러리 설치 (아직 설치되지 않았다면)
# pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time # 사용하지는 않지만, 필요시 추가 가능
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

# --- 환경 설정 및 파라미터 ---
landmarks = {
    'Tree1': np.array([2.0, 8.0]), 'BuildingA': np.array([7.0, 2.0]),
    'Statue': np.array([10.0, 10.0]), 'Bench': np.array([1.0, 1.0]),
    'LampPost': np.array([8.0, 7.0]), 'Fountain': np.array([5.0, 12.0]),
    'Sign': np.array([-2.0, 5.0]), 'Corner': np.array([13.0, 1.0])
}
initial_true_position = np.array([1.0, 1.5])
# target_position은 이제 전역 변수로 관리되며 클릭으로 변경됨
initial_target_position = np.array([12.0, 9.0]) # 초기 목표 지점
target_position = initial_target_position.copy() # 현재 목표 지점 (전역)

sensor_range = 5.0
measurement_noise_std = 0.25
max_cycles = 300       # 시뮬레이션 최대 사이클
step_distance = 0.4
heading_noise_std = 0.15
reach_threshold = 0.8
steering_gain = 0.3
REPEAT_KEY = 'r' # 현재 목표로 재시작 키
QUIT_KEY = 'q'   # 종료 키

# --- 시뮬레이션 함수 (target_position을 전역 변수로 사용) ---
def simulate_obstacle_detection(current_pos, known_landmarks, range_limit, noise_std):
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
    position_estimates = []
    if not detected_info: return None
    for name, measured_relative_pos in detected_info.items():
        if name in known_landmarks_map:
            landmark_abs_pos = known_landmarks_map[name]
            estimated_pos = landmark_abs_pos - measured_relative_pos
            position_estimates.append(estimated_pos)
    if not position_estimates: return None
    final_estimated_position = np.mean(position_estimates, axis=0)
    return final_estimated_position

# --- 전역 변수 선언 ---
fig = None
ax = None
ani = None # 현재 애니메이션 객체
# 플롯 요소 핸들
landmark_scatter = None
target_scatter = None
target_circle_patch = None
line_true = None
line_est = None
point_true = None
point_est = None
cycle_text = None
legend_handle = None

def run_simulation():
    """현재 global target_position을 사용하여 시뮬레이션을 실행하고 경로 데이터를 반환."""
    global target_position # 현재 설정된 목표 지점 사용
    print(f"--- 시뮬레이션 실행 (목표: {target_position.round(2)}) ---")

    current_true_pos = initial_true_position.copy() # 항상 같은 시작점에서 출발
    # 초기 방향은 현재 목표를 향하도록 설정
    current_heading = np.arctan2(target_position[1] - current_true_pos[1],
                               target_position[0] - current_true_pos[0])
    true_path_list = []
    estimated_path_list = []
    last_valid_estimate = current_true_pos.copy()
    # 목표까지의 거리는 현재 target_position 기준
    distance_to_target = np.linalg.norm(last_valid_estimate - target_position)
    cycle = 0

    while distance_to_target > reach_threshold and cycle < max_cycles:
        true_path_list.append(current_true_pos.copy())
        detected_info = simulate_obstacle_detection(current_true_pos, landmarks, sensor_range, measurement_noise_std)
        current_estimated_pos = estimate_my_position(detected_info, landmarks)

        if current_estimated_pos is not None:
            estimated_path_list.append(current_estimated_pos.copy())
            last_valid_estimate = current_estimated_pos.copy()
            # 목표까지 거리 업데이트 (현재 target_position 기준)
            distance_to_target = np.linalg.norm(last_valid_estimate - target_position)
        else:
            estimated_path_list.append(np.array([np.nan, np.nan]))

        # 이동 계산 (현재 target_position 기준)
        vector_to_target = target_position - current_true_pos
        target_heading = np.arctan2(vector_to_target[1], vector_to_target[0])
        heading_error = target_heading - current_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        current_heading += steering_gain * heading_error + np.random.normal(0, heading_noise_std)
        current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi
        dx = step_distance * np.cos(current_heading)
        dy = step_distance * np.sin(current_heading)
        current_true_pos += np.array([dx, dy])
        cycle += 1

    true_path_list.append(current_true_pos.copy())
    true_path = np.array(true_path_list)
    estimated_path = np.array(estimated_path_list)
    num_frames = cycle

    print(f"--- 데이터 생성 완료 ({num_frames} 프레임). ---")
    if distance_to_target <= reach_threshold: print(f"--- 결과: 목표 도달 성공! (거리: {distance_to_target:.3f}) ---")
    else: print(f"--- 결과: 최대 사이클({max_cycles}) 도달. (최종 거리: {distance_to_target:.3f}) ---")
    return true_path, estimated_path, num_frames

def start_new_simulation_and_animation():
    """
    현재 target_position으로 새 시뮬레이션을 실행하고 애니메이션을 시작/재시작합니다.
    """
    global ani, ax, fig, target_position # 전역 변수 사용/수정
    global landmark_scatter, target_scatter, target_circle_patch
    global line_true, line_est, point_true, point_est, cycle_text, legend_handle

    # --- 새 시뮬레이션 실행 ---
    true_path, estimated_path, num_frames = run_simulation()

    # --- 플롯 설정 또는 업데이트 ---
    landmark_coords = np.array(list(landmarks.values()))
    if ax is None: # 최초 실행 시
        print("최초 플롯 설정...")
        fig, ax = plt.subplots(figsize=(11, 9))
        # 이벤트 핸들러 연결 (최초 한 번)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('button_press_event', on_click) # 마우스 클릭 핸들러 연결

        # 플롯 요소 핸들 생성
        landmark_scatter = ax.scatter(landmark_coords[:, 0], landmark_coords[:, 1], c='blue', marker='x', s=80, label='랜드마크')
        target_scatter = ax.scatter([], [], c='magenta', marker='*', s=300, label='목표 지점', zorder=7) # 초기 위치는 아래에서 설정
        target_circle_patch = plt.Circle([np.nan, np.nan], reach_threshold, color='magenta', fill=False, linestyle=':', lw=1.5, label='목표 기준') # 초기 위치 아래 설정
        ax.add_patch(target_circle_patch)
        line_true, = ax.plot([], [], 'g-', lw=2, alpha=0.7, label='실제 경로')
        line_est, = ax.plot([], [], 'r--', lw=2, alpha=0.7, label='추정 경로')
        point_true, = ax.plot([], [], 'go', markersize=9, label='현재 실제 위치')
        point_est, = ax.plot([], [], 'ro', markersize=9, markerfacecolor='orange', label='현재 추정 위치')
        cycle_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        legend_handle = ax.legend(loc='upper left', fontsize='small')
        ax.set_xlabel("X 좌표")
        ax.set_ylabel("Y 좌표")
        ax.grid(True)
        ax.axis('equal')
    # else: # 반복 실행 시 (핸들은 이미 존재)
        # print("기존 플롯 업데이트...")
        # pass # 아래에서 데이터 업데이트 수행

    # --- (최초 및 반복 시 공통) 플롯 요소 데이터 업데이트 ---
    landmark_scatter.set_offsets(landmark_coords) # 랜드마크 위치 업데이트 (보통 불필요)
    target_scatter.set_offsets(target_position)    # 현재 target_position으로 목표 마커 위치 설정
    target_circle_patch.center = target_position   # 현재 target_position으로 원 중심 설정
    if target_circle_patch not in ax.patches:      # 혹시 제거되었다면 다시 추가
        ax.add_patch(target_circle_patch)

    # 동적 요소들 초기 상태로 리셋
    line_true.set_data([], [])
    line_est.set_data([], [])
    point_true.set_data([], [])
    point_est.set_data([], [])
    point_est.set_visible(False)
    cycle_text.set_text('')

    # --- 축 범위 재설정 ---
    valid_estimated_x = estimated_path[~np.isnan(estimated_path[:, 0]), 0]
    valid_estimated_y = estimated_path[~np.isnan(estimated_path[:, 1]), 1]
    all_x = np.concatenate([landmark_coords[:, 0], true_path[:, 0], valid_estimated_x, [target_position[0]], [initial_true_position[0]]]) # 시작점 포함
    all_y = np.concatenate([landmark_coords[:, 1], true_path[:, 1], valid_estimated_y, [target_position[1]], [initial_true_position[1]]]) # 시작점 포함
    if len(all_x) > 0 :
        ax.set_xlim(np.nanmin(all_x) - 1.5, np.nanmax(all_x) + 1.5)
        ax.set_ylim(np.nanmin(all_y) - 1.5, np.nanmax(all_y) + 1.5)
    else:
        ax.set_xlim(-5, 15); ax.set_ylim(-5, 15)

    ax.set_title(f"VPR 시뮬레이션 (클릭: 새 목표 / '{REPEAT_KEY}': 재시작 / '{QUIT_KEY}': 종료)")

    # --- 새 애니메이션 함수 정의 (클로저 이용) ---
    def init_animation():
        line_true.set_data([], [])
        line_est.set_data([], [])
        point_true.set_data([], [])
        point_est.set_data([], [])
        point_est.set_visible(False)
        cycle_text.set_text('')
        return line_true, line_est, point_true, point_est, cycle_text

    def update_animation(frame):
        line_true.set_data(true_path[:frame+1, 0], true_path[:frame+1, 1])
        est_segment = estimated_path[:frame+1]
        line_est.set_data(est_segment[:, 0], est_segment[:, 1])
        point_true.set_data([true_path[frame, 0]], [true_path[frame, 1]])
        if frame < len(estimated_path) and not np.isnan(estimated_path[frame, 0]):
            point_est.set_data([estimated_path[frame, 0]], [estimated_path[frame, 1]])
            point_est.set_visible(True)
        else:
            point_est.set_visible(False)
        cycle_text.set_text(f'Cycle: {frame + 1} / {num_frames}')
        return line_true, line_est, point_true, point_est, cycle_text

    # --- 새 애니메이션 생성 및 시작 ---
    if num_frames > 0:
        print(f"애니메이션 생성 ({num_frames} 프레임)...")
        ani = FuncAnimation(fig, update_animation, frames=num_frames,
                            init_func=init_animation, interval=75, blit=True, repeat=False)
        fig.canvas.draw_idle()
    else:
        print("시뮬레이션 0 프레임. 애니메이션 없음.")
        init_animation() # 동적 요소 클리어
        fig.canvas.draw_idle()

def on_key_press(event):
    """키보드 이벤트 처리"""
    global ani
    print(f"입력 키: {event.key}")
    if event.key == REPEAT_KEY:
        print(f"'{REPEAT_KEY}' 키 감지! 현재 목표({target_position.round(2)})로 시뮬레이션 재시작...")
        start_new_simulation_and_animation() # 현재 목표로 재시작
    elif event.key == QUIT_KEY:
        print(f"'{QUIT_KEY}' 키 감지. 종료합니다.")
        plt.close(fig)

def on_click(event):
    """마우스 클릭 이벤트 처리"""
    global target_position, target_scatter, target_circle_patch, ax

    # Axes 안쪽을 좌클릭했을 때만 처리
    if event.inaxes != ax: return
    if event.button == 1: # 1: 좌클릭, 2: 가운데, 3: 우클릭
        new_target = np.array([event.xdata, event.ydata])
        print(f"--- 마우스 클릭! 새 목표 지점 설정: {new_target.round(2)} ---")
        target_position = new_target # 전역 목표 지점 업데이트

        # 즉시 시각적 피드백: 목표 마커와 원 업데이트
        if target_scatter is not None:
            target_scatter.set_offsets(target_position)
        if target_circle_patch is not None:
            target_circle_patch.center = target_position
        # fig.canvas.draw_idle() # 즉시 반영 (선택적, 아래 재시작 시 어차피 업데이트됨)

        # 새 목표로 시뮬레이션 및 애니메이션 재시작
        start_new_simulation_and_animation()
    else:
        print(f"다른 마우스 버튼({event.button}) 클릭은 무시합니다.")


# --- 메인 실행 부분 ---
# 최초 Figure, Axes 생성 (이때 핸들러 연결 안 함)
if fig is None:
     # fig, ax 전역 변수에 할당됨
     start_new_simulation_and_animation() # 여기서 최초 설정 및 핸들러 연결 수행

# 핸들러 연결은 start_new_simulation_and_animation 내부에서 최초 실행 시 한 번만 됨

print(f"\n애니메이션 실행 중. 플롯 창에서 마우스 좌클릭: 새 목표 설정 / '{REPEAT_KEY}': 현재 목표로 재시작 / '{QUIT_KEY}': 종료")
plt.show() # Matplotlib 이벤트 루프 시작

print("플롯 창이 닫혔습니다.")