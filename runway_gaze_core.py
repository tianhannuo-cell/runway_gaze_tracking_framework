# -*- coding: utf-8 -*-
"""
眼动追踪校准与实时实验系统（Eye-tracking calibration and real-time experimental system）

This script integrates a complete eye-tracking application workflow:
1. Calibration: Displays 30 target points sequentially on the full-screen interface to 
   guide the user's gaze, and simultaneously records the pairing data of 
   the user's gaze angle (theta, phi) and screen coordinates (x, y).

2. Training: Using the data collected in the calibration phase, 
   trains a mapping function based on the model (homophoria transformation) selected 
   by the user via command line.

3. Experiment: Using the trained model, converts the user's gaze angle into screen coordinates 
   in real time and displays it on the screen as a visible cursor, 
   achieving the effect of "eye-tracking mouse."

Running Example:
- Using the homography transformation model:
  python gaze_calibration_system.py --model homography
  
"""
import argparse
import os
import sys
import time
from collections import deque
import seaborn as sns
import coloredlogs
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from PIL import Image, ImageDraw, ImageFont
from datasources import Webcam
from models import ELG
from draw import analyze_and_plot

def parse_model_ratio(s):
    """
    The  model_ratio  parameter can be parsed in two forms:
    1) '1x2x3'                      -> [[1, 2, 3]]
    2) '1x2x3,2x3x4,1x1x2'          -> [[1, 2, 3], [2, 3, 4], [1, 1, 2]]

    Each group of three numbers represents, in order: Head: Torso: Legs
    """
    import argparse

    ratio_list = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        nums = [float(item) for item in part.split('x')]
        if len(nums) != 3:
            raise argparse.ArgumentTypeError(
                "Each model_ratio must be in the form of 'aXbXc', such as '1x2x3' or '1x2x3,2x3x4'."
            )
        ratio_list.append(nums)

    if not ratio_list:
        raise argparse.ArgumentTypeError("model_ratio 参数不能为空")

    return ratio_list

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

def app_path():
    if getattr(sys, 'frozen', False):
        # If the program is packaged
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))

def get_primary_screen_size():
    return 2560, 1600

# --- Filter Interface and Implementation ---
class GazeFilter:
    """Filter base class interface"""

    def filter(self, point, timestamp):
        raise NotImplementedError


class NoFilter(GazeFilter):
    """Returning directly to the origin without using any filters"""

    def filter(self, point, timestamp):
        return point


class SMAFilter(GazeFilter):
    """Simple Moving Average"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.points = deque(maxlen=window_size)

    def filter(self, point, timestamp):
        self.points.append(point)
        return np.mean(self.points, axis=0)


class WMAFilter(GazeFilter):
    """Exponential Moving Average"""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last_filtered_point = None

    def filter(self, point, timestamp):
        if self.last_filtered_point is None:
            self.last_filtered_point = point
        else:
            self.last_filtered_point = self.alpha * point + (1 - self.alpha) * self.last_filtered_point
        return self.last_filtered_point


class KalmanFilter(GazeFilter):
    """Kalman filter"""

    def __init__(self):
        # State vector [x, y, vx, vy] (position and velocity)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # Process noise; the larger the value, the less confidence there is in the model's predictions, 
        # and the faster but less smooth the response
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        # Measurement noise: the higher the value, the less trust there is in the measured values, 
        # and the stronger the smoothing effect
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 5
        self.initialized = False

    def filter(self, point, timestamp):
        if not self.initialized:
            # Initialization state
            self.kalman.statePost = np.array([point[0], point[1], 0, 0], np.float32)
            self.initialized = True
            return point

        # Forecast and Update
        self.kalman.predict()
        measurement = np.array([[point[0]], [point[1]]], np.float32)
        self.kalman.correct(measurement)

        # Return to the smoothed position
        return self.kalman.statePost[:2].flatten()


class OneEuroFilter(GazeFilter):
    def __init__(self, min_cutoff= 0.2 , beta=0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # Create persistent filter states for position and derivative
        self.x_filter = self._create_low_pass_filter()
        self.y_filter = self._create_low_pass_filter()
        self.dx_filter = self._create_low_pass_filter()
        self.dy_filter = self._create_low_pass_filter()
        self.last_timestamp = None

    def _create_low_pass_filter(self):
        # 'x_prev' stores the previous raw value, 'hat_x_prev' stores the previous filtered value
        return {'hat_x_prev': None, 'x_prev': None}

    def _low_pass(self, filt, x, alpha):
        if filt['hat_x_prev'] is None:
            hat_x = x
        else:
            hat_x = alpha * x + (1.0 - alpha) * filt['hat_x_prev']
        filt['hat_x_prev'] = hat_x
        filt['x_prev'] = x
        return hat_x

    def filter(self, point, timestamp):
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            # Completely initialize all states
            self.x_filter['hat_x_prev'] = point[0]
            self.x_filter['x_prev'] = point[0]
            self.y_filter['hat_x_prev'] = point[1]
            self.y_filter['x_prev'] = point[1]
            return point

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        if dt < 1e-6:  # To prevent division by zero, return to the previous filtered value
            return np.array([self.x_filter['hat_x_prev'], self.y_filter['hat_x_prev']])

        # --- X coordinate ---
        dx = (point[0] - self.x_filter['x_prev']) / dt
        edx_alpha = 1.0 / (1.0 + (2 * np.pi * self.d_cutoff * dt) ** -1)
        edx = self._low_pass(self.dx_filter, dx, edx_alpha)

        cutoff_x = self.min_cutoff + self.beta * abs(edx)
        alpha_x = 1.0 / (1.0 + (2 * np.pi * cutoff_x * dt) ** -1)
        filtered_x = self._low_pass(self.x_filter, point[0], alpha_x)

        # --- Y coordinate ---
        dy = (point[1] - self.y_filter['x_prev']) / dt
        edy_alpha = 1.0 / (1.0 + (2 * np.pi * self.d_cutoff * dt) ** -1)
        edy = self._low_pass(self.dy_filter, dy, edy_alpha)

        cutoff_y = self.min_cutoff + self.beta * abs(edy)
        alpha_y = 1.0 / (1.0 + (2 * np.pi * cutoff_y * dt) ** -1)
        filtered_y = self._low_pass(self.y_filter, point[1], alpha_y)

        return np.array([filtered_x, filtered_y])


# --- Mapping Model Interface and Implementation ---
class GazeMapper:

    def train(self, source_points, dest_points):
        raise NotImplementedError

    def predict(self, gaze_angle):
        raise NotImplementedError


class HomographyMapper(GazeMapper):

    def __init__(self):
        self.homography_matrix = None

    def train(self, source_points, dest_points):
        # FindHomography requires at least 4 points.
        if len(source_points) < 4:
            raise ValueError("单应性变换需要至少4个校准点。")
        self.homography_matrix, _ = cv2.findHomography(source_points, dest_points)
        if self.homography_matrix is None:
            raise RuntimeError("无法计算单应性矩阵，请检查校准点是否共线。")

    def predict(self, gaze_angle):
        if self.homography_matrix is None:
            return None
        gaze_point_reshaped = np.array([[gaze_angle]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(gaze_point_reshaped, self.homography_matrix)
        return transformed_point[0][0]

class PolynomialMapper(GazeMapper):

    def __init__(self, degree=2):
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.regressor = Ridge(alpha=0.5)

    def train(self, source_points, dest_points):
        source_poly = self.poly_features.fit_transform(source_points)
        self.regressor.fit(source_poly, dest_points)

    def predict(self, gaze_angle):
        gaze_poly = self.poly_features.transform([gaze_angle])
        return self.regressor.predict(gaze_poly)[0]


class SVRMapper(GazeMapper):

    def __init__(self):
        self.multi_regressor = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))

    def train(self, source_points, dest_points):
        self.multi_regressor.fit(source_points, dest_points)

    def predict(self, gaze_angle):
        return self.multi_regressor.predict([gaze_angle])[0]


# --- Main system class ---

class GazeCalibrationSystem:
    def __init__(self, args):

        coloredlogs.install(
            datefmt='%d/%m %H:%M',
            fmt='%(asctime)s %(levelname)s %(message)s',
            level=args.v.upper(),
        )

        self.args = args
        self.screen_w, self.screen_h = args.screen_size
        self.window_name = "Gaze Calibration System"

        # Calibration data
        rows = self.args.calibrationNums[0]
        cols = self.args.calibrationNums[1]
        self.calibration_targets = self._generate_calibration_targets(rows, cols)
        self.calibration_data = []
        self.experiment_points = []

        # Initialize the model
        self.gaze_mapper = self._get_mapper(args.model)
        self.gaze_filter = self._get_filter(args.filter)
        self.elg_model = None
        self.data_source = None
        self.tf_session = None

        self.radius = 105
        self.model = [[(300,20), (510,230)],[(225,230), (585,537)],[(260,537),(550,1237)]]
        try:
            font_path = resource_path(os.path.join('data','fonts','simhei.ttf'))
            self.font = ImageFont.truetype(font_path, 40, encoding="utf-8")

        except IOError:
            print("警告: 未找到中文字体'simhei.ttf'，回退到默认字体，中文可能无法正常显示。")
            print("请将有效的中文字体（如simhei.ttf, msyh.ttf）放置在脚本可以访问的路径，或修改脚本中的字体路径。")
            self.font = ImageFont.load_default()
    def _get_mapper(self, model_name):
        if model_name == 'homography':
            return HomographyMapper()
        elif model_name == 'polynomial':
            return PolynomialMapper()
        elif model_name == 'svr':
            return SVRMapper()
        else:
            raise ValueError(f"未知的模型: {model_name}")

    def _get_filter(self, filter_name):
        if filter_name == 'sma': return SMAFilter()
        elif filter_name == 'wma': return WMAFilter()
        elif filter_name == 'kalman': return KalmanFilter()
        elif filter_name == 'one_euro': return OneEuroFilter()
        else: return NoFilter()

    def _generate_calibration_targets(self, rows=5, cols=4):
        """Generate calibration points evenly distributed on the screen"""
        targets = []
        x_margin = self.screen_w // (cols * 2)
        y_margin = self.screen_h // (rows * 2)

        x_points = np.linspace(x_margin, self.screen_w - x_margin, cols, dtype=int)
        y_points = np.linspace(y_margin, self.screen_h - y_margin, rows, dtype=int)

        for y in y_points:
            for x in x_points:
                targets.append((x, y))
        return targets

    def _draw_text(self, frame, text, position='center', color=(255, 255, 255)):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Calculate text size and position
        try:
            # Pillow >= 9.2.0
            text_bbox = draw.textbbox((0, 0), text, font=self.font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Pillow < 9.2.0
            text_w, text_h = draw.textsize(text, font=self.font)

        if position == 'center':
            pos = ((self.screen_w - text_w) // 2, (self.screen_h - text_h) // 2)
        elif position == 'top_center':
            pos = ((self.screen_w - text_w) // 2, 100)
        else:
            pos = position

        draw.text(pos, text, font=self.font, fill=color)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _get_gaze_angle(self, inference_generator):
        output = next(inference_generator)
        eye_index = 0

        # Extract key points and radius
        eye_landmarks = output['landmarks'][eye_index, :]
        eye_radius = output['radius'][eye_index][0]

        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        # Transform to the original image coordinate system
        frame_index = output['frame_index'][eye_index]
        eye_data = self.data_source._frames[frame_index]['eyes'][eye_index]

        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
        eye_landmarks = (eye_landmarks * eye_data['inv_landmarks_transform_mat'].T)[:, :2]
        eye_landmarks = np.asarray(eye_landmarks)

        iris_centre = eye_landmarks[16, :]
        eyeball_centre = eye_landmarks[17, :]

        # Calculate the angle. 
        i_x0, i_y0 = iris_centre
        e_x0, e_y0 = eyeball_centre

        # Ensure eyeball_radius is not 0
        eyeball_radius = np.linalg.norm(eye_landmarks[18, :] - eyeball_centre)
        if eyeball_radius < 1e-6:
            return None

        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))

        return np.array([theta, phi])

    def setup(self):
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.tf_session = tf.Session(config=session_config)

        self.data_source = Webcam(tensorflow_session=self.tf_session, batch_size=2,
                                  camera_id=self.args.camera_id, fps=30,
                                  data_format='NHWC', eye_image_shape=(36, 60))

        # Initialize the model
        self.elg_model = ELG(
            self.tf_session, train_data={'videostream': self.data_source},
            first_layer_stride=1, num_modules=2, num_feature_maps=32,
            learning_schedule=[{'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']}}]
        )
        self.elg_model.initialize_if_not(training=False)
        self.elg_model.checkpoint.load_all()

    def run_calibration(self):
        gen = self.elg_model.inference_generator()
        for i, target in enumerate(self.calibration_targets):
            state = "WAITING_FOR_TRIGGER"
            samples = []

            while True:
                current_gaze = self._get_gaze_angle(gen)
                frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

                if state == "WAITING_FOR_TRIGGER":
                    # --- Waiting phase ---
                    msg = f"请注视白点, 按下<空格键>开始采集 ({i + 1}/{len(self.calibration_targets)})"
                    frame = self._draw_text(frame, msg, 'top_center')
                    # Draw target points
                    cv2.circle(frame, target, 20, (255, 255, 255), -1)
                    cv2.circle(frame, target, 7, (0, 0, 0), -1)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:
                        state = "COLLECTING"
                        collection_start_time = time.time()
                    elif key == ord('q'):
                        # print("The user interrupted the calibration.")
                        return False

                elif state == "COLLECTING":
                    # --- Collection phase ---
                    elapsed = time.time() - collection_start_time
                    if elapsed < 2:
                        frame = self._draw_text(frame, "请保持注视...", 'top_center')
                        # Draw the target point (green) to indicate that data is being collected
                        cv2.circle(frame, target, 20, (0, 255, 0), -1)
                        cv2.circle(frame, target, 7, (0, 0, 0), -1)
                        if current_gaze is not None:
                            samples.append(current_gaze)
                    else:
                        # Data collection complete, exit inner loop
                        break

                cv2.imshow(self.window_name, frame)

            if not samples:
                # print(f"Warning: Calibration point {i + 1} failed to acquire valid line-of-sight data.")
                continue

            avg_gaze = np.mean(samples, axis=0)
            self.calibration_data.append((avg_gaze, target))
            # print(f"Calibration point {i + 1} has been collected.")
            # print(f"Calibration point {i + 1} data acquisition complete. Average line-of-sight angle: {avg_gaze}, Target position: {target}")

        if len(self.calibration_data) < 4:
            print("错误: 有效校准点太少，无法继续。")
            return False
        return True

    def train_model(self):
        """Training the selected mapping model"""
        frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        self._draw_text(frame, "校准完成，正在训练模型...")
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(100)

        source_points = np.array([data[0] for data in self.calibration_data])
        dest_points = np.array([data[1] for data in self.calibration_data])

        try:
            self.gaze_mapper.train(source_points, dest_points)
        except Exception as e:
            print(f"模型训练失败: {e}")
            return False

        time.sleep(2)
        return True

    def run_experiment(self):
        """Running real-time eye-tracking experiments"""
        inference_generator = self.elg_model.inference_generator()

        #  Real-time detection -- Camera mode
        if self.args.csv_path is None:
            while True:
                frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                current_gaze = self._get_gaze_angle(inference_generator)
                if current_gaze is not None:
                    raw_predicted_pos = self.gaze_mapper.predict(current_gaze)
                    if raw_predicted_pos is not None:
                        timestamp = time.time()
                        filter_pos = self.gaze_filter.filter(raw_predicted_pos, timestamp)
                        px, py = filter_pos

                        if px > 0 and py > 0:
                            self.experiment_points.append(filter_pos)

                        px = int(np.clip(px, 0, self.screen_w - 1))
                        py = int(np.clip(py, 0, self.screen_h - 1))
                        cv2.circle(frame, (px, py), 25, (0, 180, 255), -1)
                        cv2.circle(frame, (px, py), 10, (255, 255, 255), -1)

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Offline detection -- Video mode
        else:
            csv_file = pd.read_csv(self.args.csv_path)
            theta = csv_file.iloc[:, 2]
            phi = csv_file.iloc[:, 3]
            for i in range(len(theta)):
                raw_predicted_pos = self.gaze_mapper.predict(np.array([theta[i], phi[i]]))
                if raw_predicted_pos is not None:
                    timestamp = time.time()
                    filter_pos = self.gaze_filter.filter(raw_predicted_pos, timestamp)
                    px, py = filter_pos
                    if px > 0 and py > 0:
                        self.experiment_points.append(filter_pos)

    def _generate_dummy_scene_image(self):
        if not os.path.exists(self.args.scene_image):
            img = np.full((1237, 800, 3), (240, 240, 240), dtype=np.uint8)
            cv2.circle(img, (self.model[0][0][0] + self.radius, self.model[0][1][1] - self.radius), self.radius,(147,112,219) , -1)  # Head
            cv2.rectangle(img, self.model[1][0], self.model[1][1], (173, 216, 230), -1) # Torso
            cv2.rectangle(img, self.model[2][0], self.model[2][1], (220, 220, 240), -1) # Legs
            cv2.imwrite(self.args.scene_image, img)


    def _generate_multi_dummy_scene_image(self):
        img_w, img_h = self.screen_w, self.screen_h
        img = np.full((img_h, img_w, 3), (240, 240, 240), dtype=np.uint8)

        # Number of humanoids
        model_num = max(1, int(getattr(self.args, "model_num", 1)))

        # Parse model_ratio:
        # Compatible with two formats:
        # 1) [1, 2, 3]                  (single humanoid figure)
        # 2) [[1, 2, 3], [2, 3, 4], ...] (a ratio for each humanoid figure)
        model_ratios = getattr(self.args, "model_ratio", [[1, 2, 3]])
        if isinstance(model_ratios, list) and model_ratios and isinstance(model_ratios[0], int):
            base_ratio = model_ratios
            model_ratios = [base_ratio[:] for _ in range(model_num)]

        if len(model_ratios) < model_num:
            last = model_ratios[-1]
            model_ratios = model_ratios + [last[:] for _ in range(model_num - len(model_ratios))]
        model_ratios = model_ratios[:model_num]

        top_y = 0.0

        head_regions = []
        torso_regions = []
        legs_regions = []

        segment_width = img_w / float(model_num)

        for idx in range(model_num):
            person_height = self.args.model_height[idx]
            top_y = img_h - person_height
            ratio = model_ratios[idx]
            if len(ratio) != 3:
                # Revert to default if the ratio configuration is invalid
                ratio = [1, 2, 3]

            r_head, r_torso, r_legs = ratio
            ratio_sum = float(r_head + r_torso + r_legs) or 1.0

            head_h = person_height * (r_head / ratio_sum)
            torso_h = person_height * (r_torso / ratio_sum)
            legs_h = person_height * (r_legs / ratio_sum)

            cx = int((idx + 0.5) * segment_width)

            person_width = segment_width * 0.40
            half_w = int(person_width / 2.0)

            head_top = int(top_y)
            head_bottom = int(top_y + head_h)
            torso_top = head_bottom
            torso_bottom = int(torso_top + torso_h)
            legs_top = torso_bottom
            legs_bottom = int(legs_top + legs_h)

            def clamp_y(y):
                return max(0, min(img_h - 1, int(y)))

            head_top = clamp_y(head_top)
            head_bottom = clamp_y(head_bottom)
            torso_top = clamp_y(torso_top)
            torso_bottom = clamp_y(torso_bottom)
            legs_top = clamp_y(legs_top)
            legs_bottom = clamp_y(legs_bottom)

            left = max(0, int(cx - half_w))
            right = min(img_w - 1, int(cx + half_w))

            # AOI rectangle
            head_rect = ((left, head_top), (right, head_bottom))
            torso_rect = ((left, torso_top), (right, torso_bottom))
            legs_rect = ((left, legs_top), (right, legs_bottom))

            head_regions.append(head_rect)
            torso_regions.append(torso_rect)
            legs_regions.append(legs_rect)

            head_center_y = int((head_top + head_bottom) / 2)
            head_radius = max(5, int((head_bottom - head_top) / 2))
            cv2.circle(img, (cx, head_center_y), head_radius, (147, 112, 219), -1)

            cv2.rectangle(img, (left, torso_top), (right, torso_bottom), (173, 216, 230), -1)  # Torso
            cv2.rectangle(img, (left, legs_top), (right, legs_bottom), (220, 220, 240), -1)  # Legs

        def merge_regions(regions):
            if not regions:
                return ((0, 0), (0, 0))
            x1 = min(r[0][0] for r in regions)
            y1 = min(r[0][1] for r in regions)
            x2 = max(r[1][0] for r in regions)
            y2 = max(r[1][1] for r in regions)
            return (x1, y1), (x2, y2)

        self.model = [
            merge_regions(head_regions),
            merge_regions(torso_regions),
            merge_regions(legs_regions)
        ]

        cv2.imwrite(self.args.scene_image, img)

    def save_and_analyze_data(self):
        """Save experimental data to CSV and generate heatmap analysis"""
        # 1. Save data
        output_filename = (os.path.join(app_path(), f'{self.args.figName}_gaze_points_output.csv'))
        df = pd.DataFrame(self.experiment_points, columns=['x_coord', 'y_coord'])
        df.to_csv(output_filename, index=False)

        # 2. Draw a scatter plot
        analyze_and_plot(output_filename,self.args.figName,x_min =self.screen_w,x_max = self.screen_h)

        # 3. Generate heatmap
        self._generate_multi_scene_analysis_report(df)

    def _generate_scene_analysis_report(self, df):
        try:
            bg_image = cv2.imread(self.args.scene_image)
            # bg_image = cv2.imdecode(np.fromfile(self.args.scene_image, dtype=np.uint8), -1)
            if bg_image is None: raise FileNotFoundError
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = bg_image.shape
        except FileNotFoundError:
            print(f"错误: 无法加载场景图片: {self.args.scene_image}")
            return

        points = df[['x_coord', 'y_coord']].values

        # # --- 2. Conversion from screen coordinates to image pixel coordinates ---
        screen_w, screen_h = self.args.screen_size

        image_coords = points / np.array([screen_w, screen_h]) * np.array([img_w, img_h])

        # Filter out points that fall into the black border area
        valid_mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] <= img_w) & \
                     (image_coords[:, 1] >= 0) & (image_coords[:, 1] <= img_h)
        image_coords_valid = image_coords[valid_mask]

        # DBSCAN noise reduction
        db = DBSCAN(eps=100, min_samples=20).fit(image_coords_valid)
        cleaned_points = image_coords_valid[db.labels_ != -1]

        # Define AOI (px)
        aois = {
            'head': self.model[0],
            'body': self.model[1],
            'legs': self.model[2]
        }

        # Calculate AOI hit rate
        aoi_counts = {name: 0 for name in aois}
        total_valid_points = len(cleaned_points)
        for p in cleaned_points:
            for name, (p1, p2) in aois.items():
                if p1[0] <= p[0] <= p2[0] and p1[1] <= p[1] <= p2[1]:
                    aoi_counts[name] += 1
                    break

        aoi_percentages = {name: (count / total_valid_points) * 100 if total_valid_points > 0 else 0 for name, count in
                           aoi_counts.items()}

        # --- Drawing ---
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={'width_ratios': [img_w, 400]})
        fig.suptitle('Report', fontsize=20)

        # Heat map overlay
        ax1.imshow(bg_image, extent=[0, img_w, img_h, 0])
        sns.kdeplot(x=cleaned_points[:, 0], y=cleaned_points[:, 1], cmap="rocket_r",
                    fill=True, thresh=0.05, alpha=0.5, ax=ax1)

        for name, (p1, p2) in aois.items():
            rect = plt.Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1], linewidth=2, edgecolor='cyan', facecolor='none',
                                 alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(p1[0] + 5, p1[1] + 25, f'{name}\n{aoi_percentages[name]:.1f}%', color='cyan', fontsize=12,
                     weight='bold')

        ax1.set_title('Gaze Heatmap and Area of Interest (AOI)')
        ax1.set_xlabel('Horizontal pixel coordinates of the image (X)')
        ax1.set_ylabel('Vertical pixel coordinates of the image (Y)')
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0)
        ax1.set_aspect('equal', adjustable='box')

        # Bar chart of AOI analysis
        names = list(aoi_percentages.keys())
        percentages = list(aoi_percentages.values())
        ax2.barh(names, percentages, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_title('Percentage of Gaze Time Spent in Areas of Interest (AOI)')
        ax2.set_xlabel('Percentage of gaze time (%)')
        ax2.set_xlim(0, 100)
        for index, value in enumerate(percentages):
            ax2.text(value + 1, index, f'{value:.1f}%')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        outFig = (os.path.join(app_path(),f'{self.args.figName}_attention_result.png'))
        print("generate attention_result.png")
        plt.savefig(outFig, dpi=300)

    def _generate_multi_scene_analysis_report(self, df):
        try:
            bg_image = cv2.imread(self.args.scene_image)
            if bg_image is None:
                raise FileNotFoundError
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = bg_image.shape
        except FileNotFoundError:
            print(f"错误: 无法加载场景图片: {self.args.scene_image}")
            return

        # 1. Convert screen coordinates to scene image pixel coordinates
        points = df[['x_coord', 'y_coord']].values
        screen_w, screen_h = self.args.screen_size
        image_coords = points / np.array([screen_w, screen_h]) * np.array([img_w, img_h])

        # Only retain points falling within the image area
        valid_mask = (
                (image_coords[:, 0] >= 0) & (image_coords[:, 0] <= img_w) &
                (image_coords[:, 1] >= 0) & (image_coords[:, 1] <= img_h)
        )
        image_coords_valid = image_coords[valid_mask]

        if image_coords_valid.shape[0] < 2:
            print("有效注视点过少，无法生成热力图与AOI分析。")
            return

        # 2. DBSCAN Denoising
        db = DBSCAN(eps=100, min_samples=20).fit(image_coords_valid)
        cleaned_points = image_coords_valid[db.labels_ != -1]
        if cleaned_points.shape[0] < 2:
            cleaned_points = image_coords_valid

        # 3. Recalculate the AOI based on model_num / model_ratio
        model_num = max(1, int(getattr(self.args, "model_num", 1)))
        model_ratios = getattr(self.args, "model_ratio", [[1, 2, 3]])

        # --model_ratio 1x2x3` will result in `[1,2,3]
        if isinstance(model_ratios, list) and model_ratios and isinstance(model_ratios[0], int):
            base_ratio = model_ratios
            model_ratios = [base_ratio[:] for _ in range(model_num)]

        if len(model_ratios) < model_num:
            last = model_ratios[-1]
            model_ratios = model_ratios + [last[:] for _ in range(model_num - len(model_ratios))]
        model_ratios = model_ratios[:model_num]

        person_height = float(img_h)
        top_y = 0.0

        # Horizontally equidistant
        segment_width = img_w / float(model_num)

        head_regions = []
        torso_regions = []
        legs_regions = []

        for idx in range(model_num):
            ratio = model_ratios[idx]
            if len(ratio) != 3:
                ratio = [1, 2, 3]

            r_head, r_torso, r_legs = ratio
            ratio_sum = float(r_head + r_torso + r_legs) or 1.0

            head_h = person_height * (r_head / ratio_sum)
            torso_h = person_height * (r_torso / ratio_sum)
            legs_h = person_height * (r_legs / ratio_sum)

            cx = int((idx + 0.5) * segment_width)
            person_width = segment_width * 0.7
            person_width = np.clip(person_width, 270, 500)
            half_w = int(person_width / 2.0)

            head_top = int(top_y)
            head_bottom = int(top_y + head_h)
            torso_top = head_bottom
            torso_bottom = int(torso_top + torso_h)
            legs_top = torso_bottom
            legs_bottom = int(legs_top + legs_h)

            def clamp_y(y):
                return max(0, min(img_h - 1, int(y)))

            head_top = clamp_y(head_top)
            head_bottom = clamp_y(head_bottom)
            torso_top = clamp_y(torso_top)
            torso_bottom = clamp_y(torso_bottom)
            legs_top = clamp_y(legs_top)
            legs_bottom = clamp_y(legs_bottom)

            left = max(0, int(cx - half_w))
            right = min(img_w - 1, int(cx + half_w))

            head_regions.append(((left, head_top), (right, head_bottom)))
            torso_regions.append(((left, torso_top), (right, torso_bottom)))
            legs_regions.append(((left, legs_top), (right, legs_bottom)))

        def merge_regions(regions):
            if not regions:
                return (0, 0), (0, 0)
            x1 = min(r[0][0] for r in regions)
            y1 = min(r[0][1] for r in regions)
            x2 = max(r[1][0] for r in regions)
            y2 = max(r[1][1] for r in regions)
            return (x1, y1), (x2, y2)

        aois = {
            'Head': merge_regions(head_regions),
            'Body': merge_regions(torso_regions),
            'Legs': merge_regions(legs_regions)
        }

        # 4. Calculate AOI hit rate
        aoi_counts = {name: 0 for name in aois}
        total_valid_points = len(cleaned_points)

        for p in cleaned_points:
            for name, (p1, p2) in aois.items():
                if p1[0] <= p[0] <= p2[0] and p1[1] <= p[1] <= p2[1]:
                    aoi_counts[name] += 1
                    break

        aoi_percentages = {
            name: (count / total_valid_points) * 100 if total_valid_points > 0 else 0
            for name, count in aoi_counts.items()
        }

        # 5. Drawing
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 10),
            gridspec_kw={'width_ratios': [img_w, 400]}
        )
        fig.suptitle('Report', fontsize=20)

        # Heat map overlay
        ax1.imshow(bg_image, extent=[0, img_w, img_h, 0])  # Y 轴向下
        sns.kdeplot(
            x=cleaned_points[:, 0],
            y=cleaned_points[:, 1],
            cmap="rocket_r",
            fill=True,
            thresh=0.05,
            alpha=0.5,
            ax=ax1
        )

        ax1.set_title('Gaze Heatmap and Area of Interest (AOI)')
        ax1.set_xlabel('Horizontal pixel coordinates of the image (X)')
        ax1.set_ylabel('Vertical pixel coordinates of the image (Y)')
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0)
        ax1.set_aspect('equal', adjustable='box')

        # AOI Analysis Bar Chart
        names = list(aoi_percentages.keys())
        percentages = list(aoi_percentages.values())
        ax2.barh(names, percentages, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_title('Percentage of Gaze Time Spent in Areas of Interest (AOI)')
        ax2.set_xlabel('Percentage of gaze time (%)')
        ax2.set_xlim(0, 100)
        for index, value in enumerate(percentages):
            ax2.text(value + 1, index, f'{value:.1f}%')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # outFig = os.path.join(app_path(), 'attention_result.png')
        outFig = (os.path.join(app_path(), f'{self.args.figName}_attention_result.png'))
        plt.savefig(outFig, dpi=300)

    def cleanup(self):
        """Clean up resources"""
        if self.data_source:
            self.data_source.cleanup()
        if self.tf_session:
            self.tf_session.close()
        cv2.destroyAllWindows()

    def run(self):
        """Execute the complete process"""
        try:
            self._generate_multi_dummy_scene_image()

            self.setup()

            # Phase 1: Calibration
            if not self.run_calibration():
                return

            # Phase 2: Training
            if not self.train_model():
                return

            # Phase 3: Experiment
            self.run_experiment()

            # Phase 4: Save data and draw heatmap
            self.save_and_analyze_data()

        finally:
            self.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='眼动追踪校准与实时实验系统')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--model', type=str, default='homography',
                        choices=['homography', 'polynomial', 'svr'],
                        help='选择用于视线到屏幕坐标映射的模型。')
    parser.add_argument('--screen_size', type=lambda s: [int(item) for item in s.split('x')],
                        default=[2560,1600],
                        help="您的主显示器分辨率, 格式: '宽x高' (例如 '1920x1080')")
    parser.add_argument('--camera_id', type=int, default=1, help='要使用的摄像头ID。')
    parser.add_argument('--filter', type=str, default='one_euro',
                        choices=['none', 'sma', 'wma', 'kalman', 'one_euro'], help='选择用于平滑光标的滤波器。')

    parser.add_argument('--scene_image',type=str,default=(os.path.join(app_path(), 'model.jpg')),
                        help='用于热力图分析的背景场景图片。')

    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--model_height', type=lambda s: [int(item) for item in s.split(',')], default='1600,1600')

    parser.add_argument(
        '--model_ratio',
        type=parse_model_ratio,
        default=[[18, 19, 63]],
        help="每个人形的头-躯干-腿的比例。"
             "示例: '1x2x3'（单个人形）或 '1x2x3,2x3x4,1x1x2'（多个人形各自比例）"
    )
    parser.add_argument('--generate_heatmap',action='store_true',help='实验结束后，自动生成场景化热力图分析。')

    # Video upload related
    parser.add_argument('--csv_path', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--figName', default="P001")

    # Calibration point related
    parser.add_argument('--calibrationNums', type=int, nargs='+',
                        default=[5,6])
    args = parser.parse_args()


    # Operating System
    system = GazeCalibrationSystem(args)
    system.run()
