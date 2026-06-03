# -*- coding: utf-8 -*-
"""
眼动追踪校准与实时实验系统

本脚本整合了一个完整的眼动追踪应用流程:
1.  **校准 (Calibration)**: 在全屏界面上依次显示30个目标点，
    引导用户注视，并同步记录用户视线角度(theta, phi)与屏幕坐标(x, y)的配对数据。

2.  **训练 (Training)**: 使用校准阶段收集的数据，根据用户通过命令行
    选择的模型(单应性变换、多项式回归、SVR)，训练一个映射函数。

3.  **实验 (Experiment)**: 利用训练好的模型，实时将用户的视线角度
    转换为屏幕坐标，并以一个可见光标的形式在屏幕上实时显示，
    实现“眼动鼠标”的效果。

如何运行 (示例):
- 使用单应性变换模型:
  python gaze_calibration_system.py --model homography
- 使用多项式回归模型:
  python gaze_calibration_system.py --model polynomial --screen_size 1920x1080
- 使用SVR模型并指定摄像头ID:
  python gaze_calibration_system.py --model svr --camera_id 0

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

# 确保可以从src目录的父目录运行此脚本
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasources import Webcam
from models import ELG
from draw import analyze_and_plot

def parse_model_ratio(s):
    """
    解析 model_ratio 参数，支持两种形式：
    1) '1x2x3'                      -> [[1, 2, 3]]
    2) '1x2x3,2x3x4,1x1x2'          -> [[1, 2, 3], [2, 3, 4], [1, 1, 2]]

    每组三个数字依次表示：头 : 躯干 : 腿
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
                "每个 model_ratio 必须是 'aXbXc' 形式，例如 '1x2x3' 或 '1x2x3,2x3x4'"
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
    """获取应用的根目录，用于写入文件。在开发时是项目根目录，在打包后是.exe文件所在的目录。"""
    if getattr(sys, 'frozen', False):
        # 如果程序被打包
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))

def get_primary_screen_size():
    return 2560, 1600

# --- 滤波器接口与实现 ---
class GazeFilter:
    """滤波器基类接口"""

    def filter(self, point, timestamp):
        raise NotImplementedError


class NoFilter(GazeFilter):
    """不使用任何滤波器，直接返回原始点"""

    def filter(self, point, timestamp):
        return point


class SMAFilter(GazeFilter):
    """简单移动平均滤波器 (Simple Moving Average)"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.points = deque(maxlen=window_size)

    def filter(self, point, timestamp):
        self.points.append(point)
        return np.mean(self.points, axis=0)


class WMAFilter(GazeFilter):
    """指数加权移动平均滤波器 (Exponential Moving Average)"""

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
    """卡尔曼滤波器，用于平滑二维点"""

    def __init__(self):
        # 状态向量 [x, y, vx, vy] (位置和速度)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 过程噪声，数值越大，代表对模型预测越不信任，响应越快但越不平滑
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        # 测量噪声，数值越大，代表对测量值越不信任，平滑效果越强
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 5
        self.initialized = False

    def filter(self, point, timestamp):
        if not self.initialized:
            # 初始化状态
            self.kalman.statePost = np.array([point[0], point[1], 0, 0], np.float32)
            self.initialized = True
            return point

        # 预测和更新
        self.kalman.predict()
        measurement = np.array([[point[0]], [point[1]]], np.float32)
        self.kalman.correct(measurement)

        # 返回平滑后的位置
        return self.kalman.statePost[:2].flatten()


class OneEuroFilter(GazeFilter):
    def __init__(self, min_cutoff= 0.2 , beta=0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # 为位置和导数创建持久的滤波器状态
        self.x_filter = self._create_low_pass_filter()
        self.y_filter = self._create_low_pass_filter()
        self.dx_filter = self._create_low_pass_filter()
        self.dy_filter = self._create_low_pass_filter()
        self.last_timestamp = None

    def _create_low_pass_filter(self):
        # 'x_prev' 存储上一个原始值, 'hat_x_prev' 存储上一个滤波后的值
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
            # 完整初始化所有状态
            self.x_filter['hat_x_prev'] = point[0]
            self.x_filter['x_prev'] = point[0]
            self.y_filter['hat_x_prev'] = point[1]
            self.y_filter['x_prev'] = point[1]
            return point

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        if dt < 1e-6:  # 防止除零, 返回上一个滤波值
            return np.array([self.x_filter['hat_x_prev'], self.y_filter['hat_x_prev']])

        # --- X coordinate ---
        # 计算导数
        dx = (point[0] - self.x_filter['x_prev']) / dt
        # 使用持久的导数滤波器对导数进行滤波
        edx_alpha = 1.0 / (1.0 + (2 * np.pi * self.d_cutoff * dt) ** -1)
        edx = self._low_pass(self.dx_filter, dx, edx_alpha)

        # 计算自适应的截止频率
        cutoff_x = self.min_cutoff + self.beta * abs(edx)
        # 计算最终的alpha值并对位置进行滤波
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


# --- 映射模型接口与实现 ---
class GazeMapper:
    """映射模型的基类接口"""

    def train(self, source_points, dest_points):
        """使用源点和目标点训练模型"""
        raise NotImplementedError

    def predict(self, gaze_angle):
        """预测单个视线角度对应的屏幕坐标"""
        raise NotImplementedError


class HomographyMapper(GazeMapper):
    """使用单应性变换进行映射"""

    def __init__(self):
        self.homography_matrix = None

    def train(self, source_points, dest_points):
        # print("正在训练单应性变换模型...")
        # findHomography需要至少4个点
        if len(source_points) < 4:
            raise ValueError("单应性变换需要至少4个校准点。")
        self.homography_matrix, _ = cv2.findHomography(source_points, dest_points)
        if self.homography_matrix is None:
            raise RuntimeError("无法计算单应性矩阵，请检查校准点是否共线。")
        # print("单应性变换模型训练完成。")

    def predict(self, gaze_angle):
        if self.homography_matrix is None:
            return None
        # perspectiveTransform需要一个(1, 1, 2)形状的数组
        gaze_point_reshaped = np.array([[gaze_angle]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(gaze_point_reshaped, self.homography_matrix)
        return transformed_point[0][0]

class PolynomialMapper(GazeMapper):
    """使用多项式回归进行映射"""

    def __init__(self, degree=2):
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        # 使用Ridge回归增加稳定性
        self.regressor = Ridge(alpha=0.5)

    def train(self, source_points, dest_points):
        # print(f"正在训练{self.poly_features.degree}阶多项式回归模型...")
        source_poly = self.poly_features.fit_transform(source_points)
        self.regressor.fit(source_poly, dest_points)
        # print("多项式回归模型训练完成。")

    def predict(self, gaze_angle):
        gaze_poly = self.poly_features.transform([gaze_angle])
        return self.regressor.predict(gaze_poly)[0]


class SVRMapper(GazeMapper):
    """使用支持向量机回归(SVR)进行映射"""

    def __init__(self):
        # MultiOutputRegressor允许SVR用于多目标回归(x和y)
        self.multi_regressor = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))

    def train(self, source_points, dest_points):
        # print("正在训练SVR模型...")
        self.multi_regressor.fit(source_points, dest_points)
        # print("SVR模型训练完成。")

    def predict(self, gaze_angle):
        return self.multi_regressor.predict([gaze_angle])[0]


# --- 主系统类 ---

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

        # 校准数据
        rows = self.args.calibrationNums[0]
        cols = self.args.calibrationNums[1]
        self.calibration_targets = self._generate_calibration_targets(rows, cols)
        self.calibration_data = []  # 存储 ( (theta,phi), (x,y) )
        self.experiment_points = []  # 存储实验阶段，预测出来的点

        # 初始化模型
        self.gaze_mapper = self._get_mapper(args.model)
        self.gaze_filter = self._get_filter(args.filter)
        self.elg_model = None
        self.data_source = None
        self.tf_session = None

        self.radius = 105
        self.model = [[(300,20), (510,230)],[(225,230), (585,537)],[(260,537),(550,1237)]]
        try:
            # 优先使用Windows下的黑体，您也可以替换为自己的字体文件路径
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
        # print(f"使用滤波器: {filter_name}")
        if filter_name == 'sma': return SMAFilter()
        elif filter_name == 'wma': return WMAFilter()
        elif filter_name == 'kalman': return KalmanFilter()
        elif filter_name == 'one_euro': return OneEuroFilter()
        else: return NoFilter()

    def _generate_calibration_targets(self, rows=5, cols=4):
        """生成均匀分布在屏幕上的校准点"""
        targets = []
        # 在屏幕内部留出一些边距
        x_margin = self.screen_w // (cols * 2)
        y_margin = self.screen_h // (rows * 2)

        x_points = np.linspace(x_margin, self.screen_w - x_margin, cols, dtype=int)
        y_points = np.linspace(y_margin, self.screen_h - y_margin, rows, dtype=int)

        for y in y_points:
            for x in x_points:
                targets.append((x, y))
        return targets

    # def _draw_text(self, frame, text, position='center', color=(255, 255, 255)):
    #     """在屏幕上绘制居中文字"""
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1.2
    #     thickness = 2
    #     text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    #
    #     if position == 'center':
    #         pos = ((self.screen_w - text_size[0]) // 2, (self.screen_h + text_size[1]) // 2)
    #     elif position == 'top_center':
    #         pos = ((self.screen_w - text_size[0]) // 2, 100)
    #     else:
    #         pos = position
    #     cv2.putText(frame, text, pos, font, font_scale, color, thickness)

    def _draw_text(self, frame, text, position='center', color=(255, 255, 255)):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 计算文本尺寸和位置
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

        # 绘制文本
        draw.text(pos, text, font=self.font, fill=color)

        # 将Pillow图像转换回OpenCV图像(BGR)并返回
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _get_gaze_angle(self, inference_generator):
        """从ELG模型获取当前的视线角度"""
        output = next(inference_generator)
        # 简单起见，我们只使用检测到的第一个眼睛的数据
        eye_index = 0

        # 提取关键点和半径
        eye_landmarks = output['landmarks'][eye_index, :]
        eye_radius = output['radius'][eye_index][0]

        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        # 变换到原始图像坐标系
        frame_index = output['frame_index'][eye_index]
        eye_data = self.data_source._frames[frame_index]['eyes'][eye_index]

        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
        eye_landmarks = (eye_landmarks * eye_data['inv_landmarks_transform_mat'].T)[:, :2]
        eye_landmarks = np.asarray(eye_landmarks)

        iris_centre = eye_landmarks[16, :]
        eyeball_centre = eye_landmarks[17, :]

        # 计算角度
        i_x0, i_y0 = iris_centre
        e_x0, e_y0 = eyeball_centre

        # 确保eyeball_radius不为0
        eyeball_radius = np.linalg.norm(eye_landmarks[18, :] - eyeball_centre)
        if eyeball_radius < 1e-6:
            return None

        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))

        return np.array([theta, phi])

    def setup(self):
        """初始化OpenCV窗口和TensorFlow模型"""
        # 设置窗口为全屏
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 初始化Tensorflow Session
        # from tensorflow.python.client import device_lib
        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.tf_session = tf.Session(config=session_config)
        # session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        # self.tf_session = tf.compat.v1.Session(config=session_config)

        # 初始化数据源 (Webcam)
        self.data_source = Webcam(tensorflow_session=self.tf_session, batch_size=2,
                                  camera_id=self.args.camera_id, fps=30,
                                  data_format='NHWC', eye_image_shape=(36, 60))

        # 初始化模型 (ELG)
        self.elg_model = ELG(
            self.tf_session, train_data={'videostream': self.data_source},
            first_layer_stride=1, num_modules=2, num_feature_maps=32,
            learning_schedule=[{'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']}}]
        )
        self.elg_model.initialize_if_not(training=False)
        self.elg_model.checkpoint.load_all()

    # def run_calibration(self):
    #     """执行校准流程"""
    #     inference_generator = self.elg_model.inference_generator()
    #
    #     for i, target_pos in enumerate(self.calibration_targets):
    #         gaze_samples = []
    #         start_time = time.time()
    #
    #         # 校准点显示分为两个阶段：注视阶段 和 采集阶段
    #         while True:
    #             elapsed_time = time.time() - start_time
    #             frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
    #
    #             # 获取当前视线角度
    #             current_gaze = self._get_gaze_angle(inference_generator)
    #
    #             # 阶段控制
    #             if elapsed_time < 0.75:  # 0.75秒给用户找到目标
    #                 self._draw_text(frame, f"校准点: {i + 1} / {len(self.calibration_targets)}", 'top_center')
    #                 cv2.circle(frame, target_pos, 20, (255, 255, 255), -1)  # 白色大圆
    #                 cv2.circle(frame, target_pos, 7, (0, 0, 0), -1)  # 黑色小圆
    #             elif elapsed_time < 2.25:  # 1.5秒采集数据
    #                 self._draw_text(frame, "请保持注视...", 'top_center')
    #                 cv2.circle(frame, target_pos, 20, (0, 255, 0), -1)  # 绿色表示正在采集
    #                 cv2.circle(frame, target_pos, 7, (0, 0, 0), -1)
    #                 if current_gaze is not None:
    #                     gaze_samples.append(current_gaze)
    #             else:
    #                 break
    #
    #             cv2.imshow(self.window_name, frame)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 print("用户中断了校准。")
    #                 return False
    #
    #         if not gaze_samples:
    #             print(f"警告: 校准点 {i + 1} 未能采集到有效的视线数据。")
    #             continue
    #
    #         # 计算平均视线角度
    #         avg_gaze = np.mean(gaze_samples[1:], axis=0)   # 去掉第一个
    #         self.calibration_data.append((avg_gaze, target_pos))
    #         print(f"校准点 {i + 1} 采集完成。平均视线角度: {avg_gaze}, 目标位置: {target_pos}")
    #
    #     if len(self.calibration_data) < 4:
    #         print("错误：有效校准点太少，无法继续。")
    #         return False
    #     return True

    def run_calibration(self):
        """执行用户触发式的校准流程"""
        gen = self.elg_model.inference_generator()
        for i, target in enumerate(self.calibration_targets):
            state = "WAITING_FOR_TRIGGER"  # 初始状态：等待用户触发
            samples = []

            while True:
                # 持续获取视线数据，即使在等待时，以保持模型生成器运行
                current_gaze = self._get_gaze_angle(gen)
                frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

                if state == "WAITING_FOR_TRIGGER":
                    # --- 等待阶段 ---
                    msg = f"请注视白点, 按下<空格键>开始采集 ({i + 1}/{len(self.calibration_targets)})"
                    frame = self._draw_text(frame, msg, 'top_center')
                    # 绘制白色目标点
                    cv2.circle(frame, target, 20, (255, 255, 255), -1)
                    cv2.circle(frame, target, 7, (0, 0, 0), -1)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # 32是空格键的ASCII码
                        state = "COLLECTING"
                        collection_start_time = time.time()
                    elif key == ord('q'):
                        # print("用户中断了校准。")
                        return False

                elif state == "COLLECTING":
                    # --- 采集阶段 ---
                    elapsed = time.time() - collection_start_time
                    if elapsed < 2:
                        frame = self._draw_text(frame, "请保持注视...", 'top_center')
                        # 绘制绿色目标点，表示正在采集
                        cv2.circle(frame, target, 20, (0, 255, 0), -1)
                        cv2.circle(frame, target, 7, (0, 0, 0), -1)
                        if current_gaze is not None:
                            samples.append(current_gaze)
                    else:
                        # 采集结束，跳出内层循环
                        break

                cv2.imshow(self.window_name, frame)

            if not samples:
                # print(f"警告: 校准点 {i + 1} 未能采集到有效的视线数据。")
                continue

            avg_gaze = np.mean(samples, axis=0)
            self.calibration_data.append((avg_gaze, target))
            # print(f"校准点 {i + 1} 采集完成。")
            # print(f"校准点 {i + 1} 采集完成。平均视线角度: {avg_gaze}, 目标位置: {target}")

        if len(self.calibration_data) < 4:
            print("错误: 有效校准点太少，无法继续。")
            return False
        return True

    def train_model(self):
        """训练选择的映射模型"""
        frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        self._draw_text(frame, "校准完成，正在训练模型...")
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(100)  # 保证文字显示

        source_points = np.array([data[0] for data in self.calibration_data])
        dest_points = np.array([data[1] for data in self.calibration_data])

        try:
            self.gaze_mapper.train(source_points, dest_points)
        except Exception as e:
            print(f"模型训练失败: {e}")
            return False

        time.sleep(2)  # 等待用户看到训练完成的消息
        return True

    def run_experiment(self):
        """运行实时眼动追踪实验"""
        inference_generator = self.elg_model.inference_generator()

        #  实时检测--摄像头模式
        if self.args.csv_path is None:
            # print("\n实验开始！请移动您的视线。按 'q' 键退出。")
            while True:
                frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                current_gaze = self._get_gaze_angle(inference_generator)
                if current_gaze is not None:
                    raw_predicted_pos = self.gaze_mapper.predict(current_gaze)
                    if raw_predicted_pos is not None:
                        timestamp = time.time()
                        filter_pos = self.gaze_filter.filter(raw_predicted_pos, timestamp)   # 滤波
                        # 将预测点限制在屏幕范围内
                        px, py = filter_pos

                        if px > 0 and py > 0:
                            self.experiment_points.append(filter_pos)  # 记录滤波后的点

                        px = int(np.clip(px, 0, self.screen_w - 1))
                        py = int(np.clip(py, 0, self.screen_h - 1))
                        # 绘制眼动光标
                        cv2.circle(frame, (px, py), 25, (0, 180, 255), -1)  # 橙色光标
                        cv2.circle(frame, (px, py), 10, (255, 255, 255), -1)  # 白色中心

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 离线检测--视频模式
        else:
            csv_file = pd.read_csv(self.args.csv_path)
            theta = csv_file.iloc[:, 2]
            phi = csv_file.iloc[:, 3]
            # gaze_list = list(zip(theta, phi))
            for i in range(len(theta)):
                raw_predicted_pos = self.gaze_mapper.predict(np.array([theta[i], phi[i]]))
                if raw_predicted_pos is not None:
                    timestamp = time.time()
                    filter_pos = self.gaze_filter.filter(raw_predicted_pos, timestamp)  # 滤波
                    # 将预测点限制在屏幕范围内
                    px, py = filter_pos
                    if px > 0 and py > 0:
                        self.experiment_points.append(filter_pos)  # 记录滤波后的点

    def _generate_dummy_scene_image(self):
        """如果场景图片不存在，则创建一个模拟图片"""
        if not os.path.exists(self.args.scene_image):
            # print(f"未找到场景图片 '{self.args.scene_image}'，将创建一个模拟图片。")
            img = np.full((1237, 800, 3), (240, 240, 240), dtype=np.uint8)
            # 绘制头部、躯干、腿部的示意矩形
            # cv2.circle(img, (405, 125), 105,(147,112,219) , -1)  # Head
            # cv2.rectangle(img, (225, 230), (585, 537), (173,216,230), -1) # Torso
            # cv2.rectangle(img, (260, 537), (550, 1237), (220, 220, 240), -1) # Legs
            cv2.circle(img, (self.model[0][0][0] + self.radius, self.model[0][1][1] - self.radius), self.radius,(147,112,219) , -1)  # Head
            cv2.rectangle(img, self.model[1][0], self.model[1][1], (173, 216, 230), -1) # Torso
            cv2.rectangle(img, self.model[2][0], self.model[2][1], (220, 220, 240), -1) # Legs
            cv2.imwrite(self.args.scene_image, img)


    def _generate_multi_dummy_scene_image(self):
        """如果场景图片不存在，则根据屏幕尺寸生成一个包含多个人形的模拟场景图片。

        - 根据 self.args.model_num 在水平方向等距排列多个人形
        - 每个人形的总高度等于屏幕高度 screen_h
        - 每个人形使用各自的 model_ratio(头:躯干:腿) 计算三段的高度比例
        """
        # 如果用户已经提供了场景图片，就不再生成
        # if os.path.exists(self.args.scene_image):
        #     return

        # 使用屏幕分辨率作为画布大小，保持与视线坐标一致
        img_w, img_h = self.screen_w, self.screen_h
        # img_h = self.args.model_height
        img = np.full((img_h, img_w, 3), (240, 240, 240), dtype=np.uint8)

        # 人形数量
        model_num = max(1, int(getattr(self.args, "model_num", 1)))

        # 解析 model_ratio:
        # 兼容两种形式:
        # 1) [1, 2, 3]                  (旧版本, 单个人形)
        # 2) [[1, 2, 3], [2, 3, 4], ...] (新版本, 每个人形一组比例)
        model_ratios = getattr(self.args, "model_ratio", [[1, 2, 3]])
        if isinstance(model_ratios, list) and model_ratios and isinstance(model_ratios[0], int):
            # 旧形式: 单一比例, 对所有人形复用
            base_ratio = model_ratios
            model_ratios = [base_ratio[:] for _ in range(model_num)]

        # 如果数量不足, 使用最后一组比例补齐; 如果过多, 截断
        if len(model_ratios) < model_num:
            last = model_ratios[-1]
            model_ratios = model_ratios + [last[:] for _ in range(model_num - len(model_ratios))]
        model_ratios = model_ratios[:model_num]

        # 每个人形总高度 = 屏幕高度
        # person_height = float(img_h)

        # 垂直方向: 人形从顶部开始到底部结束
        top_y = 0.0

        # 用于 AOI 统计的合并区域(头/躯干/腿)
        head_regions = []
        torso_regions = []
        legs_regions = []

        # 水平方向: 将屏幕宽度划分为 model_num 个等宽区间, 每个人形居中放置在各自区间
        segment_width = img_w / float(model_num)

        for idx in range(model_num):
            person_height = self.args.model_height[idx]
            top_y = img_h - person_height
            ratio = model_ratios[idx]
            if len(ratio) != 3:
                # 防御式: 比例配置不合法时退回默认 [1,2,3]
                ratio = [1, 2, 3]

            r_head, r_torso, r_legs = ratio
            ratio_sum = float(r_head + r_torso + r_legs) or 1.0

            head_h = person_height * (r_head / ratio_sum)
            torso_h = person_height * (r_torso / ratio_sum)
            legs_h = person_height * (r_legs / ratio_sum)

            # 当前人形中心的 x 坐标（等距分布）
            cx = int((idx + 0.5) * segment_width)

            # 人形宽度, 这里取该分区宽度的 30% 作为示意
            person_width = segment_width * 0.40
            # person_width = np.clip(person_width, 270, 500)
            half_w = int(person_width / 2.0)

            # 计算三段在垂直方向上的上下边界
            head_top = int(top_y)
            head_bottom = int(top_y + head_h)
            torso_top = head_bottom
            torso_bottom = int(torso_top + torso_h)
            legs_top = torso_bottom
            legs_bottom = int(legs_top + legs_h)

            # 限制在图像范围内
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

            # AOI 矩形 ((x1,y1),(x2,y2))
            head_rect = ((left, head_top), (right, head_bottom))
            torso_rect = ((left, torso_top), (right, torso_bottom))
            legs_rect = ((left, legs_top), (right, legs_bottom))

            head_regions.append(head_rect)
            torso_regions.append(torso_rect)
            legs_regions.append(legs_rect)

            # 绘制头部 (圆形), 以 head 段中点为圆心
            head_center_y = int((head_top + head_bottom) / 2)
            head_radius = max(5, int((head_bottom - head_top) / 2))
            cv2.circle(img, (cx, head_center_y), head_radius, (147, 112, 219), -1)

            # 绘制躯干和腿部 (矩形)
            cv2.rectangle(img, (left, torso_top), (right, torso_bottom), (173, 216, 230), -1)  # Torso
            cv2.rectangle(img, (left, legs_top), (right, legs_bottom), (220, 220, 240), -1)  # Legs

        # 将多个人形的 AOI 合并成三大区域, 便于后续现有统计逻辑使用
        def merge_regions(regions):
            if not regions:
                return ((0, 0), (0, 0))
            x1 = min(r[0][0] for r in regions)
            y1 = min(r[0][1] for r in regions)
            x2 = max(r[1][0] for r in regions)
            y2 = max(r[1][1] for r in regions)
            return (x1, y1), (x2, y2)

        self.model = [
            merge_regions(head_regions),  # 头部总区域
            merge_regions(torso_regions),  # 躯干总区域
            merge_regions(legs_regions)  # 腿部总区域
        ]

        cv2.imwrite(self.args.scene_image, img)

    def save_and_analyze_data(self):
        """保存实验数据到CSV并生成热力图分析"""
        # 1. 保存数据
        # output_filename = f"data/gaze_points_output.csv"
        # output_filename = resource_path(os.path.join('data', 'gaze_points_output.csv"'))
        output_filename = (os.path.join(app_path(), f'{self.args.figName}_gaze_points_output.csv'))

        df = pd.DataFrame(self.experiment_points, columns=['x_coord', 'y_coord'])
        df.to_csv(output_filename, index=False)
        # print(f"\n实验数据已保存至: {output_filename}")

        # 2. 绘制散点图
        # print("正在生成视线点及聚类图...")
        analyze_and_plot(output_filename,self.args.figName,x_min =self.screen_w,x_max = self.screen_h)

        # 3. 生成热力图
        # print("正在生成热力图分析...")
        # self._generate_scene_analysis_report(df)
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

        # # --- 2. 核心：屏幕坐标到图像像素坐标的转换 ---
        screen_w, screen_h = self.args.screen_size

        image_coords = points / np.array([screen_w, screen_h]) * np.array([img_w, img_h])

        # 过滤掉落在黑边区域的点
        valid_mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] <= img_w) & \
                     (image_coords[:, 1] >= 0) & (image_coords[:, 1] <= img_h)
        image_coords_valid = image_coords[valid_mask]

        # DBSCAN去噪
        db = DBSCAN(eps=100, min_samples=20).fit(image_coords_valid)
        cleaned_points = image_coords_valid[db.labels_ != -1]

        # if cleaned_points.shape[0] < 20:
        #     # print("警告: 清洗后的有效注视点过少，分析可能不准确。")
        #     if cleaned_points.shape[0] < 2: return

        # 定义兴趣区域 (AOI)，单位为像素
        aois = {
            'head': self.model[0],
            'body': self.model[1],
            'legs': self.model[2]
        }

        # 计算AOI命中率
        aoi_counts = {name: 0 for name in aois}
        total_valid_points = len(cleaned_points)
        for p in cleaned_points:
            for name, (p1, p2) in aois.items():
                if p1[0] <= p[0] <= p2[0] and p1[1] <= p[1] <= p2[1]:
                    aoi_counts[name] += 1
                    break

        aoi_percentages = {name: (count / total_valid_points) * 100 if total_valid_points > 0 else 0 for name, count in
                           aoi_counts.items()}

        # --- 绘图 ---
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 10), gridspec_kw={'width_ratios': [img_w, 400]})
        fig.suptitle('Report', fontsize=20)

        # 左图: 热力图叠加
        ax1.imshow(bg_image, extent=[0, img_w, img_h, 0])  # Y轴方向正确
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
        ax1.set_ylim(img_h, 0)  # 保持(0,0)在左上角
        ax1.set_aspect('equal', adjustable='box')

        # 右图: AOI分析条形图
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

        # 1. 将屏幕坐标转换为场景图像像素坐标
        points = df[['x_coord', 'y_coord']].values
        screen_w, screen_h = self.args.screen_size
        image_coords = points / np.array([screen_w, screen_h]) * np.array([img_w, img_h])

        # 只保留落在图像范围内的点
        valid_mask = (
                (image_coords[:, 0] >= 0) & (image_coords[:, 0] <= img_w) &
                (image_coords[:, 1] >= 0) & (image_coords[:, 1] <= img_h)
        )
        image_coords_valid = image_coords[valid_mask]

        if image_coords_valid.shape[0] < 2:
            print("有效注视点过少，无法生成热力图与AOI分析。")
            return

        # 2. DBSCAN 去噪
        db = DBSCAN(eps=100, min_samples=20).fit(image_coords_valid)
        cleaned_points = image_coords_valid[db.labels_ != -1]
        if cleaned_points.shape[0] < 2:
            # 如果去噪后点太少，则退回使用原有效点
            cleaned_points = image_coords_valid

        # 3. 根据 model_num / model_ratio 重新计算 AOI（多个人形）
        model_num = max(1, int(getattr(self.args, "model_num", 1)))
        model_ratios = getattr(self.args, "model_ratio", [[1, 2, 3]])

        # 兼容旧写法: --model_ratio 1x2x3 得到 [1,2,3]
        if isinstance(model_ratios, list) and model_ratios and isinstance(model_ratios[0], int):
            base_ratio = model_ratios
            model_ratios = [base_ratio[:] for _ in range(model_num)]

        if len(model_ratios) < model_num:
            last = model_ratios[-1]
            model_ratios = model_ratios + [last[:] for _ in range(model_num - len(model_ratios))]
        model_ratios = model_ratios[:model_num]

        # 每个人形总高度 = 整张场景图高度（与前面 _generate_dummy_scene_image 保持一致）
        person_height = float(img_h)
        top_y = 0.0

        # 水平方向等距摆放
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

            # 当前人形中心 x 坐标
            cx = int((idx + 0.5) * segment_width)

            # 人形宽度：取该分区宽度的 30% 作为示意
            person_width = segment_width * 0.7
            person_width = np.clip(person_width, 270, 500)
            half_w = int(person_width / 2.0)

            # 垂直方向分三段
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

        # 将所有模特同一部位的区域合并成一个总 AOI 矩形
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

        # 4. 计算 AOI 命中率
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

        # 5. 绘图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 10),
            gridspec_kw={'width_ratios': [img_w, 400]}
        )
        fig.suptitle('Report', fontsize=20)

        # 左图: 热力图叠加
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

        # for name, (p1, p2) in aois.items():
        #     rect = plt.Rectangle(
        #         p1,
        #         p2[0] - p1[0],
        #         p2[1] - p1[1],
        #         linewidth=2,
        #         edgecolor='cyan',
        #         facecolor='none',
        #         alpha=0.7
        #     )
        #     ax1.add_patch(rect)
        #     ax1.text(
        #         p1[0] + 5,
        #         p1[1] + 25,
        #         f'{name}\n{aoi_percentages[name]:.1f}%',
        #         color='cyan',
        #         fontsize=12,
        #         weight='bold'
        #     )

        ax1.set_title('Gaze Heatmap and Area of Interest (AOI)')
        ax1.set_xlabel('Horizontal pixel coordinates of the image (X)')
        ax1.set_ylabel('Vertical pixel coordinates of the image (Y)')
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0)
        ax1.set_aspect('equal', adjustable='box')

        # 右图: AOI 分析条形图
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
        """清理资源"""
        if self.data_source:
            self.data_source.cleanup()
        if self.tf_session:
            self.tf_session.close()
        cv2.destroyAllWindows()
        # print("系统已退出。")

    def run(self):
        """执行完整流程"""
        try:
            self._generate_multi_dummy_scene_image()

            self.setup()

            # 阶段一：校准
            if not self.run_calibration():
                return

            # 阶段二：训练
            if not self.train_model():
                return

            # 阶段三：实验
            self.run_experiment()

            # 阶段四：保存数据并绘制热力图
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
    # parser.add_argument('--model_ratio',type=lambda s: [int(item) for item in s.split('x')],default=[1,2,3])
    parser.add_argument('--model_height', type=lambda s: [int(item) for item in s.split(',')], default='1600,1600')

    parser.add_argument(
        '--model_ratio',
        type=parse_model_ratio,
        default=[[18, 19, 63]],
        help="每个人形的头-躯干-腿的比例。"
             "示例: '1x2x3'（单个人形）或 '1x2x3,2x3x4,1x1x2'（多个人形各自比例）"
    )
    parser.add_argument('--generate_heatmap',action='store_true',help='实验结束后，自动生成场景化热力图分析。')

    # 上传视频相关
    parser.add_argument('--csv_path', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--figName', default="P001")
    # parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')

    # 校准点相关
    parser.add_argument('--calibrationNums', type=int, nargs='+',
                        default=[5,6])
    # parser.add_argument('--calibrationPoints', type=float, nargs='+',
    #                     default= None)
    args = parser.parse_args()


    # 运行系统
    system = GazeCalibrationSystem(args)
    system.run()
