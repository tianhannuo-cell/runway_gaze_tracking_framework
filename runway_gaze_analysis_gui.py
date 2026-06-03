# -*- coding: utf-8 -*-

import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
from PIL import Image, ImageTk
import os
import re
import sys
# 确保gaze_calibration_system.py与此脚本在同一目录下
from gaze_calbration import GazeCalibrationSystem, get_primary_screen_size
#from src.elg_demo import sight_analysis


# def resource_path(relative_path):
#     try:
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(os.path.dirname(__file__))
#     return os.path.join(base_path, relative_path)

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

def app_path():
    """获取应用的根目录，用于写入文件。在开发时是项目根目录，在打包后是.exe文件所在的目录。"""
    if getattr(sys, 'frozen', False):
        # 如果程序被打包
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
class GazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.adv_calib_file_var = None
        self.image_label = None
        self.title("Runway Show Eye Tracking Calibration and Analysis System")
        self.geometry("1200x700")
        self.minsize(1000, 650)

        # --- 数据与状态 ---
        self.backend_thread = None
        self.comm_queue = queue.Queue()
        # self.final_image_path = None

        self.final_image_path = (os.path.join(app_path(), 'attention_result.png'))
        self.heatmap_path = (os.path.join(app_path(), 'attention_heatmap.png'))
        self.scene_image = (os.path.join(app_path(), 'model321.jpg'))

        self.calibrationPoints = []
        self.calibrationNums = []

        # --- 创建UI组件 ---
        self.create_widgets()

        # --- 启动队列轮询 ---
        self.process_queue()

        self.args = None

    def   create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=tk.YES)
        # 【核心修改】: 创建并配置样式
        style = ttk.Style(self)
        # 定义主操作按钮样式 (蓝底白字)
        style.configure('Primary.TButton', foreground='blue', background='#007bff')
        # 定义取消按钮样式 (红字)
        style.configure('Cancel.TButton', foreground='red')
        # 映射样式状态，以便在禁用时也能正确显示
        style.map('Primary.TButton',
            background=[('active', '#0056b3'), ('disabled', '#c0c0c0')])
        style.map('Cancel.TButton',
            foreground=[('disabled', '#c0c0c0')])

        top_panel = ttk.Frame(main_frame)
        top_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # --- 左侧：参数设置区 ---
        # 将 settings_frame 放入 top_panel
        settings_frame = ttk.Labelframe(top_panel, text="Parameter Settings", padding="10")
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')

        # 1. 实验设置
        exp_frame = ttk.Frame(settings_frame)
        exp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(exp_frame, text="Subject ID:").pack(side=tk.LEFT, padx=5)
        self.participant_id_var = tk.StringVar(value="P001")
        ttk.Entry(exp_frame, textvariable=self.participant_id_var, width=20).pack(side=tk.LEFT, fill=tk.X,
                                                                                  expand=tk.YES)

        img_frame = ttk.Frame(settings_frame)
        img_frame.pack(fill=tk.X, pady=5)
        ttk.Label(img_frame, text="Scene Images:").pack(side=tk.LEFT, padx=5)
        self.scene_image_var = tk.StringVar(value="model321.jpg")
        img_entry = ttk.Entry(img_frame, textvariable=self.scene_image_var, state="readonly", width=15)
        img_entry.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)
        ttk.Button(img_frame, text="浏览...", command=self.browse_image).pack(side=tk.LEFT, padx=5)

        # 2. 模型配置
        model_frame = ttk.Labelframe(settings_frame, text="模型配置", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="映射模型:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.map_model_var = tk.StringVar(value="homography")
        map_model_cb = ttk.Combobox(model_frame, textvariable=self.map_model_var,
                                    values=["homography", "polynomial", "svr"], state="readonly")
        map_model_cb.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(model_frame, text="平滑滤波器:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.filter_var = tk.StringVar(value="one_euro")
        filter_cb = ttk.Combobox(model_frame, textvariable=self.filter_var,
                                 values=["none", "sma", "wma", "kalman", "one_euro"], state="readonly")
        filter_cb.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # =========== 【新增代码开始】 ===========
        # 1. 模特数量 (Spinbox类型)
        ttk.Label(model_frame, text="模特数量:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.model_num_var = tk.IntVar(value=1)
        # width设为5保持与其他控件对齐，范围设为1-10
        tk.Spinbox(model_frame, from_=1, to=10, textvariable=self.model_num_var, width=5).grid(row=2, column=1,
                                                                                               sticky="w", padx=5,
                                                                                               pady=5)
        # 2. 模特比例 (Label类型 - 实际需用Entry输入，Label作为提示)
        # 提示用户格式，例如 1x2x3
        ttk.Label(model_frame, text="模特比例 (头x身x腿):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.model_ratio_var = tk.StringVar(value="24x71x83")
        ttk.Entry(model_frame, textvariable=self.model_ratio_var, width=15).grid(row=3, column=1, sticky="ew", padx=5,
                                                                                 pady=5)

        # Row 4: 模特身高 (Label + Entry)
        ttk.Label(model_frame, text="模特身高:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.model_height = tk.StringVar(value=134)  # 默认身高 175
        ttk.Entry(model_frame, textvariable=self.model_height, width=23).grid(row=4, column=1, sticky="w", padx=5,
                                                                              pady=5)
        # =========== 【新增代码end】 ===========

        input_calib_frame = ttk.Labelframe(settings_frame, text="输入与校准", padding="10")
        input_calib_frame.pack(fill=tk.X, pady=5)

        # 3. 硬件配置
        hw_frame = ttk.Labelframe(settings_frame, text="硬件配置", padding="10")
        hw_frame.pack(fill=tk.X, pady=5)

        ttk.Label(hw_frame, text="摄像头ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.camera_id_var = tk.IntVar(value=0)
        tk.Spinbox(hw_frame, from_=0, to=10, textvariable=self.camera_id_var, width=5).grid(row=0, column=1, sticky="w",
                                                                                            padx=5, pady=5)

        ttk.Label(hw_frame, text="屏幕尺寸 (宽x高):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.screen_size_var = tk.StringVar()
        self.screen_size_entry = ttk.Entry(hw_frame, textvariable=self.screen_size_var, width=15)
        self.screen_size_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # 4.高级设置
        adv_frame = ttk.Labelframe(settings_frame, text="高级设置", padding="10")
        adv_frame.pack(fill=tk.X, pady=10)
        # 4a. 上传视频
        ttk.Label(adv_frame, text="视频文件:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.adv_video_path_var = tk.StringVar(value="")
        adv_video_entry = ttk.Entry(adv_frame, textvariable=self.adv_video_path_var, state="readonly", width=15)
        adv_video_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        adv_video_btn = ttk.Button(adv_frame, text="浏览...", command=self.browse_video_file)  # 启用并添加命令
        adv_video_btn.grid(row=0, column=2, sticky="w", padx=5, pady=2)
        # style = Style()
        # style.configure("TButton", , background="blue", relief="raised")
        self.cancel_video_btn = ttk.Button(adv_frame, text="×", command=self.cancel_video_selection, width=2, state="disabled",style='Cancel.TButton')
        self.cancel_video_btn.grid(row=0, column=3, sticky="w", padx=2)
        # 4aa.视频分析按钮
        self.video_analysis_button = ttk.Button(adv_frame, text="视频视线分析与结果导出", state="disabled", command=self.run_video_analysis, style='Primary.TButton')
        self.video_analysis_button.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=(5,0))

        # ttk.Separator(adv_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky='ew', pady=10)

        # 4b. 校准点个数
        ttk.Label(adv_frame, text="校准点网格:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(adv_frame, text="行:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.adv_rows_var = tk.IntVar(value=5)
        adv_rows_spinbox = tk.Spinbox(adv_frame, from_=2, to=10, textvariable=self.adv_rows_var, width=5)  # 启用
        adv_rows_spinbox.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(adv_frame, text="列:").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.adv_cols_var = tk.IntVar(value=6)
        adv_cols_spinbox = tk.Spinbox(adv_frame, from_=2, to=10, textvariable=self.adv_cols_var, width=5)  # 启用
        adv_cols_spinbox.grid(row=4, column=1, sticky="w", padx=5, pady=2)

        # ttk.Separator(adv_frame, orient='horizontal').grid(row=5, column=0, columnspan=3, sticky='ew', pady=5)

        # --- 右侧：状态与输出区 ---
        # 【代码修复】: 将 output_frame 放入 top_panel
        output_frame = ttk.Frame(top_panel)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # 4. 状态日志
        log_frame = ttk.Labelframe(output_frame, text="状态日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.log_text = scrolledtext.Text(log_frame, wrap=tk.WORD, height=10, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=tk.YES)

        # 5. 结果预览
        # preview_frame = ttk.Labelframe(output_frame, text="分析结果预览", padding="10")
        # preview_frame.pack(fill=tk.BOTH, expand=tk.YES, pady=(10, 0))
        # self.image_label = ttk.Label(preview_frame, text="分析完成后，结果将在此处显示")
        # self.image_label.pack(fill=tk.BOTH, expand=tk.YES)

        # 5. 结果预览 (重构为左右两个面板)
        preview_container = ttk.Frame(output_frame)
        preview_container.pack(fill=tk.BOTH, expand=tk.YES)

        preview_left_frame = ttk.Labelframe(preview_container, text="视线坐标及聚类分析结果", padding=5)
        preview_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=(0, 5))
        self.image_label_left = ttk.Label(preview_left_frame, text="视线坐标及聚类分析结果将在此处显示")
        self.image_label_left.pack(fill=tk.BOTH, expand=tk.YES)

        preview_right_frame = ttk.Labelframe(preview_container, text="分析结果", padding=5)
        preview_right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=(5, 0))
        self.image_label_right = ttk.Label(preview_right_frame, text="热图分析将在此处显示")
        self.image_label_right.pack(fill=tk.BOTH, expand=tk.YES)

        # --- 底部：主控制区 ---
        action_frame = ttk.Frame(main_frame, padding="10 0 0 0")
        action_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(action_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=tk.YES, side=tk.LEFT, padx=(0, 10))

        self.start_button = ttk.Button(action_frame, text="开始校准与实验", command=self.start_experiment, width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.exit_button = ttk.Button(action_frame, text="退出", command=self.quit)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # 控件组，方便统一禁用/启用
        self.settings_widgets = [exp_frame, model_frame, hw_frame]

        self.auto_detect_screen_size()  # 自动填充

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="选择场景图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if file_path:
            self.scene_image_var.set(os.path.basename(file_path))

    def browse_video_file(self):
        path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.adv_video_path_var.set(path)
            # 更新按钮状态
            self.start_button.config(state="disabled")
            self.video_analysis_button.config(state="normal")
            self.cancel_video_btn.config(state="normal")
            self.log_message("请先进行视频视线分析！")

    def browse_calib_file(self):
        path = filedialog.askopenfilename(title="选择校准点Excel文件", filetypes=[("Excel Files", "*.xlsx *.xls")])
        if path:
            self.adv_calib_file_var.set(path)
            # pd = pandas.read_excel(path)
            # x_coord = pd.iloc[:, 0]
            # y_coord = pd.iloc[:, 1]
            # self.calibrationPoints = list(zip(x_coord, y_coord))

    def cancel_video_selection(self, delete_video = True):
        """【新增】: 取消视频选择并重置UI状态"""
        if delete_video:
            self.adv_video_path_var.set("")
            self.start_button.config(state="normal")
            self.video_analysis_button.config(state="disabled")
            self.cancel_video_btn.config(state="disabled")
        else:
            self.start_button.config(state="normal")
            # self.video_analysis_button.config(state="disabled")
            # self.cancel_video_btn.config(state="disabled")

    def run_video_analysis(self):
        video_path = self.adv_video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("错误", "请先选择一个有效的视频文件。")
            return

        self.log_message(f"正在分析视频 '{os.path.basename(video_path)}' ...")

        parser_video = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
        parser_video.add_argument('-v', type=str, help='logging level', default='info',
                            choices=['debug', 'info', 'warning', 'error', 'critical'])

        parser_video.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
        parser_video.add_argument('--record_video', type=str, help='Output path of video of demonstration.')

        parser_video.add_argument('--fullscreen', action='store_true')
        parser_video.add_argument('--headless', action='store_true')
        parser_video.add_argument('--fps', type=int, default=20, help='Desired sampling rate of webcam')
        parser_video.add_argument('--camera_id', type=int, default=1, help='ID of webcam to use')

        args_video = parser_video.parse_args()
        # sight_analysis(args_video,0)

        # subprocess.call(['python', 'elg_demo.py'])  # 调用视频

        messagebox.showinfo("分析完成", "视频分析完成！")

        # 恢复UI状态
        self.cancel_video_selection(delete_video = False)
        self.log_message("视频视线分析完成，视线结果保存在demo_gaze_result.csv，"
                         "可以开始实验任务！")

    def auto_detect_screen_size(self):
        try:
            w, h = get_primary_screen_size()
            self.screen_size_var.set(f"{w}x{h}")
            self.log_message(f"成功自动检测到屏幕分辨率: {w}x{h}")
        except Exception as e:
            self.log_message(f"自动检测屏幕分辨率失败: {e}")

    def log_message(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def toggle_settings_enabled(self, enabled=True):
        state = "normal" if enabled else "disabled"
        for frame in self.settings_widgets:
            for widget in frame.winfo_children():
                try:
                    widget.config(state=state)
                except tk.TclError:
                    pass  # Some widgets like Labels don't have a state
        self.start_button.config(state=state)
        # # 如果有视频被选中，则start_button依然保持禁用
        # if self.adv_video_path_var.get():
        #      self.start_button.config(state="disabled")
        self.exit_button.config(state=state)

    def start_experiment(self):
        # 1. 验证输入
        screen_size_str = self.screen_size_var.get()
        if not re.match(r'^\d+x\d+$', screen_size_str):
            messagebox.showerror("输入错误", "屏幕尺寸格式不正确，应为 '宽x高' (例如 '1920x1080')。")
            return

        if not self.scene_image_var.get():
            messagebox.showerror("输入错误", "请选择一个场景图片。")
            return

        # 2. 收集参数
        settings = {
            'participant_id': self.participant_id_var.get(),
            'scene_image': self.scene_image_var.get(),
            'model': self.map_model_var.get(),
            'filter': self.filter_var.get(),
            'camera_id': self.camera_id_var.get(),
            'screen_size': [int(i) for i in screen_size_str.split('x')],
            'generate_heatmap': True  # GUI模式下总是生成热力图
        }

        parser = argparse.ArgumentParser(description='眼动追踪校准与实时实验系统')
        parser.add_argument('-v', type=str, help='logging level', default='info',
                            choices=['debug', 'info', 'warning', 'error', 'critical'])
        parser.add_argument('--model', type=str, default= self.map_model_var.get(),
                            choices=['homography', 'polynomial', 'svr'],
                            help='选择用于视线到屏幕坐标映射的模型。')
        parser.add_argument('--screen_size', type=lambda s: [int(item) for item in s.split('x')],
                            default=[int(i) for i in screen_size_str.split('x')],
                            help="您的主显示器分辨率, 格式: '宽x高' (例如 '1920x1080')")
        parser.add_argument('--camera_id', type=int, default= int(self.camera_id_var.get()), help='要使用的摄像头ID。')
        parser.add_argument('--filter', type=str, default= self.filter_var.get(),
                            choices=['none', 'sma', 'wma', 'kalman', 'one_euro'], help='选择用于平滑光标的滤波器。')
        parser.add_argument('--scene_image', type=str, default= self.scene_image, help='用于热力图分析的背景场景图片。')
        parser.add_argument('--generate_heatmap', default = True,action='store_true', help='实验结束后，自动生成场景化热力图分析。')

        # 上传视频相关
        parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam',
                            default= self.adv_video_path_var.get())
        parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.',
                            default= f'{self.adv_video_path_var.get()}_output.mp4')

        # 校准点相关
        parser.add_argument('--calibrationNums', type= int, nargs= '+',
                            default= [self.adv_rows_var.get(), self.adv_cols_var.get()])

        parser.add_argument('--figName', default=self.participant_id_var.get())

        # parser.add_argument('--scene_image', type=str, default=(os.path.join(app_path(), 'F:\sceneimage\model321.jpg')),
        #                     help='用于热力图分析的背景场景图片。')
        parser.add_argument('--model_num', type=int, default=self.model_num_var.get())
        # current_ratio_str = self.model_ratio_var.get()
        # # 使用脚本开头定义的 parse_model_ratio 函数将字符串转换为列表
        # # 注意：argparse的default参数不会自动经过type转换，所以这里必须手动转换
        # ratio_list_val = parse_model_ratio(current_ratio_str)
        parser.add_argument('--model_ratio', type=parse_model_ratio, default=self.model_ratio_var.get(),
                            help="每个人形的头-躯干-腿的比例。"
                                 "示例: '1x2x3'（单个人形）或 '1x2x3,2x3x4,1x1x2'（多个人形各自比例）"
                            )
        parser.add_argument('--model_height', type=lambda s: [float(item) for item in s.split(',')],default=self.model_height.get())

        # 摄像头模式
        if self.adv_video_path_var.get() == "":
            parser.add_argument('--csv_path', type=str)
            args = parser.parse_args()
            # 3. 禁用UI并启动后端线程
            self.toggle_settings_enabled(enabled=False)
            self.log_message("--- 开始新任务 ---")
            self.progress_bar['value'] = 0
            # self.final_image_path = None  # 重置

            # 将GazeCalibrationSystem的运行封装到新线程中
            self.run_backend_task(args,self.comm_queue)
            # self.backend_thread = threading.Thread(
            #     target=self.run_backend_task,
            #     args=(args, self.comm_queue),
            #     daemon=True
            # )
            # self.backend_thread.start()
        # 视频模式
        else:

            parser.add_argument('--csv_path', type=str, default=(os.path.join(app_path(), 'demo_gaze_result.csv')))

            args = parser.parse_args()
            # 3. 禁用UI并启动后端线程
            self.toggle_settings_enabled(enabled=False)
            self.log_message("--- 开始新任务 ---")
            self.progress_bar['value'] = 0
            # self.final_image_path = None  # 重置
            # 将GazeCalibrationSystem的运行封装到新线程中
            self.backend_thread = threading.Thread(
                target=self.run_backend_task,
                args=(args, self.comm_queue),
                daemon=True
            )
            self.backend_thread.start()

    def run_backend_task(self, args, comm_queue):
        """这个函数在独立的线程中运行，不会阻塞GUI"""
        try:
            # GazeCalibrationSystem现在需要一个队列来进行通信
            backend = GazeCalibrationSystem(args)
            backend.run()
        except Exception as e:
            # 将异常信息格式化，以便更好地调试
            import traceback
            error_info = f"后端任务发生严重错误:\n{traceback.format_exc()}"
            comm_queue.put(("ERROR", error_info))
        finally:
            comm_queue.put(("FINISHED", None))

    def process_queue(self):
        """定期检查队列中是否有来自后端的消息，并更新UI"""
        try:
            while True:
                msg_type, msg_data = self.comm_queue.get_nowait()

                if msg_type == "LOG":
                    self.log_message(msg_data)
                elif msg_type == "PROGRESS":
                    current, total = msg_data
                    self.progress_bar['value'] = (current / total) * 100
                elif msg_type == "HIDE_GUI":
                    self.withdraw()  # 隐藏主窗口，为全屏校准让路
                elif msg_type == "SHOW_GUI":
                    self.deiconify()  # 恢复主窗口
                elif msg_type == "RESULT":
                    # self.final_image_path = msg_data['img_path']
                    self.display_result_image()
                elif msg_type == "ERROR":
                    messagebox.showerror("后端错误", msg_data)
                elif msg_type == "FINISHED":
                    self.toggle_settings_enabled(enabled=True)
                    self.log_message("--- 任务完成 ---")
                    break  # 退出循环，直到下一次任务

            self.display_result_image()

        except queue.Empty:
            pass  # 队列为空，什么都不做
        finally:
            # 每100ms检查一次队列
            self.after(100, self.process_queue)

    def _load_and_display_image(self, path, label_widget):
        """辅助函数，用于加载、缩放和显示单张图片"""
        if not path or not os.path.exists(path):
            label_widget.config(text=f"未能加载图片:\n{path}")
            return
        try:
            self.update_idletasks()
            label_w = label_widget.winfo_width()
            label_h = label_widget.winfo_height()
            if label_w < 20 or label_h < 20:
                self.after(100, lambda: self._load_and_display_image(path, label_widget))
                return
            img = Image.open(path)
            thumbnail_method = getattr(Image, 'Resampling', Image).LANCZOS
            img.thumbnail((label_w, label_h), thumbnail_method)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            self.log_message(f"显示图片'{os.path.basename(path)}'时出错: {e}")
            label_widget.config(text=f"加载图片失败:\n{os.path.basename(path)}")

    def display_result_image(self):
        self.final_image_path = (os.path.join(app_path(), f'{self.participant_id_var.get()}_attention_result.png'))
        self.heatmap_path = (os.path.join(app_path(), f'{self.participant_id_var.get()}_attention_heatmap.png'))
        self._load_and_display_image(self.heatmap_path, self.image_label_left)
        self._load_and_display_image(self.final_image_path, self.image_label_right)

        # if not self.final_image_path or not os.path.exists(self.final_image_path):
        #     self.image_label.config(text="未能加载结果图片。")
        #     return
        # try:
        #     # 获取Label的尺寸以进行缩放
        #     self.update_idletasks()  # 确保获取到最新的尺寸
        #     label_w = self.image_label.winfo_width()
        #     label_h = self.image_label.winfo_height()
        #
        #     if label_w < 20 or label_h < 20:  # 窗口可能还未完全绘制
        #         self.after(100, self.display_result_image)  # 稍后重试
        #         return
        #
        #     img = Image.open(self.final_image_path)
        #     # 兼容旧版Pillow
        #     thumbnail_method = getattr(Image, 'Resampling', Image).LANCZOS
        #     img.thumbnail((label_w, label_h), thumbnail_method)
        #
        #     photo = ImageTk.PhotoImage(img)
        #     self.image_label.config(image=photo)
        #     self.image_label.image = photo  # 保持引用，防止被垃圾回收
        # except Exception as e:
        #     self.log_message(f"显示结果图片时出错: {e}")
        #     self.image_label.config(text=f"加载图片失败:\n{self.final_image_path}")

    def quit(self):
        if messagebox.askokcancel("退出", "您确定要退出程序吗?"):
            self.destroy()


if __name__ == '__main__':
    # 为了让打包成exe更容易，将ELG的tensorflow依赖检查放在这里
    try:
        import tensorflow as tf

        if not tf.__version__.startswith('1.'):
            print("警告：本程序推荐使用 TensorFlow 1.x 版本。")
    except ImportError:
        print("错误：未安装 TensorFlow。请运行 'pip install tensorflow==1.15'")
        sys.exit(1)

    app = GazeApp()
    app.mainloop()

