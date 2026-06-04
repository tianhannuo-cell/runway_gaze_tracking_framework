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
from gaze_calbration import GazeCalibrationSystem, get_primary_screen_size

def parse_model_ratio(s):
    import argparse

    ratio_list = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        nums = [float(item) for item in part.split('x')]
        if len(nums) != 3:
            raise argparse.ArgumentTypeError(
                "Each model_ratio must be in the form of 'aXbXc', For example, '1x2x3' or '1x2x3,2x3x4'"
            )
        ratio_list.append(nums)

    if not ratio_list:
        raise argparse.ArgumentTypeError("model_ratio 参数不能为空")

    return ratio_list

def app_path():
    if getattr(sys, 'frozen', False):
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

        # --- Data and Status ---
        self.backend_thread = None
        self.comm_queue = queue.Queue()
        # self.final_image_path = None

        self.final_image_path = (os.path.join(app_path(), 'attention_result.png'))
        self.heatmap_path = (os.path.join(app_path(), 'attention_heatmap.png'))
        self.scene_image = (os.path.join(app_path(), 'model321.jpg'))

        self.calibrationPoints = []
        self.calibrationNums = []

        # --- Creating UI Components ---
        self.create_widgets()

        # --- Start queue polling ---
        self.process_queue()

        self.args = None

    def   create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=tk.YES)
        style = ttk.Style(self)
        style.configure('Primary.TButton', foreground='blue', background='#007bff')
        style.configure('Cancel.TButton', foreground='red')
        style.map('Primary.TButton',
            background=[('active', '#0056b3'), ('disabled', '#c0c0c0')])
        style.map('Cancel.TButton',
            foreground=[('disabled', '#c0c0c0')])

        top_panel = ttk.Frame(main_frame)
        top_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # --- Left side: Parameter setting area ---
        settings_frame = ttk.Labelframe(top_panel, text="Parameter Settings", padding="10")
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), anchor='n')

        # 1. Experimental setup
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

        # 2. Model Configuration
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

        # 2.1. Number of models (Spinbox type)
        ttk.Label(model_frame, text="模特数量:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.model_num_var = tk.IntVar(value=1)
        tk.Spinbox(model_frame, from_=1, to=10, textvariable=self.model_num_var, width=5).grid(row=2, column=1,
                                                                                               sticky="w", padx=5,
                                                                                               pady=5)
        # 2.2. Model proportions (Label type)
        # Prompt the user to use the format, such as 1x2x3
        ttk.Label(model_frame, text="模特比例 (头x身x腿):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.model_ratio_var = tk.StringVar(value="24x71x83")
        ttk.Entry(model_frame, textvariable=self.model_ratio_var, width=15).grid(row=3, column=1, sticky="ew", padx=5,
                                                                                 pady=5)

        # 2.3. Model height (Label + Entry)
        ttk.Label(model_frame, text="模特身高:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.model_height = tk.StringVar(value=134)  # 默认身高 175
        ttk.Entry(model_frame, textvariable=self.model_height, width=23).grid(row=4, column=1, sticky="w", padx=5,
                                                                              pady=5)

        input_calib_frame = ttk.Labelframe(settings_frame, text="输入与校准", padding="10")
        input_calib_frame.pack(fill=tk.X, pady=5)

        # 3. Hardware configuration
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

        # 4.Advanced settings
        adv_frame = ttk.Labelframe(settings_frame, text="高级设置", padding="10")
        adv_frame.pack(fill=tk.X, pady=10)
        # 4.1. Upload video
        ttk.Label(adv_frame, text="视频文件:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.adv_video_path_var = tk.StringVar(value="")
        adv_video_entry = ttk.Entry(adv_frame, textvariable=self.adv_video_path_var, state="readonly", width=15)
        adv_video_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        adv_video_btn = ttk.Button(adv_frame, text="浏览...", command=self.browse_video_file)
        adv_video_btn.grid(row=0, column=2, sticky="w", padx=5, pady=2)
        # style = Style()
        # style.configure("TButton", , background="blue", relief="raised")
        self.cancel_video_btn = ttk.Button(adv_frame, text="×", command=self.cancel_video_selection, width=2, state="disabled",style='Cancel.TButton')
        self.cancel_video_btn.grid(row=0, column=3, sticky="w", padx=2)
        # 4.1.1. Video Analysis Button
        self.video_analysis_button = ttk.Button(adv_frame, text="视频视线分析与结果导出", state="disabled", command=self.run_video_analysis, style='Primary.TButton')
        self.video_analysis_button.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=(5,0))

        # ttk.Separator(adv_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky='ew', pady=10)

        # 4.2. Number of calibration points
        ttk.Label(adv_frame, text="校准点网格:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(adv_frame, text="行:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.adv_rows_var = tk.IntVar(value=5)
        adv_rows_spinbox = tk.Spinbox(adv_frame, from_=2, to=10, textvariable=self.adv_rows_var, width=5)
        adv_rows_spinbox.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(adv_frame, text="列:").grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.adv_cols_var = tk.IntVar(value=6)
        adv_cols_spinbox = tk.Spinbox(adv_frame, from_=2, to=10, textvariable=self.adv_cols_var, width=5)
        adv_cols_spinbox.grid(row=4, column=1, sticky="w", padx=5, pady=2)

        # ttk.Separator(adv_frame, orient='horizontal').grid(row=5, column=0, columnspan=3, sticky='ew', pady=5)

        # --- Right side: Status and output area ---
        output_frame = ttk.Frame(top_panel)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # 4. Status Log
        log_frame = ttk.Labelframe(output_frame, text="状态日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=tk.YES)
        self.log_text = scrolledtext.Text(log_frame, wrap=tk.WORD, height=10, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=tk.YES)

        # 5. Results Preview
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

        # ---Bottom: Main Control Area ---
        action_frame = ttk.Frame(main_frame, padding="10 0 0 0")
        action_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(action_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=tk.YES, side=tk.LEFT, padx=(0, 10))

        self.start_button = ttk.Button(action_frame, text="开始校准与实验", command=self.start_experiment, width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.exit_button = ttk.Button(action_frame, text="退出", command=self.quit)
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Control group: Convenient for unified disabling/enabling
        self.settings_widgets = [exp_frame, model_frame, hw_frame]

        self.auto_detect_screen_size()

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
        messagebox.showinfo("分析完成", "视频分析完成！")

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
        self.exit_button.config(state=state)

    def start_experiment(self):
        # 1. Validate Input
        screen_size_str = self.screen_size_var.get()
        if not re.match(r'^\d+x\d+$', screen_size_str):
            messagebox.showerror("输入错误", "屏幕尺寸格式不正确，应为 '宽x高' (例如 '1920x1080')。")
            return

        if not self.scene_image_var.get():
            messagebox.showerror("输入错误", "请选择一个场景图片。")
            return

        # 2. Collect Parameters
        settings = {
            'participant_id': self.participant_id_var.get(),
            'scene_image': self.scene_image_var.get(),
            'model': self.map_model_var.get(),
            'filter': self.filter_var.get(),
            'camera_id': self.camera_id_var.get(),
            'screen_size': [int(i) for i in screen_size_str.split('x')],
            'generate_heatmap': True
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

        # Upload Video Related
        parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam',
                            default= self.adv_video_path_var.get())
        parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.',
                            default= f'{self.adv_video_path_var.get()}_output.mp4')

        # Calibration Point Related
        parser.add_argument('--calibrationNums', type= int, nargs= '+',
                            default= [self.adv_rows_var.get(), self.adv_cols_var.get()])

        parser.add_argument('--figName', default=self.participant_id_var.get())
        parser.add_argument('--model_num', type=int, default=self.model_num_var.get())
        parser.add_argument('--model_ratio', type=parse_model_ratio, default=self.model_ratio_var.get(),
                            help="每个人形的头-躯干-腿的比例。"
                                 "示例: '1x2x3'（单个人形）或 '1x2x3,2x3x4,1x1x2'（多个人形各自比例）"
                            )
        parser.add_argument('--model_height', type=lambda s: [float(item) for item in s.split(',')],default=self.model_height.get())

        # Camera Mode
        if self.adv_video_path_var.get() == "":
            parser.add_argument('--csv_path', type=str)
            args = parser.parse_args()
            # Disable UI and start backend thread
            self.toggle_settings_enabled(enabled=False)
            self.log_message("--- Start a new task ---")
            self.progress_bar['value'] = 0
            self.run_backend_task(args,self.comm_queue)
           
        # Video mode
        else:

            parser.add_argument('--csv_path', type=str, default=(os.path.join(app_path(), 'demo_gaze_result.csv')))

            args = parser.parse_args()
            # Disable UI and start backend thread
            self.toggle_settings_enabled(enabled=False)
            self.log_message("--- Start a new task ---")
            self.progress_bar['value'] = 0
            self.backend_thread = threading.Thread(
                target=self.run_backend_task,
                args=(args, self.comm_queue),
                daemon=True
            )
            self.backend_thread.start()

    def run_backend_task(self, args, comm_queue):
        """This function runs in a separate thread and will not block the GUI"""
        try:
            backend = GazeCalibrationSystem(args)
            backend.run()
        except Exception as e:
            # Formatting of exception information
            import traceback
            error_info = f"后端任务发生严重错误:\n{traceback.format_exc()}"
            comm_queue.put(("ERROR", error_info))
        finally:
            comm_queue.put(("FINISHED", None))

    def process_queue(self):
        """Periodically check the queue for messages from the backend and update the UI accordingly"""
        try:
            while True:
                msg_type, msg_data = self.comm_queue.get_nowait()

                if msg_type == "LOG":
                    self.log_message(msg_data)
                elif msg_type == "PROGRESS":
                    current, total = msg_data
                    self.progress_bar['value'] = (current / total) * 100
                elif msg_type == "HIDE_GUI":
                    self.withdraw()
                elif msg_type == "SHOW_GUI":
                    self.deiconify()
                elif msg_type == "RESULT":
                    # self.final_image_path = msg_data['img_path']
                    self.display_result_image()
                elif msg_type == "ERROR":
                    messagebox.showerror("后端错误", msg_data)
                elif msg_type == "FINISHED":
                    self.toggle_settings_enabled(enabled=True)
                    self.log_message("--- Task completed ---")
                    break  # Exit the loop until the next task.

            self.display_result_image()

        except queue.Empty:
            pass
        finally:
            # The queue is checked every 100ms
            self.after(100, self.process_queue)

    def _load_and_display_image(self, path, label_widget):
        """Helper functions for loading, scaling, and displaying a single image"""
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

    def quit(self):
        if messagebox.askokcancel("退出", "您确定要退出程序吗?"):
            self.destroy()


if __name__ == '__main__':
    # To make packaging into an EXE easier, the ELG TensorFlow dependency check is placed here
    try:
        import tensorflow as tf

        if not tf.__version__.startswith('1.'):
            print("警告：本程序推荐使用 TensorFlow 1.x 版本。")
    except ImportError:
        print("错误：未安装 TensorFlow。请运行 'pip install tensorflow==1.15'")
        sys.exit(1)

    app = GazeApp()
    app.mainloop()

