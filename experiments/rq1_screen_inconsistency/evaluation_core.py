# evaluation_core.py

import os
import csv
import glob
import warnings
import random
from copy import deepcopy
from typing import Callable
import cv2
import numpy as np
from guipilot.entities import Screen, Widget # 确保导入 Widget
# 假设所有依赖项都已安装
# pip install python-dotenv guipilot-utils ...
from dotenv import load_dotenv

# 确保你的 guipilot, mutate, utils 等模块可以被正确导入
# 如果它们在同一目录下，这应该没问题
from guipilot.matcher import WidgetMatcher, GUIPilotV2 as GUIPilotMatcher, GVT as GVTMatcher
from guipilot.checker import ScreenChecker, GVT as GVTChecker
from mutate import insert_row, delete_row, swap_widgets, change_widgets_text, change_widgets_color
from utils import load_screen, convert_inconsistencies, filter_swapped_predictions, filter_overlap_predictions, filter_color, filter_text

# --- 初始化 (这部分代码基本不变) ---
load_dotenv()
warnings.filterwarnings("ignore")
random.seed(42)

dataset_path = os.getenv("DATASET_PATH", "./dataset") # 提供一个默认值

# 获取所有图片路径
all_paths: list[str] = []
if os.path.exists(dataset_path):
    for app_path in glob.glob(os.path.join(dataset_path, "**", "*")):
        if not os.path.isdir(app_path): continue
        image_paths = glob.glob(f"{app_path}/*.jpg")
        image_paths = [x for x in image_paths if x.split("/")[-1].replace(".jpg", "").isdigit()]
        image_paths.sort(key=lambda path: int(path.split("/")[-1].replace(".jpg", "")))
        all_paths += image_paths
else:
    print(f"警告: 数据集路径 '{dataset_path}' 不存在。请检查你的 .env 文件或路径。")


mutations = {
    "insert_row": insert_row, "delete_row": delete_row, "swap_widgets": swap_widgets,
    "change_widgets_text": change_widgets_text, "change_widgets_color": change_widgets_color
}
postprocessing = {
    "insert_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(y_pred, y_true, None, s2),
    "delete_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(y_pred, y_true, s1, None),
    "swap_widgets": lambda y_pred, y_true, s1, s2: filter_swapped_predictions(y_pred, y_true, s1, s2),
    "change_widgets_text": lambda y_pred, y_true, s1, s2: filter_color(y_pred, y_true, s1, None),
    "change_widgets_color": lambda y_pred, y_true, s1, s2: filter_text(y_pred, y_true, s1, None)
}
matchers: dict[str, Callable] = {
    "gvt": lambda screen: GVTMatcher(screen.image.shape[0] / 8),
    "guipilot": lambda screen: GUIPilotMatcher()
}
checkers: dict[str, ScreenChecker] = {"gvt": GVTChecker()}

def metrics(y_pred: set, y_true: set) -> tuple[int, int, int, int]:
    a = set([(x[0], x[1]) for x in y_pred])
    b = set([(x[0], x[1]) for x in y_true])
    cls_tp = len(y_pred.intersection(y_true))
    tp = len(a.intersection(b))
    fn = len(b.difference(a))
    fp = len(a.difference(b))
    return cls_tp, tp, fp, fn

def visualize_for_gui(screen1: Screen, screen2: Screen, pairs: set, inconsistencies: set) -> np.ndarray:
    """
    一个修改版的visualize函数，它不保存文件，而是返回一个可以在GUI中显示的图像Numpy数组。
    这个版本通过提取 widget.bbox 并使用 cv2.rectangle 来正确地绘制边界框。
    """
    s1_image = screen1.image.copy()
    s2_image = screen2.image.copy()

    # 颜色定义 (B, G, R)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    THICKNESS = 2

    # --- 绘制不一致的控件 (红色) ---
    for s1_id, s2_id, _ in inconsistencies:
        if s1_id is not None:
            widget = screen1.widgets.get(s1_id)
            if widget:
                # Bbox 是一个 (xmin, ymin, xmax, ymax) 的元组
                pt1 = (widget.bbox[0], widget.bbox[1])
                pt2 = (widget.bbox[2], widget.bbox[3])
                cv2.rectangle(s1_image, pt1, pt2, COLOR_RED, THICKNESS)
        
        if s2_id is not None:
            widget = screen2.widgets.get(s2_id)
            if widget:
                pt1 = (widget.bbox[0], widget.bbox[1])
                pt2 = (widget.bbox[2], widget.bbox[3])
                cv2.rectangle(s2_image, pt1, pt2, COLOR_RED, THICKNESS)

    # --- 绘制匹配且一致的控件 (绿色) ---
    # 首先，创建一个包含所有不一致配对的集合，以便快速查找
    inconsistent_pairs = set()
    for s1_id, s2_id, _ in inconsistencies:
        if s1_id is not None and s2_id is not None:
            inconsistent_pairs.add((s1_id, s2_id))

    for s1_id, s2_id in pairs:
        # 如果这个配对没有在不一致列表中，那么它就是一致的
        if (s1_id, s2_id) not in inconsistent_pairs:
            s1_widget = screen1.widgets.get(s1_id)
            s2_widget = screen2.widgets.get(s2_id)
            
            if s1_widget:
                pt1 = (s1_widget.bbox[0], s1_widget.bbox[1])
                pt2 = (s1_widget.bbox[2], s1_widget.bbox[3])
                cv2.rectangle(s1_image, pt1, pt2, COLOR_GREEN, THICKNESS)

            if s2_widget:
                pt1 = (s2_widget.bbox[0], s2_widget.bbox[1])
                pt2 = (s2_widget.bbox[2], s2_widget.bbox[3])
                cv2.rectangle(s2_image, pt1, pt2, COLOR_GREEN, THICKNESS)

    # --- 拼接图像和添加标签 (这部分代码不变) ---
    h1, w1, _ = s1_image.shape
    h2, w2, _ = s2_image.shape
    combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_image[:h1, :w1] = s1_image
    combined_image[:h2, w1:w1+w2] = s2_image
    
    cv2.putText(combined_image, "Original Screen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_image, "Mutated Screen", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return combined_image

def run_evaluation(image_path: str, mutation_name: str, matcher_name: str, checker_name: str):
    """
    执行单次评估并返回结果以供GUI显示。
    """
    if not image_path:
        return None, "请先选择一个UI截图。", ""

    try:
        # 1. 加载和突变
        screen1: Screen = load_screen(image_path)
        screen1.ocr()
        screen2 = deepcopy(screen1)
        mutate_func = mutations[mutation_name]
        screen2, y_true = mutate_func(screen2, 0.05) # 固定突变率为5%
        screen2.ocr()

        # 2. 匹配和检查
        init_matcher = matchers[matcher_name]
        checker = checkers[checker_name]
        
        matcher: WidgetMatcher = init_matcher(screen1)
        pairs, _, match_time = matcher.match(screen1, screen2)
        y_pred, check_time = checker.check(screen1, screen2, pairs)

        # 3. 后处理
        y_pred = postprocessing[mutation_name](y_pred, y_true, screen1, screen2)
        
        # 4. 计算指标
        cls_tp, tp, fp, fn = metrics(y_pred, y_true)
        
        try: cls_precision = round(cls_tp / tp, 2)
        except ZeroDivisionError: cls_precision = 0.0
        try: precision = round(tp / (tp + fp), 2)
        except ZeroDivisionError: precision = 0.0
        try: recall = round(tp / (tp + fn), 2)
        except ZeroDivisionError: recall = 0.0

        # 5. 准备输出结果
        result_metrics = (
            f"--- 评估指标 ---\n"
            f"分类准确率 (Cls. Precision): {cls_precision}\n"
            f"准确率 (Precision): {precision}\n"
            f"召回率 (Recall): {recall}\n\n"
            f"--- 统计 ---\n"
            f"真阳性 (TP): {tp} (正确检测到不一致)\n"
            f"假阳性 (FP): {fp} (错误报告了不一致)\n"
            f"假阴性 (FN): {fn} (未能检测到不一致)\n\n"
            f"--- 耗时 ---\n"
            f"匹配耗时: {match_time:.4f} 秒\n"
            f"检查耗时: {check_time:.4f} 秒"
        )
        
        result_details = (
            f"--- 真实的不一致 (Ground Truth) ---\n"
            f"{convert_inconsistencies(y_true)}\n\n"
            f"--- 算法预测的不一致 (Prediction) ---\n"
            f"{convert_inconsistencies(y_pred)}"
        )
        
        # 6. 生成可视化图像
        visualization_image = visualize_for_gui(screen1, screen2, pairs, y_pred)
        
        return visualization_image, result_metrics, result_details

    except Exception as e:
        import traceback
        error_message = f"处理过程中发生错误:\n{e}\n\n{traceback.format_exc()}"
        return None, error_message, ""