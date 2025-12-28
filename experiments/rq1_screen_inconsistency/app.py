# app.py

import gradio as gr
import os
import shutil
import tempfile
from evaluation_core import run_evaluation, all_paths, mutations, matchers, checkers

# --- 创建 Gradio 界面 ---

# 从核心逻辑文件中获取选项列表
image_choices = all_paths
mutation_choices = list(mutations.keys())
matcher_choices = list(matchers.keys())
checker_choices = list(checkers.keys())


# --- 中间函数 (无需改动) ---
def process_inputs_and_run(dropdown_path, uploaded_image_path, uploaded_json, mutation, matcher, checker):
    """
    处理多种输入方式（下拉选择 vs. 文件上传），并调用核心评估函数。
    它会优先使用上传的文件。
    """
    temp_dir = tempfile.mkdtemp()
    final_image_path = None

    try:
        # --- 情况1: 用户上传了图片和JSON文件 (最高优先级) ---
        # uploaded_image_path 现在会直接是一个字符串路径
        if uploaded_image_path is not None and uploaded_json is not None:
            base_filename = "uploaded_ui"
            temp_image_path = os.path.join(temp_dir, f"{base_filename}.jpg")
            temp_json_path = os.path.join(temp_dir, f"{base_filename}.json")

            # 直接使用路径进行复制
            shutil.copy(uploaded_image_path, temp_image_path)
            shutil.copy(uploaded_json.name, temp_json_path)
            
            final_image_path = temp_image_path

        # --- 情况2: 用户从下拉列表选择 ---
        elif dropdown_path is not None and uploaded_image_path is None:
            final_image_path = dropdown_path
            
        # --- 错误处理 ---
        else:
            if uploaded_image_path is not None and uploaded_json is None:
                error_msg = "错误：您上传了图片，但忘记上传对应的 JSON 标注文件。"
            elif uploaded_image_path is None and uploaded_json is not None:
                error_msg = "错误：您上传了 JSON 文件，但忘记上传对应的图片文件。"
            else:
                error_msg = "错误：请从下拉列表中选择一个文件，或者同时上传一张图片和其对应的 JSON 文件。"
            
            return None, error_msg, ""

        # --- 调用核心评估函数 ---
        return run_evaluation(final_image_path, mutation, matcher, checker)

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


# --- 定义界面 (UI修改) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GUI 一致性检测评估工具")
    gr.Markdown("选择一个UI截图和评估方法，然后点击“运行评估”来查看结果。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 上传文件或从数据集中选择")
            
            # --- 关键修改在这里 ---
            # 将 gr.Image 的 type 从 "file" 改为 "filepath"
            image_upload = gr.Image(type="filepath", label="上传UI截图 (.jpg, .png)")
            
            json_upload = gr.File(label="上传对应的JSON标注文件 (.json)", file_types=[".json"])
            
            gr.Markdown("<center>或</center>")
            
            image_dropdown = gr.Dropdown(choices=image_choices, label="从数据集中选择")
            
            gr.Markdown("### 2. 选择评估模型")
            mutation_dropdown = gr.Dropdown(choices=mutation_choices, label="选择修改（突变）类型", value="swap_widgets")
            matcher_dropdown = gr.Dropdown(choices=matcher_choices, label="选择匹配算法", value="guipilot")
            checker_dropdown = gr.Dropdown(choices=checker_choices, label="选择检查算法", value="gvt")
            run_button = gr.Button("运行评估", variant="primary")

        with gr.Column(scale=3):
            gr.Markdown("### 3. 可视化结果")
            gr.Markdown("（左：原始截图，右：修改后截图。绿色框表示匹配且一致，红色框表示检测到的不一致）")
            output_image = gr.Image(label="结果对比")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 4. 性能指标")
            output_metrics = gr.Textbox(label="评估指标", lines=10)
        with gr.Column():
            gr.Markdown("### 5. 详细对比")
            output_details = gr.Textbox(label="预测 vs. 真实不一致列表", lines=10)

    # 更新按钮点击事件的 inputs
    run_button.click(
        fn=process_inputs_and_run,
        inputs=[image_dropdown, image_upload, json_upload, mutation_dropdown, matcher_dropdown, checker_dropdown],
        outputs=[output_image, output_metrics, output_details]
    )

# 启动界面
if __name__ == "__main__":
    demo.launch()