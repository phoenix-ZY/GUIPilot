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
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=theme, title="GUIPilot Evaluation") as demo:
    gr.Markdown(
        """
        # 🧭 GUI Pilot 一致性检测平台
        
        本工具用于评估 GUI 界面在不同环境或版本下的视觉一致性。通过模拟突变（Mutation）并使用不同的匹配（Matcher）与检查（Checker）算法，检测并可视化界面中的异常。
        """
    )

    with gr.Row(equal_height=False):
        # 左侧控制面板
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 🛠️ 配置面板")
            
            with gr.Tabs():
                with gr.TabItem("📂 数据集选择"):
                    image_dropdown = gr.Dropdown(
                        choices=image_choices, 
                        label="选择测试样本", 
                        info="从预置数据集中选择一个 UI 截图",
                        interactive=True
                    )
                
                with gr.TabItem("📤 本地上传"):
                    image_upload = gr.Image(
                        type="filepath", 
                        label="上传 UI 截图", 
                        height=200,
                        sources=["upload", "clipboard"]
                    )
                    json_upload = gr.File(
                        label="上传 JSON 标注", 
                        file_types=[".json"],
                        file_count="single"
                    )
                    gr.Markdown("*注意：如果上传了文件，将优先使用上传的数据。*")

            with gr.Group():
                gr.Markdown("#### ⚙️ 算法参数")
                mutation_dropdown = gr.Dropdown(
                    choices=mutation_choices, 
                    label="突变类型 (Mutation)", 
                    value="swap_widgets",
                    info="模拟界面发生的不一致类型"
                )
                matcher_dropdown = gr.Dropdown(
                    choices=matcher_choices, 
                    label="匹配算法 (Matcher)", 
                    value="guipilot",
                    info="用于关联前后两个界面的组件 (GVT仅支持竖屏)"
                )
                checker_dropdown = gr.Dropdown(
                    choices=checker_choices, 
                    label="检查算法 (Checker)", 
                    value="gvt",
                    info="用于判定组件属性是否一致"
                )
            
            run_button = gr.Button("🚀 运行评估 (Run Evaluation)", variant="primary", size="lg")

        # 右侧结果展示
        with gr.Column(scale=2):
            gr.Markdown("### 👁️ 可视化结果")
            output_image = gr.Image(
                label="检测结果对比", 
                show_label=False,
                height=600, 
                interactive=False,
                elem_id="output_img"
            )
            gr.Markdown("*(左图：原始界面 | 右图：突变后界面 | 🟩 绿色：匹配一致 | 🟥 红色：检测到不一致)*")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📊 评估指标")
                    output_metrics = gr.Markdown() # Use Markdown for cleaner text
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 📝 详细日志")
                    output_details = gr.Textbox(
                        label="不一致详情", 
                        lines=8, 
                        show_copy_button=True,
                        text_align="left"
                    )

    # 事件绑定
    run_button.click(
        fn=process_inputs_and_run,
        inputs=[
            image_dropdown, 
            image_upload, 
            json_upload, 
            mutation_dropdown, 
            matcher_dropdown, 
            checker_dropdown
        ],
        outputs=[output_image, output_metrics, output_details]
    )

# 启动界面
if __name__ == "__main__":
    demo.launch()