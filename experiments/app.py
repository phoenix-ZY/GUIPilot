import gradio as gr
import os
import sys
import shutil
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add subdirectories to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "rq1_screen_inconsistency"))
sys.path.append(os.path.join(current_dir, "rq4_case_study"))

# Import RQ1 modules
try:
    from rq1_screen_inconsistency import evaluation_core
    from rq1_screen_inconsistency.evaluation_core import run_evaluation, all_paths, mutations, matchers, checkers
except ImportError:
    # Fallback if direct import fails (e.g. if sys.path isn't enough or structure differs)
    import evaluation_core
    from evaluation_core import run_evaluation, all_paths, mutations, matchers, checkers

# Import RQ4 modules
try:
    from rq4_case_study import case_study_core
except ImportError:
    import case_study_core

# --- RQ1 Logic ---

# Get choices from evaluation_core
image_choices = all_paths
mutation_choices = list(mutations.keys())
matcher_choices = list(matchers.keys())
checker_choices = list(checkers.keys())

def rq1_process_inputs_and_run(dropdown_path, uploaded_image_path, uploaded_json, mutation, matcher, checker):
    """
    Handles inputs for RQ1 (Dropdown vs Upload) and calls evaluation core.
    """
    temp_dir = tempfile.mkdtemp()
    final_image_path = None

    try:
        # Case 1: User uploaded image and JSON
        if uploaded_image_path is not None and uploaded_json is not None:
            base_filename = "uploaded_ui"
            temp_image_path = os.path.join(temp_dir, f"{base_filename}.jpg")
            temp_json_path = os.path.join(temp_dir, f"{base_filename}.json")

            shutil.copy(uploaded_image_path, temp_image_path)
            shutil.copy(uploaded_json.name, temp_json_path)
            
            final_image_path = temp_image_path

        # Case 2: User selected from dropdown
        elif dropdown_path is not None and uploaded_image_path is None:
            final_image_path = dropdown_path
            
        # Error handling
        else:
            if uploaded_image_path is not None and uploaded_json is None:
                error_msg = "Error: You uploaded an image but forgot the JSON file."
            elif uploaded_image_path is None and uploaded_json is not None:
                error_msg = "Error: You uploaded a JSON file but forgot the image."
            else:
                error_msg = "Error: Please select a file from the dropdown OR upload both an image and a JSON file."
            
            return None, error_msg, ""

        return run_evaluation(final_image_path, mutation, matcher, checker)

    finally:
        shutil.rmtree(temp_dir)

# --- RQ4 Logic ---

# Initial values for RQ4
initial_process = case_study_core.PROCESS_IDS[0] if case_study_core.PROCESS_IDS else None
initial_screens = case_study_core.get_screen_choices(initial_process) if initial_process else []

# --- Main UI ---

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=theme, title="GUIPilot Experiments") as demo:
    gr.Markdown("# 🧭 GUIPilot Experiments Platform")
    
    with gr.Tabs():
        
        # ================= RQ1 Tab =================
        with gr.TabItem("RQ1: Screen Inconsistency"):
            gr.Markdown("### 🔍 Screen Inconsistency Detection")
            gr.Markdown("Evaluate visual consistency under simulated mutations.")
            
            with gr.Row(equal_height=False):
                # Left Control Panel
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("#### 🛠️ Configuration")
                    
                    with gr.Tabs():
                        with gr.TabItem("📂 Dataset"):
                            rq1_image_dropdown = gr.Dropdown(
                                choices=image_choices, 
                                label="Select Sample", 
                                info="Choose a UI screenshot from the dataset",
                                interactive=True
                            )
                        
                        with gr.TabItem("📤 Upload"):
                            rq1_image_upload = gr.Image(
                                type="filepath", 
                                label="Upload UI Image", 
                                height=200,
                                sources=["upload", "clipboard"]
                            )
                            rq1_json_upload = gr.File(
                                label="Upload JSON Annotation", 
                                file_types=[".json"],
                                file_count="single"
                            )
                            gr.Markdown("*Uploads take priority over dataset selection.*")

                    with gr.Group():
                        gr.Markdown("#### ⚙️ Parameters")
                        rq1_mutation_dropdown = gr.Dropdown(
                            choices=mutation_choices, 
                            label="Mutation Type", 
                            value="swap_widgets",
                            info="Simulated inconsistency type"
                        )
                        rq1_matcher_dropdown = gr.Dropdown(
                            choices=matcher_choices, 
                            label="Matcher Algorithm", 
                            value="guipilot",
                            info="Algorithm to match widgets"
                        )
                        rq1_checker_dropdown = gr.Dropdown(
                            choices=checker_choices, 
                            label="Checker Algorithm", 
                            value="gvt",
                            info="Algorithm to check properties"
                        )
                    
                    rq1_run_button = gr.Button("🚀 Run Evaluation", variant="primary", size="lg")

                # Right Result Panel
                with gr.Column(scale=2):
                    gr.Markdown("#### 👁️ Visualization")
                    rq1_output_image = gr.Image(
                        label="Result Comparison", 
                        show_label=False,
                        height=600, 
                        interactive=False
                    )
                    gr.Markdown("*(Left: Original | Right: Mutated | 🟩 Match | 🟥 Inconsistency)*")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📊 Metrics")
                            rq1_output_metrics = gr.Markdown()
                        
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📝 Details")
                            rq1_output_details = gr.Textbox(
                                label="Inconsistency Details", 
                                lines=8, 
                                show_copy_button=True,
                                text_align="left"
                            )

            rq1_run_button.click(
                fn=rq1_process_inputs_and_run,
                inputs=[
                    rq1_image_dropdown, 
                    rq1_image_upload, 
                    rq1_json_upload, 
                    rq1_mutation_dropdown, 
                    rq1_matcher_dropdown, 
                    rq1_checker_dropdown
                ],
                outputs=[rq1_output_image, rq1_output_metrics, rq1_output_details]
            )

        # ================= RQ4 Tab =================
        with gr.TabItem("RQ4: Case Study"):
            gr.Markdown("### 📱 Case Study Explorer")
            gr.Markdown("Analyze real-world app process implementations against mockups.")
            
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("#### 📂 Selection")
                    rq4_process_dd = gr.Dropdown(
                        choices=case_study_core.PROCESS_IDS, 
                        label="Select Process", 
                        value=initial_process
                    )
                    rq4_screen_dd = gr.Dropdown(
                        label="Select Screen/Step", 
                        choices=initial_screens, 
                        value=initial_screens[0] if initial_screens else None
                    )
                    
                    # Update screen choices when process changes
                    rq4_process_dd.change(
                        fn=case_study_core.get_screen_choices, 
                        inputs=[rq4_process_dd], 
                        outputs=[rq4_screen_dd]
                    )
                    
                    rq4_load_btn = gr.Button("Load Images", variant="secondary")
                    
                    rq4_info_box = gr.Textbox(label="Step Info", lines=5)
                    
                with gr.Column(scale=2):
                    with gr.Row():
                        rq4_mock_img = gr.Image(label="Mockup", type="numpy", height=400)
                        rq4_real_img = gr.Image(label="Implementation", type="numpy", height=400)
            
            rq4_load_btn.click(
                fn=case_study_core.load_step_images, 
                inputs=[rq4_process_dd, rq4_screen_dd], 
                outputs=[rq4_mock_img, rq4_real_img, rq4_info_box]
            )
            
            gr.Markdown("---")
            gr.Markdown("#### 🧠 Analysis Tools")
            
            with gr.Row():
                # Consistency Check Column
                with gr.Column():
                    gr.Markdown("##### 1. Consistency Check")
                    rq4_check_btn = gr.Button("Run Consistency Check", variant="primary")
                    rq4_consistency_res_img = gr.Image(label="Inconsistencies Visualization", height=400)
                    rq4_consistency_text = gr.Textbox(label="Consistency Stats", lines=8)
                    
                    rq4_check_btn.click(
                        fn=case_study_core.run_consistency_check, 
                        inputs=[rq4_process_dd, rq4_screen_dd], 
                        outputs=[rq4_consistency_res_img, rq4_consistency_text]
                    )
                    
                # Agent Prediction Column
                with gr.Column():
                    gr.Markdown("##### 2. Agent Action Prediction")
                    rq4_api_key_input = gr.Textbox(
                        label="Qwen API Key (Optional if in .env)", 
                        type="password",
                        placeholder="sk-..."
                    )
                    rq4_agent_btn = gr.Button("Run Agent Prediction", variant="primary")
                    rq4_agent_res_img = gr.Image(label="Agent Annotation & Action", height=400)
                    rq4_agent_text = gr.Textbox(label="Agent Results", lines=8)
                    
                    rq4_agent_btn.click(
                        fn=case_study_core.run_agent_prediction, 
                        inputs=[rq4_process_dd, rq4_screen_dd, rq4_api_key_input], 
                        outputs=[rq4_agent_res_img, rq4_agent_text]
                    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
