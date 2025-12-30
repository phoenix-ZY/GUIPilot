import os
import json
import cv2
import numpy as np
from dotenv import load_dotenv
from guipilot.agent import QwenAgent
from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher
from guipilot.checker import GVT as GVTChecker
from guipilot.entities import Screen
# Assuming utils is in the same directory as this file
try:
    from .utils import get_screen, get_scores, get_action_completion, visualize, check_action
except ImportError:
    from utils import get_screen, get_scores, get_action_completion, visualize, check_action

# --- Initialization ---
load_dotenv()
mockups_path = os.getenv("RQ4_DATASET_PATH", os.getenv("DATASET_PATH"))
if not mockups_path:
    # Fallback or warning if env var is not set
    print("Warning: DATASET_PATH not set in .env")
    # Fix: Use correct relative path from this script to datasets
    # Assuming this file is in experiments/rq4_case_study/
    mockups_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/rq4"))

print(f"Using mockups path: {mockups_path}")

def get_processes(mockups_path: str) -> dict:
    """Get all paths of the nested processes in the case study dataset, mapped by ID."""
    processes = {}
    if not os.path.exists(mockups_path):
        print(f"Error: Mockups path does not exist: {mockups_path}")
        return processes
        
    for root, dirs, _ in os.walk(mockups_path):
        if "mockup" in dirs:
            full_path = os.path.abspath(root)
            # Create a readable ID relative to mockups_path
            rel_path = os.path.relpath(full_path, mockups_path)
            if rel_path == ".":
                rel_path = os.path.basename(full_path)
            processes[rel_path] = full_path
    
    print(f"Found {len(processes)} processes: {list(processes.keys())}")
    return processes

# Global state to cache loaded process data to avoid reloading json constantly
PROCESS_CACHE = {}
PROCESS_MAP = get_processes(mockups_path)
PROCESS_IDS = sorted(list(PROCESS_MAP.keys()))

def get_process_data(process_id):
    if process_id not in PROCESS_CACHE:
        process_path = PROCESS_MAP[process_id]
        json_path = os.path.join(process_path, "implementation", "process.json")
        print(f"Loading process data from: {json_path}")
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            PROCESS_CACHE[process_id] = data
            print(f"Loaded {len(data)} steps for process {process_id}")
        except Exception as e:
            print(f"Error loading process json: {e}")
            return []
    return PROCESS_CACHE[process_id]

def get_screen_choices(process_id):
    if not process_id:
        return []
    data = get_process_data(process_id)
    choices = []
    for i, step in enumerate(data):
        screen_name = step.get("screen", f"Screen {i}")
        choices.append(f"{i}: {screen_name}")
    return choices

def load_step_images(process_id, screen_choice):
    if not process_id or not screen_choice:
        return None, None, "Please select a process and a screen."
    
    try:
        step_idx = int(screen_choice.split(":")[0])
        process_path = PROCESS_MAP[process_id]
        data = get_process_data(process_id)
        step = data[step_idx]
        screen_filename = step["screen"]
        
        implementation_path = os.path.join(process_path, "implementation")
        mockup_path = os.path.join(process_path, "mockup")
        
        # Use utils.get_screen logic but just return image for display
        # Note: get_screen resizes images. We should probably show the resized ones 
        # to match what the algorithm sees, or raw ones. 
        # Let's use get_screen to ensure consistency with the algorithm.
        
        real_screen = get_screen(implementation_path, screen_filename)
        mock_screen = get_screen(mockup_path, screen_filename)
        
        # Convert BGR to RGB for Gradio
        real_img_rgb = cv2.cvtColor(real_screen.image, cv2.COLOR_BGR2RGB)
        mock_img_rgb = cv2.cvtColor(mock_screen.image, cv2.COLOR_BGR2RGB)
        
        info_text = f"Loaded Step {step_idx}\nScreen File: {screen_filename}\n"
        info_text += f"Mock Actions: {step.get('mock_actions')}\n"
        info_text += f"True Actions: {step.get('actions')}"
        
        return mock_img_rgb, real_img_rgb, info_text
        
    except Exception as e:
        return None, None, f"Error loading images: {str(e)}"

def run_consistency_check(process_id, screen_choice):
    if not process_id or not screen_choice:
        return None, "Please select a process and a screen."
    
    try:
        step_idx = int(screen_choice.split(":")[0])
        process_path = PROCESS_MAP[process_id]
        data = get_process_data(process_id)
        step = data[step_idx]
        screen_filename = step["screen"]
        
        implementation_path = os.path.join(process_path, "implementation")
        mockup_path = os.path.join(process_path, "mockup")
        
        matcher = GUIPilotMatcher()
        checker = GVTChecker()
        
        real_screen = get_screen(implementation_path, screen_filename)
        mock_screen = get_screen(mockup_path, screen_filename)
        
        pairs, score, match_time = get_scores(mock_screen, real_screen, matcher)
        inconsistencies, check_time = checker.check(mock_screen, real_screen, pairs)
        
        # visualize returns: image, bbox_image, match_image
        # We usually want the one showing inconsistencies (image) or matches (match_image)
        # The main.py saves all three. Let's return the main 'image' which usually has inconsistencies marked.
        image, bbox_image, match_image = visualize(mock_screen, real_screen, pairs, inconsistencies)
        
        res_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result_text = f"Consistency Score: {score:.4f}\n"
        result_text += f"Inconsistencies Found: {len(inconsistencies)}\n"
        result_text += f"Match Time: {match_time:.2f}ms\n"
        result_text += f"Check Time: {check_time:.2f}ms"
        
        return res_img_rgb, result_text
        
    except Exception as e:
        return None, f"Error during analysis: {str(e)}"

def run_agent_prediction(process_id, screen_choice, api_key):
    if not process_id or not screen_choice:
        return None, "Please select a process and a screen."
    
    if not api_key:
        # Try to get from env
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            return None, "Please provide Qwen API Key."

    try:
        step_idx = int(screen_choice.split(":")[0])
        process_path = PROCESS_MAP[process_id]
        data = get_process_data(process_id)
        step = data[step_idx]
        screen_filename = step["screen"]
        mock_actions = step["mock_actions"]
        true_actions = step["actions"]
        
        implementation_path = os.path.join(process_path, "implementation")
        real_screen = get_screen(implementation_path, screen_filename)
        
        agent = QwenAgent(api_key=api_key)
        
        # Run agent (single trial for GUI to save time/cost, or maybe loop?)
        # Let's do one trial per click
        agent_image, action_names, actions_raw, actions, raw_response = get_action_completion(agent, real_screen, mock_actions)
        
        # Check correctness
        is_correct_length = len(actions) == len(true_actions)
        
        action_checks = []
        if is_correct_length:
            for true_action, action_name, action in zip(true_actions, action_names, actions):
                check = check_action(true_action, action_name, action)
                action_checks.append(check)
            all_correct = all(action_checks)
        else:
            all_correct = False
            
        # Convert agent image (annotated) to RGB
        # agent_image from get_action_completion is likely BGR if it came from cv2
        res_img_rgb = cv2.cvtColor(agent_image, cv2.COLOR_BGR2RGB)
        
        result_text = f"Full Model Response:\n{raw_response}\n\n"
        result_text += f"Raw Agent Output: {actions_raw}\n"
        result_text += f"Parsed Actions: {actions}\n"
        result_text += f"True Actions: {true_actions}\n"
        result_text += f"Length Correct: {is_correct_length}\n"
        result_text += f"Actions Correct: {all_correct}\n"
        
        return res_img_rgb, result_text
    except Exception as e:
        import traceback
        return None, f"Error running agent: {str(e)}\n{traceback.format_exc()}"
