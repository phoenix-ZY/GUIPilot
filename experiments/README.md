# GUIPilot Experiments Platform

This directory contains the unified experimental platform for GUIPilot, combining the evaluation interfaces for **RQ1 (Screen Inconsistency)** and **RQ4 (Case Study)** into a single application.

## 🔄 Recent Updates

1.  **Unified Interface**: 
    - A new `app.py` has been created in the root of the `experiments/` folder.
    - It uses a tabbed interface to switch seamlessly between the RQ1 and RQ4 experiments without restarting the server.

2.  **Refactored RQ4 Code**:
    - The logic for RQ4 has been separated from the UI.
    - **Old**: `rq4_case_study/app.py` contained both logic and UI.
    - **New**: Core logic is now in `rq4_case_study/case_study_core.py`, making it importable and cleaner.

3.  **Centralized Configuration**:
    - A single `.env` file in the `experiments/` folder now manages configuration for both modules.
    - Specific environment variables (`RQ1_DATASET_PATH`, `RQ4_DATASET_PATH`) allow for precise dataset targeting.

## 📂 Directory Structure

```text
experiments/
├── app.py                  # <--- Main Entry Point (Unified App)
├── .env                    # <--- Central Configuration
├── README.md               # <--- This file
├── rq1_screen_inconsistency/
│   ├── evaluation_core.py  # Core logic for RQ1
│   └── ...
└── rq4_case_study/
    ├── case_study_core.py  # Core logic for RQ4 (Refactored)
    └── ...
```

## 🚀 How to Run

### 1. Environment Setup

Ensure you have the necessary dependencies installed (Gradio, OpenCV, GUIPilot, etc.).

Check the `.env` file in this directory to ensure your paths and API keys are correct:

```dotenv
RQ1_DATASET_PATH=/home/huangtianhao/GUIPilot/datasets
RQ4_DATASET_PATH=/home/huangtianhao/GUIPilot/datasets/rq4
QWEN_API_KEY=sk-...
```

### 2. Launch the Application

Run the unified app from the `experiments` directory:

```bash
cd /home/huangtianhao/GUIPilot/experiments
python app.py
```

### 3. Using the Interface

The application will launch on `http://0.0.0.0:7860`.

*   **Tab 1: RQ1 Screen Inconsistency**
    *   Select a sample from the dataset or upload your own UI screenshot + JSON.
    *   Choose mutation, matcher, and checker types.
    *   Click "Run Evaluation" to see visual inconsistencies and metrics.

*   **Tab 2: RQ4 Case Study**
    *   Select a real-world process implementation from the dropdown.
    *   Load the Mockup vs. Implementation images.
    *   **Consistency Check**: Run the visual consistency algorithms.
    *   **Agent Prediction**: Use the LLM Agent (Qwen) to predict actions and compare them with ground truth.
