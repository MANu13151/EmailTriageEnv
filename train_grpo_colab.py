# OmniTriageEnv — GRPO Training Notebook
# ========================================
# This notebook trains an LLM to perform omnichannel customer triage
# using GRPO (Group Relative Policy Optimization).
#
# ⚡ Designed for Google Colab with T4 GPU
# 📦 Uses Unsloth + HuggingFace TRL
#
# HOW TO RUN:
# 1. Open in Google Colab
# 2. Select GPU runtime (T4)
# 3. Run All Cells
# 4. Results will be in ./training_output/

# %% [markdown]
# # 🚀 OmniTriageEnv — GRPO Training
# Train an LLM to triage customer communications (emails, grievances, social media)
# using reinforcement learning with our custom environment as the reward signal.

# %% Cell 1: Install Dependencies
# !pip install -q unsloth trl datasets matplotlib pydantic fastapi

# %% Cell 2: Clone Environment
# !git clone https://huggingface.co/spaces/Prakhar132/email-triage-env OmniTriageEnv
# %cd OmniTriageEnv

# %% Cell 3: Run Training
# !python train_grpo.py --model unsloth/Llama-3.2-1B-Instruct --steps 200 --batch-size 4

# %% Cell 4: Display Results
# from IPython.display import Image, display
# display(Image("./training_output/reward_curve.png"))
# display(Image("./training_output/comparison.png"))

# %% Cell 5: Show Comparison Data
# import json
# with open("./training_output/comparison_results.json") as f:
#     results = json.load(f)
# print(json.dumps(results, indent=2))
