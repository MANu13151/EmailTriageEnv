# OmniTriageEnv: Product Requirements Document (PRD)

## 1. Executive Summary

**Product Name:** OmniTriageEnv
**Tagline:** An RL environment that teaches LLMs to read customer emails like a human — and know when to hand off to one.
**Hackathon:** OpenEnv Hackathon India 2026

**OmniTriageEnv** is an OpenEnv-compliant reinforcement learning environment simulating a high-stakes customer support inbox. It trains LLMs not just to classify text, but to *reason* about implicit context (like fraud or legal threats) and make critical decisions about priority, department routing, and human escalation.

## 2. Problem Statement

Traditional customer support triage relies on static keyword matching (e.g., if body contains "fraud", route to security). This approach fails catastrophically in the real world:
- **Phishing and Fraud:** Attackers don't use the word "fraud". They use urgency and links.
- **Ambiguity:** Customers often don't know what they need ("Not sure if I should refund or return").
- **Legal/Compliance Risk:** GDPR requests or legal threats require immediate human handling, but might look like general inquiries to simple classifiers.

LLMs can understand this context, but out-of-the-box models struggle with strict JSON formatting, multi-step reasoning, and knowing *when* they are out of their depth (escalation). 

## 3. Target Audience / Personas

1. **Customer Support Managers:** Need to reduce average handling time (AHT) while ensuring high-risk tickets (fraud, legal, angry customers) are never mishandled by AI.
2. **AI/ML Engineers:** Need a standardized, deterministic environment to train, benchmark, and evaluate LLMs on complex text triage tasks using Reinforcement Learning (RL).
3. **Hackathon Judges:** Need an easily verifiable, demonstrably trained environment that showcases a novel application of RL for text-based decision-making.

## 4. Product Vision & Solution Overview

OmniTriageEnv provides a closed-loop system for training and deploying a support triage agent:

1. **The Environment:** A deterministic, 30-email corpus with strict ground truth, spanning 3 difficulty levels (Easy, Medium, Hard). It provides dense, per-action reward signals.
2. **The Training Pipeline:** A GRPO (Group Relative Policy Optimization) pipeline that trains small open-source models (like Llama-3.2-1B) to achieve near-perfect JSON parsing and positive rewards on complex cases.
3. **The Live Demo (Inference):** A beautiful, interactive dashboard that allows users to test the trained model on custom emails or hook it up to a real live Gmail inbox to see real-time triage, department routing, and auto-reply generation.

## 5. Key Features & Requirements

### 5.1 The RL Environment (OpenEnv Compliant)
- **State Representation:** Must provide current email details (ID, subject, body, sender, tier), queue length, and session score.
- **Action Space:** Must support 6 discrete actions: `classify_priority`, `assign_department`, `draft_response`, `escalate`, `archive`, `skip`.
- **Dense Rewards:** Must provide immediate fractional rewards for correct sub-actions (e.g., +0.15 for correct department, -0.15 for missed escalation) rather than sparse episode-end rewards.
- **Loop Prevention:** Must penalize the agent for repeating the same action on the same email endlessly.

### 5.2 The Triage Agent Logic
- **Priority Assessment:** Categorize into `urgent`, `normal`, or `low`.
- **Department Routing:** Route to `billing`, `technical`, `returns`, or `general`.
- **Human Escalation:** Accurately flag emails that require human review based on:
  - Concept-matched fraud patterns (e.g., Link + Banking info + Urgency).
  - Legal, regulatory, or PR threats.
  - High emotional distress.
  - Intent ambiguity.
- **Auto-Reply Generation:** Draft a contextual, polite response of at least 10 characters.

### 5.3 Live Demo Dashboard UI
- **Queue Visualization:** Display incoming emails and their current processing status.
- **Agent Action Log:** Show a real-time feed of the agent's actions and corresponding rewards.
- **Judge Test Panel:** Allow manual input of Subject/Body to instantly test the agent's logic.
- **Live Gmail Integration:** Must optionally connect via IMAP to a real Gmail account, polling every 10 seconds, and visualising live triage.

## 6. Evaluation Metrics & Success Criteria

The primary success metric is the **Session Score (Reward)** achieved across the environment's difficulties:

| Difficulty | Passing Threshold | Description |
|---|---|---|
| **Easy** | ≥ 0.70 | Explicit signals, clear keywords, category hints provided. |
| **Medium** | ≥ 0.60 | Implicit signals, nuanced technical language, no hints. |
| **Hard** | ≥ 0.50 | Complex domain knowledge required (GDPR, security), heavy penalties for missed escalations. |

*Note: After GRPO training, the Llama-3.2-1B model improved its average reward by 59% and successfully moved the Hard difficulty score from negative (-0.133) to positive (+0.100).*

## 7. Technical Architecture

- **Backend / Environment:** Python 3.10+, FastAPI (for OpenEnv REST compliance), Pydantic for strict schema validation.
- **Inference Agent:** OpenAI-compatible client integration for querying local/remote LLMs.
- **Frontend Dashboard:** Vanilla HTML/CSS/JS (no heavy frameworks, single file for easy HF Spaces deployment).
- **Training:** Hugging Face `trl` (GRPO Trainer), `unsloth` for fast 4-bit LoRA fine-tuning.
- **Deployment:** Dockerized and hosted on Hugging Face Spaces.

## 8. Out of Scope (For Now)
- Multi-agent collaboration (e.g., one agent drafts, a second agent reviews).
- Native database integrations for historical ticket retrieval (RAG).
- Real-time chat/voice triage (restricted to asynchronous email/ticket format).
