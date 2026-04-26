# OmniTriageEnv: Product Requirements Document (PRD)

**Version:** 1.0
**Date:** April 26, 2026
**Authors:** Prakhar, Manu
**Status:** Released (OpenEnv Hackathon India 2026)

---

## 1. Executive Summary

**Product Name:** OmniTriageEnv
**Tagline:** An RL environment that teaches LLMs to read customer emails like a human — and know when to hand off to one.

OmniTriageEnv is an OpenEnv-compliant reinforcement learning environment simulating a high-stakes customer support inbox. It trains LLMs not just to classify text, but to *reason* about implicit context (fraud, legal threats, emotional distress) and make critical multi-step decisions about priority, department routing, response drafting, and human escalation.

---

## 2. Problem Statement

Traditional customer support triage relies on static keyword matching (e.g., if body contains "fraud", route to security). This approach fails catastrophically in the real world:

- **Phishing and Fraud:** Attackers don't use the word "fraud". They use urgency and deceptive links.
- **Ambiguity:** Customers often don't know what they need ("Not sure if I should refund or return").
- **Legal/Compliance Risk:** GDPR requests or legal threats require immediate human handling, but may look like general inquiries to simple classifiers.
- **Emotional Distress:** ALL CAPS, exclamation marks, and denial language signal customer frustration that automated replies will worsen.

LLMs can understand this context, but out-of-the-box models struggle with strict JSON formatting, multi-step reasoning, and knowing *when* they are out of their depth (escalation).

---

## 3. Target Audience / Personas

| Persona | Need |
|---------|------|
| **Customer Support Managers** | Reduce average handling time (AHT) while ensuring high-risk tickets (fraud, legal, angry customers) are never mishandled by AI. |
| **AI/ML Engineers** | A standardized, deterministic environment to train, benchmark, and evaluate LLMs on complex text triage using RL. |
| **Hackathon Judges** | An easily verifiable, demonstrably trained environment showcasing a novel RL application for text-based decision-making. |

---

## 4. Product Vision & Solution Overview

OmniTriageEnv provides a closed-loop system for training and deploying a support triage agent:

1. **The Environment:** A deterministic, 30-email corpus with strict ground truth, spanning 3 difficulty levels (Easy, Medium, Hard). It provides dense, per-action reward signals.
2. **The Training Pipeline:** A GRPO (Group Relative Policy Optimization) pipeline that trains small open-source models (like Llama-3.2-1B) to achieve 100% JSON parse rate and positive rewards on complex cases.
3. **The Live Demo (Inference):** An interactive single-page dashboard (<3s load time) enabling users to test the trained model on custom emails or connect it to a real live Gmail inbox for real-time triage, department routing, and auto-reply generation.

---

## 5. Key Features & Requirements

### 5.1 The RL Environment (OpenEnv Compliant)

- **State Representation:** Must provide current email details (ID, subject, body, sender, tier), queue length, and session score.
- **Action Space:** Must support 6 discrete actions:

| Action | Required Fields | Constraints |
|--------|----------------|-------------|
| `classify_priority` | `priority` (urgent/normal/low) | Must be performed before `assign_department` |
| `assign_department` | `department` (billing/technical/general/returns) | Must be performed before `draft_response` |
| `draft_response` | `response_text` (≥ 10 chars) | Requires prior classification and routing |
| `escalate` | — | Can only be called once per email |
| `archive` | — | Can only be called once per email; signals completion |
| `skip` | — | Budget-limited; excess skips penalized |

- **Recommended Sequence:** `classify_priority → assign_department → draft_response → [escalate?] → archive`
- **Dense Rewards:** Must provide immediate fractional rewards for correct sub-actions (e.g., +0.15 for correct department, −0.15 for missed escalation) rather than sparse episode-end rewards.
- **Loop Prevention:** Must penalize the agent (−0.05) for repeating the same action on the same email for the 3rd+ time.

### 5.2 The Triage Agent Logic

- **Priority Assessment:** Categorize into `urgent`, `normal`, or `low`.
- **Department Routing:** Route to `billing`, `technical`, `returns`, or `general`. Must disambiguate service-failure refunds (→ billing) from product returns (→ returns).
- **Human Escalation:** Flag emails requiring human review based on:
  - Concept-matched fraud patterns (e.g., Link + Banking info + Urgency).
  - Legal, regulatory, or PR threats.
  - High emotional distress (CAPS, exclamation marks, denial language).
  - Intent ambiguity (customer unsure what they need).
  - Insufficient context (≤ 15 words, no clear department signal).
- **Auto-Reply Generation:** Draft a contextual, polite response containing required keywords.

### 5.3 Live Demo Dashboard UI

- **Queue Visualization:** Display incoming emails and their current processing status.
- **Agent Action Log:** Show a real-time feed of the agent's actions and corresponding rewards.
- **Judge Test Panel:** Allow manual input of Subject/Body to instantly test the agent's logic.
- **Live Gmail Integration:** Optionally connect via IMAP to a real Gmail account, polling every 10 seconds, and visualizing live triage with auto-reply.
- **Setup Guide:** Interactive 3-step collapsible wizard (auto-hides when Gmail is connected).

---

## 6. Evaluation Metrics & Success Criteria

### 6.1 Primary Metrics

| Metric | Baseline (Untrained) | After GRPO Training | Target |
|--------|---------------------|---------------------|--------|
| **Average Reward** | 0.276 | **0.438** (+59%) | > 0.40 |
| **JSON Parse Rate** | 93.3% | **100.0%** | 100% |
| **Easy Score** | 0.405 | 0.543 | ≥ 0.70 (passing) |
| **Medium Score** | 0.205 | 0.381 | ≥ 0.60 (passing) |
| **Hard Score** | −0.133 | **+0.100** | ≥ 0.50 (passing) |

### 6.2 Safety Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Fraud Detection Rate** | 7/7 patterns | All tested fraud types correctly identified |
| **Human Escalation Triggers** | 5 categories | Legal, high-risk, ambiguity, distress, insufficient context |
| **False Positive (Fraud)** | Quarantined, not deleted | 7-day retention in Fraud-Quarantine Gmail label |

### 6.3 Non-Functional Requirements (NFRs)

| Requirement | Target | Status |
|-------------|--------|--------|
| API response latency (per `/step`) | ≤ 500ms | ✅ Met (typically <100ms) |
| Dashboard load time | < 3 seconds | ✅ Met (single HTML file, no frameworks) |
| HF Spaces uptime | Best-effort (free tier) | ✅ Deployed |
| Max email corpus (current) | 30 emails | Extensible via `emails.py` |
| Docker image size | < 500MB | ✅ Met |

---

## 7. Technical Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend / Environment** | Python 3.10+, FastAPI, Pydantic | OpenEnv REST compliance, strict schema validation |
| **Inference Agent** | OpenAI-compatible client | Querying local/remote LLMs (Llama, GPT, etc.) |
| **Frontend Dashboard** | Vanilla HTML/CSS/JS | Single-file, zero-dependency, HF Spaces compatible |
| **Training** | HF `trl` (GRPO Trainer), `unsloth` | 4-bit LoRA fine-tuning on free Colab T4 |
| **Deployment** | Docker, HF Spaces | Containerized, auto-deploying on push |
| **Live Email** | IMAP (Gmail), SMTP | Real-time inbox polling + auto-reply |

---

## 8. Risk Assessment & Mitigation

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **GRPO training diverges** (reward collapse) | Medium | Low | Cosine learning rate schedule (5e-6), early stopping. GRPO's group-relative scoring is inherently more stable than PPO. Training logs show noisy but consistently upward trend. |
| **Gmail IMAP connection fails during live demo** | Medium | Medium | Dashboard degrades gracefully: setup guide is shown instead of live inbox. Pre-recorded email demo always available via the Judge Test Panel. |
| **LLM hallucinates non-JSON output** | High | Low (post-training) | Multi-layer defense: (1) structured system prompt with JSON schema, (2) regex-based JSON extraction fallback, (3) keyword heuristic override if LLM output is unparseable. Post-training parse rate: 100%. |
| **Fraud false positive deletes legitimate email** | High | Low | Fraud emails are quarantined (moved to `Fraud-Quarantine` label), never deleted. 7-day retention policy allows manual review and recovery. |
| **HF Spaces goes down** | Low | Low | Environment is fully reproducible locally via `pip install + python server.py`. Docker build also works standalone. |

---

## 9. Out of Scope (& Rationale)

| Feature | Why Deferred |
|---------|-------------|
| **Multi-agent collaboration** (e.g., drafter + reviewer) | Adds orchestration complexity without improving single-agent triage accuracy. Would require a fundamentally different reward structure. |
| **RAG / database integration** for historical ticket retrieval | Introduces non-determinism and external dependencies, conflicting with OpenEnv's requirement for reproducible, self-contained environments. |
| **Real-time chat/voice triage** | The environment is designed for asynchronous email/ticket format. Chat requires streaming, latency optimization, and turn-taking logic that is a separate product. |
| **Larger model training** (7B+ parameters) | Exceeds free Colab T4 GPU memory. 1B model demonstrates the RL methodology; scaling is a straightforward engineering step, not a research gap. |

---

## 10. Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | April 26, 2026 | Initial release for OpenEnv Hackathon India 2026 submission. |
