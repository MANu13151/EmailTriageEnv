---
title: "We Trained an LLM to Handle Customer Emails — Here's What It Learned"
thumbnail: https://huggingface.co/spaces/Prakhar132/email-triage-env/resolve/main/training_output/comparison.png
authors:
- user: Prakhar132
---

# We Trained an LLM to Handle Customer Emails — Here's What It Learned

Ever worked in customer support? If not, picture this: you open your inbox Monday morning and there are 200 unread emails. One is someone asking to reset their password. Another is an enterprise client threatening legal action over a data breach. A third is a confused user asking if your phishing email is legit.

They all look the same in the inbox. But the consequences of handling them wrong are wildly different.

That password reset? Low priority, route to tech support, send a template link. The data breach? Drop everything — that's urgent, legal needs to know, and a human agent needs to take over immediately. The phishing email? That's a fraud case disguised as a support query, and most AI systems would miss it entirely.

**This is the problem we tried to solve.** Not with better keyword matching — we tried that first, and it sucked. Instead, we built an RL environment where an LLM learns to make these judgment calls by practicing on 30 realistic emails, getting feedback after every single decision.

## What We Actually Built

**OmniTriageEnv** is a reinforcement learning environment (built on [OpenEnv](https://openenv.dev)) that simulates a customer support inbox. You give it an email, and the agent has to:

1. **Read it and decide: how urgent is this?** (urgent / normal / low)
2. **Figure out who should handle it** (billing / technical / returns / general)
3. **Write a short reply** that actually addresses the issue
4. **Decide: should a human take over?** This is the hardest part — and the most important one.
5. **Archive it** and move to the next email

The agent gets a score after every action. Not just at the end — after *every single step*. Get the priority right? +0.15. Route to the wrong department? -0.075. Miss an escalation on a fraud email? -0.15. This dense feedback is what makes the model actually learn, instead of just guessing.

## The Part That Surprised Us: Fraud Detection is Hard

Here's what broke our first version. We had this email:

> *"I received an email asking me to click a link and update my banking details to avoid account suspension. The email looks like it's from your company, but something feels off."*

Our keyword system classified it as **"General Inquiry"** because the word "fraud" appears nowhere in the text. But any human reading this would immediately think: *that's a phishing attempt*.

So we stopped thinking like engineers ("what keywords should we add?") and started thinking like humans. What *concepts* make this suspicious?

- Someone received a link ✓
- They're being asked to enter banking details ✓  
- Something "feels off" to them ✓

None of these words are alarming on their own. But *together*, they scream phishing. So we built a **concept-matching system** — instead of checking if "fraud" appears, we check if combinations of ideas appear together. "Link + banking details + suspicious feeling" = Phishing Attempt, risk score 40/100, escalate to human.

We also added **emotional tone analysis**. When someone writes "PLEASE HELP – money deducted without my permission" in all caps with exclamation marks and denial language ("I didn't authorize"), the distress signal alone should flag this as high-risk. Our system scores that at 78/100 distress, which combined with the fraud pattern, gives a risk score of 70/100 and immediately routes to a human agent.

After these changes, our system correctly identifies all 7 types of fraud we tested: account takeover, phishing, social engineering, unauthorized charges, double billing, account compromise, and unauthorized subscriptions — without the word "fraud" appearing in any of them.

## Training: 57 Minutes on a Free GPU

We used **GRPO (Group Relative Policy Optimization)** to train Llama-3.2-1B. The idea is simple: for each email, the model generates 4 different triage attempts. Our reward function scores all 4. GRPO reinforces the best ones and suppresses the worst. No separate critic model needed — which means it fits on a free Colab T4 GPU.

200 training steps. 57 minutes. Here's what happened:

![Before vs After](training_output/comparison.png)

The overall reward went from **0.276 to 0.438** — a 59% improvement. The model went from outputting valid JSON 93% of the time to **100% of the time** (turns out that's learnable too).

But the number that matters most: **hard difficulty went from -0.133 to +0.100**. That's GDPR requests, chargeback disputes, security breach emails — the stuff that actually causes legal and financial damage in the real world. Before training, the model was worse than random on these. After training, it handles them.

![Reward Curve](training_output/reward_curve.png)

The reward curve is noisy (as expected with RL), but the trend is clearly upward. The red moving average goes from ~0.1 to ~0.3 over the training run.

## What We'd Do With More Time

If we had another weekend, we'd:
- **Scale the email corpus** from 30 to 300+ to reduce overfitting
- **Add multi-agent dynamics** — what happens when two agents handle the same inbox?
- **Try larger models** — Llama-3.2-1B is tiny; a 7B model would likely learn faster
- **Deploy the trained model** as the actual agent in our dashboard, so judges can see it triage live

## Try It Yourself

The best part of our submission is the **Judge Test Panel**. Go to our dashboard, type any email you want — a fraud scenario, a billing complaint, a password reset — and watch the system analyze it in real-time. You'll see the fraud patterns it detects, the emotional distress score, and whether it decides to escalate to a human.

- **Live Dashboard:** [prakhar132-email-triage-env.hf.space/dashboard](https://prakhar132-email-triage-env.hf.space/dashboard)
- **Training Notebook:** [Open in Colab](https://huggingface.co/spaces/Prakhar132/email-triage-env/blob/main/OmniTriageEnv_GRPO_Training.ipynb)  
- **Source Code:** [GitHub](https://github.com/MANu13151/OmniTriageEnv)

---

*Built for the OpenEnv Hackathon India 2026.*
