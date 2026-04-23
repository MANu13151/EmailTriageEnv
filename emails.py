"""
Deterministic email corpus with ground-truth labels.
All emails are fixed — no randomness.
Ground truth is used ONLY by graders, never exposed to the agent directly.
"""
from __future__ import annotations
from typing import Dict, Any, List

# ── Ground truth reference (used by graders only) ────────────────────────────
# Structure: email_id -> {priority, department, escalate, keywords_required}

GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    # ── EASY TASK emails (E001-E010) ─────────────────────────────────────────
    "E001": {
        "priority": "urgent",
        "department": "billing",
        "escalate": False,
        "response_keywords": ["refund", "apologize", "processed"],
    },
    "E002": {
        "priority": "low",
        "department": "general",
        "escalate": False,
        "response_keywords": ["thank", "password", "reset"],
    },
    "E003": {
        "priority": "normal",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["team", "issue", "investigate"],
    },
    "E004": {
        "priority": "urgent",
        "department": "technical",
        "escalate": True,
        "response_keywords": ["escalat", "senior", "urgent"],
    },
    "E005": {
        "priority": "low",
        "department": "returns",
        "escalate": False,
        "response_keywords": ["return", "label", "ship"],
    },
    "E006": {
        "priority": "normal",
        "department": "billing",
        "escalate": False,
        "response_keywords": ["invoice", "send", "detail"],
    },
    "E007": {
        "priority": "low",
        "department": "general",
        "escalate": False,
        "response_keywords": ["hour", "support", "contact"],
    },
    "E008": {
        "priority": "normal",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["error", "log", "check"],
    },
    "E009": {
        "priority": "urgent",
        "department": "billing",
        "escalate": True,
        "response_keywords": ["fraud", "secur", "block"],
    },
    "E010": {
        "priority": "low",
        "department": "returns",
        "escalate": False,
        "response_keywords": ["condition", "policy", "return"],
    },
    # ── MEDIUM TASK emails (M001-M010) ────────────────────────────────────────
    "M001": {
        "priority": "urgent",
        "department": "billing",
        "escalate": True,
        "response_keywords": ["escalat", "account", "charge"],
    },
    "M002": {
        "priority": "normal",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["integrat", "api", "document"],
    },
    "M003": {
        "priority": "low",
        "department": "general",
        "escalate": False,
        "response_keywords": ["feature", "roadmap", "future"],
    },
    "M004": {
        "priority": "urgent",
        "department": "technical",
        "escalate": True,
        "response_keywords": ["data loss", "backup", "immedi"],
    },
    "M005": {
        "priority": "normal",
        "department": "returns",
        "escalate": False,
        "response_keywords": ["exchange", "size", "availab"],
    },
    "M006": {
        "priority": "normal",
        "department": "billing",
        "escalate": False,
        "response_keywords": ["discount", "plan", "upgrade"],
    },
    "M007": {
        "priority": "urgent",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["restart", "clear", "cache"],
    },
    "M008": {
        "priority": "low",
        "department": "general",
        "escalate": False,
        "response_keywords": ["certif", "complet", "course"],
    },
    "M009": {
        "priority": "normal",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["webhook", "configur", "endpoint"],
    },
    "M010": {
        "priority": "urgent",
        "department": "billing",
        "escalate": False,
        "response_keywords": ["refund", "24 hour", "process"],
    },
    # ── HARD TASK emails (H001-H010) ──────────────────────────────────────────
    "H001": {
        "priority": "urgent",
        "department": "technical",
        "escalate": True,
        "response_keywords": ["compli", "legal", "escalat"],
    },
    "H002": {
        "priority": "normal",
        "department": "billing",
        "escalate": False,
        "response_keywords": ["tax", "invoice", "correct"],
    },
    "H003": {
        "priority": "low",
        "department": "general",
        "escalate": False,
        "response_keywords": ["partner", "team", "contact"],
    },
    "H004": {
        "priority": "urgent",
        "department": "billing",
        "escalate": True,
        "response_keywords": ["chargeback", "bank", "document"],
    },
    "H005": {
        "priority": "normal",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["migrat", "export", "format"],
    },
    "H006": {
        "priority": "urgent",
        "department": "technical",
        "escalate": True,
        "response_keywords": ["breach", "secur", "notify"],
    },
    "H007": {
        "priority": "low",
        "department": "returns",
        "escalate": False,
        "response_keywords": ["inspect", "photo", "review"],
    },
    "H008": {
        "priority": "normal",
        "department": "billing",
        "escalate": False,
        "response_keywords": ["prorat", "credit", "adjust"],
    },
    "H009": {
        "priority": "urgent",
        "department": "general",
        "escalate": True,
        "response_keywords": ["press", "pr", "statement"],
    },
    "H010": {
        "priority": "normal",
        "department": "technical",
        "escalate": False,
        "response_keywords": ["rate limit", "throttl", "quota"],
    },
}


# ── Email corpus ──────────────────────────────────────────────────────────────

EMAILS: Dict[str, Dict[str, Any]] = {
    # ── EASY ──────────────────────────────────────────────────────────────────
    "E001": {
        "email_id": "E001",
        "subject": "Double charge on my account - URGENT",
        "body": (
            "Hello, I was charged twice for my subscription this month. "
            "The duplicate charge of $49.99 appeared on 2024-03-01 and again on 2024-03-02. "
            "I need this resolved immediately and the extra charge refunded. "
            "This is unacceptable for a paying customer."
        ),
        "sender": "james.wilson@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T09:15:00Z",
        "category_hint": "billing_dispute",
    },
    "E002": {
        "email_id": "E002",
        "subject": "How do I reset my password?",
        "body": (
            "Hi support team, I forgot my password and cannot log in. "
            "I tried the reset link but didn't get an email. "
            "Can you help me reset it manually? My username is sarah_t."
        ),
        "sender": "sarah.thomas@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T10:00:00Z",
        "category_hint": "account_access",
    },
    "E003": {
        "email_id": "E003",
        "subject": "App crashes when exporting reports",
        "body": (
            "The export feature in the dashboard has been crashing for the past two days. "
            "I get error code 503 every time I click 'Export to CSV'. "
            "This is blocking my weekly reporting process. "
            "Please investigate and fix this issue."
        ),
        "sender": "mike.chen@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T11:30:00Z",
        "category_hint": "bug_report",
    },
    "E004": {
        "email_id": "E004",
        "subject": "CRITICAL: Production database corrupted",
        "body": (
            "Our entire production database appears to be corrupted after your last update. "
            "We cannot access any customer records. This is causing a complete business outage. "
            "We are an enterprise customer paying $5000/month. "
            "We need your senior engineering team involved IMMEDIATELY. "
            "Every minute of downtime costs us $10,000."
        ),
        "sender": "cto@bigcorp.com",
        "sender_tier": "enterprise",
        "received_at": "2024-03-02T08:00:00Z",
        "category_hint": "critical_outage",
    },
    "E005": {
        "email_id": "E005",
        "subject": "Return request for order #45231",
        "body": (
            "Hello, I would like to return the item I purchased last week. "
            "Order number 45231. The color was different from the website photo. "
            "Please send me a return label. Thank you."
        ),
        "sender": "linda.park@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T14:00:00Z",
        "category_hint": "return_request",
    },
    "E006": {
        "email_id": "E006",
        "subject": "Need copy of January invoice",
        "body": (
            "Hi, I need a copy of my January 2024 invoice for expense reporting purposes. "
            "Could you please resend it to my email or provide a download link? "
            "Account ID: ACC-8821."
        ),
        "sender": "robert.gray@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T13:00:00Z",
        "category_hint": "invoice_request",
    },
    "E007": {
        "email_id": "E007",
        "subject": "What are your support hours?",
        "body": (
            "Hi there, I just wanted to know what your customer support hours are. "
            "I tried calling yesterday at 8pm and no one answered. "
            "Do you have weekend support available?"
        ),
        "sender": "amy.jones@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T15:45:00Z",
        "category_hint": "general_inquiry",
    },
    "E008": {
        "email_id": "E008",
        "subject": "Getting 404 errors on API endpoints",
        "body": (
            "I'm integrating your REST API and keep getting 404 errors on /api/v2/users endpoint. "
            "My API key is valid (checked the dashboard). "
            "Error appears consistently for GET requests. "
            "Can you check if the endpoint is down or if my config is wrong?"
        ),
        "sender": "dev.team@startup.io",
        "sender_tier": "pro",
        "received_at": "2024-03-02T12:00:00Z",
        "category_hint": "api_error",
    },
    "E009": {
        "email_id": "E009",
        "subject": "FRAUD ALERT: Unauthorized transactions on my account",
        "body": (
            "I am seeing multiple unauthorized charges on my account totaling $847.50. "
            "I did NOT authorize these transactions. This looks like fraud. "
            "I need my account blocked immediately and these charges reversed. "
            "I am also contacting my bank. Please respond URGENTLY."
        ),
        "sender": "victim123@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T07:30:00Z",
        "category_hint": "fraud_report",
    },
    "E010": {
        "email_id": "E010",
        "subject": "Can I return a used item?",
        "body": (
            "Hello, I bought a product 45 days ago and have been using it regularly. "
            "I'm not completely satisfied with it. Is it possible to return it? "
            "I know your policy says 30 days but wondered if there are exceptions."
        ),
        "sender": "curious.customer@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T16:00:00Z",
        "category_hint": "return_policy_question",
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    "M001": {
        "email_id": "M001",
        "subject": "Unauthorized renewal charge after cancellation",
        "body": (
            "I cancelled my subscription on February 15th and received a confirmation email. "
            "Despite this, you charged my credit card $299 on March 1st for an annual renewal. "
            "I have the cancellation confirmation email with reference #CANC-4421. "
            "This is an unauthorized charge. I expect an immediate refund and an explanation "
            "of how this happened. If not resolved today, I will dispute with my credit card company."
        ),
        "sender": "angry.customer@email.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T08:45:00Z",
        "category_hint": "billing_dispute_escalation",
    },
    "M002": {
        "email_id": "M002",
        "subject": "API rate limits too restrictive for our use case",
        "body": (
            "We are building a high-volume data pipeline using your API. "
            "The current rate limit of 100 requests/minute is insufficient for our batch jobs. "
            "We need at least 1000 req/min. We are on the Pro tier ($299/month). "
            "Can you provide documentation on higher-tier plans or custom rate limit options? "
            "We are evaluating whether to stay with your platform or migrate to a competitor."
        ),
        "sender": "backend@techfirm.io",
        "sender_tier": "pro",
        "received_at": "2024-03-02T10:15:00Z",
        "category_hint": "api_limits",
    },
    "M003": {
        "email_id": "M003",
        "subject": "Feature request: Dark mode for dashboard",
        "body": (
            "Hi, love the product but would really appreciate a dark mode option for the dashboard. "
            "Many of us work late and the bright white interface is hard on the eyes. "
            "Is this on your roadmap? Several of my colleagues agree this would be a great addition. "
            "Happy to be a beta tester if you implement it!"
        ),
        "sender": "feature.requester@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T14:30:00Z",
        "category_hint": "feature_request",
    },
    "M004": {
        "email_id": "M004",
        "subject": "DATA LOSS after migration tool ran",
        "body": (
            "We ran your migration tool (v2.3.1) to move from legacy to new platform. "
            "After completion, approximately 15% of our customer records are missing — "
            "roughly 4,500 customer profiles with order history. "
            "We have a backup from 3 days ago but it does not include recent orders. "
            "This is a production emergency. We cannot process new orders. "
            "We need your data recovery team NOW."
        ),
        "sender": "ops@retailco.com",
        "sender_tier": "enterprise",
        "received_at": "2024-03-02T06:30:00Z",
        "category_hint": "data_loss",
    },
    "M005": {
        "email_id": "M005",
        "subject": "Exchange request - wrong size delivered",
        "body": (
            "Order #78432: I ordered a Large but received a Medium in my shipment. "
            "I'd prefer an exchange for the correct size rather than a return/refund. "
            "Please let me know if the Large is still in stock and how to proceed with the exchange."
        ),
        "sender": "shopping.fan@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T11:00:00Z",
        "category_hint": "exchange_request",
    },
    "M006": {
        "email_id": "M006",
        "subject": "Interested in upgrading - what discount is available?",
        "body": (
            "Hi billing team, we're considering upgrading from Pro to Enterprise plan. "
            "We currently have 12 seats and expect to grow to 50 by end of year. "
            "Are there any promotional discounts available for annual commitments? "
            "We're also evaluating your competitor so timeline is important."
        ),
        "sender": "procurement@growingco.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T13:45:00Z",
        "category_hint": "upgrade_inquiry",
    },
    "M007": {
        "email_id": "M007",
        "subject": "Dashboard completely blank after Chrome update",
        "body": (
            "Since Chrome updated to version 122 yesterday, our entire dashboard shows a blank page. "
            "The browser console shows: 'Uncaught TypeError: Cannot read properties of undefined'. "
            "We have 50 users who cannot work. We tried Firefox — same issue. "
            "This needs to be fixed ASAP. Is there a known fix or workaround?"
        ),
        "sender": "it.admin@company.com",
        "sender_tier": "enterprise",
        "received_at": "2024-03-02T09:00:00Z",
        "category_hint": "browser_bug",
    },
    "M008": {
        "email_id": "M008",
        "subject": "Certificate of completion not received",
        "body": (
            "I completed the Advanced Analytics certification course 2 weeks ago (completion ID: CERT-9921). "
            "I still haven't received my certificate PDF or the LinkedIn badge. "
            "I need this for a job application deadline next week. Please help."
        ),
        "sender": "job.applicant@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T15:00:00Z",
        "category_hint": "certification",
    },
    "M009": {
        "email_id": "M009",
        "subject": "Webhook not firing for payment events",
        "body": (
            "Our webhook endpoint is not receiving payment.completed events. "
            "We've verified our endpoint (https://api.ourapp.com/webhooks) is live and responding 200. "
            "The webhook is enabled in our dashboard. Last successful delivery was 3 days ago. "
            "We're missing critical payment data. Please check your webhook delivery system."
        ),
        "sender": "devops@saasco.io",
        "sender_tier": "pro",
        "received_at": "2024-03-02T10:45:00Z",
        "category_hint": "webhook_issue",
    },
    "M010": {
        "email_id": "M010",
        "subject": "Charged wrong price - need refund now",
        "body": (
            "Your website showed $19/month for the Starter plan but I was charged $29/month. "
            "I have a screenshot of the pricing page from when I signed up. "
            "I have been overcharged for the past 3 months ($30 total). "
            "Please refund the difference and correct my subscription price immediately."
        ),
        "sender": "price.dispute@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T12:30:00Z",
        "category_hint": "pricing_dispute",
    },
    # ── HARD ──────────────────────────────────────────────────────────────────
    "H001": {
        "email_id": "H001",
        "subject": "GDPR Data Request - Legal Obligation",
        "body": (
            "Under Article 17 of GDPR, I formally request deletion of all my personal data "
            "from your systems within 30 days. This includes all account data, usage logs, "
            "payment history, and any third-party data processors you've shared my data with. "
            "Failure to comply is a regulatory violation subject to fines of up to 4% of annual revenue. "
            "I expect written confirmation of deletion with a data processor list. "
            "My account email: gdpr.requester@eu.com. Reference: GDPR-REQ-2024-0301."
        ),
        "sender": "gdpr.requester@eu.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T08:00:00Z",
        "category_hint": None,  # hidden in hard mode
    },
    "H002": {
        "email_id": "H002",
        "subject": "VAT invoice correction needed for audit",
        "body": (
            "Our accounting department discovered that invoices from Q4 2023 (INV-4401 through INV-4488) "
            "show incorrect VAT numbers for our EU subsidiary. "
            "We are currently under tax audit and need corrected invoices urgently. "
            "The correct VAT number is DE123456789. "
            "This affects 88 invoices totaling €45,200. Please issue corrected invoices ASAP."
        ),
        "sender": "accounting@eurosubsidiary.de",
        "sender_tier": "enterprise",
        "received_at": "2024-03-02T09:30:00Z",
        "category_hint": None,
    },
    "H003": {
        "email_id": "H003",
        "subject": "Partnership inquiry - white label opportunity",
        "body": (
            "We are a SaaS company serving the healthcare vertical with 200+ SMB clients. "
            "We believe your platform could be white-labeled and integrated into our offering. "
            "We'd like to explore a partnership where we resell your technology under our brand. "
            "Could you put us in touch with your partnerships or business development team? "
            "We have budget allocated for Q2 2024."
        ),
        "sender": "bd@healthtech-partners.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T14:00:00Z",
        "category_hint": None,
    },
    "H004": {
        "email_id": "H004",
        "subject": "Chargeback initiated - account at risk",
        "body": (
            "I have initiated a chargeback with my bank (Citibank, dispute #CB-2024-88732) "
            "for $1,247 in charges I believe are fraudulent. "
            "I did not authorize subscription renewals after I requested cancellation on Jan 5. "
            "I have all correspondence. If you want to avoid the chargeback, "
            "provide a full refund immediately and I will cancel the bank dispute. "
            "Otherwise my bank will investigate and you will lose anyway plus fees."
        ),
        "sender": "chargeback.customer@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T07:15:00Z",
        "category_hint": None,
    },
    "H005": {
        "email_id": "H005",
        "subject": "Need to migrate 500K records from legacy system",
        "body": (
            "We are ready to proceed with migrating our legacy database (PostgreSQL 9.6) "
            "to your platform. We have approximately 500,000 customer records with custom fields. "
            "Our data schema is non-standard (attached separately). "
            "We need guidance on: 1) export format supported, 2) migration tooling available, "
            "3) whether custom field mapping is supported, 4) estimated migration timeline. "
            "We have a hard go-live deadline of April 15."
        ),
        "sender": "migration.lead@enterprise.com",
        "sender_tier": "enterprise",
        "received_at": "2024-03-02T10:00:00Z",
        "category_hint": None,
    },
    "H006": {
        "email_id": "H006",
        "subject": "Security breach suspected - customer data exposed",
        "body": (
            "One of our system administrators noticed unusual access patterns in your platform logs. "
            "It appears that an unauthorized third party may have accessed our customer database "
            "through your API. We see API calls from IP 192.168.99.44 which is not in our system. "
            "This potentially affects 12,000 of our customers' PII. "
            "We need your security team to investigate IMMEDIATELY and notify us "
            "of the scope. We have legal obligations to notify affected customers within 72 hours."
        ),
        "sender": "ciso@affectedcompany.com",
        "sender_tier": "enterprise",
        "received_at": "2024-03-02T06:00:00Z",
        "category_hint": None,
    },
    "H007": {
        "email_id": "H007",
        "subject": "Return claim - item arrived damaged",
        "body": (
            "The item from order #99102 arrived with a cracked casing. "
            "I have photos documenting the damage. It appears the damage occurred during shipping. "
            "I'd like either a replacement or a full refund. "
            "Please advise on how to submit the photos and what the process is for damaged goods."
        ),
        "sender": "damaged.goods@example.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T15:30:00Z",
        "category_hint": None,
    },
    "H008": {
        "email_id": "H008",
        "subject": "Pro-rata billing confusion after mid-cycle upgrade",
        "body": (
            "I upgraded from Starter to Pro on March 15th (mid-billing cycle). "
            "My invoice shows a charge of $67.74 which I don't understand. "
            "Can you explain how pro-rata billing works and confirm this amount is correct? "
            "If possible, please provide a breakdown showing the calculation. "
            "Also confirm whether my next billing date changes after a mid-cycle upgrade."
        ),
        "sender": "billing.confused@example.com",
        "sender_tier": "pro",
        "received_at": "2024-03-02T13:00:00Z",
        "category_hint": None,
    },
    "H009": {
        "email_id": "H009",
        "subject": "Media inquiry - comment on data breach reports",
        "body": (
            "Hi, I am a journalist at TechNews Daily working on a story about a reported "
            "data breach at several SaaS companies including yours. "
            "Multiple sources have told me that customer data was compromised in February 2024. "
            "I am reaching out for an official comment before publication tomorrow at 5pm. "
            "Please have your PR or communications team contact me urgently. "
            "No response will be noted as 'declined to comment'."
        ),
        "sender": "reporter@technewsdaily.com",
        "sender_tier": "free",
        "received_at": "2024-03-02T11:00:00Z",
        "category_hint": None,
    },
    "H010": {
        "email_id": "H010",
        "subject": "API quota exhausted - production system down",
        "body": (
            "Our production system has hit the monthly API quota limit and all API calls "
            "are now returning 429 errors. This has taken down our customer-facing application. "
            "We are on the Business plan (50K calls/month) and have used 51,247 calls by March 2nd. "
            "We clearly need a higher quota. Can you: 1) temporarily increase our limit to restore service, "
            "2) tell us the process for emergency quota increases, 3) advise on which plan supports "
            "higher volumes. Every minute costs us ~$500 in lost revenue."
        ),
        "sender": "platform.team@rapidgrowth.io",
        "sender_tier": "pro",
        "received_at": "2024-03-02T08:30:00Z",
        "category_hint": None,
    },
}

# ── GRIEVANCE FORM ground truth ───────────────────────────────────────────────
GROUND_TRUTH.update({
    "GE01": {"priority": "urgent", "department": "billing", "escalate": True,
             "response_keywords": ["overcharg", "refund", "investigat"]},
    "GE02": {"priority": "normal", "department": "technical", "escalate": False,
             "response_keywords": ["access", "account", "resolv"]},
    "GM01": {"priority": "urgent", "department": "technical", "escalate": True,
             "response_keywords": ["data", "breach", "notif"]},
    "GM02": {"priority": "normal", "department": "returns", "escalate": False,
             "response_keywords": ["defect", "replac", "qualit"]},
    "GH01": {"priority": "urgent", "department": "billing", "escalate": True,
             "response_keywords": ["harass", "collect", "legal"]},
    "GH02": {"priority": "normal", "department": "general", "escalate": False,
             "response_keywords": ["accessib", "disab", "accommod"]},
})

# ── SOCIAL MEDIA ground truth ─────────────────────────────────────────────────
GROUND_TRUTH.update({
    "SE01": {"priority": "normal", "department": "technical", "escalate": False,
             "response_keywords": ["outage", "team", "fix"]},
    "SE02": {"priority": "low", "department": "general", "escalate": False,
             "response_keywords": ["thank", "feedback", "improv"]},
    "SM01": {"priority": "urgent", "department": "billing", "escalate": True,
             "response_keywords": ["scam", "unauthor", "secur"]},
    "SM02": {"priority": "normal", "department": "technical", "escalate": False,
             "response_keywords": ["bug", "update", "fix"]},
    "SH01": {"priority": "urgent", "department": "general", "escalate": True,
             "response_keywords": ["viral", "pr", "respond"]},
    "SH02": {"priority": "normal", "department": "technical", "escalate": False,
             "response_keywords": ["privac", "data", "sett"]},
})

# ── GRIEVANCE FORM corpus ─────────────────────────────────────────────────────
EMAILS.update({
    "GE01": {
        "email_id": "GE01", "channel": "grievance",
        "subject": "Formal Grievance: Systematic Overcharging",
        "body": (
            "GRIEVANCE FORM — Ref: GRV-2024-0301\n"
            "Complainant: Margaret Stevens, Account #AC-5509\n"
            "Nature of Grievance: Systematic overcharging over 6 months.\n"
            "I have been overcharged $15-$25 each month since September 2023. "
            "Total disputed amount: $127.50. I have raised this via chat 3 times "
            "with no resolution. I am requesting a full refund and formal investigation "
            "into your billing practices. If unresolved within 14 days, I will file "
            "a complaint with the Consumer Financial Protection Bureau."
        ),
        "sender": "margaret.stevens@example.com", "sender_tier": "pro",
        "received_at": "2024-03-02T08:30:00Z", "category_hint": "billing_grievance",
    },
    "GE02": {
        "email_id": "GE02", "channel": "grievance",
        "subject": "Grievance: Account Locked Without Explanation",
        "body": (
            "GRIEVANCE FORM — Ref: GRV-2024-0302\n"
            "Complainant: David Park, Account #AC-7712\n"
            "Nature of Grievance: Account access denied.\n"
            "My account was locked on Feb 28 without any notification or explanation. "
            "I have active projects depending on this service. Your support chatbot "
            "keeps saying 'escalated' but nothing happens. I need immediate account "
            "restoration and an explanation of why it was locked."
        ),
        "sender": "david.park@example.com", "sender_tier": "pro",
        "received_at": "2024-03-02T09:45:00Z", "category_hint": "account_grievance",
    },
    "GM01": {
        "email_id": "GM01", "channel": "grievance",
        "subject": "URGENT Grievance: Potential Data Breach Notification",
        "body": (
            "GRIEVANCE FORM — Ref: GRV-2024-0303\n"
            "Complainant: IT Security Team, OrganiCorp Ltd, Account #ENT-2201\n"
            "Nature of Grievance: Suspected unauthorized data access.\n"
            "Our internal monitoring detected that customer PII stored on your platform "
            "was accessed by an IP address (45.33.12.88) not associated with our organization. "
            "We require: 1) Full access logs for the past 90 days, 2) Confirmation of what data "
            "was accessed, 3) Your incident response timeline. We have regulatory obligations "
            "under CCPA to notify affected individuals within 72 hours."
        ),
        "sender": "security@organicorp.com", "sender_tier": "enterprise",
        "received_at": "2024-03-02T06:15:00Z", "category_hint": None,
    },
    "GM02": {
        "email_id": "GM02", "channel": "grievance",
        "subject": "Grievance: Defective Product — Third Replacement Request",
        "body": (
            "GRIEVANCE FORM — Ref: GRV-2024-0304\n"
            "Complainant: Lisa Chen, Order #ORD-34521\n"
            "Nature of Grievance: Repeated product defects.\n"
            "This is my THIRD replacement request for the same item. Each unit has had "
            "the same manufacturing defect (loose hinge). I have spent over 4 hours on "
            "support calls. I am requesting a full refund plus compensation for the defective "
            "products and my time. Your quality control needs serious review."
        ),
        "sender": "lisa.chen@example.com", "sender_tier": "free",
        "received_at": "2024-03-02T11:30:00Z", "category_hint": None,
    },
    "GH01": {
        "email_id": "GH01", "channel": "grievance",
        "subject": "Formal Grievance: Harassment by Collections Department",
        "body": (
            "GRIEVANCE FORM — Ref: GRV-2024-0305\n"
            "Complainant: Robert Okafor, Former Account #AC-3301\n"
            "Nature of Grievance: Aggressive and harassing collection practices.\n"
            "Despite cancelling my account on Jan 15 (confirmation #CANC-8812), your "
            "collections department has called me 14 times in the past 2 weeks, including "
            "calls at 6:30 AM. This violates FDCPA regulations. I have recorded these calls. "
            "I demand: 1) Immediate cessation of contact, 2) Written confirmation of zero balance, "
            "3) Removal of any negative credit reporting. My attorney is CC'd on this grievance."
        ),
        "sender": "robert.okafor@example.com", "sender_tier": "free",
        "received_at": "2024-03-02T07:00:00Z", "category_hint": None,
    },
    "GH02": {
        "email_id": "GH02", "channel": "grievance",
        "subject": "Grievance: ADA Accessibility Compliance Failure",
        "body": (
            "GRIEVANCE FORM — Ref: GRV-2024-0306\n"
            "Complainant: Jennifer Walsh, Account #AC-9903\n"
            "Nature of Grievance: Platform inaccessible to users with disabilities.\n"
            "As a visually impaired user relying on screen readers, your recent dashboard "
            "redesign has made the platform nearly unusable. Key issues: no alt text on icons, "
            "color-only status indicators, keyboard navigation broken on settings page. "
            "Under ADA Title III, digital services must be accessible. I request a timeline "
            "for remediation and interim accommodations."
        ),
        "sender": "jennifer.walsh@example.com", "sender_tier": "pro",
        "received_at": "2024-03-02T13:00:00Z", "category_hint": None,
    },
})

# ── SOCIAL MEDIA corpus ───────────────────────────────────────────────────────
EMAILS.update({
    "SE01": {
        "email_id": "SE01", "channel": "social_media",
        "subject": "[Twitter @techuser99] Your service is down AGAIN",
        "body": (
            "SOCIAL MEDIA POST — Platform: Twitter/X\n"
            "Author: @techuser99 (1.2K followers)\n"
            "Post: '@YourCompany your API has been returning 500 errors for the last "
            "2 hours. This is the third outage this month. My entire app is broken. "
            "Seriously considering switching to @CompetitorCo. #disappointed #outage'\n"
            "Engagement: 45 likes, 12 retweets, 8 replies"
        ),
        "sender": "@techuser99", "sender_tier": "pro",
        "received_at": "2024-03-02T10:20:00Z", "category_hint": "social_complaint",
    },
    "SE02": {
        "email_id": "SE02", "channel": "social_media",
        "subject": "[Instagram @happy_customer] Great experience with support",
        "body": (
            "SOCIAL MEDIA POST — Platform: Instagram\n"
            "Author: @happy_customer (350 followers)\n"
            "Post: 'Just had the best customer support experience with @YourCompany! "
            "They resolved my issue in under 10 minutes. The new dashboard is amazing. "
            "Highly recommend! 🌟 #customerservice #recommended'\n"
            "Engagement: 23 likes, 2 comments"
        ),
        "sender": "@happy_customer", "sender_tier": "free",
        "received_at": "2024-03-02T14:15:00Z", "category_hint": "social_positive",
    },
    "SM01": {
        "email_id": "SM01", "channel": "social_media",
        "subject": "[Reddit r/scams] Warning: unauthorized charges from this company",
        "body": (
            "SOCIAL MEDIA POST — Platform: Reddit (r/scams)\n"
            "Author: u/careful_shopper (post has 234 upvotes)\n"
            "Post: 'WARNING: I signed up for a free trial with this company and they charged "
            "my card $499 without any notification. When I tried to cancel, their website "
            "conveniently had errors. I had to call my bank to dispute. Several others in "
            "the comments are reporting the same thing. This looks like a systematic scam. "
            "Has anyone filed with the FTC?'\n"
            "Engagement: 234 upvotes, 89 comments, trending in subreddit"
        ),
        "sender": "u/careful_shopper", "sender_tier": "free",
        "received_at": "2024-03-02T08:00:00Z", "category_hint": None,
    },
    "SM02": {
        "email_id": "SM02", "channel": "social_media",
        "subject": "[Twitter @dev_sarah] Bug report: dark mode breaks charts",
        "body": (
            "SOCIAL MEDIA POST — Platform: Twitter/X\n"
            "Author: @dev_sarah (5.8K followers, verified developer)\n"
            "Post: '@YourCompany found a bug — when dark mode is enabled, all chart labels "
            "become invisible (white text on white background). Screenshot attached. "
            "This affects the analytics dashboard. Reproducible on Chrome 122 and Firefox 123. "
            "Happy to file a proper bug report if you have a tracker. #bugreport'\n"
            "Engagement: 67 likes, 23 retweets"
        ),
        "sender": "@dev_sarah", "sender_tier": "pro",
        "received_at": "2024-03-02T11:45:00Z", "category_hint": None,
    },
    "SH01": {
        "email_id": "SH01", "channel": "social_media",
        "subject": "[TikTok @influencer_kate] VIRAL: Company exposed customer data",
        "body": (
            "SOCIAL MEDIA POST — Platform: TikTok\n"
            "Author: @influencer_kate (850K followers)\n"
            "Post: 'STORY TIME: So I just found out that @YourCompany has been exposing "
            "customer data through their public API. I can literally see other people's "
            "order histories and email addresses. I have screenshots and screen recordings. "
            "This is going viral — 2M views in 6 hours. Your move, @YourCompany. "
            "#dataprivacy #exposed #techscandal'\n"
            "Engagement: 2.1M views, 180K likes, 45K comments, trending #1"
        ),
        "sender": "@influencer_kate", "sender_tier": "free",
        "received_at": "2024-03-02T06:00:00Z", "category_hint": None,
    },
    "SH02": {
        "email_id": "SH02", "channel": "social_media",
        "subject": "[LinkedIn Post] Concerned about data privacy practices",
        "body": (
            "SOCIAL MEDIA POST — Platform: LinkedIn\n"
            "Author: James Porter, CISO at MidCorp (12K connections)\n"
            "Post: 'After reviewing @YourCompany's updated privacy policy, I have concerns "
            "about Section 7.3 which grants broad data sharing rights with unnamed third parties. "
            "As a customer, I'd like clarity on: 1) Who are these third parties? 2) Can we opt out? "
            "3) How is data anonymized? This seems to conflict with their SOC 2 certification claims. "
            "Tagging @YourCompany for a public response. #privacy #infosec #datarights'\n"
            "Engagement: 1.2K reactions, 340 comments, shared by 89 people"
        ),
        "sender": "james.porter@midcorp.com", "sender_tier": "enterprise",
        "received_at": "2024-03-02T09:30:00Z", "category_hint": None,
    },
})


# ── Task email sets ───────────────────────────────────────────────────────────

TASK_EMAIL_IDS = {
    "easy":   ["E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008", "E009", "E010",
               "GE01", "GE02", "SE01", "SE02"],
    "medium": ["M001", "M002", "M003", "M004", "M005", "M006", "M007", "M008", "M009", "M010",
               "GM01", "GM02", "SM01", "SM02"],
    "hard":   ["H001", "H002", "H003", "H004", "H005", "H006", "H007", "H008", "H009", "H010",
               "GH01", "GH02", "SH01", "SH02"],
}

