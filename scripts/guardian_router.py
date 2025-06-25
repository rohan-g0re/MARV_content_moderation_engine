# guardian_router.py

from database_filter import DatabaseFilter
from detoxify import Detoxify

class GuardianAIRouter:
    def __init__(self):
        print("🛡️ Initializing GuardianAI...")
        self.keyword_filter = DatabaseFilter()
        print("🧠 Loading Detoxify model...")
        self.detox_model = Detoxify('original')

    def moderate_post(self, post: str) -> dict:
        # Phase 1: Rule-based filter
        rule_result = self.keyword_filter.scan(post)

        # Phase 2: Detoxify ML model
        detox_result = self.detox_model.predict(post)
        toxicity_score = detox_result["toxicity"]
        context_label = "toxic" if toxicity_score >= 0.5 else "non-toxic"

        # Merge decision logic
        if rule_result["threat_level"] == "high":
            action = "Block"
        elif rule_result["threat_level"] == "medium":
            action = "Block" if context_label == "toxic" else "Flag"
        else:  # low
            action = "Flag" if toxicity_score >= 0.75 else "Accept"

        return {
            "post": post,
            "matched_terms": rule_result["matched_terms"],
            "severity_score": rule_result["total_severity"],
            "threat_level": rule_result["threat_level"],
            "toxicity_score": round(toxicity_score, 4),
            "context_prediction": context_label,
            "moderation_action": action
        }

    def close(self):
        self.keyword_filter.close()
