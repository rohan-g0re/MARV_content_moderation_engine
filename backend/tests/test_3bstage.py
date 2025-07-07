# test_stage3b.py

from app.core.moderation import Stage3bFraudClassifier, ModerationResult

def test_stage3b_examples():
    # Instantiate the classifier (change if you need to pass API keys, etc.)
    classifier = Stage3bFraudClassifier()

    # List of test posts (borderline/manipulative)
    test_posts = [
        "This isn’t financial advice, but I found a new platform that multiplies your deposit really fast. I’ve had crazy results, and it’s totally legit. Message me for the invite link if you want in before it fills up.",
        "Lots of people I know have doubled their savings with this system. No risks, payouts every day. Can’t say too much here, but if you’re interested, reply below.",
        "At first I was skeptical, but after trying this, my returns went through the roof. Not sure if it’s for everyone, but DM me if you want details. Only a few spots left.",
        "Heard from an insider about an opportunity where your investment is almost guaranteed to grow. Not public info yet. Ask me privately if you want in.",
    ]

    # Run each post through the fraud classifier
    for idx, post in enumerate(test_posts, 1):
        print(f"\n--- Test Post {idx} ---")
        print(f"Content: {post}")
        result: ModerationResult = classifier.process(post)
        print("Decision:", result.decision)
        print("Reason:", result.reason)
        print("Confidence:", result.confidence)
        print("Threat level:", result.threat_level)

if __name__ == "__main__":
    test_stage3b_examples()
