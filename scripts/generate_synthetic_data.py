import os
import csv
import random
import re
from langchain_groq import ChatGroq
from tqdm import tqdm
from dotenv import load_dotenv
from collections import OrderedDict
load_dotenv()

# === CONFIG ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_MODEL = "llama3-8b-8192"
OUTPUT_FILE = "synthetic_moderation_data.csv"

NUM_PASS = 600
NUM_FLAG = 450
NUM_BLOCK = 450
N_PER_BATCH = 10  # Number of posts to generate per batch
PARAPHRASE_PROB = 0.25  # Fraction of posts to paraphrase
NOISE_PROB = 0.2        # Fraction of posts to add noise to

# --- Prompt templates for each class ---
PROMPT_TEMPLATES = {
    "PASS": [
        "You are an expert at writing synthetic samples for content moderation. Generate {n} unique, realistic examples of short user posts for class 'PASS'. Posts should be harmless, safe, normal conversation or finance advice. Vary tone, length, and style. Include a wide variety of greetings (e.g., 'Hi!', 'Hello!', 'Good morning!', 'Hey there!'), short exclamatory messages, posts with emojis, posts with exclamation marks, posts with some uppercase words, and posts with minor typos. Make sure to include excited, friendly, and informal posts that are clearly safe. Mix in some posts with punctuation, emojis, and a few with all-caps words, but all should be non-controversial and safe.",
        "Write {n} short, safe, and friendly user posts that would be classified as 'PASS' (not suspicious). Use a mix of formal and informal language, and include a few with emojis, exclamation marks, or minor typos. Make sure to include greetings, excited messages, and harmless advice.",
        "Generate {n} harmless, everyday user posts for a finance forum. These should be safe, non-controversial, and could include greetings, questions, or simple advice. Include posts with exclamation marks, emojis, and some with excited tone.",
        "Produce {n} unique, safe user posts that are clearly not scams or spam. Include a variety of greetings (e.g., 'Hi!', 'Hello!', 'Good morning!'), friendly exclamations, and posts with emojis or punctuation. Make sure some posts are short and excited, and all are harmless."
    ],
    "FLAG": [
        "Generate {n} unique, realistic user posts for class 'FLAG'. These should be somewhat suspicious, possibly manipulative, borderline scammy or ambiguous, but not a clear fraud. Vary the style, and include some posts that are hard to judge.",
        "Write {n} user posts that are ambiguous or could be interpreted as manipulative or suspicious, but not outright scams. Include some with financial jargon, some with subtle persuasion, and some with odd phrasing.",
        "Create {n} user posts that might raise a red flag for content moderation, but are not clearly fraudulent. Use a mix of tones, and include some with slang or emojis."
    ],
    "BLOCK": [
        "Generate {n} unique, realistic user posts for class 'BLOCK'. These should be definite scams, clear fraud, abusive, spam, phishing, or explicit scam attempts. Vary the style, and include some with aggressive language, links, or obvious scam tactics.",
        "Write {n} user posts that would be immediately blocked by a content moderation system for being scams, spam, or abusive. Include some with links, some with all-caps, and some with explicit scam offers.",
        "Create {n} posts that are clear examples of fraud, phishing, or abusive content. Use a mix of short and long posts, and include some with typos or emojis."
    ]
}

PARAPHRASE_PROMPT = "Paraphrase the following post to mean the same thing, but use different words, tone, or style. Output only the new version. Post: {post}"

NOISE_PATTERNS = [
    lambda s: re.sub(r'([aeiou])', lambda m: m.group(1) + random.choice(['', '', 'x']), s, count=1),  # random typo
    lambda s: s + random.choice([' 😅', ' 🤔', '!!', '...']),
    lambda s: s.replace('s', '$', 1) if 's' in s else s,
    lambda s: s.lower() if random.random() < 0.5 else s.upper(),
    lambda s: s + random.choice([' #finance', ' #advice', ' #scamalert'])
]

def get_llm():
    if not GROQ_API_KEY:
        raise EnvironmentError("Set your GROQ_API_KEY environment variable!")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0.85,
        max_tokens=1200,
    )

def generate_samples(llm, label, n_samples):
    all_posts = []
    n_batches = (n_samples + N_PER_BATCH - 1) // N_PER_BATCH
    prompt_templates = PROMPT_TEMPLATES[label]
    for i in tqdm(range(n_batches), desc=f"Generating {label}"):
        prompt_str = random.choice(prompt_templates).format(n=N_PER_BATCH)
        response = llm.invoke(prompt_str)
        posts = [
            line.strip().split('.', 1)[-1].strip()
            for line in response.content.strip().splitlines()
            if line.strip() and any(c.isalpha() for c in line)
        ]
        all_posts.extend(posts)
        if len(all_posts) >= n_samples:
            break
    return all_posts[:n_samples]

def paraphrase_posts(llm, posts):
    new_posts = []
    for post in posts:
        try:
            prompt = PARAPHRASE_PROMPT.format(post=post)
            response = llm.invoke(prompt)
            new_text = response.content.strip()
            if new_text and new_text.lower() != post.lower():
                new_posts.append(new_text)
        except Exception:
            continue
    return new_posts

def add_noise(posts):
    noisy_posts = []
    for post in posts:
        pattern = random.choice(NOISE_PATTERNS)
        noisy_posts.append(pattern(post))
    return noisy_posts

def deduplicate(posts):
    # Remove near-duplicates (case-insensitive, ignore punctuation)
    def normalize(s):
        return re.sub(r'\W+', '', s).lower()
    seen = set()
    unique = []
    for post in posts:
        norm = normalize(post)
        if norm not in seen:
            seen.add(norm)
            unique.append(post)
    return unique

def main():
    llm = get_llm()
    data = []
    for label, n in [("PASS", NUM_PASS), ("FLAG", NUM_FLAG), ("BLOCK", NUM_BLOCK)]:
        posts = generate_samples(llm, label, n)
        # Paraphrase a subset
        n_para = int(PARAPHRASE_PROB * len(posts))
        if n_para > 0:
            para_posts = paraphrase_posts(llm, random.sample(posts, n_para))
            posts.extend(para_posts)
        # Add noise to a subset
        n_noise = int(NOISE_PROB * len(posts))
        if n_noise > 0:
            noisy_posts = add_noise(random.sample(posts, n_noise))
            posts.extend(noisy_posts)
        # Deduplicate
        posts = deduplicate(posts)
        # Truncate to n
        posts = posts[:n]
        data += [(post, label) for post in posts]
    random.shuffle(data)
    with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(data)
    print(f"Done! Saved {len(data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
