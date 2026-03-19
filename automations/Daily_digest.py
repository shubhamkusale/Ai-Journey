import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# ============================================================
#   SHUBHAM'S DAILY AI DIGEST BOT
#   Fetches top AI news from HackerNews
#   Sends to Telegram every morning
#   Part of: Ai-Journey/automations/
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- STEP 1: Fetch top AI stories from HackerNews ---
def fetch_ai_news():
    print("Fetching top stories from HackerNews...")
    top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    story_ids = requests.get(top_stories_url).json()[:30]

    ai_keywords = ["AI", "LLM", "GPT", "machine learning", "neural", 
                   "OpenAI", "Anthropic", "Gemini", "Claude", "model",
                   "agent", "automation", "deep learning", "transformer"]

    ai_stories = []
    for sid in story_ids:
        story = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json").json()
        title = story.get("title", "")
        if any(kw.lower() in title.lower() for kw in ai_keywords):
            ai_stories.append({
                "title": title,
                "url": story.get("url", "https://news.ycombinator.com/item?id=" + str(sid)),
                "score": story.get("score", 0)
            })
        if len(ai_stories) == 5:
            break

    print(f"Found {len(ai_stories)} AI stories.")
    return ai_stories

# --- STEP 2: Summarize using Groq AI ---
def summarize_with_groq(stories):
    print("Summarizing with Groq AI...")

    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY (set it in automations/.env)")
    
    stories_text = "\n".join([f"- {s['title']}" for s in stories])
    
    prompt = f"""You are a sharp AI news summarizer for a student named Shubham who is 18, 
learning AI/ML and building projects. 

Here are today's top AI headlines:
{stories_text}

Write a crisp morning digest in this exact format:
- Start with: Good morning Shubham! Here's your AI digest for {datetime.now().strftime('%d %B %Y')} 🚀
- Then 5 bullet points, one per story, max 1 sentence each
- End with one motivational line connecting AI to his goal of building Jarvis
- Keep total message under 300 words
- No markdown, plain text only"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400
        }
    )
    
    result = response.json()
    summary = result["choices"][0]["message"]["content"]
    print("Summary generated.")
    return summary

# --- STEP 3: Send to Telegram ---
def send_to_telegram(message):
    print("Sending to Telegram...")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (set them in automations/.env)")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Failed to send: {response.text}")

# --- MAIN ---
def main():
    print(f"\n=== SHUBHAM'S AI DIGEST | {datetime.now().strftime('%d-%m-%Y %H:%M')} ===\n")
    
    stories = fetch_ai_news()
    
    if not stories:
        send_to_telegram(f"Good morning Shubham! No major AI news found today. Keep building Jarvis! 🤖")
        return
    
    summary = summarize_with_groq(stories)
    
    # Add story links at bottom
    links = "\n\nToday's sources:"
    for i, s in enumerate(stories, 1):
        links += f"\n{i}. {s['url']}"
    
    full_message = summary + links
    send_to_telegram(full_message)

if __name__ == "__main__":
    main()