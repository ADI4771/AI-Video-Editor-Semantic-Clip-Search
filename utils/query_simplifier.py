import re
import json
import urllib.request


SYSTEM_PROMPT = """You are a visual search query optimizer for a CLIP-based video search engine.

CLIP understands simple, concrete visual descriptions. Your job:
1. Take a user's natural language query
2. Return a simplified, CLIP-friendly version — short, visual, concrete
3. Also return a list of 2-3 alternative phrasings to improve recall

Rules:
- Keep it under 10 words
- Use visual nouns and adjectives (colors, objects, scenes, lighting)
- Remove abstract concepts, emotions, narrative phrases
- Return ONLY valid JSON, no markdown

Output format:
{
  "primary": "simplified query for CLIP",
  "alternatives": ["alt phrasing 1", "alt phrasing 2"]
}"""


def simplify_query(user_query: str) -> dict:

    try:
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"User query: {user_query}"}
            ]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        text = data["content"][0]["text"].strip()

        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        parsed = json.loads(text)

        return {
            "original": user_query,
            "primary": parsed.get("primary", user_query),
            "alternatives": parsed.get("alternatives", []),
        }

    except Exception as e:
        print(f" LLM query simplification failed ({e}), using original query")
        return {
            "original": user_query,
            "primary": user_query,
            "alternatives": [],
        }
