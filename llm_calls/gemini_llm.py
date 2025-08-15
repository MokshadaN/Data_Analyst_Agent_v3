import os
import re
import json
import time
from google import genai
from google.genai.types import GenerateContentConfig

GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

def gemini_call_for_code(
    system_prompt: str,
    user_prompt: str,
    content=None,
    max_retries=5,
    retry_delay=1,
    models_cycle=("gemini-2.5-pro", "gemini-2.5-flash"),
) -> str:
    import re, json, time

    def _clean_code(text: str) -> str:
        text = text.strip()
        # Prefer last fenced code block if any
        blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.I)
        if blocks:
            return blocks[-1].strip()
        # Else, drop non-code leading lines until we hit code-like syntax
        lines = text.splitlines()
        pat = re.compile(r'^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|#|"""|\'\'\')')
        for i, ln in enumerate(lines):
            if pat.search(ln):
                return "\n".join(lines[i:]).strip()
        return text  # fallback

    attempt = 0
    while attempt < max_retries:
        # Pick model by attempt index: 0->flash, 1->pro, 2->flash, ...
        model_to_use = models_cycle[attempt % len(models_cycle)]
        try:
            contents = user_prompt
            if content is not None:
                content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
                contents += f"\n\n--- Additional Context ---\n{content_str}"

            print(f"[Gemini] Attempt {attempt+1}/{max_retries} using model={model_to_use}")
            response = client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.35  # deterministic
                )
            )

            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("No response from Gemini model.")

            raw_text = response.candidates[0].content.parts[0].text.strip()
            with open("temp.py","w", encoding="utf-8") as f:
                f.write(raw_text)
            return _clean_code(raw_text)

        except Exception as e:
            attempt += 1
            print(f"[Retry {attempt}/{max_retries}] ({model_to_use}) Gemini API call failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Gemini API call failed after {max_retries} retries.") from e

def test_gemini_call_for_code():
    # System prompt: minimal, code-only instruction
    # system_prompt = "You are a Python code generator. Respond ONLY with executable Python code."
    system_prompt = "You are a Data Analyst Agent for solving a particular question with data sourcing/scraping "
    with open("E:/BS/Sem_May_2025/TDS_Project_2/Data_Analyst_Agent_v2/questions/question_url_1.txt","r") as f:
        text = f.read()
    # User prompt: trivial coding task
    user_prompt = (
        {text}
    )

    try:
        code = gemini_call_for_code(system_prompt, user_prompt)
        print("\n--- Cleaned Code from Gemini ---\n")
        print(code)

        # Optional: Execute the code to see if it runs
        exec_globals = {}
        exec(code, exec_globals)
        print("\n--- Execution Completed ---\n")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_gemini_call_for_code()
