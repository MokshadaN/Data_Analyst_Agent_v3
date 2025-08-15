import os
import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=API_KEY)

def openai_call_for_code_responses(
    system_prompt: str,
    user_prompt: str,
    content=None,
    max_retries=5,
    retry_delay=1,
    models_cycle=("gpt-5", "gpt-4.1"),
) -> str:
    """
    Calls OpenAI responses API to generate Python code, with retries and cleaning logic.
    """
    def _clean_code(text: str) -> str:
        text = text.strip()
        blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.I)
        if blocks:
            return blocks[-1].strip()
        lines = text.splitlines()
        pat = re.compile(r'^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|#|\"\"\"|\'\'\')')
        for i, ln in enumerate(lines):
            if pat.search(ln):
                return "\n".join(lines[i:]).strip()
        return text

    attempt = 0
    while attempt < max_retries:
        model_to_use = models_cycle[attempt % len(models_cycle)]
        try:
            contents = user_prompt
            if content is not None:
                content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
                contents += f"\n\n--- Additional Context ---\n{content_str}"

            print(f"[OpenAI] Attempt {attempt+1}/{max_retries} using model={model_to_use}")

            # Build input in the new Responses format
            input_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": contents}
            ]

            # Create request (no temperature for gpt-5 / gpt-4.1)
            response = client.responses.create(
                model=model_to_use,
                input=input_payload
            )

            if not hasattr(response, "output_text") or not response.output_text:
                raise ValueError("No response from OpenAI model.")

            raw_text = response.output_text.strip()

            with open("temp.py", "w", encoding="utf-8") as f:
                f.write(raw_text)

            return _clean_code(raw_text)

        except Exception as e:
            attempt += 1
            print(f"[Retry {attempt}/{max_retries}] ({model_to_use}) OpenAI API call failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"OpenAI API call failed after {max_retries} retries.") from e


def test_openai_call_for_code_responses():
    system_prompt = "You are a Data Analyst Agent for solving a particular question with data sourcing/scraping"
    with open("E:/BS/Sem_May_2025/TDS_Project_2/Data_Analyst_Agent_v2/questions/question_url_1.txt", "r", encoding="utf-8") as f:
        text = f.read()

    try:
        code = openai_call_for_code_responses(system_prompt, text)
        print("\n--- Cleaned Code from OpenAI ---\n")
        print(code)

        exec_globals = {}
        exec(code, exec_globals)
        print("\n--- Execution Completed ---\n")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_openai_call_for_code_responses()
