import json
import os
import subprocess
import tempfile

from llm_calls.claude_call import claude_call_for_code
from prompts import PromptManager
from llm_calls.gemini_llm import gemini_call_for_code  # updated import
from llm_calls.openai_call import openai_call_for_code_responses
import time

os.makedirs("generated_code", exist_ok=True)

def execute_plan_v1(plan, questions, data_files, timeout_for_execution=300, max_retries=3):
    """Generate and execute code for a given plan with fallback LLMs."""
    print("[EXECUTE] 1 Executing Plan")
    pm = PromptManager()

    # Detect S3 presence
    s3_present = "s3://" in questions.lower() or "s3://" in json.dumps(plan).lower()
    if s3_present:
        print("[EXECUTE] Detected S3-related questions → using execute_s3 prompt")
        system_prompt, user_prompt = pm.execute_s3(plan, questions, data_files)
        max_retries = 8
    else:
        print("[EXECUTE] No S3 detected → using execute_entire_plan_v2 prompt")
        system_prompt, user_prompt = pm.execute_entire_plan_v2(plan, questions, data_files)

    system_prompt += "\nIMPORTANT: Final output must be valid JSON only. " \
                     "Always do: import json; print(json.dumps(result, ensure_ascii=False, indent=2)) " \
                     "at the end of the script. No other prints."

    # --------- Primary: Claude ----------
    print("[EXECUTE] 2 Generating Code with Claude")
    try:
        code = claude_call_for_code(system_prompt, user_prompt, None)
    except Exception as e:
        print(f"[EXECUTE] Claude call failed immediately: {e}")
        code = None

    # If Claude fails completely, fallback to OpenAI
    if not code:
        print("[EXECUTE] Claude failed, trying OpenAI")
        try:
            code = openai_call_for_code_responses(system_prompt, user_prompt, None)
        except Exception as e:
            print(f"[EXECUTE] OpenAI call failed immediately: {e}")
            code = None

    # If both Claude and OpenAI fail, fallback to Gemini
    if not code:
        print("[EXECUTE] Both Claude and OpenAI failed, trying Gemini")
        try:
            code = gemini_call_for_code(system_prompt, user_prompt, None)
        except Exception as e:
            print(f"[EXECUTE] Gemini call failed too: {e}")
            return {"error": "All LLMs failed to generate code"}

    # Save generated code
    with open("temp.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    with open(f"generated_code/initial_{int(time.time()*1000)}.py", "w", encoding="utf-8") as _f:
        _f.write(code)

    print("[EXECUTE] 3 Code successfully generated (after fallback if needed)")

    # --------- Execute initial code ----------
    ok, output, error = _run_and_validate_json(code, timeout_sec=timeout_for_execution)
    with open(f"generated_code/run_{int(time.time()*1000)}.txt", "w", encoding="utf-8") as _f:
        _f.write(f"ok={ok}\n")
        _f.write((output or "") + "\n")
        _f.write((error or "") + "\n")
    print("[EXECUTE] 4 After execution:", ok, str(output)[:100], error)

    if ok:
        return output

    # --------- Retry Loop (still with fallback) ----------
    for attempt in range(max_retries):
        print(f"[EXECUTE][RETRY {attempt+1}] Building repair prompt")
        sys_repair_prompt, user_repair_prompt = _build_repair_prompt(system_prompt, plan, questions, data_files, code, error)

        # Try Claude repair first
        try:
            code = claude_call_for_code(str(sys_repair_prompt), str(user_repair_prompt), str(questions))
        except Exception as e:
            print(f"[RETRY] Claude repair failed: {e}")
            code = None

        # If Claude fails, try OpenAI
        if not code:
            try:
                code = openai_call_for_code_responses(str(sys_repair_prompt), str(user_repair_prompt), str(questions))
            except Exception as e:
                print(f"[RETRY] OpenAI repair failed: {e}")
                code = None

        # If both fail, try Gemini
        if not code:
            try:
                code = gemini_call_for_code(str(sys_repair_prompt), str(user_repair_prompt), str(questions))
            except Exception as e:
                print(f"[RETRY] Gemini repair failed: {e}")
                continue  # try next attempt

        # Save repaired code
        with open("temp.py", "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        with open(f"generated_code/retry_{int(time.time()*1000)}.py", "w", encoding="utf-8") as _f:
            _f.write(code)

        ok, output, error = _run_and_validate_json(code, timeout_sec=timeout_for_execution)
        with open(f"generated_code/retry_result_{int(time.time()*1000)}.txt", "w", encoding="utf-8") as _f:
            _f.write(f"ok={ok}\n")
            _f.write((output or "") + "\n")
            _f.write((error or "") + "\n")

        if ok:
            return output

    return error


# --- in plan_execution.py ---
import sys, os, json, tempfile, subprocess

def _run_and_validate_json(code: str, timeout_sec: int = 300):
    """
    Run Python code in a clean subprocess and ensure stdout is valid JSON.
    - Uses the *same* interpreter as the FastAPI app (sys.executable).
    - Forces UTF-8 streams to avoid init_sys_streams errors on Windows.
    - Uses a headless plotting backend.
    """
    # Write the temp script
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8", newline="\n") as tmp:
        tmp.write(code)
        code_path = tmp.name

    try:
        env = os.environ.copy()
        # Force UTF-8 I/O to avoid fatal init on Windows / service contexts
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        # Ensure matplotlib is headless
        env.setdefault("MPLBACKEND", "Agg")

        # IMPORTANT: use the same Python that runs the server
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", code_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,                 # decode using locale/utf-8 (we force utf-8 above)
            env=env
        )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            # Bubble up the real stderr so your repair loop can use it
            return False, stdout, stderr or "Script exited with non-zero status."

        # Validate JSON
        try:
            json.loads(stdout)
            return True, stdout, None
        except json.JSONDecodeError as je:
            return False, stdout, f"Invalid JSON output: {je}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    except subprocess.TimeoutExpired:
        return False, stdout, f"Execution timed out after {timeout_sec}s."
    finally:
        try:
            os.remove(code_path)
        except OSError:
            pass

def _build_repair_prompt(system_prompt, plan,questions, data_files, prev_code, error):
    """Builds a repair system and user prompt for Gemini."""
    print("[EXECUTE] Building Repair Prompt")
    pm = PromptManager()
    s3_present = "s3://" in questions.lower() or "s3://" in json.dumps(plan).lower()
    if s3_present:
        print("[RETRY] Detected S3-related questions → using execute_s3 prompt")
        system_prompt, base_user_prompt = pm.execute_s3(plan, questions, data_files)
    else:
        print("[RETRY] No S3 detected → using execute_entire_plan_v2 prompt")
        system_prompt, base_user_prompt = pm.execute_entire_plan_v2(plan, questions, data_files)# First attempt

    repair_system_prompt = f"{system_prompt.strip()}\n\nIMPORTANT: Always print json.dumps(result).\n- Never create dummy data for any type of source , try sourcing again if not executed the first time"
    repair_user_prompt = (
        f"{base_user_prompt}\n\n"
        "Your previous code failed.\n"
        "If you the answers are correct only the json formatting is incorrect then take those answers and format it as json as required in the questions"
        "----- PREVIOUS CODE -----\n"
        f"{prev_code}\n"
        "----- ERROR / OUTPUT -----\n"
        f"{error}\n"
        "----- PLAN -----\n"
        f"{plan}\n"
        "Please fix the issues and return only working Python code. "
        "Ensure the script ends with:\n"
        "import json\nprint(json.dumps(result))"
        "Ensure that you do not create a dummy data",
    )

    return repair_system_prompt, repair_user_prompt
