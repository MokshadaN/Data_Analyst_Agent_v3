# /// script
# dependencies = ["fastapi", "uvicorn", "python-multipart","google-genai","pydantic"]
# ///

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import pathlib
import os
import uvicorn
from google import genai
from google.genai import types
import json
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
from enum import Enum
from prompts import PromptManager
import re

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
prompt_manager = PromptManager()

def run_planner_agent_json_with_feedback_looping(questions ,files, max_retries=2):
    print("4: In Plan Generation ")
    general_sys_prompt, general_user_prompt = prompt_manager.general_json_planner_prompt(questions, files)
    print("5: Got the General Prompts")

    csv_present = any(file.get("extension").lower() == ".csv" for file in files)
    # print("6")
    json_present = any(file.get("extension").lower() == ".json" or file.get("type", "").lower() in {"application/json", "text/json"} for file in files)
    # print("7")
    excel_present = any(file.get("extension").lower() in {".xls", ".xlsx"} for file in files)
    # print("8")
    pdf_present = any(file.get("extension").lower() == ".pdf" for file in files)
    # print("9")
    html_present  = any(f.get("source_type")=="html" for f in files)
    # print("10")
    s3_present = ("s3://" in questions) or ("s3_region=" in questions) or ("s3://" in files) or ("s3_region=" in files)
    # print("11")
    url_present = any(f.get("source_type")=="url" for f in files)
    file_prompts = []
    if s3_present:
        # file_prompts.append(prompt_manager.new_planner_agent_prompt())
        file_prompts.append(prompt_manager.s3_instructions())
    elif url_present:
        print("Url present")
        for f in files:
            if f.get("source_type") == "url":
                if f.get("has_dynamic_params"):
                    file_prompts.append(prompt_manager.url_dynamic_params_prompt())
                if f.get("js_rendering"):
                    file_prompts.append(prompt_manager.url_js_rendering_prompt())

                if f.get("pagination"):
                    file_prompts.append(prompt_manager.url_pagination_prompt())

                if f.get("has_tables"):
                    file_prompts.append(prompt_manager.url_table_prompt())

                if f.get("is_api"):
                    file_prompts.append(prompt_manager.url_api_prompt())

                if not any([f.get("has_tables"), f.get("is_api"), f.get("pagination") , f.get("")]):
                    file_prompts.append(prompt_manager.url_text_only_prompt())
    elif html_present:
        print("Html present")
        # file_prompts.append(prompt_manager.html_instructions())
        file_prompts.append(prompt_manager.html_instructions_planning())
    if csv_present:
        file_prompts.append(prompt_manager.csv_instructions())
    if json_present:
        file_prompts.append(prompt_manager.json_instructions())
    if excel_present:
        file_prompts.append(prompt_manager.excel_instructions())
    if pdf_present:
        file_prompts.append(prompt_manager.pdf_instructions())
    # if html_present:
    #     file_prompts.append(prompt_manager.url_new_instructions())
    # if not csv_present and not json_present and not excel_present and not pdf_present and not html_present:
    #     file_prompts.append(prompt_manager.url_new_instructions())
        

    final_user_prompt = general_user_prompt
    if file_prompts:
        general_sys_prompt += "\n\n" + "\n\n".join(file_prompts)

    print("6: Got the Final Prompts and saved it to plan_prompts")
    with open("plan_prompts.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write(str(general_sys_prompt))
        f.write(str(final_user_prompt))

    attempt = 0
    while attempt <= max_retries:
        try:
            print("7: Calling the gemini api for plan generation")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=general_sys_prompt,temperature=0.5),
                contents=final_user_prompt
            )
            plan_text = response.candidates[0].content.parts[0].text.strip()

            print("8: Got the Response from Planning ")
            # Remove markdown fences if present
            if plan_text.startswith("```"):
                plan_text = re.sub(r"^```[a-zA-Z]*\n", "", plan_text)
                plan_text = re.sub(r"\n```$", "", plan_text)

            # Try parsing JSON — if it fails, just return the text as is
            try:
                print("9: JSON plan saved")
                return json.loads(plan_text)
            except json.JSONDecodeError:
                print("9: Text plan saved")
                return plan_text

        except Exception as e:
            print(f"❌ Gemini API call failed (attempt {attempt+1}/{max_retries+1}): {e}")
            if attempt == max_retries:
                raise RuntimeError("Gemini planner failed after maximum retries.") from e
            attempt += 1


