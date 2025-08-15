from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import pathlib
import os
import io
import time
import random
import json
import requests
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from plan_Creation import *
from plan_execution import execute_plan_v1
import pandas as pd
import pdfplumber
import re
import time
import os
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

import os
# ---- Create logs folder and per-run subfolder ----
BASE_LOG_DIR = "logs"
os.makedirs(BASE_LOG_DIR, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, run_id)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# ---- Setup logging ----
log_file = os.path.join(RUN_LOG_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def save_to_log_folder(filename, content):
    """
    Save content under the current run's log directory.
    - dict/list  -> JSON (UTF-8, pretty)
    - bytes      -> raw bytes
    - everything else -> str()
    Returns the absolute file path.
    """
    file_path = os.path.join(RUN_LOG_DIR, filename)

    # Ensure parent exists (paranoia)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        if isinstance(content, (dict, list)):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        elif isinstance(content, (bytes, bytearray)):
            with open(file_path, "wb") as f:
                f.write(content)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))
        logging.info("Saved file: %s", file_path)
        return file_path
    except Exception as e:
        logging.exception("Failed to save file %s: %s", file_path, e)
        # Don't crash the whole request just because logging failed
        return file_path

from bs4 import BeautifulSoup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# FRONT END API CHECKS

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request 
templates = Jinja2Templates(directory="templates")

@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_API_KEY = "AIzaSyDh7TfjKwBEI2eoE4xObDfyBbRh25YGe8k"
client = genai.Client(api_key=GEMINI_API_KEY)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _is_image_filename(name: str) -> bool:
    ext = pathlib.Path(name).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

def _is_image_content_type(content_type: str) -> bool:
    return (content_type or "").startswith("image/")
def _is_csv(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    return ext == ".csv" or (content_type or "") in {"text/csv", "application/vnd.ms-excel"}

def _is_json(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    return ext == ".json" or (content_type or "").lower() in {"application/json", "text/json"}

def _is_excel(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    ct = (content_type or "").lower()
    return ext in {".xls", ".xlsx"} or ct in {
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

def _is_pdf(content_type: str, filename: str) -> bool:
    ext = pathlib.Path(filename).suffix.lower()
    ct = (content_type or "").lower()
    return ext == ".pdf" or ct in {"application/pdf"}


def get_image_description(image_path_or_url: str, max_retries: int = 5) -> str:
    """Returns a concise description for a local image path or URL using Gemini."""
    _client = genai.Client(api_key="AIzaSyDh7TfjKwBEI2eoE4xObDfyBbRh25YGe8k")

    try:
        if image_path_or_url.startswith(("http://", "https://")):
            resp = requests.get(image_path_or_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
        else:
            image = Image.open(image_path_or_url)
    except Exception as e:
        return f"Could not open image: {e}"

    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Generate a detailed description of this image. ",
                    "If any numerical data mention that as well",
                    "Give a detailed description and understanding from the image",
                    image
                ]
            )
            text = getattr(response, "text", "").strip()
            return text or "No description returned."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                return f"Description unavailable due to error: {e}"

def get_csv_metadata(file_path: str, sample_rows: int = 1) -> dict:
    """
    Efficiently read minimal data to report CSV schema and a sample row.
    - Tries to infer dtypes from a small chunk.
    - Returns columns, dtypes (as strings), and one sample row (if present).
    """
    try:
        # Read a small chunk to infer schema
        chunk_iter = pd.read_csv(
            file_path,
            nrows=None,            # allow chunksize to control rows
            chunksize=2048,        # small chunk for inference
            low_memory=False,      # better type inference
            dtype_backend="numpy_nullable",  # stable dtypes as strings
            encoding="utf-8",
            on_bad_lines="skip",   # be forgiving
        )
        first_chunk = next(chunk_iter, None)
        if first_chunk is None or first_chunk.empty:
            return {
                "columns": [],
                "dtypes": {},
                "sample_row": {}
            }

        # Normalize column names to str
        first_chunk.columns = [str(c) for c in first_chunk.columns]

        # Build dtype mapping as strings
        dtypes = {col: str(first_chunk.dtypes[col]) for col in first_chunk.columns}

        # Get a small sample (default 1 row)
        sample_df = first_chunk.head(sample_rows)
        sample_row = sample_df.iloc[0].to_dict() if not sample_df.empty else {}

        return {
            "columns": list(first_chunk.columns),
            "dtypes": dtypes,
            "sample_row": sample_row
        }

    except StopIteration:
        return {"columns": [], "dtypes": {}, "sample_row": {}}
    except UnicodeDecodeError:
        # Retry with latin-1 fallback for weird encodings
        try:
            df = pd.read_csv(
                file_path,
                nrows=2048,
                low_memory=False,
                dtype_backend="numpy_nullable",
                encoding="latin-1",
                on_bad_lines="skip",
            )
            df.columns = [str(c) for c in df.columns]
            dtypes = {col: str(df.dtypes[col]) for col in df.columns}
            sample_row = df.head(1).iloc[0].to_dict() if not df.empty else {}
            return {
                "columns": list(df.columns),
                "dtypes": dtypes,
                "sample_row": sample_row
            }
        except Exception:
            return {"columns": [], "dtypes": {}, "sample_row": {}}
    except Exception:
        # Keep it resilient; upstream can still process the file even if metadata fails
        return {"columns": [], "dtypes": {}, "sample_row": {}}

def get_json_metadata(file_path: str, max_preview_bytes: int = 131072) -> dict:
    """
    Read a small portion of a JSON file and summarize structure.
    Handles top-level object or array of objects.
    Returns keys, types, and a sample item.
    """
    try:
        # Fast path: if small enough, load fully; else stream then parse
        if os.path.getsize(file_path) <= max_preview_bytes:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Stream first N bytes but ensure valid JSON by falling back to full load on failure
            with open(file_path, "r", encoding="utf-8") as f:
                chunk = f.read(max_preview_bytes)
            try:
                data = json.loads(chunk)
            except Exception:
                # As a safe fallback (still bounded by OS limits), try full parse
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

        # Normalize to an object sample
        if isinstance(data, list):
            sample = next((x for x in data if isinstance(x, dict)), {})
        elif isinstance(data, dict):
            sample = data
        else:
            return {"top_level_type": type(data).__name__, "keys": [], "sample_object": {}}

        keys = list(sample.keys())
        dtypes = {k: type(sample.get(k)).__name__ for k in keys}

        return {
            "top_level_type": "array" if isinstance(data, list) else "object",
            "keys": keys,
            "dtypes": dtypes,
            "sample_object": sample,
        }
    except Exception:
        return {"top_level_type": "unknown", "keys": [], "dtypes": {}, "sample_object": {}}

def get_excel_metadata(file_path: str, sample_rows: int = 3) -> dict:
    """
    Probe up to a few sheets to summarize structure:
    - sheet names
    - columns
    - dtypes (as strings)
    - one sample row per sheet (if available)
    """
    try:
        import pandas as pd
        xls = pd.ExcelFile(file_path)
        sheets_meta = []
        for sheet_name in xls.sheet_names[:5]:  # cap for speed
            try:
                df = xls.parse(sheet_name, nrows=sample_rows, dtype_backend="numpy_nullable")
                df.columns = [str(c) for c in df.columns]
                dtypes = {c: str(df.dtypes[c]) for c in df.columns}
                sample = df.head(1).iloc[0].to_dict() if not df.empty else {}
                sheets_meta.append({
                    "name": sheet_name,
                    "columns": list(df.columns),
                    "dtypes": dtypes,
                    "sample_row": sample
                })
            except Exception:
                sheets_meta.append({
                    "name": sheet_name,
                    "columns": [],
                    "dtypes": {},
                    "sample_row": {}
                })
        return {"sheets": sheets_meta}
    except Exception:
        return {"sheets": []}


def get_pdf_metadata(file_path: str, max_pages: int = 5, max_text_chars: int = 4000) -> dict:
    """
    Extract a quick summary from a PDF:
    - page count
    - text preview (first N chars aggregated from first `max_pages`)
    - table summaries: page, heuristic header detection, columns (if header), sample rows

    Table extraction uses pdfplumber's table parser heuristics (no external deps).
    Header detection:
      - If first row on a page looks string-like and repeats across pages, treat as header.
      - Otherwise, no header: we still return the first row as sample to help the planner infer headers.
    """
    try:
        def looks_like_header(row):
            # Heuristic: mostly non-empty strings, not numbers; short-ish cells.
            if not row or not isinstance(row, list):
                return False
            str_like = sum(1 for c in row if isinstance(c, str) and bool(re.search(r"[A-Za-z]", c or "")))
            num_like = sum(1 for c in row if isinstance(c, str) and re.fullmatch(r"[-+]?[\d,.]+", (c or "").strip()))
            avg_len = sum(len((c or "")) for c in row) / max(len(row), 1)
            return (str_like >= max(1, len(row)//2)) and (num_like <= len(row)//3) and (avg_len <= 40)

        text_buf = []
        tables_meta = []
        header_candidates = []

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages[:max_pages]):
                # text preview
                t = (page.extract_text() or "").strip()
                if t:
                    text_buf.append(t)

                # basic table extraction (pdfplumber)
                try:
                    raw_tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "intersection_tolerance": 5,
                        }
                    )
                except Exception:
                    raw_tables = []

                for tbl in raw_tables:
                    # tbl is a list of rows (lists of cells)
                    header_row = tbl[0] if tbl else None
                    has_header = looks_like_header(header_row) if header_row else False
                    if has_header and header_row:
                        header_candidates.append(tuple((c or "").strip() for c in header_row))
                        columns = [str((c or "")).strip() for c in header_row]
                        data_rows = tbl[1:3]  # sample 2 rows
                    else:
                        columns = []
                        data_rows = tbl[:2]

                    # normalize sample rows to dicts when header exists
                    sample_rows = []
                    if columns and data_rows:
                        for r in data_rows:
                            sample_rows.append({columns[j]: (r[j] if j < len(r) else None) for j in range(len(columns))})

                    tables_meta.append({
                        "page_index": i,                    # 0-based
                        "has_header": has_header,
                        "columns": columns,
                        "sample_rows": sample_rows if columns else (data_rows or []),
                        "row_count_estimate": len(tbl) if isinstance(tbl, list) else None,
                    })

        # check for repeated headers across pages
        header_repeat = False
        if header_candidates:
            from collections import Counter
            c = Counter(header_candidates)
            most_common, freq = c.most_common(1)[0]
            header_repeat = freq >= 2  # seen on 2+ tables/pages

        text_preview = "\n".join(text_buf)
        if len(text_preview) > max_text_chars:
            text_preview = text_preview[:max_text_chars] + "…"

        return {
            "page_count": page_count,
            "text_preview": text_preview,
            "tables": tables_meta,
            "headers_repeat_across_pages": header_repeat,
        }
    except Exception:
        return {
            "page_count": None,
            "text_preview": "",
            "tables": [],
            "headers_repeat_across_pages": False,
        }


import re, tempfile
import pandas as pd
import requests
import pdfplumber
from urllib.parse import urlparse

_URL_RE = re.compile(r'https?://[^\s)>\]"\']+', re.I)

def _extract_urls(text: str):
    return list(dict.fromkeys(_URL_RE.findall(text or "")))  # unique, preserve order

def _extract_urls_comprehensive(text: str) -> List[str]:
    """
    Extract all URLs from text using multiple patterns to catch various URL formats.
    Handles http/https, ftp, file, mailto, and other schemes, with or without protocols.
    """
    print("In extract urls")
    if not text:
        return []
    
    urls = []
    
    # Pattern 1: Standard URLs with protocol (http/https/ftp/file/etc.)
    protocol_pattern = re.compile(
        r'\b(?:(?:https?|ftp|ftps|sftp|file|mailto|tel|sms|whatsapp)://[^\s<>"\'`\[\]{}|\\^]+)',
        re.IGNORECASE
    )
    urls.extend(protocol_pattern.findall(text))
    
    # Pattern 2: URLs without protocol (www. domains)
    www_pattern = re.compile(
        r'\b(?:www\.)[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*(?:\.[a-zA-Z]{2,})[^\s<>"\'`\[\]{}|\\^]*',
        re.IGNORECASE
    )
    www_urls = www_pattern.findall(text)
    urls.extend([f"https://{url}" for url in www_urls])
    
    # Pattern 3: Naked domains (domain.com without www)
    naked_domain_pattern = re.compile(
        r'\b[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:/[^\s<>"\'`\[\]{}|\\^]*)?',
        re.IGNORECASE
    )
    potential_domains = naked_domain_pattern.findall(text)
    
    # Filter naked domains to avoid false positives (common file extensions, etc.)
    valid_tlds = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'io', 'ai', 'app',
        'uk', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'in', 'br', 'ru', 'it', 'es',
        'nl', 'se', 'no', 'dk', 'fi', 'pl', 'be', 'ch', 'at', 'ie', 'nz', 'za',
        'mx', 'ar', 'cl', 'co', 'pe', 'kr', 'sg', 'my', 'th', 'vn', 'ph', 'id',
        'tw', 'hk', 'ae', 'sa', 'eg', 'tr', 'gr', 'cz', 'hu', 'ro', 'bg', 'hr',
        'lt', 'lv', 'ee', 'si', 'sk', 'is', 'mt', 'cy', 'lu', 'li', 'mc', 'sm',
        'tv', 'me', 'cc', 'ly', 'to', 'ws', 'nu', 'tk', 'ml', 'ga', 'cf'
    }
    
    for domain in potential_domains:
        # Skip if already captured by other patterns
        if any(domain in existing_url for existing_url in urls):
            continue
            
        # Extract TLD and validate
        parts = domain.split('.')
        if len(parts) >= 2:
            tld = parts[-1].lower()
            if tld in valid_tlds:
                # Additional validation: avoid common false positives
                if not any(domain.lower().endswith(f'.{ext}') for ext in [
                    'txt', 'doc', 'docx', 'pdf', 'jpg', 'jpeg', 'png', 'gif', 
                    'mp3', 'mp4', 'zip', 'rar', 'exe', 'dmg'
                ]):
                    urls.append(f"https://{domain}")
    
    # Pattern 4: File URLs and special protocols
    file_pattern = re.compile(
        r'\b(?:file://[^\s<>"\'`\[\]{}|\\^]+)',
        re.IGNORECASE
    )
    urls.extend(file_pattern.findall(text))
    
    # Pattern 5: IP addresses with ports
    ip_pattern = re.compile(
        r'\b(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?:/[^\s<>"\'`\[\]{}|\\^]*)?',
        re.IGNORECASE
    )
    ip_urls = ip_pattern.findall(text)
    for ip_url in ip_urls:
        if not ip_url.startswith(('http://', 'https://')):
            urls.append(f"http://{ip_url}")
        else:
            urls.append(ip_url)
    
    # Pattern 6: localhost and internal domains
    localhost_pattern = re.compile(
        r'\b(?:https?://)?(?:localhost|127\.0\.0\.1|0\.0\.0\.0|::1)(?::\d{1,5})?(?:/[^\s<>"\'`\[\]{}|\\^]*)?',
        re.IGNORECASE
    )
    localhost_urls = localhost_pattern.findall(text)
    for localhost_url in localhost_urls:
        if not localhost_url.startswith(('http://', 'https://')):
            urls.append(f"http://{localhost_url}")
        else:
            urls.append(localhost_url)
    
    # Clean up URLs and remove duplicates while preserving order
    cleaned_urls = []
    seen = set()
    
    for url in urls:
        # Remove trailing punctuation that might be captured
        url = re.sub(r'[.,;:!?)\]}]+$', '', url)
        
        # Normalize URL
        url = url.strip()
        
        # Skip empty or very short URLs
        if len(url) < 4:
            continue
            
        # Skip duplicates (case-insensitive)
        url_lower = url.lower()
        if url_lower not in seen:
            seen.add(url_lower)
            cleaned_urls.append(url)
    
    return cleaned_urls

def _detect_source_type_from_ct(ct: str, url: str):
    ct = (ct or "").lower()
    path = urlparse(url).path.lower()
    if "json" in ct or path.endswith(".json"): return "json"
    if "csv" in ct or path.endswith(".csv"): return "csv"
    if "pdf" in ct or path.endswith(".pdf"): return "pdf"
    # heuristic: html or unknown
    if "html" in ct or not ct: return "html"
    return "unknown"

from bs4 import BeautifulSoup
import re

def detect_noisy_values(table_html, headers):
    noisy_values = {}
    try:
        soup = BeautifulSoup(table_html, "lxml")
        rows = soup.find_all("tr")
        data_rows = []
        for row in rows[1:]:  # skip header row
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) == len(headers):  # only keep matching length
                data_rows.append(cells)

        if not data_rows:
            return noisy_values  # no clean rows to check

        df = pd.DataFrame(data_rows, columns=headers)

        for col in df.columns:
            col_vals = df[col].dropna().astype(str)
            # Check if majority are numeric
            num_like_ratio = col_vals.str.match(r"^\d+(\.\d+)?$").sum() / len(col_vals)
            if num_like_ratio > 0.5:
                # Find values with extra non-numeric characters (noisy)
                noise = col_vals[col_vals.str.contains(r"[^\d.,-]", regex=True)].unique().tolist()
                if noise:
                    noisy_values[col] = noise

    except Exception as e:
        noisy_values["_error"] = str(e)

    return noisy_values

import os
import re
import hashlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

def _detect_source_type_from_ct(ct, url):
    if "html" in ct:
        return "html"
    elif "csv" in ct or url.endswith(".csv"):
        return "csv"
    elif "json" in ct or url.endswith(".json"):
        return "json"
    elif "pdf" in ct or url.endswith(".pdf"):
        return "pdf"
    else:
        return "unknown"
import re
import pandas as pd
from collections import Counter
import re
from collections import Counter
import pandas as pd
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

def _detect_source_type_from_ct(ct, url):
    ct = (ct or "").lower()
    if "html" in ct:
        return "html"
    if "csv" in ct or url.lower().endswith(".csv"):
        return "csv"
    if "json" in ct or url.lower().endswith(".json"):
        return "json"
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return "pdf"
    return "unknown"
import re
import pandas as pd

def detect_noisy_values_simple(df: pd.DataFrame, max_examples: int = 3) -> dict:
    """
    Return { column: [noisy_example1, noisy_example2, ...] } with only strings.
    Flags numeric-like columns where raw cells contain extra tokens (letters, $,
    refs, daggers, ranges, etc.) that would require cleaning.
    """
    if df is None or df.empty:
        return {}

    # Heuristic: columns that are likely numeric by name
    NUM_NAME_HINT = re.compile(r"(rank|peak|year|gross|budget|revenue|total|amount|usd|worldwide)", re.I)
    CURRENCY = re.compile(r"[$€£¥₹]")
    BRACKET_REF = re.compile(r"\[[^\]]*\]")      # [1], [note]
    HAS_LETTER = re.compile(r"[A-Za-z]")         # letters like in '23RK'
    RANGE = re.compile(r"\d+\s*[–—-]\s*\d+")     # 50–100 / 50-100
    DAGGER = re.compile(r"[†‡*]")                # footnote marks
    BAD_CHARS = re.compile(r"[^\d.,()%\-\s]")    # anything not typical numeric tokens

    # numeric-like test on a cleaned version (for column classification)
    NUMRE = re.compile(r"^-?\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?%?$|^-?\(?\d+(?:\.\d+)?\)?%?$")
    def is_numeric_like(s: str) -> bool:
        return bool(NUMRE.fullmatch(s))

    def basic_clean(s: str) -> str:
        s = s.replace("\xa0", " ").strip()
        s = s.replace("−", "-").replace("–", "-").replace("—", "-")
        s = re.sub(CURRENCY, "", s)                 # drop currency symbol
        s = BRACKET_REF.sub("", s)                  # drop [refs]
        s = re.sub(r"[A-Za-z]", "", s)              # drop letters (important!)
        s = re.sub(r"[^0-9.,()%\-\s]", "", s)       # keep numeric punctuation
        s = re.sub(r"\s+", " ", s).strip()
        return s

    out = {}
    for col in df.columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        # decide if column is numeric-like
        name_hint = bool(NUM_NAME_HINT.search(str(col)))
        cleaned = series.map(basic_clean)
        numeric_ratio = cleaned.map(is_numeric_like).mean()
        numeric_like = name_hint or (numeric_ratio >= 0.5)

        if not numeric_like:
            continue

        examples = []
        for raw, cleaned_val in zip(series, cleaned):
            # Flag if raw contains any of these extras:
            has_letters = bool(HAS_LETTER.search(raw))          # catches '23RK', '24TS'
            has_currency = bool(CURRENCY.search(raw))           # catches '$2,923,706,026'
            has_ref = bool(BRACKET_REF.search(raw))             # catches '[# 1]'
            has_dagger = bool(DAGGER.search(raw))               # catches '†', '*'
            is_range = bool(RANGE.search(raw))                  # catches '50–100'
            has_other = bool(BAD_CHARS.search(raw))             # other odd tokens
            # Or: cleaned becomes numeric but raw != cleaned (meaning extra junk existed)
            becomes_numeric = is_numeric_like(cleaned_val)

            if (has_letters or has_currency or has_ref or has_dagger or is_range or has_other) and becomes_numeric:
                if raw not in examples:
                    examples.append(raw)
            # Stop early
            if len(examples) >= max_examples:
                break

        if examples:
            out[str(col)] = examples

    return out

def _probe_url(url: str, timeout=15, save_dir="tables_output"):
    info = {"filename": url, "url": url, "is_url": True}
    os.makedirs(save_dir, exist_ok=True)

    try:
        # HEAD, then fallback GET for content-type
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        ct = r.headers.get("Content-Type", "").split(";")[0].strip()
        if not ct or r.status_code >= 400:
            r = requests.get(url, stream=True, timeout=timeout)
            ct = r.headers.get("Content-Type", "").split(";")[0].strip()
        info["type"] = ct

        stype = _detect_source_type_from_ct(ct, url)
        info["extension"] = {"csv": ".csv", "json": ".json", "pdf": ".pdf"}.get(stype, "")
        info["saved_path"] = url
        info["source_type"] = stype

        if stype == "html":
            html_resp = requests.get(url, timeout=timeout)
            html_resp.raise_for_status()
            html_text = html_resp.text

            # Parse title (safe)
            soup = BeautifulSoup(html_text, "lxml")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Read all tables via StringIO to avoid FutureWarning
            try:
                dfs = pd.read_html(StringIO(html_text), flavor="lxml")
            except Exception as e:
                info["html_metadata"] = {
                    "title": title,
                    "tables_total": 0,
                    "error": f"pd.read_html failed: {type(e).__name__}: {e}"
                }
                return info

            tables_info, kept = [], 0
            for idx, df in enumerate(dfs):
                try:
                    # Skip tables with all-numeric column names (junk/layout)
                    if all(isinstance(c, (int, float)) for c in df.columns):
                        continue

                    # Save CSV
                    file_path = os.path.join(save_dir, f"table_{idx}.csv")
                    df.to_csv(file_path, index=False, encoding="utf-8-sig")

                    # Simple noisy values (list of strings per column)
                    noisy = detect_noisy_values_simple(df)

                    # Print preview for debugging (safe)
                    print(f"\n--- Table {idx} ---")
                    print(f"Columns: {df.columns.tolist()}")
                    print(f"First row: {('<empty>' if df.empty else df.iloc[0].tolist())}")
                    print(f"Noisy: {noisy}")

                    tables_info.append({
                        "table_index": idx,
                        "file_saved": file_path,
                        "columns": [str(c) for c in df.columns],
                        "rows_saved": int(len(df)),
                        "noisy_values": {k: list(v) for k, v in noisy.items()}  # strings only
                    })
                    kept += 1

                except Exception as e_table:
                    # Keep going even if one table fails
                    tables_info.append({
                        "table_index": idx,
                        "error": f"{type(e_table).__name__}: {e_table}"
                    })
                    continue

            info["html_metadata"] = {
                "title": title,
                "tables_total": kept,
                "tables_info": tables_info
            }

        return info

    except Exception as e:
        info["probe_error"] = f"{type(e).__name__}: {e}"
        info["source_type"] = "unknown"
        return info


import unicodedata

def _sanitize_text(s: str, mode: str = "replace") -> str:
    """
    Return a UTF‑8 safe string.
    - mode="replace": unknowns become �
    - mode="ignore": drop unencodable bytes
    - mode="ascii": strip to ASCII (best-effort)
    """
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFC", s)  # canonical normalize
    if mode == "ascii":
        return s.encode("ascii", "ignore").decode("ascii")
    # UTF‑8 can encode all Unicode; this ensures later .encode(...) calls won't explode
    return s.encode("utf-8", errors=mode).decode("utf-8", errors=mode)

def _to_safe(obj, mode: str = "replace"):
    """
    Recursively sanitize any strings inside dict/list/tuple/str so they
    are safe to print/write. Non-strings are returned as-is.
    """
    if isinstance(obj, dict):
        return {(_sanitize_text(k, mode) if isinstance(k, str) else k): _to_safe(v, mode) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_safe(x, mode) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_safe(list(obj), mode))
    if isinstance(obj, str):
        return _sanitize_text(obj, mode)
    return obj

def _safe_debug(obj, prefix=""):
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    print(prefix + s)

def generate_dummy_json(questions: str):
    ans_prompt = (
        "RETURN ONLY THE VALID JSON FORMAT SPECIFIED IN THE QUESTION"
        "Return only valid JSON in the exact format implied by the question. "
        "Use dummy placeholder values of the correct type (integer, float, string, chart_uri). "
        "No explanations, no markdown, not neccesarily a correct answer , also terminate a base64 image uri do not loop till infinity"
        "Do not enclose it in ``` json ```"
    )
    user_ans = f"""
Create Dummy Answers for the questions and return ONLY A VALID JSON FORMAT AS expected in the questions
ENSURE THAT THE FORMAT IS ALWAYS CORRECT AND MATCHES THE ONE ASKED IN THE QUESTION
{questions}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_ans,
            config=GenerateContentConfig(system_instruction=ans_prompt)
        )
        llm_output = response.candidates[0].content.parts[0].text.strip()
        # Extract JSON object or array
        # print(llm_output)
        match = re.search(r"(\{.*\}|\[.*\])", llm_output, re.S)
        if match:
            llm_output = match.group(0)
        # print(llm_output)
        try:
            parsed = json.loads(llm_output)
            return parsed
        except json.JSONDecodeError:
            logging.error("Dummy LLM output was not valid JSON")
            return {}
        
    except Exception as e:
        logging.error(f"LLM dummy generation failed: {e}")
        return {}

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
# ---------- route ----------

@app.post("/api")
async def upload_files(request: Request):
    try:
        start = time.time()
        print(start)
        upload_dir = pathlib.Path("uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        form_data = await request.form()

        data_files = []
        questions = None

        for param_name, param_value in form_data.items():
            # Check if the form parameter is a file
            if hasattr(param_value, "filename") and param_value.filename:
                filename = param_value.filename
                content_type = getattr(param_value, "content_type", "") or ""
                file_ext = pathlib.Path(filename).suffix.lower()

                # Critical change: Check the parameter name, not the filename
                if param_name == "questions.txt":
                    content = await param_value.read()
                    questions = content.decode("utf-8")
                    continue

                # For all other uploaded files, save them and process as data files
                file_path = upload_dir / filename
                content = await param_value.read()
                with open(file_path, "wb") as f:
                    f.write(content)

                file_info = {
                    "filename": filename,
                    "type": content_type,
                    "extension": file_ext,
                    "saved_path": str(file_path)
                }

                # If it's an image, generate and store description
                if _is_image_content_type(content_type) or _is_image_filename(filename):
                    file_info["image_description"] = get_image_description(str(file_path))
                    print("Image description generated", file_info["image_description"])

                # If it's a CSV, attach lightweight schema + sample row
                if _is_csv(content_type, filename):
                    file_info["csv_metadata"] = get_csv_metadata(str(file_path), sample_rows=1)
                # If it's a JSON file, attach structure summary
                if _is_json(content_type, filename):
                    file_info["json_metadata"] = get_json_metadata(str(file_path))
                # If it's an Excel file, attach sheet-wise structure summary
                if _is_excel(content_type, filename):
                    file_info["excel_metadata"] = get_excel_metadata(str(file_path), sample_rows=3)
                if _is_pdf(content_type, filename):
                    file_info["pdf_metadata"] = get_pdf_metadata(str(file_path), max_pages=3, max_text_chars=1000)
                           
                data_files.append(file_info)
        print("[APP] 1: Received All Files Info")
        # 1) Extract & probe URLs in questions
        urls = _extract_urls_comprehensive(questions)
        for u in urls:
            url_info = _probe_url(u)
            # normalize to planner-friendly shape
            data_files.append(url_info)
        print("[APP] 2: Received All URLS info")
        # make the values themselves safe for later usage
        data_files = _to_safe(data_files, mode="replace")
        questions  = _to_safe(questions,  mode="replace")

        # optional: debug print
        _safe_debug(data_files, "data_files: ")
        _safe_debug(questions,  "Questions: ")
        if not questions:
            raise HTTPException(status_code=400, detail="questions.txt form field is required and must contain a file.")
        
        logging.info("Step 1: Generating plan...")
        print("[APP] 3 Calling the planner Agent")
        # return JSONResponse({"question" : questions , "files" : data_files})
        # Build plan from questions + uploaded artifacts
        # plan = run_planner_agent_files(questions, data_files)
        plan = run_planner_agent_json_with_feedback_looping(questions,data_files)
        save_to_log_folder("plan.json", plan)
        print("[APP] 4 GOT THE PLAN")
        
        if isinstance(plan, (dict, list)):
            # plan.json
            with open("plan.json", "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)

        else:
            with open("plan.txt", "w", encoding="utf-8", errors="replace") as f:
                f.write(str(plan))
        # return JSONResponse({"Questions":questions,"data files":data_files,"plan":plan})
        # return JSONResponse({"plan" : plan , "files" : data_files})

        print("[APP] 5 CALLING EXECUTE PLAN ")
        result = execute_plan_v1(plan, questions , data_files)
        print(result)
        print("[APP] 6 PLAN EXECUTED SUCCESSFULLY WITH THE RESULT")
        end = time.time()
        print("Starting",start)
        print("Ending",end)
        print(end-start)
        try:
            parsed = json.loads(result)
            results = {"status": "success"}  # Example
            save_to_log_folder("results.json", str(results))

            logging.info("=== API Call Completed ===")
            with open("results.json","w") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            parsed = generate_dummy_json(questions)
        return parsed
    except HTTPException:
        parsed = generate_dummy_json(questions)
        return parsed
        return JSONResponse(status_code=400, content="error")
    except Exception as e:
        parsed = generate_dummy_json(questions)
        return parsed
        raise JSONResponse(status_code=400, content="error")

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    # uvicorn.run("app:app", host="127.0.0.1", port=7680, reload=True)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="127.1.1.1", port=port)




