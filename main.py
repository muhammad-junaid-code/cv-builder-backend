"""
CV Builder AI - FastAPI Backend v11
Port: 8001  |  Start: uvicorn main:app --host 0.0.0.0 --port 8001

Providers: Groq, Cerebras, Gemini, DeepSeek, OpenAI, Ollama
100% Dynamic - No hardcoded technologies, no static fallbacks, no predefined categories
Everything derived from Job Description + Job Title only
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import httpx, json, re, math, io, asyncio, secrets, string
from datetime import date, datetime, timedelta

# reportlab - PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

app = FastAPI(title="CV Builder AI", version="11.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models"
OLLAMA_URL   = "http://localhost:11434"

# Only static data: company names (users can edit via profile)
CANDIDATE_COMPANIES = [
    {"name": "MULTYLOGICS SOLUTIONS",        "start": "May 2024",  "end": "Present"},
    {"name": "ENCS NETWORKS",                "start": "May 2022",  "end": "May 2024"},
    {"name": "NOW TECHNOLOGIES (NOW.NET.PK)","start": "May 2020",  "end": "May 2022"},
]

# Month helpers
_MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"]
_MONTH_MAP = {m.lower(): i+1 for i, m in enumerate(_MONTH_NAMES)}
_MONTH_MAP.update({m.lower()[:3]: i+1 for i, m in enumerate(_MONTH_NAMES)})

def _month_name(n: int) -> str:
    return _MONTH_NAMES[(n - 1) % 12]

def _parse_month_year(s: str) -> date:
    s = s.strip()
    if s.lower() == "present":
        return date.today()
    parts = s.split()
    if len(parts) == 2:
        return date(int(parts[1]), _MONTH_MAP.get(parts[0].lower(), 1), 1)
    raise ValueError(f"Cannot parse date: {s!r}")

def _months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))

def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    return date(year, month, 1)

def _subtract_months(d: date, months: int) -> date:
    return _add_months(d, -months)

def _calc_total_years(years_exp: str = "") -> str:
    if years_exp:
        try:
            n = float(years_exp.strip().replace("+", ""))
            return str(int(n)) if n == int(n) else str(round(n, 1))
        except ValueError:
            pass
    total_months = 0
    for co in CANDIDATE_COMPANIES:
        try:
            start = _parse_month_year(co["start"])
            end = _parse_month_year(co["end"])
            total_months += _months_between(start, end)
        except Exception:
            pass
    y = total_months / 12
    if y >= 5: return "5+"
    elif y >= 4: return "4+"
    elif y >= 3: return "3+"
    elif y >= 2: return "2+"
    else: return "1+"

def _build_dynamic_companies(years_exp: str, num_companies: int = 0) -> list:
    if not years_exp:
        n_fallback = num_companies if num_companies > 0 else 3
        return CANDIDATE_COMPANIES[:n_fallback]
    try:
        n = float(years_exp.strip().replace("+", ""))
    except ValueError:
        return CANDIDATE_COMPANIES[:3]
    total_months = int(round(n * 12))
    today = date.today()
    def fmt(d: date) -> str:
        return f"{_month_name(d.month)} {d.year}"
    if num_companies > 0:
        num_cos = num_companies
    elif n <= 1.4:
        num_cos = 1
    elif n <= 2.4:
        num_cos = 2
    else:
        num_cos = 3
    each = total_months // num_cos if num_cos > 0 else total_months
    remainder = total_months - each * num_cos
    result = []
    cursor = today
    for i in range(num_cos):
        span = each + (remainder if i == 0 else 0)
        co_start = _subtract_months(cursor, span)
        co_end = "Present" if i == 0 else fmt(cursor)
        name = CANDIDATE_COMPANIES[i]["name"] if i < len(CANDIDATE_COMPANIES) else f"Company {i+1}"
        result.append({"name": name, "start": fmt(co_start), "end": co_end})
        cursor = co_start
    return result

def _build_education_year(years_exp: str, profile_edu: list = None) -> dict:
    today = date.today()
    if profile_edu:
        e = profile_edu[0]
        p_from = str(e.get("from", "") or "").strip()
        p_to = str(e.get("to", "") or "").strip()
        if p_from and p_to:
            return {"start": p_from, "end": p_to}
        if p_to and not p_from:
            try:
                end_yr = int(p_to[:4])
                return {"start": str(end_yr - 4), "end": p_to}
            except (ValueError, TypeError):
                pass
        if p_from and not p_to:
            try:
                start_yr = int(p_from[:4])
                return {"start": p_from, "end": str(start_yr + 4)}
            except (ValueError, TypeError):
                pass
    if years_exp:
        try:
            n = int(float(years_exp.strip().replace("+", "")))
            end_year = today.year - max(n - 1, 0)
            start_year = end_year - 4
            return {"start": str(start_year), "end": str(end_year)}
        except ValueError:
            pass
    return {"start": "2017", "end": "2021"}

# ==============================================================================
# AUTH SYSTEM
# ==============================================================================
import os

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))
AUTH_FILE = os.path.join(_DATA_DIR, "auth_keys.json")
STATIC_ACCESS_KEY = "CVAI-A927-42F8-1E31"

def _load_auth_keys() -> dict:
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, "r") as f:
                keys = json.load(f)
        except Exception:
            keys = {}
    else:
        keys = {}
    if STATIC_ACCESS_KEY not in keys:
        keys[STATIC_ACCESS_KEY] = {
            "label": "Static Key",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": None,
            "active": True,
            "last_used": None,
        }
        _save_auth_keys(keys)
    return keys

def _save_auth_keys(keys: dict):
    with open(AUTH_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def _is_token_valid(token_data: dict) -> bool:
    if not token_data.get("active", True):
        return False
    expires_at = token_data.get("expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at)
            if datetime.utcnow() > exp:
                return False
        except Exception:
            return False
    return True

class GenerateKeyRequest(BaseModel):
    label: str = "Access Key"
    days_valid: int = 30
    admin_pass: str = ""

class LoginRequest(BaseModel):
    token: str

class VerifyRequest(BaseModel):
    token: str

class RevokeRequest(BaseModel):
    token: str
    admin_pass: str = ""

ADMIN_PASSWORD = os.environ.get("CV_ADMIN_PASS", "admin1234")

@app.post("/auth/generate-key")
async def generate_key(req: GenerateKeyRequest):
    if req.admin_pass != ADMIN_PASSWORD:
        raise HTTPException(403, "Invalid admin password")
    keys = _load_auth_keys()
    raw = secrets.token_hex(8).upper()
    token = f"CVAI-{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"
    now = datetime.utcnow()
    expires_at = (now + timedelta(days=req.days_valid)).isoformat() if req.days_valid > 0 else None
    keys[token] = {
        "label": req.label,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
        "active": True,
        "last_used": None,
    }
    _save_auth_keys(keys)
    return {"token": token, "label": req.label, "expires_at": expires_at, "message": "Key generated successfully"}

@app.post("/auth/login")
async def login(req: LoginRequest):
    keys = _load_auth_keys()
    token = req.token.strip().upper()
    entry = keys.get(token)
    if not entry:
        raise HTTPException(401, "Invalid key - not found")
    if not _is_token_valid(entry):
        raise HTTPException(401, "Key is expired or revoked")
    entry["last_used"] = datetime.utcnow().isoformat()
    keys[token] = entry
    _save_auth_keys(keys)
    return {"ok": True, "token": token, "label": entry.get("label", ""), "expires_at": entry.get("expires_at"), "message": "Login successful"}

@app.post("/auth/verify")
async def verify_token(req: VerifyRequest):
    keys = _load_auth_keys()
    token = req.token.strip().upper()
    entry = keys.get(token)
    if not entry or not _is_token_valid(entry):
        return {"valid": False}
    return {"valid": True, "label": entry.get("label", ""), "expires_at": entry.get("expires_at")}

@app.post("/auth/revoke")
async def revoke_key(req: RevokeRequest):
    if req.admin_pass != ADMIN_PASSWORD:
        raise HTTPException(403, "Invalid admin password")
    keys = _load_auth_keys()
    token = req.token.strip().upper()
    if token not in keys:
        raise HTTPException(404, "Key not found")
    keys[token]["active"] = False
    _save_auth_keys(keys)
    return {"ok": True, "message": "Key revoked"}

# ==============================================================================
# KEY MANAGEMENT
# ==============================================================================
_key_usage: dict = {}
_key_rate_limited_until: dict = {}
_debug_log: list = []

def mask(key: str) -> str:
    k = (key or "").strip()
    return k[:8] + "..." + k[-4:] if len(k) > 12 else "***"

def est_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.8)

def _log_generation(job_title: str, key_masked: str, key_index: int, prompt_tokens: int, model: str, success: bool, error: str = ""):
    _debug_log.insert(0, {"job_title": job_title, "key_used": key_masked, "key_index": key_index, "model": model, "prompt_tokens": prompt_tokens, "success": success, "error": error})
    if len(_debug_log) > 10:
        _debug_log.pop()

def _prioritised_keys(valid_keys: list) -> list:
    import time
    now = time.time()
    def _sort_key(k):
        mk = mask(k)
        cooldown_until = _key_rate_limited_until.get(mk, 0)
        is_limited = 1 if cooldown_until > now else 0
        return (is_limited, _key_usage.get(mk, 0))
    return sorted(valid_keys, key=_sort_key)

# ==============================================================================
# CV Request Model
# ==============================================================================
class CVRequest(BaseModel):
    job_title: str
    job_description: str
    years_exp: Optional[str] = ""
    provider: str = "cerebras"
    model: str = "llama3.1-8b"
    groq_keys: Optional[List[str]] = []
    cerebras_keys: Optional[List[str]] = []
    deepseek_keys: Optional[List[str]] = []
    openai_keys: Optional[List[str]] = []
    gemini_keys: Optional[List[str]] = []
    ollama_model: Optional[str] = "qwen2.5:7b"
    profile: str = ""
    profile_data: Optional[dict] = None
    static_data: Optional[dict] = None
    company_name: Optional[str] = ""
    company_context: Optional[str] = ""

# ==============================================================================
# DYNAMIC PROMPT BUILDER - EVERYTHING FROM JD ONLY
# ==============================================================================
def build_dynamic_prompt(req: CVRequest) -> tuple:
    """
    Build a prompt that forces the AI to derive EVERYTHING from the JD.
    NO hardcoded technologies, NO regex extraction, NO static content.
    The AI reads the JD and decides everything.
    """
    jd = req.job_description.strip()
    job_title = req.job_title.strip()
    years_exp = (req.years_exp or "").strip()
    
    # Format years display for the prompt
    if years_exp:
        years_display = f"{years_exp}+" if years_exp.isdigit() and "+" not in years_exp else years_exp
    else:
        years_display = _calc_total_years(years_exp)
        if years_display.isdigit():
            years_display = f"{years_display}+"
    
    companies = _build_dynamic_companies(years_exp)
    num_cos = len(companies)
    edu = _build_education_year(years_exp)
    
    # Build company list for the prompt
    co_lines = "\n".join(
        f'Company {i+1}: "{c["name"]}" (Dates: {c["start"]} - {c["end"]})'
        for i, c in enumerate(companies)
    )
    
    # Company context - just pass raw data, AI analyzes it
    company_context_block = ""
    if req.company_context and req.company_name:
        company_context_block = f"""
TARGET COMPANY CONTEXT:
Company Name: {req.company_name}
Company Data: {req.company_context[:1500]}

Read and understand this company data. Use it to make projects relevant to this company.
"""
    elif req.company_name:
        company_context_block = f"""
TARGET COMPANY: {req.company_name}
Use your knowledge of this company to create relevant projects.
"""
    else:
        company_context_block = "No target company provided."
    
    # Profile block - just pass raw data
    profile_block = ""
    if req.profile_data:
        profile_name = req.profile_data.get("name", "")
        profile_links = req.profile_data.get("links", [])
        
        if profile_name:
            profile_block = f"""
CANDIDATE NAME: {profile_name}
CONTACT: {', '.join([l.get('value', '') for l in profile_links[:4]])}
"""
    
    system_prompt = f"""You are an expert CV writer. Generate a complete professional CV based ONLY on the job description below.

JOB DETAILS:
- Job Title: {job_title}
- Experience: {years_display} years (use EXACTLY this format with the + sign in your output)
- Companies: {num_cos} positions

COMPANIES (use these exact names and dates):
{co_lines}

EDUCATION YEARS: {edu['start']} - {edu['end']}

{company_context_block}

{profile_block}

========================================
CRITICAL TITLE FORMAT REQUIREMENT:
========================================

The title MUST follow this EXACT format:

[Position Title] | [Tech1], [Tech2], [Tech3] | [Experience]

EXPLANATION OF EACH PART:

1. POSITION TITLE:
   - MUST be derived from the Job Title
   - NOT an exact copy - rephrase it
   - Keep the SAME meaning and seniority level
   - Example: Job Title "Python Developer" → "Python Software Engineer"
   - Example: Job Title "Senior SEO Specialist" → "Senior Search Engine Optimization Expert"
   - Example: Job Title "Generative AI Engineer" → "LLM Application Developer"

2. TECHNICAL STACK (after first pipe):
   - Pick 3-5 key technologies from the Job Description
   - Separate with commas
   - Example: "Python, FastAPI, PostgreSQL, Docker"
   - Example: "Generative Engine Optimization, LLM, AI Visibility"

3. EXPERIENCE (after second pipe):
   - Use EXACTLY: "{years_display}+"
   - Example: "5+" or "3+" or "8+"

VALID TITLE EXAMPLES:
- "Python Software Engineer | FastAPI, PostgreSQL, Docker, Redis | 5+"
- "Senior SEO Specialist | Generative Engine Optimization, LLM, AI Visibility | 4+"
- "Generative AI Engineer | LangChain, OpenAI API, Vector Databases, RAG | 3+"
- "Full Stack Developer | React, Node.js, MongoDB, AWS | 6+"
- "Data Engineer | Python, Spark, Airflow, BigQuery | 5+"

INVALID TITLE EXAMPLES (DO NOT USE):
- "AI-DRIVEN SEARCH STRATEGIST | 5+" (missing technical stack)
- "5+ | Python Developer" (wrong order)
- "Senior SEO Specialist | 5+" (missing technical stack)
- Exact copy of Job Title without rephrasing

========================================
CRITICAL SUMMARY FORMAT REQUIREMENT:
========================================

The summary MUST start with: "{years_display}+ years of experience in [domain from JD]..."

Example: "5+ years of experience in Generative Engine Optimization and Large Language Models..."

========================================
TASK: Generate a complete CV in JSON format
========================================

Read the job description carefully. Extract ALL information from it. Then generate:

1. **title**: MUST follow format: "[Rephrased Position Title] | [Tech1], [Tech2], [Tech3] | {years_display}+"
   - Rephrase the Job Title (not exact copy)
   - Pick 3 technologies from the JD for the middle section
   - Add experience at the end with + sign

2. **summary**: Write 6-7 sentences (120-180 words) that:
   - Start EXACTLY with: "{years_display}+ years of experience in [domain from JD]..."
   - Naturally include 6-8 different technologies from the JD
   - Never repeat the same technology twice
   - Sound professional and human-written

3. **competencies**: 10 domain-specific phrases separated by " * " - all from the JD

4. **keywords**: 18-20 ATS keywords from the JD - comma separated

5. **technologies**: 
   - mustHave: technologies explicitly required in the JD (10-14 items)
   - niceToHave: technologies listed as preferred/nice-to-have (8-12 items)
   - additional: related technologies from the same ecosystem (8-10 items)

6. **skills**: 5 categories. For each category:
   - Name a category that makes sense for THIS job
   - List 10-12 technologies from the JD

7. **companies**: For each company, generate:
   - role: Seniority level based on experience + domain from JD + function word
   - bullets: 4 achievements (20-30 words each) using technologies from the JD
   - tech: 6-8 technologies from the JD separated by |

8. **projects**: EXACTLY 4 projects with:
   - name: Descriptive name (NEVER "Project 1", "Project 2")
   - overview: 3-4 sentences telling a complete story
   - bullets: 3 achievements with unique metrics
   - techTags: 5-7 technologies from the JD

9. **relatedTech**: 5 category boxes, 5 items each, all from the JD

10. **education**: Use the provided university and degree, add the years

========================================

OUTPUT FORMAT (JSON only, no markdown):

{{
  "title": "[Rephrased Position Title] | [Tech1], [Tech2], [Tech3] | {years_display}+",
  "summary": "{years_display}+ years of experience in [domain]... (6-7 sentences total, 120-180 words)",
  "competencies": "Phrase1 * Phrase2 * Phrase3 * Phrase4 * Phrase5 * Phrase6 * Phrase7 * Phrase8 * Phrase9 * Phrase10",
  "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, keyword7, keyword8, keyword9, keyword10, keyword11, keyword12, keyword13, keyword14, keyword15, keyword16, keyword17, keyword18",
  "technologies": {{
    "mustHave": ["tech1", "tech2", "tech3", "tech4", "tech5", "tech6", "tech7", "tech8", "tech9", "tech10", "tech11", "tech12"],
    "niceToHave": ["tech1", "tech2", "tech3", "tech4", "tech5", "tech6", "tech7", "tech8", "tech9", "tech10"],
    "additional": ["tech1", "tech2", "tech3", "tech4", "tech5", "tech6", "tech7", "tech8", "tech9", "tech10"]
  }},
  "skills": [
    "Category Name 1: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10, tech11, tech12",
    "Category Name 2: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10, tech11, tech12",
    "Category Name 3: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10, tech11, tech12",
    "Category Name 4: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10, tech11, tech12",
    "Category Name 5: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10, tech11, tech12"
  ],
  "companies": [
    {{
      "company": "Company Name",
      "role": "Seniority Domain Function",
      "dateRange": "Start - End",
      "bullets": [
        "Achievement description with technology and metric (20-30 words)",
        "Achievement description with different technology and metric (20-30 words)",
        "Achievement description with technology and metric (20-30 words)",
        "Achievement description with business impact (20-30 words)"
      ],
      "tech": "Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6 | Tech7 | Tech8"
    }}
  ],
  "projects": [
    {{
      "name": "Descriptive Project Name",
      "overview": "3-4 sentences explaining the problem, solution, functionality, and impact.",
      "bullets": [
        "Specific accomplishment with technology (20-30 words)",
        "Technical challenge solved with unique metric (20-30 words)",
        "Business outcome with measurable result (20-30 words)"
      ],
      "techTags": ["Tech1", "Tech2", "Tech3", "Tech4", "Tech5", "Tech6"]
    }},
    {{
      "name": "Another Descriptive Project Name",
      "overview": "3-4 sentences for a different project...",
      "bullets": ["Bullet 1", "Bullet 2", "Bullet 3"],
      "techTags": ["Tech1", "Tech2", "Tech3", "Tech4", "Tech5", "Tech6"]
    }},
    {{
      "name": "Company-Relevant Project Name",
      "overview": "3-4 sentences based on company context...",
      "bullets": ["Bullet 1", "Bullet 2", "Bullet 3"],
      "techTags": ["Tech1", "Tech2", "Tech3", "Tech4", "Tech5", "Tech6"]
    }},
    {{
      "name": "Industry Innovation Project Name",
      "overview": "3-4 sentences for industry-wide problem...",
      "bullets": ["Bullet 1", "Bullet 2", "Bullet 3"],
      "techTags": ["Tech1", "Tech2", "Tech3", "Tech4", "Tech5", "Tech6"]
    }}
  ],
  "education": {{
    "university": "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
    "degree": "Bachelor of Science in Computer Science (BSCS)",
    "cgpa": "3.97/4.0",
    "years": "{edu['start']} - {edu['end']}",
    "achievement": "Gold Medalist for Academic Excellence"
  }},
  "relatedTech": [
    {{"category": "Category Name 1", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
    {{"category": "Category Name 2", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
    {{"category": "Category Name 3", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
    {{"category": "Category Name 4", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
    {{"category": "Category Name 5", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}}
  ]
}}

========================================
TITLE TRANSFORMATION RULES:
========================================

When rephrasing the Job Title:
- Keep the SAME seniority level (Senior, Junior, Lead, etc.)
- Keep the SAME core domain
- Change the wording slightly
- Examples:
  - "Python Developer" → "Python Software Engineer"
  - "Senior SEO Specialist" → "Senior Search Engine Optimization Expert"
  - "Generative AI Engineer" → "LLM Application Developer"
  - "Frontend Developer" → "UI Engineering Specialist"
  - "Data Scientist" → "Machine Learning Analyst"
  - "DevOps Engineer" → "Cloud Infrastructure Engineer"

========================================
REMEMBER:
- Title format: "[Rephrased Position] | [Tech1], [Tech2], [Tech3] | {years_display}+"
- Summary starts with: "{years_display}+ years of experience in [domain]..."
- Read the job description below
- Extract EVERYTHING from it
- Generate ALL content based ONLY on what you read
- The CV must be DIFFERENT for every different job description
========================================
"""

    user_prompt = f"""
JOB DESCRIPTION:
{jd}

Generate the complete CV now.
CRITICAL TITLE FORMAT: "[Rephrased Position Title] | [Tech1], [Tech2], [Tech3] | {years_display}+"
CRITICAL SUMMARY START: "{years_display}+ years of experience in..."

Example for SEO role: "Generative AI Search Specialist | LLM Optimization, AI Visibility, Entity-Based SEO | 5+"
Example for Python role: "Python Software Engineer | FastAPI, PostgreSQL, Docker, Redis | 4+"
Example for .NET role: "Senior .NET Developer | ASP.NET Core, C#, Azure, Entity Framework | 6+"

Everything must come from this job description.
"""

    return system_prompt, user_prompt
# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def extract_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object in model response")
    end = raw.rfind("}")
    if end == -1:
        raw = raw[start:] + "}"
        j = raw
    else:
        j = raw[start:end + 1]
    j = re.sub(r",\s*([}\]])", r"\1", j)
    try:
        return json.loads(j)
    except json.JSONDecodeError:
        j = re.sub(r'[\x00-\x1f\x7f]', ' ', j)
        j = re.sub(r',\s*([}\]])', r'\1', j)
        return json.loads(j)

def esc_html(s: str) -> str:
    if not s:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def _is_real_tech(token: str) -> bool:
    token = token.strip()
    if len(token) < 2:
        return False
    if token.isdigit():
        return False
    return True

def _normalize_job_title(title: str) -> str:
    if not title:
        return title
    words = title.split()
    seen = []
    seen_lower = []
    for w in words:
        if w.lower() not in seen_lower:
            seen.append(w)
            seen_lower.append(w.lower())
    return " ".join(seen)

def sanitise_cv(cv: dict) -> dict:
    if not isinstance(cv, dict):
        return {}
    
    for field in ("totalYears", "title", "summary", "competencies", "keywords"):
        cv[field] = str(cv.get(field, "")).strip()
    
    if cv.get("title"):
        cv["title"] = _normalize_job_title(cv["title"])
    
    companies = cv.get("companies")
    if isinstance(companies, list):
        clean_companies = []
        for co in companies:
            if not isinstance(co, dict):
                continue
            bullets = co.get("bullets", [])
            if not isinstance(bullets, list):
                bullets = [str(bullets)]
            tech = co.get("tech", "")
            if isinstance(tech, list):
                tech = " | ".join(str(t) for t in tech if t)
            clean_companies.append({
                "company": str(co.get("company", "")),
                "role": str(co.get("role", "")),
                "dateRange": str(co.get("dateRange", "")),
                "bullets": [str(b).strip() for b in bullets if b],
                "tech": str(tech),
            })
        cv["companies"] = clean_companies
    
    skills = cv.get("skills", [])
    if isinstance(skills, list):
        clean_skills = []
        for s in skills:
            if s and isinstance(s, str):
                clean_skills.append(s.strip())
        cv["skills"] = clean_skills
    
    projects = cv.get("projects", [])
    if isinstance(projects, list):
        clean_projects = []
        for p in projects:
            if isinstance(p, dict):
                clean_projects.append({
                    "name": str(p.get("name", "")),
                    "overview": str(p.get("overview", "")),
                    "bullets": [str(b).strip() for b in (p.get("bullets") or []) if b],
                    "techTags": p.get("techTags", []) if isinstance(p.get("techTags"), list) else [str(p.get("techTags", ""))],
                })
        cv["projects"] = clean_projects
    
    related = cv.get("relatedTech", [])
    if isinstance(related, list):
        clean_related = []
        for r in related:
            if isinstance(r, dict):
                items = r.get("items", [])
                if isinstance(items, list):
                    clean_related.append({
                        "category": str(r.get("category", "")),
                        "items": [str(i).strip() for i in items if i],
                    })
        cv["relatedTech"] = clean_related
    
    techs = cv.get("technologies", {})
    if isinstance(techs, dict):
        cv["technologies"] = {
            "mustHave": [str(t).strip() for t in (techs.get("mustHave") or []) if t],
            "niceToHave": [str(t).strip() for t in (techs.get("niceToHave") or []) if t],
            "additional": [str(t).strip() for t in (techs.get("additional") or []) if t],
        }
    
    return cv

def final_polish(cv: dict, years_exp: str = "") -> dict:
    """Final polishing - deduplicates tech tags and ensures proper experience display"""
    
    # Use the exact years_exp from UI, don't recalculate
    if years_exp:
        real_years = years_exp.strip()
    else:
        real_years = _calc_total_years(years_exp)
    
    cv["totalYears"] = real_years
    
    # Update title to include experience with + sign
    title = cv.get("title", "")
    if title and real_years:
        # Check if title already has experience
        if "|" in title:
            parts = title.split("|")
            # Remove any existing experience from last part
            last_part = parts[-1].strip()
            if "+" in last_part or any(c.isdigit() for c in last_part):
                parts.pop()
            # Add experience as new last part
            title = " | ".join(parts + [real_years + "+" if real_years.isdigit() else real_years])
        else:
            title = f"{title} | {real_years}+" if real_years.isdigit() else f"{title} | {real_years}"
        cv["title"] = title
    
    # Update summary to use the same years with + sign
    summary = cv.get("summary", "")
    if summary and real_years:
        # Replace any existing year mention with the exact format from UI
        years_display = f"{real_years}+" if real_years.isdigit() and "+" not in real_years else real_years
        summary = re.sub(r'\b\d+\+?\s*years?\b', f"{years_display} years", summary, count=1)
        # Also handle the first sentence that might start with the year
        if summary.lower().startswith(("with", "over")):
            summary = re.sub(r'(with|over)\s+\d+\+?\s*years?', f"\\1 {years_display} years", summary, count=1, flags=re.IGNORECASE)
        cv["summary"] = summary
    
    # Deduplicate tech tags across companies
    companies = cv.get("companies", [])
    used_techs = set()
    
    for i, co in enumerate(companies):
        tech_str = co.get("tech", "")
        if tech_str:
            techs = [t.strip() for t in tech_str.split("|") if t.strip()]
            unique_techs = []
            for t in techs:
                if t.lower() not in used_techs:
                    unique_techs.append(t)
                    used_techs.add(t.lower())
            if len(unique_techs) < 4 and len(techs) >= 4:
                for t in techs:
                    if len(unique_techs) >= 6:
                        break
                    if t.lower() not in used_techs:
                        unique_techs.append(t)
                        used_techs.add(t.lower())
            if unique_techs:
                co["tech"] = " | ".join(unique_techs[:8])
    
    # Deduplicate project tech tags
    projects = cv.get("projects", [])
    for proj in projects:
        tech_tags = proj.get("techTags", [])
        if tech_tags and isinstance(tech_tags, list):
            seen = set()
            unique = []
            for t in tech_tags:
                if t and t.lower() not in seen:
                    unique.append(t)
                    seen.add(t.lower())
            proj["techTags"] = unique[:7]
    
    return cv

def fix_companies(cv: dict) -> dict:
    """Fix company names and role titles - dynamic, no hardcoded techs"""
    companies = cv.get("companies", [])
    real_years_str = cv.get("totalYears", _calc_total_years())
    try:
        real_years = float(real_years_str.replace("+", "").strip())
    except Exception:
        real_years = 3.0
    
    num_cos = len(companies)
    if num_cos == 1:
        tier_labels = ["Junior"]
    elif num_cos == 2:
        tier_labels = ["", "Junior"]
    else:
        tier_labels = ["Senior", "", "Junior"]
    
    for i, co in enumerate(companies):
        if i < len(CANDIDATE_COMPANIES):
            real = CANDIDATE_COMPANIES[i]
            name = co.get("company", "")
            if not name or name.lower() in ("placeholder", "example", "company"):
                co["company"] = real["name"]
                co["dateRange"] = f"{real['start']} - {real['end']}"
        co.setdefault("company", CANDIDATE_COMPANIES[i]["name"] if i < len(CANDIDATE_COMPANIES) else f"Company {i+1}")
        
        tier = tier_labels[min(i, len(tier_labels) - 1)]
        role = co.get("role", "")
        role = re.sub(r'Co\d+\s*', '', role).strip()
        if tier and not role.lower().startswith(tier.lower()):
            co["role"] = f"{tier} {role}".strip()
        else:
            co["role"] = role
    
    return cv

# ==============================================================================
# UNIVERSAL LLM CALLER
# ==============================================================================
async def call_llm_atomic(client, key: str, model: str, url: str,
                          system: str, user: str, stage: str,
                          headers: dict, max_tokens: int = 4000,
                          _deadline: float = 0.0) -> dict:
    import time as _t
    
    if url == CEREBRAS_URL:
        per_call_timeout = 90
    elif url == GROQ_URL:
        per_call_timeout = 90
    else:
        per_call_timeout = 90
    
    if _deadline and _t.time() >= _deadline:
        raise ValueError(f"Stage {stage} skipped — deadline exceeded")
    
    for attempt in range(2):
        try:
            r = await client.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
                timeout=per_call_timeout,
            )
        except httpx.TimeoutException:
            raise ValueError(f"Stage {stage} timed out after {per_call_timeout}s")
        except Exception as e:
            raise ValueError(f"Stage {stage} failed: {str(e)}")
        
        if r.status_code == 200:
            raw = r.json()["choices"][0]["message"]["content"]
            return extract_json(raw)
        elif r.status_code == 429:
            wait = min(int(r.headers.get("retry-after", 30)), 20)
            if attempt == 0 and wait > 0:
                await asyncio.sleep(wait)
                continue
            raise ValueError(f"Rate limited on {stage}")
        elif r.status_code in (401, 403):
            raise ValueError(f"Invalid key on {stage}")
        else:
            raise ValueError(f"HTTP {r.status_code} on {stage}")
    
    raise ValueError(f"Rate limited on {stage}")

# ==============================================================================
# PROVIDER CALLERS
# ==============================================================================
async def generate_cv_dynamic(req: CVRequest, client, key: str, model: str, 
                               url: str, headers: dict) -> dict:
    """Generate CV using single dynamic prompt - everything from JD"""
    import time as _t
    
    _deadline = _t.time() + 270
    years_exp = (req.years_exp or "").strip()
    total_years = _calc_total_years(years_exp)
    companies_list = _build_dynamic_companies(years_exp)
    edu = _build_education_year(years_exp)
    
    system_prompt, user_prompt = build_dynamic_prompt(req)
    
    result = await call_llm_atomic(client, key, model, url, system_prompt, user_prompt,
                                    "FullCV", headers, max_tokens=4500, _deadline=_deadline)
    
    if not result:
        raise ValueError("AI returned empty response")
    
    if "companies" not in result:
        result["companies"] = []
    
    for i, co in enumerate(result.get("companies", [])):
        if i < len(companies_list):
            co["company"] = companies_list[i]["name"]
            co["dateRange"] = f"{companies_list[i]['start']} - {companies_list[i]['end']}"
    
    if "projects" not in result:
        result["projects"] = []
    
    cv = {
        "totalYears": total_years,
        "title": result.get("title", req.job_title),
        "summary": result.get("summary", ""),
        "skills": result.get("skills", []),
        "competencies": result.get("competencies", ""),
        "keywords": result.get("keywords", ""),
        "technologies": result.get("technologies", {"mustHave": [], "niceToHave": [], "additional": []}),
        "companies": result.get("companies", []),
        "projects": result.get("projects", []),
        "relatedTech": result.get("relatedTech", []),
        "education": result.get("education", {
            "university": "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
            "degree": "Bachelor of Science in Computer Science (BSCS)",
            "cgpa": "3.97/4.0",
            "years": f"{edu['start']} - {edu['end']}",
            "achievement": "Gold Medalist for Academic Excellence",
        }),
    }
    
    cv_sanitised = sanitise_cv(cv)
    cv_companies = fix_companies(cv_sanitised)
    cv_polished = final_polish(cv_companies, years_exp=years_exp)
    
    return cv_polished

async def call_cerebras(req: CVRequest) -> tuple:
    raw_keys = req.cerebras_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Cerebras keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid Cerebras keys found.")
    
    model = req.model or "llama3.1-8b"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=180, write=15, pool=10)) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            
            try:
                probe = await client.post(
                    CEREBRAS_URL,
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=15,
                )
                if probe.status_code in (401, 403):
                    errors_by_key.append(f"Key {i+1} ({mk}): invalid key")
                    continue
                if probe.status_code == 429:
                    errors_by_key.append(f"Key {i+1} ({mk}): rate limited")
                    continue
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): probe failed - {str(e)[:50]}")
                continue
            
            try:
                cv = await generate_cv_dynamic(req, client, key, model, CEREBRAS_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                return cv, mk, i
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue
    
    raise HTTPException(502, f"All Cerebras keys failed: {'; '.join(errors_by_key[:3])}")

async def call_groq(req: CVRequest) -> tuple:
    raw_keys = req.groq_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Groq keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip().startswith("gsk_")]
    if not valid_keys:
        raise HTTPException(400, "No valid Groq keys (must start with gsk_).")
    
    model = req.model or "llama-3.1-8b-instant"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=180, write=15, pool=10)) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            
            try:
                cv = await generate_cv_dynamic(req, client, key, model, GROQ_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                return cv, mk, i
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue
    
    raise HTTPException(502, f"All Groq keys failed: {'; '.join(errors_by_key[:3])}")

async def call_gemini(req: CVRequest) -> tuple:
    raw_keys = req.gemini_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Gemini keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid Gemini keys found.")
    
    model = req.model or "gemini-2.0-flash"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    
    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            
            try:
                sys_p, usr_p = build_dynamic_prompt(req)
                url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
                payload = {
                    "systemInstruction": {"parts": [{"text": sys_p}]},
                    "contents": [{"role": "user", "parts": [{"text": usr_p}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4000},
                }
                r = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
                
                if r.status_code == 200:
                    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    result = extract_json(raw)
                    
                    years_exp = (req.years_exp or "").strip()
                    total_years = _calc_total_years(years_exp)
                    companies_list = _build_dynamic_companies(years_exp)
                    edu = _build_education_year(years_exp)
                    
                    for j, co in enumerate(result.get("companies", [])):
                        if j < len(companies_list):
                            co["company"] = companies_list[j]["name"]
                            co["dateRange"] = f"{companies_list[j]['start']} - {companies_list[j]['end']}"
                    
                    cv = {
                        "totalYears": total_years,
                        "title": result.get("title", req.job_title),
                        "summary": result.get("summary", ""),
                        "skills": result.get("skills", []),
                        "competencies": result.get("competencies", ""),
                        "keywords": result.get("keywords", ""),
                        "technologies": result.get("technologies", {}),
                        "companies": result.get("companies", []),
                        "projects": result.get("projects", []),
                        "relatedTech": result.get("relatedTech", []),
                        "education": result.get("education", {
                            "university": "QURTUBA UNIVERSITY",
                            "degree": "BSCS",
                            "cgpa": "3.97/4.0",
                            "years": f"{edu['start']} - {edu['end']}",
                        }),
                    }
                    
                    cv_sanitised = sanitise_cv(cv)
                    cv_companies = fix_companies(cv_sanitised)
                    cv_polished = final_polish(cv_companies, years_exp=years_exp)
                    
                    _key_usage[mk] = _key_usage.get(mk, 0) + 1
                    return cv_polished, mk, i
                else:
                    errors_by_key.append(f"Key {i+1} ({mk}): HTTP {r.status_code}")
                    continue
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue
    
    raise HTTPException(502, f"All Gemini keys failed: {'; '.join(errors_by_key[:3])}")

async def call_deepseek(req: CVRequest) -> tuple:
    raw_keys = req.deepseek_keys or []
    if not raw_keys:
        raise HTTPException(400, "No DeepSeek keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid DeepSeek keys found.")
    
    model = req.model or "deepseek-chat"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    
    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            
            try:
                cv = await generate_cv_dynamic(req, client, key, model, DEEPSEEK_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                return cv, mk, i
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue
    
    raise HTTPException(502, f"All DeepSeek keys failed: {'; '.join(errors_by_key[:3])}")

async def call_openai(req: CVRequest) -> tuple:
    raw_keys = req.openai_keys or []
    if not raw_keys:
        raise HTTPException(400, "No OpenAI keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid OpenAI keys found.")
    
    model = req.model or "gpt-4o-mini"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    
    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            
            try:
                cv = await generate_cv_dynamic(req, client, key, model, OPENAI_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                return cv, mk, i
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue
    
    raise HTTPException(502, f"All OpenAI keys failed: {'; '.join(errors_by_key[:3])}")

# ==============================================================================
# HEALTH CHECK
# ==============================================================================
@app.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {"status": "ok", "ollama": "ok" if ollama_ok else "unreachable"}

# ==============================================================================
# MAIN GENERATION ENDPOINT
# ==============================================================================
@app.post("/generate-cv")
async def generate_cv(req: CVRequest):
    try:
        if req.provider == "cerebras":
            cv_data, key_used, key_idx = await call_cerebras(req)
        elif req.provider == "groq":
            cv_data, key_used, key_idx = await call_groq(req)
        elif req.provider == "gemini":
            cv_data, key_used, key_idx = await call_gemini(req)
        elif req.provider == "deepseek":
            cv_data, key_used, key_idx = await call_deepseek(req)
        elif req.provider == "openai":
            cv_data, key_used, key_idx = await call_openai(req)
        else:
            raise HTTPException(400, f"Unsupported provider: {req.provider}")
        
        return {
            "cv": cv_data,
            "provider": req.provider,
            "model": req.model,
            "key_used": key_used,
            "key_index": key_idx,
        }
    except asyncio.TimeoutError:
        raise HTTPException(504, "CV generation timed out (> 5 minutes). Try again or switch provider.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ==============================================================================
# PDF GENERATION
# ==============================================================================
class PDFRequest(BaseModel):
    cv: dict
    filename: str = "CV.pdf"
    profileData: Optional[dict] = None

def build_cv_pdf(cv: dict, profile_data: dict = None) -> bytes:
    """Build PDF from CV JSON - preserves all dynamic content with green medal"""
    from reportlab.platypus import KeepTogether
    
    _pd = profile_data or {}
    p_name = (_pd.get("name") or "").strip() or "CANDIDATE"
    p_links = _pd.get("links") or []
    p_work = _pd.get("work") or []
    p_edu = _pd.get("edu") or []
    
    buf = io.BytesIO()
    PAGE_W, _ = A4
    ML, MR, MT, MB = 13 * mm, 13 * mm, 11 * mm, 11 * mm
    PAGE_H_SINGLE = 841.89 * 2.2
    
    doc = SimpleDocTemplate(
        buf, pagesize=(PAGE_W, PAGE_H_SINGLE),
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title=f"{p_name} CV", author=p_name,
    )
    TW = PAGE_W - ML - MR
    
    def ps(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0, spaceBefore=0, textColor=colors.HexColor("#111111"))
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)
    
    S = {
        "name": ps("name", fontName="Helvetica-Bold", fontSize=18, leading=24, alignment=TA_CENTER),
        "role": ps("role", fontName="Helvetica", fontSize=8, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
        "contact": ps("contact", fontName="Helvetica", fontSize=8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#0057A8")),
        "sec_title": ps("sec", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#222222"), spaceBefore=4, spaceAfter=2),
        "company": ps("co", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#111111")),
        "role_title": ps("rt", fontName="Helvetica-Oblique", fontSize=10, leading=13, textColor=colors.HexColor("#555555")),
        "bullet": ps("bul", fontName="Helvetica", fontSize=9.5, leading=13, leftIndent=12, spaceAfter=2),
        "tech_line": ps("tech", fontName="Helvetica", fontSize=8.5, leading=11, leftIndent=12, textColor=colors.HexColor("#666666")),
        "skill_items": ps("sitm", fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#333333")),
        "proj_name": ps("pn", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=colors.HexColor("#111111")),
        "proj_body": ps("pb", fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#333333")),
        "proj_bullet": ps("pbul", fontName="Helvetica", fontSize=9.5, leading=12.5, leftIndent=12, spaceAfter=2),
        "proj_stack": ps("pst", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor("#555555")),
        "competency": ps("comp", fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#333333")),
        "edu_uni": ps("uni", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#111111")),
        "edu_deg": ps("deg", fontName="Helvetica", fontSize=10, leading=13, textColor=colors.HexColor("#444444")),
        "edu_medal": ps("med", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=colors.HexColor("#166534")),  # Green color for medal
    }
    
    def HR():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=3, spaceBefore=1)
    
    story = []
    
    # Header
    story.append(Paragraph(p_name.upper(), S["name"]))
    title = cv.get("title", "")
    if title:
        story.append(Paragraph(title.upper(), S["role"]))
    story.append(HR())
    
    # Contact strip
    if p_links:
        contact_items = []
        for l in p_links:
            val = l.get("value", "")
            if val:
                contact_items.append(val)
        if contact_items:
            story.append(Paragraph("  |  ".join(contact_items[:4]), S["contact"]))
    story.append(HR())
    
    # Summary
    summary = cv.get("summary", "")
    if summary:
        story.append(Paragraph("PROFESSIONAL SUMMARY", S["sec_title"]))
        story.append(Paragraph(summary, S["bullet"]))
        story.append(Spacer(1, 3 * mm))
    
    # Experience
    companies = cv.get("companies", [])
    if companies:
        story.append(Paragraph("WORK EXPERIENCE", S["sec_title"]))
        for co in companies:
            company = co.get("company", "")
            role = co.get("role", "")
            date_range = co.get("dateRange", "")
            bullets = co.get("bullets", [])
            tech = co.get("tech", "")
            
            row = [[Paragraph(company.upper(), S["company"]), Paragraph(date_range, ps("dr", fontName="Helvetica", fontSize=10, alignment=TA_RIGHT, textColor=colors.HexColor("#666666")))]]
            t = Table(row, colWidths=[TW * 0.65, TW * 0.35])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
            story.append(t)
            if role:
                story.append(Paragraph(role, S["role_title"]))
            for b in bullets[:4]:
                story.append(Paragraph(f"\u2022 {b}", S["bullet"]))
            if tech:
                story.append(Paragraph(f"Technologies: {tech}", S["tech_line"]))
            story.append(Spacer(1, 4 * mm))
    
    # Skills
    skills = cv.get("skills", [])
    if skills:
        story.append(Paragraph("TECHNICAL SKILLS", S["sec_title"]))
        for s in skills[:5]:
            story.append(Paragraph(s, S["skill_items"]))
            story.append(Spacer(1, 2 * mm))
    
    # Projects
    projects = cv.get("projects", [])
    if projects:
        story.append(Paragraph("KEY PROJECTS", S["sec_title"]))
        for p in projects[:4]:
            name = p.get("name", "")
            overview = p.get("overview", "")
            bullets = p.get("bullets", [])
            tech_tags = p.get("techTags", [])
            
            if name:
                story.append(Paragraph(name, S["proj_name"]))
            if overview:
                story.append(Paragraph(overview, S["proj_body"]))
            for b in bullets[:3]:
                story.append(Paragraph(f"\u2022 {b}", S["proj_bullet"]))
            if tech_tags:
                story.append(Paragraph(f"Stack: {', '.join(tech_tags[:6])}", S["proj_stack"]))
            story.append(Spacer(1, 4 * mm))
    
    # Competencies
    competencies = cv.get("competencies", "")
    if competencies:
        story.append(Paragraph("KEY COMPETENCIES", S["sec_title"]))
        comp_display = competencies.replace(" * ", ", ").replace("* ", ", ").replace(" *", ", ")
        story.append(Paragraph(comp_display, S["competency"]))
        story.append(Spacer(1, 2 * mm))
    
    # Education with GREEN Gold Medalist
    story.append(Paragraph("EDUCATION", S["sec_title"]))
    edu = cv.get("education", {})
    uni = edu.get("university", "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY")
    degree = edu.get("degree", "Bachelor of Science in Computer Science (BSCS)")
    years = edu.get("years", "")
    achievement = edu.get("achievement", "")
    cgpa = edu.get("cgpa", "3.97/4.0")
    
    story.append(Paragraph(uni.upper(), S["edu_uni"]))
    deg_text = f"{degree} | {years}"
    if cgpa:
        deg_text += f" | CGPA: {cgpa}"
    story.append(Paragraph(deg_text, S["edu_deg"]))
    
    # Gold Medalist in GREEN with medal emoji
    if achievement and "gold" in achievement.lower():
        story.append(Paragraph(f"🏅 {achievement}", S["edu_medal"]))
    elif achievement:
        story.append(Paragraph(f"✓ {achievement}", S["edu_deg"]))
    
    # Build PDF
    doc.build(story)
    
    # Crop to content
    last_y = doc.frame._y
    tight_h = (PAGE_H_SINGLE - MT) - last_y + MT + MB + 4 * mm
    tight_h = max(tight_h, 100 * mm)
    crop_bottom = PAGE_H_SINGLE - tight_h
    
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "--quiet"])
        from pypdf import PdfReader, PdfWriter
    
    buf.seek(0)
    reader = PdfReader(buf)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    page.mediabox.lower_left = (0, crop_bottom)
    page.mediabox.upper_right = (PAGE_W, PAGE_H_SINGLE)
    
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()

@app.post("/generate-pdf")
async def generate_pdf(req: PDFRequest):
    try:
        pdf_bytes = build_cv_pdf(req.cv, profile_data=req.profileData)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{req.filename}"'}
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

# ==============================================================================
# KEY CHECK ENDPOINTS (simplified)
# ==============================================================================
@app.post("/check-cerebras-keys")
async def check_cerebras_keys(body: dict):
    keys = body.get("keys", [])
    model = body.get("model", "llama3.1-8b")
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"})
                continue
            try:
                r = await client.post(CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
                status = "ok" if r.status_code == 200 else "rate_limited" if r.status_code == 429 else "invalid"
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

@app.post("/check-groq-keys")
async def check_groq_keys(body: dict):
    keys = body.get("keys", [])
    model = body.get("model", "llama-3.1-8b-instant")
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        for key in keys:
            key = (key or "").strip()
            if not key or not key.startswith("gsk_"):
                results.append({"key": mask(key), "status": "invalid_format"})
                continue
            try:
                r = await client.post(GROQ_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
                status = "ok" if r.status_code == 200 else "rate_limited" if r.status_code == 429 else "invalid"
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

@app.post("/check-gemini-keys")
async def check_gemini_keys(body: dict):
    keys = body.get("keys", [])
    model = body.get("model", "gemini-2.0-flash")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"})
                continue
            try:
                url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
                r = await client.post(url, headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}})
                status = "ok" if r.status_code == 200 else "rate_limited" if r.status_code == 429 else "invalid"
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

@app.post("/check-deepseek-keys")
async def check_deepseek_keys(body: dict):
    keys = body.get("keys", [])
    model = body.get("model", "deepseek-chat")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"})
                continue
            try:
                r = await client.post(DEEPSEEK_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
                status = "ok" if r.status_code == 200 else "rate_limited" if r.status_code == 429 else "invalid"
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

@app.post("/check-openai-keys")
async def check_openai_keys(body: dict):
    keys = body.get("keys", [])
    model = body.get("model", "gpt-4o-mini")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"})
                continue
            try:
                r = await client.post(OPENAI_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
                status = "ok" if r.status_code == 200 else "rate_limited" if r.status_code == 429 else "invalid"
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)