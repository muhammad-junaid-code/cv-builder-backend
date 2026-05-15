"""
CV Builder AI - FastAPI Backend v14
Port: 8001  |  Start: uvicorn main:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import httpx, json, re, math, io, asyncio, secrets
from datetime import date, datetime, timedelta

# reportlab - PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

app = FastAPI(title="CV Builder AI", version="14.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Base company data - only names and default dates
CANDIDATE_COMPANIES = [
    {"name": "MULTYLOGICS SOLUTIONS", "start": "May 2024", "end": "Present"},
    {"name": "ENCS NETWORKS", "start": "May 2022", "end": "May 2024"},
    {"name": "NOW TECHNOLOGIES (NOW.NET.PK)", "start": "May 2020", "end": "May 2022"},
]

_MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"]
_MONTH_MAP = {m.lower(): i+1 for i, m in enumerate(_MONTH_NAMES)}
_MONTH_MAP.update({m.lower()[:3]: i+1 for i, m in enumerate(_MONTH_NAMES)})

def _month_name(n: int) -> str:
    return _MONTH_NAMES[(n - 1) % 12]

def _parse_month_year(s: str) -> date:
    s = s.strip()
    if not s or s.lower() == "present":
        return date.today()
    parts = s.split()
    if len(parts) != 2:
        return date.today()
    try:
        year = int(parts[1])
        if year < 2000 or year > 2030:
            year = date.today().year
        month = _MONTH_MAP.get(parts[0].lower(), 1)
        month = max(1, min(12, month))
        return date(year, month, 1)
    except:
        return date.today()

def _months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))

def _calc_total_years(years_exp: str = "") -> str:
    if years_exp:
        try:
            clean = years_exp.strip().replace("+", "")
            n = float(clean)
            if n == int(n):
                return f"{int(n)}+"
            return f"{round(n, 1)}+"
        except:
            pass
    total_months = 0
    for co in CANDIDATE_COMPANIES:
        try:
            start = _parse_month_year(co["start"])
            end = _parse_month_year(co["end"])
            total_months += _months_between(start, end)
        except:
            pass
    years = total_months / 12
    if years >= 5: return "5+"
    elif years >= 4: return "4+"
    elif years >= 3: return "3+"
    elif years >= 2: return "2+"
    else: return "1+"

def _build_dynamic_companies(years_exp: str, profile_work: list = None) -> list:
    if profile_work and len(profile_work) > 0:
        result = []
        for i, w in enumerate(profile_work[:3]):
            company = w.get("company", "")
            from_date = w.get("from", "")
            to_date = w.get("to", "Present")
            if company:
                result.append({
                    "name": company.upper(),
                    "start": from_date if from_date else "May 2024",
                    "end": to_date if to_date else "Present"
                })
        if result:
            return result
    
    if not years_exp:
        return CANDIDATE_COMPANIES[:3]
    try:
        n = float(years_exp.strip().replace("+", ""))
    except:
        return CANDIDATE_COMPANIES[:3]
    
    total_months = int(round(n * 12))
    today = date.today()
    
    def fmt(d: date) -> str:
        return f"{_month_name(d.month)} {d.year}"
    
    if n <= 1.4:
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
        years_to_subtract = span // 12
        months_to_subtract = span % 12
        
        new_year = cursor.year - years_to_subtract
        new_month = cursor.month - months_to_subtract
        
        while new_month < 1:
            new_month += 12
            new_year -= 1
        
        new_month = max(1, min(12, new_month))
        if new_year < 2000:
            new_year = 2000
        
        co_start = date(new_year, new_month, 1)
        co_end = "Present" if i == 0 else fmt(cursor)
        name = CANDIDATE_COMPANIES[i]["name"] if i < len(CANDIDATE_COMPANIES) else f"Company {i+1}"
        result.append({"name": name, "start": fmt(co_start), "end": co_end})
        cursor = co_start
    
    return result

def _build_education_year(years_exp: str, profile_edu: list = None) -> dict:
    if profile_edu and len(profile_edu) > 0:
        e = profile_edu[0]
        p_from = str(e.get("from", "") or "").strip()
        p_to = str(e.get("to", "") or "").strip()
        if p_from and p_to:
            return {"start": p_from, "end": p_to}
        if p_to and not p_from:
            try:
                end_yr = int(p_to[:4])
                start_yr = max(2000, end_yr - 4)
                return {"start": str(start_yr), "end": p_to}
            except:
                pass
        if p_from and not p_to:
            try:
                start_yr = int(p_from[:4])
                end_yr = min(date.today().year, start_yr + 4)
                return {"start": p_from, "end": str(end_yr)}
            except:
                pass
    
    if years_exp:
        try:
            n = int(float(years_exp.strip().replace("+", "")))
            end_year = max(2015, date.today().year - max(n - 1, 0))
            start_year = max(2010, end_year - 4)
            return {"start": str(start_year), "end": str(end_year)}
        except:
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
        except:
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
        except:
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
# Helper Functions
# ==============================================================================
def mask(key: str) -> str:
    k = (key or "").strip()
    return k[:8] + "..." + k[-4:] if len(k) > 12 else "***"

def extract_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object in model response")
    end = raw.rfind("}")
    if end == -1:
        j = raw[start:] + "}"
    else:
        j = raw[start:end + 1]
    j = re.sub(r",\s*([}\]])", r"\1", j)
    try:
        return json.loads(j)
    except:
        j = re.sub(r'[\x00-\x1f\x7f]', ' ', j)
        j = re.sub(r',\s*([}\]])', r'\1', j)
        return json.loads(j)

# ==============================================================================
# SIMPLIFIED PROMPT - SHORT AND TO THE POINT
# ==============================================================================
def build_prompt(req: CVRequest) -> tuple:
    jd = req.job_description.strip()
    job_title = req.job_title.strip()
    years_exp = (req.years_exp or "").strip()
    
    if years_exp:
        years_display = f"{years_exp}+" if years_exp.isdigit() and "+" not in years_exp else years_exp
    else:
        years_display = _calc_total_years(years_exp)
        if years_display.isdigit():
            years_display = f"{years_display}+"
    
    profile_work = req.profile_data.get("work", []) if req.profile_data else []
    companies = _build_dynamic_companies(years_exp, profile_work)
    edu = _build_education_year(years_exp, req.profile_data.get("edu", []) if req.profile_data else [])
    
    co_lines = "\n".join([f'"{c["name"]}" ({c["start"]} - {c["end"]})' for i, c in enumerate(companies)])
    
    company_info = ""
    if req.company_context and req.company_name:
        company_info = f"\nTarget Company: {req.company_name}\nContext: {req.company_context[:500]}\n"
    elif req.company_name:
        company_info = f"\nTarget Company: {req.company_name}\n"
    
    system = f"""Generate a CV JSON based on this job description.

Job Title: {job_title}
Experience: {years_display} years
Companies: {co_lines}
Education Years: {edu['start']} - {edu['end']}
{company_info}

CRITICAL RULES:
1. Title format: "Position Name | Technology1,Technology2,Technology3 | {years_display}+"
   - Only THREE technologies, separated by commas (no spaces after commas)
2. Summary: 6-7 sentences starting with "{years_display}+ years of experience in..."
3. ALL technologies must come from the job description only
4. Each company's tech tags must be different

Output valid JSON only. Do not add any text before or after the JSON."""

    user = f"""Job Description:
{jd[:3000]}

Generate CV JSON now. Remember: title has only THREE technologies separated by commas (no spaces)."""
    return system, user

# ==============================================================================
# SANITIZE FUNCTIONS
# ==============================================================================
def sanitize_cv(cv: dict) -> dict:
    if not isinstance(cv, dict):
        return {}
    
    for field in ["totalYears", "title", "summary", "competencies", "keywords"]:
        cv[field] = str(cv.get(field, "")).strip()
    
    companies = cv.get("companies", [])
    if isinstance(companies, list):
        clean = []
        for co in companies[:3]:
            if isinstance(co, dict):
                clean.append({
                    "company": str(co.get("company", "")),
                    "role": str(co.get("role", "")),
                    "dateRange": str(co.get("dateRange", "")),
                    "bullets": [str(b).strip() for b in (co.get("bullets") or [])[:4] if b],
                    "tech": str(co.get("tech", "")),
                })
        cv["companies"] = clean
    
    projects = cv.get("projects", [])
    if isinstance(projects, list):
        clean = []
        for p in projects[:4]:
            if isinstance(p, dict):
                tags = p.get("techTags", [])
                if isinstance(tags, list):
                    tags = [str(t).strip() for t in tags[:7] if t]
                clean.append({
                    "name": str(p.get("name", "")),
                    "overview": str(p.get("overview", "")),
                    "bullets": [str(b).strip() for b in (p.get("bullets") or [])[:3] if b],
                    "techTags": tags,
                })
        cv["projects"] = clean
    
    skills = cv.get("skills", [])
    if isinstance(skills, list):
        cv["skills"] = [str(s).strip() for s in skills[:5] if s]
    
    return cv

def final_polish(cv: dict, years_exp: str = "") -> dict:
    if years_exp:
        real_years = years_exp.strip()
    else:
        real_years = _calc_total_years(years_exp)
    
    cv["totalYears"] = real_years
    years_display = f"{real_years}+" if real_years.isdigit() and "+" not in real_years else real_years
    
    # Fix title - ensure only 3 technologies
    title = cv.get("title", "")
    if title and "|" in title:
        parts = title.split("|")
        if len(parts) >= 2:
            tech_part = parts[1].strip()
            tech_list = [t.strip() for t in tech_part.split(",")]
            tech_list = tech_list[:3]
            tech_part = ",".join(tech_list)
            
            if len(parts) >= 3:
                cv["title"] = f"{parts[0].strip()} | {tech_part} | {years_display}"
            else:
                cv["title"] = f"{parts[0].strip()} | {tech_part} | {years_display}"
    
    # Fix summary double ++
    summary = cv.get("summary", "")
    if summary:
        summary = summary.replace("++", "+")
        cv["summary"] = summary
    
    return cv

# ==============================================================================
# UNIVERSAL LLM CALLER
# ==============================================================================
async def call_llm(client, key: str, model: str, url: str, system: str, user: str, headers: dict) -> dict:
    for attempt in range(2):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system[:4000]},
                    {"role": "user", "content": user[:3000]}
                ],
                "temperature": 0.2,
                "max_tokens": 3500,
            }
            
            r = await client.post(url, headers=headers, json=payload, timeout=90)
            
            if r.status_code == 200:
                raw = r.json()["choices"][0]["message"]["content"]
                return extract_json(raw)
            elif r.status_code == 429:
                if attempt == 0:
                    await asyncio.sleep(5)
                    continue
                raise ValueError(f"Rate limited after retry")
            else:
                error_text = r.text[:200]
                raise ValueError(f"HTTP {r.status_code}: {error_text}")
        except httpx.TimeoutException:
            if attempt == 0:
                continue
            raise ValueError("Request timed out")
    
    raise ValueError("Max retries exceeded")

# ==============================================================================
# GROQ CALLER
# ==============================================================================
async def call_groq(req: CVRequest) -> tuple:
    keys = [k.strip() for k in (req.groq_keys or []) if k and k.strip().startswith("gsk_")]
    if not keys:
        raise HTTPException(400, "No valid Groq keys provided")
    
    model = req.model or "llama-3.1-8b-instant"
    system, user = build_prompt(req)
    
    async with httpx.AsyncClient(timeout=120) as client:
        for i, key in enumerate(keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                result = await call_llm(client, key, model, GROQ_URL, system, user, headers)
                
                years_exp = (req.years_exp or "").strip()
                total_years = _calc_total_years(years_exp)
                profile_work = req.profile_data.get("work", []) if req.profile_data else []
                companies_list = _build_dynamic_companies(years_exp, profile_work)
                edu = _build_education_year(years_exp, req.profile_data.get("edu", []) if req.profile_data else [])
                
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
                
                cv = sanitize_cv(cv)
                cv = final_polish(cv, years_exp=years_exp)
                return cv, mk, i
                
            except Exception as e:
                print(f"Groq key {i+1} ({mk}) failed: {str(e)}")
                continue
    
    raise HTTPException(502, "All Groq keys failed. Try switching to Cerebras or Gemini.")

# ==============================================================================
# CEREBRAS CALLER
# ==============================================================================
async def call_cerebras(req: CVRequest) -> tuple:
    keys = [k.strip() for k in (req.cerebras_keys or []) if k and k.strip()]
    if not keys:
        raise HTTPException(400, "No Cerebras keys provided")
    
    model = req.model or "llama3.1-8b"
    system, user = build_prompt(req)
    
    async with httpx.AsyncClient(timeout=120) as client:
        for i, key in enumerate(keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                result = await call_llm(client, key, model, CEREBRAS_URL, system, user, headers)
                
                years_exp = (req.years_exp or "").strip()
                total_years = _calc_total_years(years_exp)
                profile_work = req.profile_data.get("work", []) if req.profile_data else []
                companies_list = _build_dynamic_companies(years_exp, profile_work)
                edu = _build_education_year(years_exp, req.profile_data.get("edu", []) if req.profile_data else [])
                
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
                
                cv = sanitize_cv(cv)
                cv = final_polish(cv, years_exp=years_exp)
                return cv, mk, i
                
            except Exception as e:
                print(f"Cerebras key {i+1} ({mk}) failed: {str(e)}")
                continue
    
    raise HTTPException(502, "All Cerebras keys failed. Try switching to Groq or Gemini.")

# ==============================================================================
# GEMINI CALLER
# ==============================================================================
async def call_gemini(req: CVRequest) -> tuple:
    keys = [k.strip() for k in (req.gemini_keys or []) if k and k.strip()]
    if not keys:
        raise HTTPException(400, "No Gemini keys provided")
    
    model = req.model or "gemini-2.0-flash"
    system, user = build_prompt(req)
    
    async with httpx.AsyncClient(timeout=120) as client:
        for i, key in enumerate(keys):
            mk = mask(key)
            try:
                url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
                payload = {
                    "systemInstruction": {"parts": [{"text": system[:3000]}]},
                    "contents": [{"role": "user", "parts": [{"text": user[:3000]}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 3500},
                }
                r = await client.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
                
                if r.status_code == 200:
                    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    result = extract_json(raw)
                    
                    years_exp = (req.years_exp or "").strip()
                    total_years = _calc_total_years(years_exp)
                    profile_work = req.profile_data.get("work", []) if req.profile_data else []
                    companies_list = _build_dynamic_companies(years_exp, profile_work)
                    edu = _build_education_year(years_exp, req.profile_data.get("edu", []) if req.profile_data else [])
                    
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
                    
                    cv = sanitize_cv(cv)
                    cv = final_polish(cv, years_exp=years_exp)
                    return cv, mk, i
                else:
                    print(f"Gemini key {i+1} ({mk}) failed: HTTP {r.status_code}")
                    continue
                    
            except Exception as e:
                print(f"Gemini key {i+1} ({mk}) failed: {str(e)}")
                continue
    
    raise HTTPException(502, "All Gemini keys failed. Try switching to Groq or Cerebras.")

# ==============================================================================
# HEALTH CHECK
# ==============================================================================
@app.get("/health")
async def health():
    return {"status": "ok"}

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
        else:
            raise HTTPException(400, f"Unsupported provider: {req.provider}")
        
        return {
            "cv": cv_data,
            "provider": req.provider,
            "model": req.model,
            "key_used": key_used,
            "key_index": key_idx,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ==============================================================================
# PDF GENERATION - SINGLE PAGE, NO PAGE BREAK
# ==============================================================================
class PDFRequest(BaseModel):
    cv: dict
    filename: str = "CV.pdf"
    profileData: Optional[dict] = None

@app.post("/generate-pdf")
async def generate_pdf(req: PDFRequest):
    try:
        _pd = req.profileData or {}
        p_name = (_pd.get("name") or "").strip() or "CANDIDATE"
        p_links = _pd.get("links") or []
        
        companies_from_cv = req.cv.get("companies", [])
        
        buf = io.BytesIO()
        PAGE_W, PAGE_H = A4
        ML, MR, MT, MB = 13 * mm, 13 * mm, 15 * mm, 10 * mm  # Reduced bottom margin
        
        # Build document with exact page size - prevent page breaks
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
            title=f"{p_name} CV", author=p_name,
        )
        TW = PAGE_W - ML - MR
        
        def ps(name, **kw):
            defaults = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#111111"))
            defaults.update(kw)
            return ParagraphStyle(name, **defaults)
        
        S = {
            "name": ps("name", fontName="Helvetica-Bold", fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=8),  # Added space after name
            "role": ps("role", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=4),
            "contact": ps("contact", fontSize=8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#0057A8")),
            "sec_title": ps("sec", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=4, spaceAfter=3),
            "company": ps("co", fontName="Helvetica-Bold", fontSize=11, leading=14),
            "role_title": ps("rt", fontName="Helvetica-Oblique", fontSize=10, leading=13, textColor=colors.HexColor("#555555"), spaceAfter=2),
            "bullet": ps("bul", fontSize=9.5, leading=13, leftIndent=12, spaceAfter=2),
            "tech_line": ps("tech", fontSize=8.5, leading=11, leftIndent=12, textColor=colors.HexColor("#666666"), spaceAfter=3),
            "skill": ps("skill", fontSize=9, leading=12, spaceAfter=2),
            "proj_name": ps("pn", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceAfter=2),
            "proj_body": ps("pb", fontSize=9.5, leading=13, spaceAfter=2),
            "proj_bullet": ps("pbul", fontSize=9.5, leading=12.5, leftIndent=12, spaceAfter=2),
            "proj_stack": ps("pst", fontName="Helvetica-Bold", fontSize=8.5, leading=11, spaceAfter=4),
            "competency": ps("comp", fontSize=9.5, leading=13, spaceAfter=3),
            "edu_uni": ps("uni", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceAfter=2),
            "edu_deg": ps("deg", fontSize=10, leading=13, textColor=colors.HexColor("#444444"), spaceAfter=2),
            "edu_medal": ps("med", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=colors.HexColor("#166534"), spaceAfter=0),  # No space after medal
        }
        
        def HR():
            return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=3, spaceBefore=1)
        
        story = []
        
        # Header - with proper spacing
        story.append(Paragraph(p_name.upper(), S["name"]))
        title = req.cv.get("title", "")
        if title:
            story.append(Paragraph(title.upper(), S["role"]))
        story.append(Spacer(1, 2 * mm))
        story.append(HR())
        
        # Contact
        if p_links:
            contacts = [l.get("value", "") for l in p_links[:4] if l.get("value")]
            if contacts:
                story.append(Spacer(1, 1 * mm))
                story.append(Paragraph("  |  ".join(contacts), S["contact"]))
        story.append(Spacer(1, 1 * mm))
        story.append(HR())
        
        # Summary
        summary = req.cv.get("summary", "")
        if summary:
            story.append(Paragraph("PROFESSIONAL SUMMARY", S["sec_title"]))
            story.append(Paragraph(summary, S["bullet"]))
            story.append(Spacer(1, 2 * mm))
        
        # Experience
        if companies_from_cv:
            story.append(Paragraph("WORK EXPERIENCE", S["sec_title"]))
            for idx, co in enumerate(companies_from_cv):
                company = co.get("company", "")
                role = co.get("role", "")
                date_range = co.get("dateRange", "")
                bullets = co.get("bullets", [])
                tech = co.get("tech", "")
                
                header = Table([[Paragraph(company.upper(), S["company"]), Paragraph(date_range, ps("dr", fontSize=10, alignment=TA_RIGHT, textColor=colors.HexColor("#666666")))]], colWidths=[TW * 0.65, TW * 0.35])
                header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
                story.append(header)
                if role:
                    story.append(Paragraph(role, S["role_title"]))
                for b in bullets[:4]:
                    if b:
                        story.append(Paragraph(f"\u2022 {b}", S["bullet"]))
                if tech:
                    story.append(Paragraph(f"Technologies: {tech}", S["tech_line"]))
                if idx < len(companies_from_cv) - 1:
                    story.append(Spacer(1, 2 * mm))
        
        # Skills
        skills = req.cv.get("skills", [])
        if skills:
            story.append(Paragraph("TECHNICAL SKILLS", S["sec_title"]))
            for s in skills[:5]:
                if s:
                    story.append(Paragraph(s, S["skill"]))
        
        # Projects - limit to keep on one page
        projects = req.cv.get("projects", [])
        if projects:
            story.append(Paragraph("KEY PROJECTS", S["sec_title"]))
            for idx, p in enumerate(projects[:3]):  # Only show 3 projects to save space
                name = p.get("name", "")
                overview = p.get("overview", "")
                bullets = p.get("bullets", [])
                tags = p.get("techTags", [])
                
                if name:
                    story.append(Paragraph(name, S["proj_name"]))
                if overview:
                    # Shorten overview to save space
                    if len(overview) > 250:
                        overview = overview[:250] + "..."
                    story.append(Paragraph(overview, S["proj_body"]))
                for b in bullets[:2]:  # Only 2 bullets per project
                    if b:
                        if len(b) > 150:
                            b = b[:150] + "..."
                        story.append(Paragraph(f"\u2022 {b}", S["proj_bullet"]))
                if tags and isinstance(tags, list):
                    clean = [str(t) for t in tags[:4] if t]
                    if clean:
                        story.append(Paragraph(f"Stack: {', '.join(clean)}", S["proj_stack"]))
                if idx < len(projects[:3]) - 1:
                    story.append(Spacer(1, 1 * mm))
        
        # Competencies
        competencies = req.cv.get("competencies", "")
        if competencies:
            story.append(Paragraph("KEY COMPETENCIES", S["sec_title"]))
            comp_display = competencies.replace(" * ", ", ").replace("* ", ", ").replace(" *", ", ")
            # Limit length
            if len(comp_display) > 200:
                comp_display = comp_display[:200] + "..."
            story.append(Paragraph(comp_display, S["competency"]))
        
        # Education - LAST SECTION (no extra space after)
        story.append(Paragraph("EDUCATION", S["sec_title"]))
        edu = req.cv.get("education", {})
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
        if achievement and "gold" in achievement.lower():
            story.append(Paragraph(f"🏅 {achievement}", S["edu_medal"]))
        
        # Build PDF - NO page break allowed
        doc.build(story, onLaterPages=lambda *args: None)  # Prevent new pages
        
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{req.filename}"'})
        
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

# ==============================================================================
# KEY CHECK ENDPOINTS
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
            except:
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
            except:
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
            except:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)