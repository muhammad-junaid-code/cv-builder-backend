"""
CV Builder AI - FastAPI Backend v9
Port: 8001  |  Start: uvicorn main:app --host 0.0.0.0 --port 8001

Providers: Groq, Cerebras (cloud.cerebras.ai), Ollama
Key strategy: round-robin across all keys, least-used first.
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

app = FastAPI(title="CV Builder AI", version="9.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models"
OLLAMA_URL   = "http://localhost:11434"

# -- Real candidate constants --------------------------------------------------
CANDIDATE_COMPANIES = [
    {"name": "MULTYLOGICS SOLUTIONS",        "start": "May 2024",  "end": "Present"},
    {"name": "ENCS NETWORKS",                "start": "May 2022",  "end": "May 2024"},
    {"name": "NOW TECHNOLOGIES (NOW.NET.PK)","start": "May 2020",  "end": "May 2022"},
]


# -- Month name map & date helpers --------------------------------------------
_MONTH_NAMES = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
_MONTH_MAP = {m.lower(): i+1 for i, m in enumerate(_MONTH_NAMES)}
_MONTH_MAP.update({m.lower()[:3]: i+1 for i, m in enumerate(_MONTH_NAMES)})

def _month_name(n: int) -> str:
    return _MONTH_NAMES[(n - 1) % 12]

def _parse_month_year(s: str) -> date:
    """Parse 'Month YYYY' or 'Present' -> date."""
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
    """Add months to a date, clamping to valid day."""
    total = d.month - 1 + months
    year  = d.year + total // 12
    month = total % 12 + 1
    return date(year, month, 1)

def _subtract_months(d: date, months: int) -> date:
    return _add_months(d, -months)


def _years_label(total_months: float) -> str:
    y = total_months / 12
    if y >= 5:   return "5+"
    elif y >= 4: return "4+"
    elif y >= 3: return "3+"
    elif y >= 2: return "2+"
    else:        return "1+"


def _calc_total_years(years_exp: str = "") -> str:
    """
    If years_exp is provided by the UI (e.g. "5"), return it exactly as-is (e.g. "5").
    Otherwise calculate from CANDIDATE_COMPANIES date ranges and return a label like "2+".
    """
    if years_exp:
        try:
            n = float(years_exp.strip().replace("+", ""))
            # Return exact integer if whole number, else one decimal
            return str(int(n)) if n == int(n) else str(round(n, 1))
        except ValueError:
            pass

    # Fallback: sum actual company durations
    total_months = 0
    for co in CANDIDATE_COMPANIES:
        try:
            start = _parse_month_year(co["start"])
            end   = _parse_month_year(co["end"])
            total_months += _months_between(start, end)
        except Exception:
            pass
    return _years_label(total_months)


def _build_dynamic_companies(years_exp: str, num_companies: int = 0) -> list:
    """
    Return N company date ranges (N = num_companies if >0, else inferred from years_exp).
    Experience is divided equally across all companies, working backwards from today.
    Remainder months go to the most recent company.
    Rules:
      - If num_companies given by profile, always honour it (cap at 3 for AI bullet slots).
      - Otherwise infer: <=1.4yr->1, <=2.4yr->2, 3+yr->3.
    """
    if not years_exp:
        n_fallback = num_companies if num_companies > 0 else 3
        return CANDIDATE_COMPANIES[:n_fallback]

    try:
        n = float(years_exp.strip().replace("+", ""))
    except ValueError:
        n_fallback = num_companies if num_companies > 0 else 3
        return CANDIDATE_COMPANIES[:n_fallback]

    total_months = int(round(n * 12))
    today        = date.today()

    def fmt(d: date) -> str:
        return f"{_month_name(d.month)} {d.year}"

    # Determine number of companies
    if num_companies > 0:
        num_cos = num_companies
    elif n <= 1.4:
        num_cos = 1
    elif n <= 2.4:
        num_cos = 2
    else:
        num_cos = 3

    # Divide experience equally, remainder to most recent
    each      = total_months // num_cos if num_cos > 0 else total_months
    remainder = total_months - each * num_cos

    result = []
    cursor = today
    for i in range(num_cos):
        span    = each + (remainder if i == 0 else 0)
        co_start = _subtract_months(cursor, span)
        co_end   = "Present" if i == 0 else fmt(cursor)
        name     = CANDIDATE_COMPANIES[i]["name"] if i < len(CANDIDATE_COMPANIES) else f"Company {i+1}"
        result.append({"name": name, "start": fmt(co_start), "end": co_end})
        cursor = co_start

    return result


def _build_education_year(years_exp: str, profile_edu: list = None) -> dict:
    """
    Priority:
      1. Profile edu entry with both from+to  → use as-is (static)
      2. Profile edu entry with only one date → fill the gap automatically
      3. years_exp provided                   → calculate: end = today - (exp-1), start = end-4
      4. No info                              → keep static 2017-2021

    Formula: graduation year = current_year - (years_exp - 1)
    e.g. 5 years exp → graduated current_year - 4  → studied current_year-8 to current_year-4
    """
    today = date.today()

    # 1 & 2: profile edu entries
    if profile_edu:
        e = profile_edu[0]  # use first edu entry for auto-fill reference
        p_from = str(e.get("from", "") or "").strip()
        p_to   = str(e.get("to",   "") or "").strip()
        if p_from and p_to:
            return {"start": p_from, "end": p_to}   # fully static from profile
        if p_from or p_to:
            # One date provided — fill the other using years_exp or 4-year default
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

    # 3: calculate from years_exp
    if years_exp:
        try:
            n = int(float(years_exp.strip().replace("+", "")))
            end_year   = today.year - max(n - 1, 0)
            start_year = end_year - 4
            return {"start": str(start_year), "end": str(end_year)}
        except ValueError:
            pass

    # 4: fallback static
    return {"start": "2017", "end": "2021"}


# -- Token budgets -------------------------------------------------------------
MODEL_PROMPT_BUDGET = {
    "llama-3.1-8b-instant":          2400,
    "llama3-8b-8192":                2400,
    "gemma2-9b-it":                  2400,
    "llama-3.3-70b-versatile":       8000,
    "deepseek-r1-distill-llama-70b": 8000,
    "mixtral-8x7b-32768":           28000,
    "llama3.1-8b":                             7000,
    "gpt-oss-120b":                            8000,
    "zai-glm-4.7":                             8000,
    "qwen-3-235b-a22b-instruct-2507":          8000,
    "gemini-2.0-flash":              8000,
    "gemini-2.0-flash-lite":         8000,
    "gemini-1.5-flash":              8000,
}
DEFAULT_PROMPT_BUDGET = 2400

# -- Model name mapping - converts shorthand to full provider-specific names ---
MODEL_ALIASES = {
    # Groq model aliases
    "deepseek-r1":            "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill":    "deepseek-r1-distill-llama-70b",
    "llama-3.3":              "llama-3.3-70b-versatile",
    "llama-3.3-70b":          "llama-3.3-70b-versatile",
    "llama3.3-70b":           "llama-3.3-70b-versatile",
    "llama-3.1":              "llama-3.1-8b-instant",
    "llama3.1":               "llama-3.1-8b-instant",
    "mixtral":                "mixtral-8x7b-32768",
    "gemma2":                 "gemma2-9b-it",
    # Map common friendly names to the strong Groq model
    "best":                   "llama-3.3-70b-versatile",
}

def _normalize_model_name(model: str, provider: str = "groq") -> str:
    """Convert shorthand model names to full provider-specific names."""
    if not model:
        return model
    
    m = model.strip().lower()
    
    # Check aliases first
    if m in MODEL_ALIASES:
        return MODEL_ALIASES[m]
    
    # Check if it starts with any alias key
    for alias_key, full_name in MODEL_ALIASES.items():
        if m.startswith(alias_key.lower()):
            return full_name
    
    # Return original if no mapping found
    return model

# -- In-memory key usage counter -----------------------------------------------
_key_usage: dict = {}
_key_rate_limited_until: dict = {}   # mask(key) -> timestamp when safe to retry
_debug_log: list = []


class CVRequest(BaseModel):
    job_title:        str
    job_description:  str
    years_exp:        Optional[str]       = ""
    provider:         str                 = "cerebras"
    model:            str                 = "llama3.1-8b"
    groq_keys:        Optional[List[str]] = []
    cerebras_keys:    Optional[List[str]] = []
    deepseek_keys:    Optional[List[str]] = []
    openai_keys:      Optional[List[str]] = []
    gemini_keys:      Optional[List[str]] = []
    ollama_model:     Optional[str]       = "qwen2.5:7b"
    profile:          str                 = ""
    profile_data:     Optional[dict]      = None   # full structured profile from extension
    static_data:      Optional[dict]      = None
    company_name:     Optional[str]       = ""
    company_context:  Optional[str]       = ""


# ==============================================================================
# AUTH SYSTEM - live key management
# Keys are stored in auth_keys.json next to main.py
# ==============================================================================
import os

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))
AUTH_FILE  = os.path.join(_DATA_DIR, "auth_keys.json")

STATIC_ACCESS_KEY = "CVAI-A927-42F8-1E31"

def _load_auth_keys() -> dict:
    """Load auth keys from disk. Always ensures the static hardcoded key is present."""
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, "r") as f:
                keys = json.load(f)
        except Exception:
            keys = {}
    else:
        keys = {}

    # Always guarantee the static key exists and is active
    if STATIC_ACCESS_KEY not in keys:
        keys[STATIC_ACCESS_KEY] = {
            "label":      "Static Key",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": None,
            "active":     True,
            "last_used":  None,
        }
        _save_auth_keys(keys)
        print(f"\n{'='*60}")
        print(f"  CV Builder AI — Static Access Key")
        print(f"  Key: {STATIC_ACCESS_KEY}")
        print(f"{'='*60}\n")

    return keys

def _save_auth_keys(keys: dict):
    """Persist auth keys to disk."""
    with open(AUTH_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def _is_token_valid(token_data: dict) -> bool:
    """Check if a token is active and not expired."""
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
    label:      str  = "Access Key"
    days_valid: int  = 30   # 0 = never expires
    admin_pass: str  = ""

class LoginRequest(BaseModel):
    token: str

class VerifyRequest(BaseModel):
    token: str

class RevokeRequest(BaseModel):
    token:      str
    admin_pass: str = ""

# Simple admin password - change this or move to env var
ADMIN_PASSWORD = os.environ.get("CV_ADMIN_PASS", "admin1234")

@app.post("/auth/generate-key")
async def generate_key(req: GenerateKeyRequest):
    """Generate a new access key. Protected by admin password."""
    if req.admin_pass != ADMIN_PASSWORD:
        raise HTTPException(403, "Invalid admin password")

    keys = _load_auth_keys()

    # Generate a readable token: CVAI-XXXX-XXXX-XXXX
    raw = secrets.token_hex(8).upper()
    token = f"CVAI-{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"

    now = datetime.utcnow()
    expires_at = (now + timedelta(days=req.days_valid)).isoformat() if req.days_valid > 0 else None

    keys[token] = {
        "label":      req.label,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
        "active":     True,
        "last_used":  None,
    }
    _save_auth_keys(keys)

    return {
        "token":      token,
        "label":      req.label,
        "expires_at": expires_at,
        "message":    "Key generated successfully"
    }

@app.post("/auth/login")
async def login(req: LoginRequest):
    """Validate a token and return session info."""
    keys = _load_auth_keys()
    token = req.token.strip().upper()
    entry = keys.get(token)

    if not entry:
        raise HTTPException(401, "Invalid key - not found")
    if not _is_token_valid(entry):
        raise HTTPException(401, "Key is expired or revoked")

    # Update last_used
    entry["last_used"] = datetime.utcnow().isoformat()
    keys[token] = entry
    _save_auth_keys(keys)

    return {
        "ok":         True,
        "token":      token,
        "label":      entry.get("label", ""),
        "expires_at": entry.get("expires_at"),
        "message":    "Login successful"
    }

@app.post("/auth/verify")
async def verify_token(req: VerifyRequest):
    """Check if a stored token is still valid (used on extension open)."""
    keys = _load_auth_keys()
    token = req.token.strip().upper()
    entry = keys.get(token)

    if not entry or not _is_token_valid(entry):
        return {"valid": False}

    return {
        "valid":      True,
        "label":      entry.get("label", ""),
        "expires_at": entry.get("expires_at"),
    }

@app.post("/auth/revoke")
async def revoke_key(req: RevokeRequest):
    """Revoke a key. Protected by admin password."""
    if req.admin_pass != ADMIN_PASSWORD:
        raise HTTPException(403, "Invalid admin password")
    keys = _load_auth_keys()
    token = req.token.strip().upper()
    if token not in keys:
        raise HTTPException(404, "Key not found")
    keys[token]["active"] = False
    _save_auth_keys(keys)
    return {"ok": True, "message": "Key revoked"}

@app.get("/auth/list-keys")
async def list_keys(admin_pass: str = ""):
    """List all keys with their status. Protected by admin password."""
    if admin_pass != ADMIN_PASSWORD:
        raise HTTPException(403, "Invalid admin password")
    keys = _load_auth_keys()
    result = []
    for token, data in keys.items():
        result.append({
            "token":      token,
            "label":      data.get("label", ""),
            "created_at": data.get("created_at"),
            "expires_at": data.get("expires_at"),
            "last_used":  data.get("last_used"),
            "active":     data.get("active", True),
            "valid":      _is_token_valid(data),
        })
    return {"keys": result}


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


@app.post("/check-keys")
async def check_keys(body: dict):
    keys  = body.get("keys", [])
    model = _normalize_model_name(body.get("model", "llama-3.1-8b-instant"), "groq")
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        for key in keys:
            if not key or not key.strip().startswith("gsk_"):
                results.append({"key": mask(key), "status": "invalid_format"})
                continue
            try:
                r = await client.post(GROQ_URL,
                    headers={"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
                status = ("ok"           if r.status_code == 200 else
                          "rate_limited" if r.status_code == 429 else
                          "restricted"   if r.status_code == 403 else
                          "invalid_key"  if r.status_code == 401 else
                          f"error_{r.status_code}")
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}


@app.get("/cerebras-models")
async def cerebras_models(key: str = ""):
    if not key:
        return {"error": "Pass ?key=csk-... to check available models"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.cerebras.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"}
            )
            if r.status_code == 200:
                data   = r.json()
                models = [m.get("id") for m in data.get("data", [])]
                return {"models": models, "count": len(models)}
            return {"error": f"HTTP {r.status_code}", "body": r.text[:500]}
    except Exception as e:
        return {"error": str(e)}


@app.post("/check-cerebras-keys")
async def check_cerebras_keys(body: dict):
    keys  = body.get("keys", [])
    model = body.get("model", "llama3.1-8b")
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"}); continue
            try:
                r = await client.post(CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
                status = ("ok"           if r.status_code == 200 else
                          "rate_limited" if r.status_code == 429 else
                          "invalid_key"  if r.status_code in (401, 403) else
                          f"error_{r.status_code}")
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}





@app.post("/check-gemini-keys")
async def check_gemini_keys(body: dict):
    keys  = body.get("keys", [])
    model = body.get("model", "gemini-2.0-flash")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"}); continue
            try:
                url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
                r = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": "hi"}]}],
                          "generationConfig": {"maxOutputTokens": 1}}
                )
                status = ("ok"           if r.status_code == 200 else
                          "rate_limited" if r.status_code == 429 else
                          "invalid_key"  if r.status_code in (400, 401, 403) else
                          f"error_{r.status_code}")
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

@app.post("/check-deepseek-keys")
async def check_deepseek_keys(body: dict):
    keys  = body.get("keys", [])
    model = body.get("model", "deepseek-chat")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"}); continue
            try:
                r = await client.post(
                    DEEPSEEK_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
                )
                status = ("ok"           if r.status_code == 200 else
                          "rate_limited" if r.status_code == 429 else
                          "invalid_key"  if r.status_code in (401, 403) else
                          f"error_{r.status_code}")
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}


@app.post("/check-openai-keys")
async def check_openai_keys(body: dict):
    keys  = body.get("keys", [])
    model = body.get("model", "gpt-4o-mini")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"}); continue
            try:
                r = await client.post(
                    OPENAI_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
                )
                status = ("ok"           if r.status_code == 200 else
                          "rate_limited" if r.status_code == 429 else
                          "invalid_key"  if r.status_code in (401, 403) else
                          f"error_{r.status_code}")
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}


@app.get("/key-stats")
async def key_stats():
    return {
        "session_usage": _key_usage,
        "explanation": (
            "Each CV is generated by EXACTLY ONE key in a single API call. "
            "Keys are rotated so the least-used key goes first."
        )
    }


@app.get("/debug-log")
async def debug_log():
    return {"last_generations": _debug_log}


@app.get("/test-cerebras")
async def test_cerebras(key: str = "", model: str = "llama3.1-8b"):
    """
    Quick diagnostic endpoint - visit in browser:
      http://localhost:8001/test-cerebras?key=csk-YOUR_KEY&model=llama3.1-8b
    Returns exactly what Cerebras returns so you can see the real error.
    """
    if not key:
        return {
            "error": "Pass ?key=csk-... to test",
            "usage": "http://localhost:8001/test-cerebras?key=csk-YOUR_KEY&model=llama3.1-8b",
        }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
            )
            if r.status_code == 200:
                data = r.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "status": "ok",
                    "model": model,
                    "key_masked": mask(key),
                    "response": text,
                    "message": "Key and model are working correctly!",
                }
            else:
                return {
                    "status": f"HTTP {r.status_code}",
                    "key_masked": mask(key),
                    "body": r.text[:500],
                    "fix": (
                        "429 = rate limited (add more keys)" if r.status_code == 429 else
                        "401/403 = invalid key (get a new one at cloud.cerebras.ai)" if r.status_code in (401, 403) else
                        f"404 = model '{model}' not on your plan (try llama3.1-8b)" if r.status_code == 404 else
                        f"Unexpected {r.status_code} - see body above"
                    ),
                }
    except httpx.ConnectError as e:
        return {
            "status": "connection_error",
            "error": str(e),
            "fix": "Cannot reach api.cerebras.ai - check internet connection / VPN / firewall",
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}




def _log_generation(job_title: str, key_masked: str, key_index: int,
                    prompt_tokens: int, model: str, success: bool, error: str = ""):
    entry = {
        "job_title":     job_title,
        "key_used":      key_masked,
        "key_index":     key_index,
        "model":         model,
        "prompt_tokens": prompt_tokens,
        "success":       success,
        "error":         error,
    }
    _debug_log.insert(0, entry)
    if len(_debug_log) > 10:
        _debug_log.pop()


def mask(key: str) -> str:
    k = (key or "").strip()
    return k[:8] + "..." + k[-4:] if len(k) > 12 else "***"


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.8)


def _normalize_job_title(title: str) -> str:
    """
    Remove duplicated words/tokens in a job title.
    e.g. "Senior Senior Angular Developer" -> "Senior Angular Developer"
    e.g. "Developer Developer" -> "Developer"
    Preserves original casing of first occurrence.
    """
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





# -- Prompt builder -------------------------------------------------------------
def _co_line(companies: list) -> str:
    return " | ".join(
        f'"{c["name"]}" {c["start"]} - {c["end"]}' for c in companies
    )


def build_prompt(req: CVRequest, jd_chars: int = 1600) -> tuple:
    """Returns (system_prompt, user_prompt) tuple."""
    # -- CRITICAL: Normalize job title - remove duplicate words --------------
    req = req.copy(update={"job_title": _normalize_job_title(req.job_title.strip())})

    jd          = req.job_description.strip()[:jd_chars]
    #profile     = (req.profile.strip()[:150] if req.profile else "Full-stack developer, 3+ years")
    years_exp   = (req.years_exp or "").strip()
    total_years = _calc_total_years(years_exp)   # dynamic - uses frontend input if given

    # Profile-driven company count and dates
    _profile_work = []
    _profile_edu  = []
    if req.profile_data and isinstance(req.profile_data, dict):
        _profile_work = req.profile_data.get("work") or []
        _profile_edu  = req.profile_data.get("edu")  or []

    _num_profile_cos = min(len(_profile_work), 3)  # cap at 3 for AI bullet generation

    if _profile_work:
        # Build company slots from profile data; auto-fill missing dates via years_exp split
        _dynamic_fallback = _build_dynamic_companies(years_exp, num_companies=_num_profile_cos)
        companies = []
        for _i, _w in enumerate(_profile_work[:3]):
            _frm  = str(_w.get("from") or "").strip()
            _to   = str(_w.get("to")   or "").strip()
            _ds   = _frm  if _frm  else (_dynamic_fallback[_i]["start"] if _i < len(_dynamic_fallback) else "")
            _de   = _to   if _to   else ("Present" if _i == 0 else (_dynamic_fallback[_i]["end"] if _i < len(_dynamic_fallback) else ""))
            _name = str(_w.get("company") or "").strip()
            if not _name and _i < len(CANDIDATE_COMPANIES):
                _name = CANDIDATE_COMPANIES[_i]["name"]
            elif not _name:
                _name = f"Company {_i + 1}"
            companies.append({"name": _name, "start": _ds, "end": _de})
    else:
        companies = _build_dynamic_companies(years_exp)

    edu = _build_education_year(years_exp, profile_edu=_profile_edu)
    company_name    = (req.company_name    or "").strip()
    company_context = (req.company_context or "").strip()[:2000]

    try:
        _yrs_float = float(total_years)
    except ValueError:
        _yrs_float = 3.0

    if _yrs_float <= 1.4:
        _seniority_rule = (
            f"SENIORITY RULE - {total_years} year experience: "
            "Co1 MUST use 'Junior'. Domain = primary NAMED TECHNOLOGY from JD "
            "(e.g. 'Junior WordPress Developer' — NOT 'Junior DevOps Engineer').\n\n"
        )
        _seniority_ban = (
            "- Missing 'Junior' on Co1\n"
            "- Generic domain (DevOps/Web/Software/Backend/Frontend) in role title\n"
            "- Using 'Intern' or 'Internship' in any role title\n"
        )
    elif _yrs_float <= 2.4:
        _seniority_rule = (
            f"SENIORITY RULE - {total_years} years experience: "
            "Co1 NO prefix. Domain = PRIMARY named tech from JD (e.g. 'WordPress Developer'). "
            "Co2 MUST use 'Junior'. Domain = DIFFERENT named tech (e.g. 'Junior Webflow Specialist'). "
            "BANNED: DevOps, Web, Software, Backend, Frontend.\n\n"
        )
        _seniority_ban = (
            "- Missing 'Junior' on Co2\n"
            "- Seniority prefix on Co1\n"
            "- Generic domain (DevOps/Web/Software/Backend/Frontend)\n"
        )
    else:
        _seniority_rule = (
            f"SENIORITY RULE - {total_years} years experience: "
            "Co1 MUST use 'Senior'. Domain = PRIMARY named tech from JD "
            "(e.g. 'Senior WordPress Engineer' — NEVER 'Senior DevOps Engineer'). "
            "Co2 NO prefix. Domain = DIFFERENT named tech (e.g. 'Webflow Developer'). "
            "Co3 MUST use 'Junior'. Domain = THIRD named tech (e.g. 'Junior JavaScript Specialist'). "
            "BANNED: DevOps, Web, Software, Backend, Frontend, Full-Stack, Digital, IT.\n\n"
        )
        _seniority_ban = (
            "- Missing 'Senior' on Co1\n"
            "- Seniority prefix on Co2\n"
            "- Missing 'Junior' on Co3\n"
            "- Generic domain (DevOps/Web/Software/Backend/Frontend/Digital)\n"
            "- Invented adjective in title (Transformed/Innovative/Dynamic/Versatile)\n"
        )

    system = (
        "You are a professional CV writer with 15 years of experience placing candidates at enterprise companies. "
        "You write CVs that read as if a real, experienced professional wrote them — not an AI. "
        "Output ONLY valid JSON. No text before/after, no markdown, no backticks. Start { end }.\n\n"

        "=== CRITICAL PRIORITY RULE (HIGHEST PRIORITY) ===\n"
        "JD relevance ALWAYS overrides AI defaults or training assumptions.\n"
        "EVERY technology, skill, section, and project MUST be strictly derived from:\n"
        "  (1) The job title and job description provided\n"
        "  (2) The company name and company context provided (website data, web search results)\n"
        "  (3) Your knowledge of what this company does based on their public profile\n"
        "NOTHING is hardcoded. Nothing carries over from a previous generation. Each CV is 100%% unique.\n"
        "NEVER include a technology not in the JD or its standard ecosystem companions.\n"
        "NEVER generate a project unrelated to the company's actual industry domain.\n"
        "NEVER assign a system type prefix that doesn't match what the project actually does.\n"
        "For non-technical roles (PM, SEO, Marketing, Finance, Design): generate domain-appropriate\n"
        "  projects (campaigns, strategies, analytics programmes) — NOT software engineering projects.\n\n"

        "=== HUMANIZATION RULES (CRITICAL - apply everywhere) ===\n"
        "The CV must read like it was written by a real person, not generated by AI.\n"
        "BANNED everywhere in experience bullets, summaries, and project overviews:\n"
        "  X 'Highly motivated', 'Results-driven', 'Dynamic professional', 'Passionate about'\n"
        "  X 'Leveraged', 'Utilized', 'Revolutionary', 'Cutting-edge', 'Next-generation'\n"
        "  X 'AI-powered platform', 'Smart ecosystem', 'Innovative solution', 'Synergy'\n"
        "  X Any sentence starting with 'I' — all bullets are third-person implied\n"
        "  X Triple repetition of the same verb across bullets (e.g. Developed, Developed, Developed)\n"
        "REQUIRED for natural, human writing:\n"
        "  - Include realistic enterprise work: bug fixes, legacy maintenance, support tasks, internal dashboards\n"
        "  - Use varied sentence openers: 'Collaborated with', 'Maintained', 'Refactored', 'Assisted in', 'Participated in'\n"
        "  - Bullets should reflect what a real developer actually does day-to-day — not just launches and wins\n"
        "  - Metrics must be believable: avoid 99.99%%, 10x improvements — use 30%%, 2.4s, 18 clients, 400ms\n"
        "  - Not every bullet needs a metric — some can describe team collaboration or process contributions\n\n"

        "=== TECHNICAL vs NON-TECHNICAL ROLE DETECTION ===\n"
        "BEFORE generating anything, classify the role:\n"
        "  TECHNICAL: Software engineer, developer, DevOps, data engineer, QA, architect, ML engineer\n"
        "  NON-TECHNICAL: Product manager, SEO, digital marketing, content, finance, HR, operations, design\n"
        "  HYBRID: Business analyst, project manager, technical writer, scrum master, solutions consultant\n"
        "Rules by type:\n"
        "  TECHNICAL: Use 5 skill categories — names and groupings derived entirely from the JD's actual technology domains. Tech-heavy bullets. JD-driven projects.\n"
        "  NON-TECHNICAL: Replace skill buckets with domain-specific categories (e.g. 'SEO & Analytics', 'Content Strategy', "
        "'Campaign Management', 'Market Research Tools', 'Collaboration & PM Tools'). NO backend/infra injected. "
        "Bullets focus on strategy, stakeholders, KPIs, campaigns, business outcomes. Projects = initiatives, campaigns, or programmes.\n"
        "  HYBRID: Mix domain knowledge with relevant tooling. Avoid deep engineering stacks unless JD requires.\n\n"

        "=== STEP 0: JOB TITLE NORMALIZATION (CRITICAL - do this BEFORE anything else) ===\n"
        "Inspect the raw job title. Clean it:\n"
        "  1. Remove any duplicated words (e.g. 'Senior Senior Angular Developer' -> 'Senior Angular Developer').\n"
        "  2. Remove unnecessary filler prefixes/suffixes not in the JD (e.g. 'Required:' or '(Urgent)').\n"
        "  3. Never inflate - do NOT add seniority words not implied by the JD.\n"
        "  4. Keep it concise, accurate, and strictly aligned with the Job Description.\n"
        "Store this cleaned title internally. Use it as the baseline for all role title and 'title' field derivations.\n\n"

        "=== STEP 1: TECHNOLOGY EXTRACTION (do this before writing anything) ===\n"
        "Read the entire JD. Extract every named technology into:\n"
        "  CORE: tools/languages/frameworks in Requirements, Responsibilities, Must Have\n"
        "  PREFERRED: tools in Nice-to-Have, Preferred, Bonus sections\n"
        "  ECOSYSTEM: for each CORE and PREFERRED tool, add its standard companion tools\n"
        "    (e.g. if JD names a backend framework -> add its ORM, auth library, test framework, logging library)\n"
        "    (e.g. if JD names a cloud platform -> add the specific services mentioned + closely related managed services)\n"
        "    (e.g. if JD names a frontend framework -> add its state manager, router, UI lib, build tool, test util)\n"
        "    (e.g. if JD names a database -> add its migration tool, admin tool, ORM, query optimiser, connection pooler)\n"
        "    (e.g. if JD names a DevOps tool -> add its CI/CD companion, registry, orchestration or IaC tool)\n"
        "    (e.g. if JD names an SEO tool -> add its companion analytics, crawling, rank tracking, and reporting tools)\n"
        "CRITICAL - THIN JD WARNING:\n"
        "  Some JDs are written as plain English bullets (e.g. '- Good understanding of REST APIs').\n"
        "  In these cases, the words 'Good', 'Understanding', 'Hands', 'Ability', 'Strong', 'Web', 'Dev',\n"
        "  'Setup', 'Mindset', 'Remote', 'Problem-solving' are DESCRIPTION WORDS, NOT technologies.\n"
        "  NEVER extract plain English adjectives, nouns, or adverbs as technology names.\n"
        "  ONLY extract real named software tools, frameworks, languages, libraries, platforms, or services.\n"
        "  If the JD uses plain English without naming specific tools, infer the most appropriate\n"
        "  technology stack purely from the job title and the role domain described.\n"
        "  Do NOT use any predefined stacks. Generate skills from context alone.\n"
        "ABSOLUTE RULE: Every technology written ANYWHERE in the output MUST come from CORE, PREFERRED, or ECOSYSTEM.\n"
        "ABSOLUTE RULE: Technologies NOT extracted from the JD are BANNED - do not add technologies from your training data.\n"
        "ABSOLUTE RULE: The output changes completely for every different JD - nothing is hardcoded.\n"
        "\n"
        "=== ANTI-FAKE-TECH ENFORCEMENT (CRITICAL) ===\n"
        "BANNED from appearing ANYWHERE in the tech, skills, or technologies fields:\n"
        "  - Common English words (e.g. 'English', 'Resumes', 'Manager', 'Director', 'Applicants', 'Position', 'Overview', 'Key', 'Responsibilities')\n"
        "  - Job title words (e.g. 'Senior', 'Junior', 'Lead', 'Role', 'Specialist')\n"
        "  - Action verbs used as tools - this is the #1 failure mode. NEVER use any of these as skill items:\n"
        "    Write, Read, Test, Debug, Troubleshoot, Configure, Deploy, Design, Implement, Develop, Build,\n"
        "    Create, Maintain, Monitor, Scale, Optimize, Architect, Operate, Manage, Execute, Conduct,\n"
        "    Determine, Provide, Support, Ensure, Deliver, Run, Use, Apply, Review, Audit, Document,\n"
        "    Report, Track, Analyze, Integrate, Migrate, Secure, Automate, Orchestrate, Coordinate\n"
        "  - Generic infrastructure nouns that are NOT product names:\n"
        "    Requirements, Architecture, Infrastructure, Environment, System, Server, Network, Platform,\n"
        "    Application, Solution, Service, Module, Component, Interface, Integration\n"
        "  - Partial product names: 'NET' alone is BANNED - always write the full name: 'ASP.NET Core'\n"
        "  - AI assistant / IDE tools are NEVER skill items: Claude Code, Cursor, Copilot, GitHub Copilot, Anthropic, ChatGPT, OpenAI\n"
        "    These are productivity tools, not technical skills a recruiter evaluates.\n"
        "  - Any word that is NOT the name of a real software tool, platform, API, framework, language, or service\n"
        "RULE: If a word could appear in a sentence as a common English verb or noun, it is NOT a technology. Reject it.\n"
        "RULE: Technologies MUST be real named products: e.g. 'SQL Server', 'ASP.NET Core', 'Entity Framework', 'React', 'PostgreSQL'\n"
        "RULE: Every tech tag for every company MUST be a real named tool from the JD domain only.\n\n"

        "=== COMPANY TECH TAG DIVERSITY (CRITICAL — do this AFTER extracting all JD technologies) ===\n"
        "Each company MUST show a MEANINGFULLY DIFFERENT technology mix from the other companies.\n"
        "This signals career progression and breadth — not repetitive single-stack experience.\n"
        "HOW TO ASSIGN TECH TAGS per company:\n"
        "  Co1 (most recent / senior role): Core JD primary stack + 1-2 cloud/DevOps tools + 1 testing/quality tool\n"
        "  Co2 (mid-level role): Related secondary stack tools + 1-2 different JD ecosystem tools + different DB/ORM\n"
        "  Co3 (oldest / junior role): Foundational tools + different framework or library + different tooling angle\n"
        "HARD RULES for company tech diversity:\n"
        "  - No single technology name should appear in ALL 3 companies' tech tags (it's OK to share 1-2 across any two)\n"
        "  - Each company must introduce at least 2-3 technologies NOT used by any other company\n"
        "  - Technologies must reflect realistic career evolution: junior → broader ownership → senior specialisation\n"
        "  - NEVER give all companies identical or near-identical tech tag lists\n"
        "  - Co3 (junior) should lean more foundational/simpler tools (e.g. jQuery, MySQL, basic MVC, vanilla JS)\n"
        "  - Co1 (senior) should include more advanced/enterprise tools (e.g. microservices, cloud-native, advanced ORM)\n"
        "EACH COMPANY'S BULLETS must also mention technologies specific to THAT company's tech tags.\n"
        "  - Do NOT write a Co1 bullet referencing a technology that only appears in Co3's tech tags.\n"
        "  - Technology mentioned in a bullet must be traceable to that company's tech tag list.\n\n"

        "=== STEP 1.5: CLOUD & HOSTING INFERENCE (mandatory - execute right after STEP 1) ===\n"
        "After extracting JD technologies, determine the most relevant cloud/hosting platform for this role.\n"
        "CLOUD RULE: If the JD names a cloud provider, include only the specific services mentioned\n"
        "or directly implied by the role context. If no cloud is mentioned but the role is\n"
        "cloud-adjacent, infer the most commonly used provider for that stack based on the JD alone.\n"
        "Do not inject cloud services that are not implied by the JD.\n"
        "If the role is non-technical or cloud is not relevant, omit cloud entirely.\n"
        "  - JD names multiple providers -> include ALL of them; merge into one 'Cloud & Hosting' category\n"
        "  - NEVER default to a fixed provider - always derive from the actual JD content\n"
        "MANDATORY PLACEMENT of inferred cloud services:\n"
        "  1. One dedicated skill category named after the platform (e.g. 'Azure Cloud Services',\n"
        "     'AWS Infrastructure', 'GCP & Cloud DevOps') - never use a generic label like 'Cloud Tools'\n"
        "  2. At least 2 work-experience bullets referencing deployment, hosting, scaling, or cloud-native ops\n"
        "  3. Tech tags for at least one company MUST include 1-2 cloud platform services\n"
        "  4. At least one project's techTags/stack MUST reference the inferred cloud platform\n"
        "HARD RULE: The cloud category MUST contain exactly 11-13 items - expand with the platform's\n"
        "  CI/CD pipelines, IaC tools, monitoring/alerting, secrets management, and serverless services.\n\n"

        "=== STEP 2: SKILLS PIPELINE — EXECUTE AS A STRICT ALGORITHM ===\n"
        "You MUST execute every sub-step below IN ORDER before writing a single skill item.\n"
        "This is not a guideline — it is a deterministic pipeline. Each step gates the next.\n\n"

        "── SUB-STEP 2.1: DETECT PRIMARY ECOSYSTEM ──────────────────────────────────\n"
        "Read the Job Title and Job Description. Identify ONE primary ecosystem from this list:\n"
        "  DOTNET   → signals: C#, .NET, ASP.NET, MVC, Blazor, MAUI, Entity Framework, NuGet\n"
        "  ANGULAR  → signals: Angular, NgRx, RxJS, TypeScript (with Angular), Angular Material\n"
        "  REACT    → signals: React, React Native, Redux, Next.js, React Query (without Angular)\n"
        "  NODE     → signals: Node.js, Express.js, NestJS, npm ecosystem (without React/Angular)\n"
        "  JAVA     → signals: Java, Spring Boot, Maven, Gradle, Hibernate, JPA\n"
        "  PHP      → signals: PHP, Laravel, Symfony, Composer, Blade, Eloquent\n"
        "  PYTHON   → signals: Python, Django, FastAPI, Flask, SQLAlchemy, Pandas\n"
        "  DEVOPS   → signals: Kubernetes, Terraform, Helm, ArgoCD, Ansible, Jenkins\n"
        "  DATA     → signals: Spark, Airflow, dbt, BigQuery, Snowflake, Databricks\n"
        "  MOBILE   → signals: Flutter, Dart, Swift, Kotlin, Xcode, Android Studio\n"
        "  FULLSTACK→ signals: JD explicitly requires BOTH a frontend AND a backend framework by name\n"
        "Write your detection result as: DETECTED_ECOSYSTEM = <name>\n"
        "If multiple ecosystems appear, list them ALL. FULLSTACK applies only when JD names BOTH sides.\n\n"

        "── SUB-STEP 2.2: BUILD THE ALLOWED-TECH WHITELIST ─────────────────────────\n"
        "Create a WHITELIST of every technology permitted in this CV's skills section.\n"
        "Sources (in priority order):\n"
        "  SOURCE-A: Every named technology found in the JD Requirements / Must Have section\n"
        "  SOURCE-B: Every named technology found in Nice-to-Have / Preferred / Bonus sections\n"
        "  SOURCE-C: Official ecosystem companions of SOURCE-A and SOURCE-B items ONLY —\n"
        "            companions must be standard, well-known tools used together in that ecosystem.\n"
        "            A companion of a .NET tool must itself be a .NET ecosystem tool.\n"
        "            A companion of an Angular tool must itself be an Angular ecosystem tool.\n"
        "            Cross-ecosystem companions are BANNED.\n\n"
        "COMPANION RULE: For every named technology found in the JD, infer its standard ecosystem\n"
        "companions (ORMs, test frameworks, build tools, auth libraries, state managers, etc.) from\n"
        "your knowledge of that technology's official ecosystem. Do NOT use a predefined list.\n"
        "Only add companions that are strongly associated with the specific version or variant\n"
        "mentioned in the JD. Cross-ecosystem companions are BANNED.\n"
        "CRITICAL: This companion rule is EXHAUSTIVE. If a companion is not strongly associated\n"
        "with the source technology's ecosystem, do NOT add it. Particularly:\n"
        "  .NET / C# role → NEVER add React, Vue.js, Next.js, Node.js, Express.js, Django,\n"
        "                    Flask, MongoDB, RabbitMQ, gRPC as companions. They are NOT .NET companions.\n"
        "  Angular role   → NEVER add React, Vue.js, Next.js, Redux as companions.\n"
        "  React role     → NEVER add Angular, NgRx, RxJS as companions.\n\n"

        "── SUB-STEP 2.3: ASSIGN EVERY WHITELISTED TECH TO EXACTLY ONE DOMAIN BUCKET ─\n"
        "For every technology in the WHITELIST, assign it to exactly ONE bucket below.\n"
        "Use these permanent, non-negotiable domain definitions:\n\n"
        "  BUCKET: Backend & API Development\n"
        "    Belongs here: server-side frameworks, API paradigms, middleware, ORMs, message brokers\n"
        "    Examples: ASP.NET Core, C#, .NET 8, Entity Framework Core, Dapper, MediatR, AutoMapper,\n"
        "              SignalR, gRPC, REST APIs, GraphQL, RabbitMQ, Kafka, Hangfire, Polly, NestJS,\n"
        "              Express.js, Django, FastAPI, Flask, Spring Boot, Laravel\n"
        "    NEVER put here: React, Vue.js, Angular, CSS frameworks, CI/CD tools, databases, cloud services\n\n"
        "  BUCKET: Frontend & UI\n"
        "    Belongs here: browser-side frameworks, UI libraries, CSS tools, build tools, state managers\n"
        "    Examples: Angular, React, Vue.js, Next.js, TypeScript (when used for UI), Tailwind CSS,\n"
        "              SCSS, Webpack, Vite, RxJS, NgRx, Redux, Angular Material, Storybook\n"
        "    NEVER put here: Node.js, Express.js, any server framework, any database, any CI/CD tool\n\n"
        "  BUCKET: Database & Storage\n"
        "    Belongs here: relational DBs, NoSQL, caching, search engines, migration tools\n"
        "    Examples: SQL Server, PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, DynamoDB,\n"
        "              Cosmos DB, Entity Framework Core (migrations), Flyway, Liquibase, T-SQL,\n"
        "              Azure SQL Database, Azure Cache for Redis, PgBouncer\n"
        "    NEVER put here: any framework, any cloud compute service, any CI/CD tool\n\n"
        "  BUCKET: Cloud & Infrastructure  (name after actual platform: 'Azure Cloud Services', 'AWS Infrastructure', etc.)\n"
        "    Belongs here: cloud platform services, hosting, compute, storage, serverless, managed services\n"
        "    Examples: Azure App Service, Azure Functions, Azure Blob Storage, Azure Service Bus,\n"
        "              Azure AD, Azure Key Vault, Azure Monitor, Azure CDN, Azure Container Registry,\n"
        "              AWS EC2, AWS S3, AWS Lambda, AWS RDS, GCP Cloud Run, Firebase\n"
        "    NEVER put here: Docker/Kubernetes (those go in DevOps), databases, backend frameworks\n\n"
        "  BUCKET: DevOps & CI/CD\n"
        "    Belongs here: containerisation, orchestration, pipelines, IaC, registries, version control\n"
        "    Examples: Docker, Kubernetes, Helm, Azure DevOps, GitHub Actions, Jenkins, Terraform,\n"
        "              ArgoCD, Git, GitHub, GitLab CI, Ansible, Dockerfile\n"
        "    NEVER put here: React, Vue.js, Angular, any frontend tool, any database, any cloud service\n\n"
        "  BUCKET: Testing & Quality\n"
        "    Belongs here: test frameworks, mocking, quality gates, code analysis\n"
        "    Examples: xUnit, NUnit, MSTest, Moq, Jest, Jasmine, Karma, Pytest, JUnit, SonarQube,\n"
        "              Selenium, Cypress, Playwright, Postman, Swagger, OpenAPI\n"
        "    NEVER put here: backend frameworks, databases, or cloud tools\n\n"
        "  BUCKET: Security & Auth\n"
        "    Belongs here: authentication libraries, authorisation protocols, secrets management, scanning\n"
        "    Examples: JWT, OAuth 2.0, Azure AD, IdentityServer, OpenID Connect, OWASP ZAP,\n"
        "              Azure Key Vault, AWS Secrets Manager, HashiCorp Vault, ASP.NET Core Identity\n"
        "    NEVER put here: Node.js, Django, Express.js, or any general-purpose framework\n\n"
        "  BUCKET: Monitoring & Observability\n"
        "    Belongs here: logging, metrics, tracing, alerting, dashboards\n"
        "    Examples: Azure Monitor, Application Insights, Prometheus, Grafana, Datadog,\n"
        "              Serilog, NLog, ELK Stack, Jaeger, OpenTelemetry, PagerDuty, Seq\n"
        "    NEVER put here: databases, backend frameworks, or cloud compute services\n\n"
        "ASSIGNMENT RULE: Each technology goes into the bucket where it PRIMARILY FUNCTIONS.\n"
        "JWT → Security (not Backend). Redis → Database (not Cloud). Docker → DevOps (not Cloud).\n"
        "If a technology serves multiple roles, pick the one most emphasised in the JD.\n\n"

        "── SUB-STEP 2.4: SELECT 5 BUCKETS AND NAME THEM ───────────────────────────\n"
        "From the populated buckets above, select exactly 5 that have the most JD-relevant items.\n"
        "Name each selected bucket using the ACTUAL technology domain of its contents.\n"
        "  CORRECT: 'ASP.NET Core & Backend APIs', '.NET Backend & API Development',\n"
        "           'Azure Cloud Services', 'Database & Storage', 'DevOps & CI/CD',\n"
        "           'Testing & Quality Assurance', 'Frontend & UI' (only if JD has frontend)\n"
        "  WRONG:   'Cloud & Infrastructure' containing React\n"
        "           'Backend & API' containing Vue.js or Tailwind CSS or Webpack\n"
        "           'Scripting & Version Control' containing only GitHub and C# (too sparse)\n"
        "           'Security & Compliance' containing Node.js or Django\n"
        "Rename a bucket only if the JD's technology makes a more specific name accurate.\n\n"

        "── SUB-STEP 2.5: POPULATE EACH BUCKET WITH 10+ ITEMS ──────────────────────\n"
        "Fill each selected bucket with at least 10 technologies.\n"
        "All items MUST be from the WHITELIST built in Sub-Step 2.2.\n"
        "If a bucket has fewer than 10 whitelist items, expand using ONLY companions from the\n"
        "COMPANION RULE in Sub-Step 2.2 that belong to that specific bucket's domain.\n"
        "DEDUPLICATION: Maintain a global 'used' set. Once a technology is placed in a bucket,\n"
        "add it to 'used'. Any technology already in 'used' CANNOT be placed in any other bucket.\n\n"

        "── SUB-STEP 2.6: CROSS-CONTAMINATION GATE (final check before output) ─────\n"
        "Before writing the skills JSON, run this explicit checklist:\n"
        "  □ DETECTED_ECOSYSTEM is DOTNET → skills contain ZERO of: React, Vue.js, Next.js,\n"
        "    Node.js, Express.js, Django, Flask, FastAPI, RabbitMQ (unless in JD), gRPC (unless in JD),\n"
        "    MongoDB (unless in JD), DynamoDB (unless in JD), Cosmos DB (unless in JD)\n"
        "  □ DETECTED_ECOSYSTEM is REACT → skills contain ZERO of: Angular, NgRx, Vue.js, Django,\n"
        "    Spring Boot, Laravel, ASP.NET Core (unless JD names them)\n"
        "  □ DETECTED_ECOSYSTEM is ANGULAR → skills contain ZERO of: React, Redux, Next.js, Vue.js,\n"
        "    Django, Spring Boot (unless JD names them)\n"
        "  □ No technology appears in more than ONE bucket\n"
        "  □ No bucket contains technologies from a different domain type\n"
        "    (e.g. Frontend bucket must not contain any backend framework or database)\n"
        "  □ Every technology in every bucket is a real named software product\n"
        "  □ No English verbs, nouns, adjectives, or soft skills appear as technologies\n"
        "If ANY checkbox fails, remove the violating item and replace with a valid one.\n\n"

        "=== COMPANY TECH TAGS ===\n"
        "Each company MUST have a 'tech' field with exactly 6-8 pipe-separated technologies from the JD.\n"
        "Tech tags MUST come from the same WHITELIST built in Sub-Step 2.2 for the detected ecosystem.\n"
        "Fewer than 6 tech tags is a HARD FAILURE.\n\n"

        f"=== ROLE TITLES ===\n"
        f"Produce EXACTLY {len(companies)} role titles using NAMED TECHNOLOGIES from the JD.\n"
        "FORMAT: '[Seniority] [Named JD Tech] [Function Word]'\n"
        "Derive the named technology directly from the JD — use the actual tool names, not generic categories.\n"
        "BANNED domain words — NEVER use these: DevOps, Web, Software, Backend, Frontend, Full-Stack, IT, Tech, Digital\n"
        "Each company: different named-tech domain AND different function word.\n"
        "Function pool (no repeats): Engineer, Developer, Specialist, Analyst, Programmer, Consultant, Designer, Technologist\n"
        + _seniority_rule +

        "=== SUMMARY ===\n"
        "Exactly 4 sentences, minimum 70 words.\n"
        "S1: Start with '{total_years} years of experience in [domain from JD]...'.\n"
        "S2: Name 4-5 technologies FROM THE JD + the specific system types built.\n"
        "S3: Scale and complexity metrics derived from the JD context.\n"
        "S4: Methodology/business outcome relevant to the target company's industry.\n"
        "Use ONLY technologies from the WHITELIST — never add technologies from outside the JD ecosystem.\n\n"

        "=== SKILLS OUTPUT FORMAT ===\n"
        "Exactly 5 categories selected from the populated domain buckets in Sub-Step 2.4.\n"
        "Each category minimum 10 items. ZERO items repeated across ANY category.\n"
        "Category name must match what is actually IN the bucket — name and contents must be consistent:\n"
        "  A category named 'Backend & API' must contain ONLY server-side frameworks, ORMs, APIs.\n"
        "  A category named 'DevOps & CI/CD' must contain ONLY pipelines, containers, IaC, registries.\n"
        "  A category named 'Database & Storage' must contain ONLY databases, caches, search engines.\n"
        "  A category named 'Cloud Services' must contain ONLY cloud platform services.\n"
        "  A category named 'Frontend & UI' must contain ONLY UI frameworks, CSS, state managers.\n"
        "All items come from the WHITELIST only. Cross-contamination gate (Sub-Step 2.6) must pass.\n\n"

        "=== TECHNOLOGIES OBJECT ===\n"
        "All items MUST come from the WHITELIST built in Sub-Step 2.2. No exceptions.\n"
        "mustHave: 10-14 items — JD CORE technologies and their direct ecosystem companions.\n"
        "niceToHave: 8-12 items — JD PREFERRED / Nice-to-Have technologies and their companions.\n"
        "additional: 8-12 items — complementary tools from the SAME detected ecosystem only.\n"
        "ZERO duplicates across the three arrays.\n"
        "ECOSYSTEM GATE: If DETECTED_ECOSYSTEM is DOTNET, mustHave/niceToHave/additional must contain\n"
        "  ZERO of: React, Vue.js, Next.js, Node.js, Express.js, Django, Flask, MongoDB (unless in JD).\n"
        "  Apply the same gate logic for every other detected ecosystem.\n\n"

        "=== ARCHITECTURES ===\n"
        "3-5 objects, each with 'name' and 'description' (25-40 words, JD technologies + concrete outcome).\n"
        "Derive patterns from what the JD actually says - not from generic architecture patterns.\n\n"

        "=== PROFESSIONAL SUMMARY (CRITICAL — read every rule) ===\n"
        "LENGTH: Exactly 7-8 full sentences. Word count: 120-150 words. Fewer than 7 sentences is a HARD FAILURE.\n"
        "TECHNOLOGY INTEGRATION (MOST IMPORTANT): The summary MUST naturally embed NAMED technologies from the JD.\n"
        "  - Use at least 4-6 distinct real technology names from CORE + ECOSYSTEM extraction.\n"
        "  - NEVER use the same technology name twice in the summary.\n"
        "  - Technologies must read naturally in prose — not as a list or stack dump.\n"
        "  - Each technology mention should state what the candidate DID with it (built, optimised, integrated, deployed, designed).\n"
        "  - Example correct: '...has built production APIs using ASP.NET Core and Entity Framework Core, with deployment pipelines on Azure DevOps...'\n"
        "  - Example WRONG: '...experienced in ASP.NET Core, Entity Framework, Azure DevOps, Docker...' (list-style = BANNED)\n"
        "SENTENCE STRUCTURE: Vary sentence openers across all 7-8 sentences.\n"
        "  BANNED openers: 'With', 'Highly', 'I am', 'As a', 'This candidate', 'Passionate', 'Results-driven'\n"
        "  USE openers like: the total_years value + 'years of experience...', 'Throughout...', 'Across...', 'Working on...', 'From building...', 'Over the course of...', direct verb: 'Designed...', 'Built...', 'Collaborated...'\n"
        "CONTENT DISTRIBUTION — sentences must cover ALL of:\n"
        "  Sentence 1: Years of experience + primary JD tech domain + 1-2 named technologies\n"
        "  Sentence 2: A specific type of system or challenge this candidate solves — name 1-2 more technologies\n"
        "  Sentence 3: Collaboration style, team context, or delivery approach (agile, cross-functional, client-facing)\n"
        "  Sentence 4: A concrete capability with a named technology from the JD ecosystem\n"
        "  Sentence 5: Another capability or domain area — different technology angle (cloud, testing, architecture, database)\n"
        "  Sentence 6: Engineering practices, code quality, or professional habits (CI/CD, TDD, code reviews, documentation)\n"
        "  Sentence 7: Impact framing — what this candidate delivers for a business (realistic, not grandiose)\n"
        "  Sentence 8 (optional): Forward-looking or values statement — must reference the JD's domain specifically\n"
        "TONE: Written by a real professional — confident but not boastful. No AI buzzwords.\n"
        "ABSOLUTE BANS: 'Highly motivated', 'Results-driven', 'Dynamic professional', 'Passionate about', 'Leveraged', 'Revolutionised', 'Next-generation', 'AI-powered', 'Innovative solutions', any sentence starting with 'I'.\n\n"

        "=== CORE COMPETENCIES ===\n"
        "Exactly 10 phrases separated by ' * '. Each 2-4 words.\n"
        "Span 4 areas: Technical Practices, Domain Expertise, Engineering Process, Impact Areas.\n"
        "ALL derived from the JD domain - zero generic phrases.\n\n"

        "=== CV HEADLINE TITLE ===\n"
"FORMAT: '[Named JD Tech] [Function Word] | [Tech1], [Tech2], [Tech3]'\n"
"Example: 'WordPress & Webflow Engineer | WordPress, Webflow, Elementor'\n"
"BANNED words: Transformed, Innovative, Dynamic, Versatile, Experienced, Seasoned, Digital Solutions,\n"
"  Web, Software, Backend, Frontend, DevOps, Digital, Full-Stack\n"
"Every word MUST come from the JD. Exactly 3 techs after pipe. Use '|'.\n\n"

        "=== BANNED (instant failure) ===\n"
	"- Job title duplication in the title field (e.g. 'Senior Senior Angular Developer') - normalize before outputting\n"
	"- Location words (Dallas, Texas, Pakistan, Remote, USA) in title or technologies\n"
	"- Copying the exact job title string into the 'title' field - VERBATIM MATCH IS INSTANT FAILURE\n"
	"- More than 3 technologies after the pipe in the title - EXACTLY 3 REQUIRED\n"
        "- Percentage skill bars\n"
        "- Placeholder text (Tech1, Category, kw1, JD-WebDev, JD-Backend)\n"
        "- Technologies not found in the JD or its ecosystem\n"
        "- Same bullet verb repeated across companies\n"
        "- Same metric repeated anywhere in the CV\n"
        "- Fewer than 10 items in any skill category\n"
        "- Fewer than 5 skill categories - EXACTLY 5 are required\n"
        "- More than 5 skill categories - EXACTLY 5 are required\n"
        "- Any technology repeated across two or more skill categories\n"
        "- Database engines (PostgreSQL, MySQL, Redis, etc.) placed in the Frontend, Backend, Cloud, or Testing category\n"
        "- Backend frameworks (Express, Django, Laravel, etc.) placed in the Frontend, Database, Cloud, or Testing category\n"
        "- Frontend libraries (React, Angular, Vue, Tailwind, etc.) placed in the Backend, Database, Cloud, or Testing category\n"
        "- Cloud/DevOps tools (Docker, K8s, GitHub Actions, etc.) placed in the Frontend, Backend, Database, or Testing category\n"
        "- Fewer than 6 tech tags in any company\n"
        "- Summary shorter than 120 words or shorter than 7 sentences\n"
        "- Generic role titles without JD tech domain\n"
        "- Role title using DevOps, Web, Software, Backend, Frontend, Digital as domain word\n"
        "- Invented adjectives in title: Transformed, Innovative, Dynamic, Versatile, Seasoned, Digital Solutions\n"
        "- Same domain word in more than one role title\n"
        "- 1-2 sentence project overviews\n"
        "- Technology-first project names (e.g. 'Azure App', 'Angular Application', 'React Dashboard')\n"
        "- Generic project names (e.g. 'ERP Platform', 'Web Application', 'Social Media Platform')\n"
        "- Real company names inside project names or overviews\n"
        "- Two or more projects with the same purpose, domain, or functionality (reworded duplicates)\n"
        "- Same sentence structure repeated across bullets or projects\n"
        "- Same metric format repeated across projects\n"
        "- Project name separator using ' - ' instead of ': ' — ALWAYS use 'PREFIX: Name' with a colon, NEVER 'PREFIX - Name'\n"
        "- Generic competency phrases like 'Problem Solving' or 'Teamwork'\n"
        "- Hardcoding AWS/Azure/GCP when the JD does not mention or imply them - always infer from the JD\n"
        "- Omitting a cloud/devops skill category - every CV MUST have one dedicated Cloud & DevOps category (Category 4)\n"
        "- Cloud category with fewer than 10 items - expand with CI/CD, IaC, monitoring, and serverless tools\n"
        + _seniority_ban
    ).replace("{total_years}", total_years)

    # -- Dynamic per-company variables ----------------------------------------
    num_cos = len(companies)

    # Seniority fully dynamic based on company count:
    # 1 company  -> Junior only (sole role, entry-level)
    # 2 companies -> plain (current) + Junior (previous)
    # 3 companies -> Senior (current) + plain (mid) + Junior (oldest)
    if num_cos == 1:
        seniority_labels = ["current / only role - Junior"]
        r3_levels = [
            "Co1: Junior prefix. Domain = top NAMED TECH from JD (e.g. 'WordPress' not 'DevOps')."
        ]
        json_seniority = ["Junior"]
    elif num_cos == 2:
        seniority_labels = ["current / plain no prefix", "previous / Junior"]
        r3_levels = [
            "Co1: NO prefix. Domain = PRIMARY named tech from JD (e.g. 'WordPress'). NOT Web/DevOps/Software.",
            "Co2: Junior prefix. Domain = DIFFERENT named tech (e.g. 'Webflow' if Co1=WordPress).",
        ]
        json_seniority = ["", "Junior"]
    else:
        seniority_labels = ["current / Senior", "mid-level / plain no prefix", "oldest / Junior"]
        r3_levels = [
            "Co1: Senior prefix. Domain = PRIMARY named tech (e.g. 'Senior WordPress Engineer' NOT 'Senior DevOps Engineer').",
            "Co2: NO prefix. Domain = DIFFERENT named tech (e.g. 'Webflow Developer').",
            "Co3: Junior prefix. Domain = THIRD named tech (e.g. 'Junior JavaScript Specialist').",
        ]
        json_seniority = ["Senior", "", "Junior"]

    r3_block = "\n".join(r3_levels)

    co_prompt_lines = "\n".join(
        f'Co{i+1} ({seniority_labels[i]}):  "{companies[i]["name"]}"   '
        f'{companies[i]["start"]} - {companies[i]["end"]}'
        for i in range(num_cos)
    )

    function_words = ["Engineer", "Developer", "Specialist"]  # never "Intern"
    # Build tech diversity hints per company based on seniority
    _tech_hints = [
        "PrimaryTech | AdvancedTool | CloudService | ORM | TestingTool | DevOpsTool | ArchitectureTool",  # Co1 senior
        "SecondaryTech | DifferentFramework | DifferentDB | MiddlewareTool | DifferentTestTool | QualityTool | BuildTool",  # Co2 mid
        "FoundationalTech | BasicFramework | SimpleDB | BasicTooling | CoreLanguage | SimpleCITool | LintTool",  # Co3 junior
    ]
    json_companies = ",".join(
        f'{{"company":"{companies[i]["name"]}",'
        f'"role":"{"" if not json_seniority[i] else json_seniority[i] + " "}[Domain] [{function_words[i]}/etc]",'
        f'"dateRange":"{companies[i]["start"]} - {companies[i]["end"]}",'
        f'"bullets":["Achievement + tech + metric","Achievement","Achievement","Achievement"],'
        f'"tech":"{_tech_hints[i] if i < len(_tech_hints) else "Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6"}" }}'
        for i in range(num_cos)
    )

    # -- Build company intelligence block ------------------------------------
    if company_context:
        # Detect whether this came from the website or a web search
        if company_context.startswith("[Company website:"):
            site_label = "COMPANY WEBSITE DATA (use for project 3 - website-driven):"
            google_label = "GOOGLE / GENERAL KNOWLEDGE (use for project 4 - reason from the company name and sector):"
        elif company_context.startswith("[Web search for:"):
            site_label = "WEB SEARCH RESULTS ABOUT THE COMPANY (use for project 3 - website-driven):"
            google_label = "GOOGLE / GENERAL KNOWLEDGE (use for project 4 - reason from the company name and sector):"
        else:
            site_label = "COMPANY CONTEXT (use for project 3 - website-driven):"
            google_label = "GOOGLE / GENERAL KNOWLEDGE (use for project 4 - reason from the company name and sector):"

        co_intel_block = f"""
TARGET COMPANY INTELLIGENCE:
Company: {company_name or "the target company"}

{site_label}
--- Company context start ---
{company_context}
--- Company context end ---

EXTRACT FROM THIS DATA:
1. INDUSTRY: What sector is this company in? (e.g. logistics, healthcare, fintech, retail, SaaS, telecoms, manufacturing)
2. OPERATIONS: What do they actually DO day-to-day? What workflows, systems, or services do they run?
3. USERS: Who are their customers or internal users? (e.g. field agents, drivers, patients, merchants, enterprise clients)
4. SCALE: Regional/national/global? SME/mid-market/enterprise?
5. TECH SIGNALS: Any technology stack hints in their context that align with the JD?

{google_label}
Using your knowledge of "{company_name or "this company"}" — their public reputation, product lines, sector positioning, known clients, and operational model — derive what kinds of internal systems they would realistically build. Use this to make ALL 4 projects feel like they could genuinely exist inside this company. Technology for all projects must come ONLY from the JD.
"""
    elif company_name:
        co_intel_block = f"""
TARGET COMPANY: {company_name}

Use your knowledge of {company_name} to answer:
1. INDUSTRY: What sector do they operate in?
2. OPERATIONS: What are their core business workflows and services?
3. USERS: Who do they serve — internal teams, enterprise clients, consumers, field workers?
4. KNOWN PRODUCTS/PLATFORMS: What are their publicly known offerings or platforms?
5. SCALE & MARKET: Regional/global? B2B/B2C? SME/enterprise?

Use ALL of this to make every generated project feel like it could realistically exist inside {company_name}.
Projects 1-4 must all belong to the same industry as {company_name} — derived from their actual domain.
Technology must come ONLY from the JD — never from what {company_name} actually uses unless it matches.
Do NOT copy their exact product name or literal service description — use the same operational PATTERN in a parallel implementation angle.
"""
    else:
        co_intel_block = ""

    edu_start = edu["start"];   edu_end = edu["end"]

    # Profile context block - used for multi-post / different applications
    profile_block = ""
    if req.profile and req.profile.strip():
        profile_block = f"\nCANDIDATE PROFILE (use personal info like name/email/phone from here ONLY - do NOT use tech stack from profile if it conflicts with JD):\n{req.profile.strip()[:300]}\n"

    # Pre-compute company project rules block (avoids nested f-string)
    _co_name_str = company_name or "the target company"
    if company_context or company_name:
        _company_proj_block = (
            "COMPANY-DRIVEN PROJECT RULES (CRITICAL — projects 1-4 must all reflect the company domain):\n"
            "You have been given intelligence about the target company (" + _co_name_str + "). Use it fully.\n\n"
            "=== STEP A: EXTRACT THE COMPANY DOMAIN ===\n"
            "From the company context provided, identify:\n"
            "  - The company's PRIMARY industry (e.g. logistics, healthcare, fintech, retail, telecoms, SaaS, manufacturing)\n"
            "  - Their core business operations (what they DO day-to-day)\n"
            "  - Their main users/customers (e.g. enterprise clients, field agents, patients, drivers, merchants)\n"
            "  - Any known product lines, platforms, or services mentioned in their website/search data\n"
            "  - The scale they operate at (regional, national, global; SME vs enterprise)\n"
            "Store this as: COMPANY_DOMAIN, COMPANY_OPERATIONS, COMPANY_USERS\n\n"
            "=== STEP B: DERIVE ALL 4 PROJECTS FROM COMPANY DOMAIN + JD ===\n"
            "Every project must solve a real problem that a company like this ACTUALLY faces in their domain.\n"
            "Projects 1 & 2: core internal systems a company in this domain typically needs, built with JD technologies\n"
            "Project 3: aligned with the company's known product/service — use an adjacent implementation angle\n"
            "Project 4: aligned with the company's broader sector — use a different operational area same domain\n"
            "HARD RULE: ALL 4 projects must feel like they could realistically exist inside " + _co_name_str + "\n"
            "HARD RULE: NEVER mix industries — if company is logistics, all 4 projects must be logistics-domain\n\n"
            "=== STEP C: ASSIGN SYSTEM TYPE PREFIX FOR EACH PROJECT ===\n"
            "Each project name MUST begin with: 'PREFIX: Full Descriptive Project Name'\n"
            "The PREFIX must be derived from what the project ACTUALLY DOES — not randomly assigned.\n"
            "Derive the prefix by asking: What class of system is this? What would a CTO call it?\n"
            "  ETL               → data ingestion, transformation, pipeline processing, batch loading\n"
            "  ERP               → enterprise resource planning, finance modules, operations, inventory\n"
            "  CRM               → customer/client records, sales tracking, relationship management\n"
            "  Workflow System   → approval chains, multi-step routing, process automation\n"
            "  Monitoring System → live dashboards, health checks, SLA tracking, alerts\n"
            "  Reporting System  → KPI dashboards, scheduled reports, business intelligence\n"
            "  Inventory System  → stock control, procurement, warehouse management\n"
            "  Dispatch System   → fleet routing, driver assignment, delivery tracking\n"
            "  Billing System    → invoicing, payments, subscription management, reconciliation\n"
            "  Ticketing System  → support requests, issue tracking, helpdesk, SLA compliance\n"
            "  Claims System     → insurance/legal claims, case management, status tracking\n"
            "  Scheduling System → appointments, calendars, bookings, queue management\n"
            "  Compliance System → audit trails, regulatory tracking, risk controls\n"
            "  Backend System    → API services, business logic, server-side processing\n"
            "  Integration System → third-party APIs, data sync, middleware, webhooks\n"
            "  Analytics System  → data visualisation, aggregated metrics, BI, OLAP\n"
            "  Notification System → email/SMS/push alerts, event-driven messaging\n"
            "  Document System   → file management, version control, digital archiving\n"
            "  Identity System   → authentication, authorisation, SSO, RBAC\n"
            "  CI/CD System      → build pipelines, deployment automation, DevOps workflows\n"
            "  LMS               → course delivery, assessments, learner progress tracking\n"
            "  CMS               → content publishing, editorial workflows, web content\n"
            "  + any other precise technical class label that accurately describes the system\n"
            "ALL 4 projects must have DIFFERENT prefixes — no two projects share the same class label\n\n"
            "=== STEP D: WRITE EACH PROJECT ===\n"
            "REQUIRED format for name field: 'PREFIX: Full Descriptive Project Name'\n"
            "  - Project name after the colon must be 5-10 descriptive words explaining what it does\n"
            "  - Name must NOT contain: the company's real product name, brand, or exact service\n"
            "  - Use technologies ONLY from the JD\n\n"
            "Each project MUST have:\n"
            "- name: 'PREFIX: Full Descriptive Project Name' (PREFIX derived from project function, not randomly)\n"
            "- systemType: SAME string as the PREFIX before the colon\n"
            "- overview: 3-4 sentences: OPERATIONAL PROBLEM faced in this domain -> SOLUTION using named JD techs\n"
            "    -> FUNCTIONALITY (concrete features users interact with) -> BUSINESS IMPACT (believable metric)\n"
            "  The problem must be domain-realistic: something a real " + _co_name_str + "-type company actually faces\n"
            "- bullets: exactly 3 strings (20-30 words each), each describing a DIFFERENT feature/challenge:\n"
            "    bullet 1: enterprise-realistic component built + JD technology used + what operational problem it solved\n"
            "    bullet 2: technical challenge + how solved + UNIQUE believable metric (not reused across projects)\n"
            "    bullet 3: business outcome with UNIQUE believable number\n"
            "- No two projects may share the same bullet structure, metric format, or system type prefix\n"
        )
    else:
        _company_proj_block = (
            "=== PROJECT GENERATION — NO COMPANY PROVIDED ===\n"
            "With no company name/context, derive ALL project domain signals from the JD itself.\n\n"
            "=== STEP A: EXTRACT DOMAIN FROM JD ===\n"
            "Read the JD carefully and identify:\n"
            "  - The INDUSTRY this role operates in (infer from responsibilities, domain keywords, user types)\n"
            "  - The OPERATIONAL CONTEXT (what kinds of systems does a company in this JD domain typically run?)\n"
            "  - The USERS the candidate will serve (internal staff, customers, field agents, enterprise clients)\n"
            "  - The SCALE implied (startup/SME/enterprise, single-office/multi-branch/global)\n\n"
            "=== STEP B: GENERATE 4 PROJECTS ALIGNED TO THAT DOMAIN ===\n"
            "All 4 projects must feel like they belong to the same company portfolio — same industry, coherent scope.\n"
            "Each project must solve a REAL OPERATIONAL PROBLEM a company in this JD domain actually faces.\n"
            "Each project must use a DIFFERENT part of the company's operational workflow:\n"
            "  e.g. if domain=logistics: project 1=dispatch, project 2=inventory, project 3=billing, project 4=compliance\n"
            "  e.g. if domain=healthcare: project 1=scheduling, project 2=records, project 3=billing, project 4=reporting\n"
            "  e.g. if domain=fintech: project 1=payments, project 2=compliance, project 3=reporting, project 4=identity\n"
            "NEVER mix unrelated industries across the 4 projects — consistency signals real enterprise experience\n\n"
            "=== STEP C: ASSIGN SYSTEM TYPE PREFIX — DERIVED FROM PROJECT FUNCTION ===\n"
            "Each project name MUST begin with: 'PREFIX: Full Descriptive Project Name'\n"
            "The PREFIX is determined by what the system ACTUALLY DOES — ask: what class of system is this?\n"
            "  ETL               → data ingestion, transformation, pipeline, batch loading\n"
            "  ERP               → enterprise resource planning, finance, inventory, operations\n"
            "  CRM               → customer records, sales tracking, relationship management\n"
            "  Workflow System   → approval chains, routing, process automation\n"
            "  Monitoring System → dashboards, health checks, SLA tracking, live alerts\n"
            "  Reporting System  → KPI dashboards, scheduled reports, BI exports\n"
            "  Inventory System  → stock control, procurement, warehouse management\n"
            "  Dispatch System   → fleet routing, driver assignment, delivery tracking\n"
            "  Billing System    → invoicing, payments, subscriptions, reconciliation\n"
            "  Ticketing System  → support requests, issue tracking, SLA compliance\n"
            "  Claims System     → insurance/legal claims, case management\n"
            "  Scheduling System → appointments, calendars, bookings, queues\n"
            "  Compliance System → audit trails, regulatory controls, risk tracking\n"
            "  Backend System    → API services, business logic, server-side processing\n"
            "  Integration System → third-party APIs, data sync, middleware, webhooks\n"
            "  Analytics System  → data visualisation, aggregated metrics, OLAP, BI\n"
            "  Notification System → email/SMS/push alerts, event-driven messaging\n"
            "  Document System   → file management, version control, digital archiving\n"
            "  Identity System   → authentication, authorisation, SSO, RBAC\n"
            "  CI/CD System      → build pipelines, deployment automation, DevOps\n"
            "  + any other precise class label that accurately describes the system's function\n"
            "ALL 4 projects must have DIFFERENT prefixes — each covers a distinct operational area\n\n"
            "=== STEP D: WRITE EACH PROJECT ===\n"
            "REQUIRED name format: 'PREFIX: Full Descriptive Project Name (5-10 words after the colon)'\n"
            "Single-word or blended names like 'EcoCycle', 'SmartFarm' are an INSTANT FAILURE\n"
            "Use technologies ONLY from the JD — nothing from outside the JD ecosystem\n\n"
            "Each project MUST have:\n"
            "- name: 'PREFIX: Full Descriptive Project Name' — PREFIX logically derived from project function\n"
            "- systemType: SAME string as the PREFIX before the colon\n"
            "- overview: 3-4 sentences: OPERATIONAL PROBLEM in this domain -> SOLUTION using named JD techs\n"
            "    -> FUNCTIONALITY (concrete features users interact with) -> BUSINESS IMPACT (unique believable metric)\n"
            "  The problem must be domain-realistic — something real companies in this JD industry actually face\n"
            "- bullets: exactly 3 strings (20-30 words each), each covering a DIFFERENT feature/challenge:\n"
            "    bullet 1: enterprise-realistic component built + JD technology + what operational problem it solved\n"
            "    bullet 2: technical challenge + how resolved + UNIQUE believable metric\n"
            "    bullet 3: business outcome + UNIQUE believable number\n"
            "- No two projects share the same prefix, bullet structure, metric format, or problem domain\n"
        )

    user = f"""Generate CV JSON for this job.

JOB: {req.job_title}
JD: {jd}
{co_intel_block}{profile_block}
TOTAL EXPERIENCE: {total_years} years
You MUST use exactly "{total_years}" in totalYears and start the summary with "{total_years} years of experience".
Do NOT change this number. Do NOT round up or down.

COMPANIES - produce EXACTLY {num_cos} company objects (use EXACTLY these names and date ranges - do not change them):
{co_prompt_lines}

EDUCATION:
Degree dates: {edu_start} - {edu_end}  (use EXACTLY these years for the education section)

PRE-GENERATION ANALYSIS (execute silently before writing any JSON):

STEP A — JD DOMAIN EXTRACTION:
  Read the job title, job description, and any company intelligence provided.
  Extract and store internally:
  A1. PRIMARY_TECH_STACK: The main technology/framework this role is built around (e.g. ASP.NET Core, React, Django, Node.js)
  A2. ROLE_TYPE: TECHNICAL / NON-TECHNICAL / HYBRID (see classification rules)
  A3. INDUSTRY_DOMAIN: The industry this company operates in (e.g. logistics, healthcare, fintech, retail, SaaS, HR tech)
  A4. COMPANY_OPERATIONS: What the company actually does operationally (infer from JD + company name if provided)
  A5. TARGET_USERS: Who the candidate will build systems for (e.g. warehouse staff, patients, field agents, finance teams)
  A6. SENIORITY_LEVEL: Derived from years_exp and JD language (junior/mid/senior)

STEP B — PROJECT DOMAIN SELECTION:
  Using A3 (INDUSTRY_DOMAIN) and A4 (COMPANY_OPERATIONS):
  Select 4 distinct operational areas within that domain that a real company would build internal systems for.
  These become the 4 project domains — they must all be from the SAME industry, covering DIFFERENT workflow areas.
  Examples:
    Logistics company → dispatch system, inventory system, billing system, compliance system
    Healthcare company → scheduling system, records system, billing system, reporting system
    Fintech company → payments system, compliance system, reporting system, identity system
    Retail company → inventory system, CRM, analytics system, workflow system
  Store these as PROJECT_DOMAIN_1, PROJECT_DOMAIN_2, PROJECT_DOMAIN_3, PROJECT_DOMAIN_4

STEP C — PREFIX ASSIGNMENT:
  For each of the 4 selected project domains, assign the logically correct system type prefix:
  Ask: "What class of system does a {INDUSTRY_DOMAIN} company build to handle {PROJECT_DOMAIN_N}?"
  The answer IS the prefix. Never assign randomly. All 4 must differ.

RULES:

R1 DOMAIN: Identify the tech stack from JD keywords. Use matching tools + closely related ecosystem tools.

"R2 TITLE: Your title MUST be derived from the JD, not generic. Extract the primary domain and a DIFFERENT technology from the JD's mustHave list.\n\n"

ANTI-COPY ENFORCEMENT (HARD RULE):
"FIRST: Extract the primary technology domain from THIS SPECIFIC JD (not a generic placeholder).\n"
STEP 1 - Write down the exact job title: "{req.job_title}".
STEP 2 - Your proposed title MUST differ in at least TWO of the following ways:
  a) Function word swapped (Engineer -> Developer, Specialist -> Architect, Trainee -> Engineer, Developer -> Programmer)
  b) Scope broadened (e.g. 'PHP Developer' -> 'Full-Stack Web Engineer', 'Laravel Trainee' -> 'Backend Application Developer')
  c) Platform angle shifted (e.g. 'Laravel Developer' -> 'PHP & MySQL Application Engineer')
  d) Domain emphasis changed (e.g. 'Backend Developer' -> 'Server-Side Systems Architect')
STEP 3 - Read your title and the job title side by side. If they look the same or share all the same words -> REJECT and rederive.
Step 4 - Your output title MUST NOT contain the exact original job title string as a substring.
"Step 5 - Add a pipe and exactly 3 technologies extracted from THIS SPECIFIC JD (e.g. if JD mentions Laravel, PHP, MySQL, pick 3 from those).\n"
HOW TO DERIVE: Extract the PRIMARY tech domain from the JD requirements. Apply STEP 2 transformations.
Pick 3 REAL technologies from the JD only for the subtitle after the pipe.
ALSO BANNED: Do NOT copy or reuse the title from the candidate profile (e.g. if profile says "Laravel & PHP Development Specialist", that string is forbidden in the output title).
Outputting the job title OR the candidate's existing profile title verbatim is an INSTANT FAILURE.

R3 SENIORITY + ROLE TITLES (critical):
Each company MUST have a UNIQUE role title. BOTH seniority AND function word MUST differ:
{r3_block}
Function word pool (ONE per company, NO repeats): Engineer, Developer, Specialist, Analyst, Programmer, Consultant, Designer, Technologist
HARD RULE: No function word repeats.
HARD RULE: Domain = real named JD tech. DevOps/Web/Software/Backend/Frontend = BANNED.
HARD RULE: Transformed/Innovative/Dynamic/Digital Solutions = BANNED in CV title.
NOTE: post-processor will auto-fix any role with a banned domain word.

R4 BULLETS: 4 per company, 20-30 words each. Every bullet: >=1 JD technology + specific named system + unique metric OR collaboration detail.
TECH ENFORCEMENT: Use ONLY technologies extracted from the JD. Never add technologies not in the JD.
Every bullet must name at least one technology from the JD's REQUIRED or PREFERRED list.
CLOUD BULLET RULE: Across all companies combined, at least 2 bullets MUST reference the cloud/hosting
platform inferred from the JD (e.g. deployment to cloud, hosted on platform, CI/CD pipeline, auto-scaling,
cloud-native service integration). Derive the platform from the JD - do NOT hardcode AWS/Azure/GCP.

REALISTIC WORK MIX (critical for authenticity):
  - Across the 12 bullets total, include at least 3 bullets describing realistic day-to-day work:
      * Bug investigation or production issue resolution (e.g. "Diagnosed and resolved a memory leak in the X service reducing p99 latency from 3.2s to 800ms")
      * Legacy system maintenance or refactoring (e.g. "Refactored legacy payment processing module from synchronous to async improving throughput by 40%")
      * Cross-team collaboration (e.g. "Collaborated with QA and DevOps teams to coordinate staged rollout of X feature across 3 environments")
      * Support, documentation, or knowledge transfer (e.g. "Maintained internal API documentation and onboarded 3 junior developers on X framework patterns")
  - These realistic bullets make the CV feel human and stop it reading like a launch list.
  - Co1 (most senior) can have 2-3 achievement-style bullets; Co2 and Co3 should have more mixed/realistic bullets.

Verb guide (derive from the JD's domain - do not use generic verbs):
  Co1 (most senior): high-ownership verbs reflecting leadership - Architected, Engineered, Led, Spearheaded, Established, Designed, Launched, Directed, Oversaw
  {"Co2 (mid): improvement and maintenance verbs - Optimised, Refactored, Migrated, Integrated, Streamlined, Resolved, Maintained, Collaborated, Extended" if num_cos >= 2 else ""}
  {"Co3 (junior): delivery and support verbs - Implemented, Built, Configured, Automated, Assisted, Participated, Documented, Debugged, Supported" if num_cos >= 3 else ""}

BULLET DIVERSITY (all 12 bullets across all companies must be unique):
- Each bullet describes a DIFFERENT specific system or feature type - named precisely (e.g. 'real-time notification engine', 'multi-tenant billing service', 'role-based access middleware').
- Each bullet uses a DIFFERENT primary technology from the JD.
- Each metric is UNIQUE - no two bullets share the same number, percentage, or user count.
- Each bullet has a DIFFERENT sentence structure - no two bullets follow the same grammatical template.
- NEVER use 'web application' or 'full stack application' as the deliverable - name the SPECIFIC system type.
- ZERO verbs repeated across any of the 12 bullets.
- NOT every bullet needs a numeric metric — some bullets can end with a collaboration, process, or quality outcome.

R5 SUMMARY: Exactly 4 sentences, minimum 70 words.
S1: Start "{total_years} years of experience in [domain from JD], with a strong background in [2-3 JD areas]."
    BANNED openers: 'Highly motivated', 'Results-driven', 'Dynamic', 'Passionate', 'Dedicated professional'
S2: "Proficient in [JD-tech1], [JD-tech2], [JD-tech3], [JD-tech4], building [specific system type from JD]."
    Keep this factual — name the actual tools and what was built with them.
S3: "Proven ability to [concrete, realistic contribution] handling [scale/complexity from JD context]."
    This should feel like something a real person would say: not 'revolutionized' but 'improved', 'maintained', 'delivered'
S4: "Committed to [methodology], delivering [business outcome] through [practice]."
    Methodology = what the JD expects (Agile, TDD, CI/CD, etc.). Business outcome = real, believable.
ABSOLUTE RULE: Do NOT use buzzwords — no 'synergy', 'leverage', 'paradigm', 'AI-powered', 'ecosystem', 'next-generation'.
ABSOLUTE RULE: Use ONLY technologies extracted from the JD. Count words - under 70 is a FAILURE.

R6 SKILLS — FULLY DYNAMIC (derive everything from the JD):

  Step 1: From your STEP 1 technology extraction, list every real named tool.
  Step 2: Group them into 5 natural technical domains that reflect how this specific role actually works.
    - Group names come from the JD's own technology areas — not from generic labels.
    - Which tools go in which group is decided by you based on what makes sense for this role.
    - Do NOT use preset bucket names. Do NOT apply rules from other CVs. Each JD produces unique groups.
  Step 3: Each group must have at least 10 real named tools. Expand with direct ecosystem companions if needed.
  Step 4: Zero items repeated across any group.
  Step 5: Output as "Group Name: tool1, tool2, tool3, ..." — exactly 5 rows.

  ONLY RULE: Every item must be a real named tool from the JD ecosystem.
  BANNED as skill items: verbs (Configure, Deploy, Monitor), generic nouns (System, Platform, Service, Infrastructure),
    soft skills (Leadership, Teamwork), adjectives (Strong, Good), or anything not a real named product.


R7 PROJECTS: Produce EXACTLY 4 projects split as follows:
  PROJECT 1 & 2 - JD-driven: Strictly aligned with the job description tech stack and role domain.
  PROJECT 3 - Company-aligned: Based on what the company website/context reveals about their core product/platform domain.
  PROJECT 4 - Company-sector: Based on broader public knowledge of what this company's sector/industry is known for.
  If no company name or URL was provided, replace projects 3 and 4 with two additional domain-specific JD projects covering DIFFERENT system types.

PROJECT SYSTEM TYPE PREFIX (mandatory for every project — embedded in the name field):
  Every project name MUST begin with a short system-type prefix followed by a colon and the descriptive name.

  REQUIRED FORMAT (exact):
    "systemType: Descriptive Project Name"

  EXAMPLES of correct format:
    "ETL: Real-Time Data Ingestion and Transformation Pipeline"
    "ERP: Internal Procurement and Purchase Order Management System"
    "CRM: Customer Support and Complaint Resolution Tracking Platform"
    "Backend System: Order Processing and Fulfilment API Service"
    "Workflow System: Multi-Stage Leave Approval and HR Management Portal"
    "Monitoring System: Infrastructure Health and Alerting Dashboard"
    "Data Processing System: Batch Invoice Extraction and Reconciliation Engine"
    "Reporting System: Multi-Branch Sales Performance and KPI Dashboard"
    "Inventory System: Warehouse Stock Control and Shipment Tracking Hub"
    "Scheduling System: Clinical Appointment Booking and Queue Management Portal"
    "Billing System: Subscription Payment Tracking and Invoice Management Platform"
    "Compliance System: Regulatory Audit Trail and Risk Assessment Portal"
    "Document System: Contract Storage, Version Control and Approval Workflow"
    "Dispatch System: Fleet Route Optimisation and Live Driver Assignment Console"
    "Ticketing System: IT Support Request Tracking and SLA Monitoring Platform"
    "Claims System: Insurance Claims Validation and Status Tracking Portal"
    "Identity System: Role-Based Access Control and SSO Management Platform"
    "Analytics System: Business Intelligence and Operational Metrics Dashboard"
    "Integration System: Third-Party API Gateway and Data Synchronisation Service"
    "Notification System: Multi-Channel Alert and Event-Driven Messaging Service"
    "LMS: Employee Training Course Delivery and Assessment Tracking System"
    "CMS: Web Content Publishing and Editorial Approval Workflow"
    "CI/CD System: Automated Build, Test and Deployment Pipeline"
    "SEO System: Keyword Ranking Audit and Organic Traffic Reporting Dashboard"
    "Marketing System: Campaign Management and Lead Generation Tracking Platform"

  HOW TO PICK THE RIGHT PREFIX (derive from JD — NEVER random, NEVER hardcoded):
    Step 1 — Read the project's actual business function (what it does operationally).
    Step 2 — Read the JD's domain, industry, and tech stack.
    Step 3 — Pick the prefix that most precisely describes the system CLASS:
      ETL             → data ingestion, transformation, loading, pipeline processing
      ERP             → enterprise resource planning, inventory, finance, operations modules
      CRM             → customer/client records, sales pipeline, relationship tracking
      Backend System  → API services, business logic layers, server-side processing
      Workflow System → approval chains, routing, multi-step business process automation
      Monitoring System → live metrics, alerts, health checks, SLA tracking, dashboards
      Data Processing System → batch jobs, bulk data handling, aggregation, reconciliation
      Reporting System → BI dashboards, KPI tracking, scheduled report generation
      Inventory System → stock management, warehouse ops, procurement, shipment tracking
      Scheduling System → bookings, calendars, queue management, appointment systems
      Billing System  → invoicing, payment tracking, subscription management, finance
      Compliance System → audit logs, regulatory tracking, risk scoring, controls
      Document System → file management, version control, digital archiving, approvals
      Dispatch System → fleet management, route optimisation, driver assignment, logistics
      Ticketing System → support requests, issue tracking, SLA enforcement, helpdesk
      Claims System   → insurance/legal claims, case management, status tracking
      Identity System → authentication, authorisation, SSO, RBAC, access control
      Analytics System → data visualisation, BI, aggregated metrics, OLAP
      Integration System → API integration, middleware, webhooks, data sync
      Notification System → email/SMS/push alerts, event-driven messaging, queues
      LMS             → learning management, course delivery, assessments
      CMS             → content publishing, editorial workflows, web content
      CI/CD System    → build pipelines, deployment automation, DevOps workflows
      SEO System      → search optimisation, keyword tracking, rank monitoring
      Marketing System → campaign management, lead tracking, digital marketing

  ABSOLUTE RULES:
    - The prefix MUST appear inside the "name" JSON field itself — format: "PREFIX: Full Project Name"
    - All 4 projects MUST use DIFFERENT prefixes — no two projects can share the same prefix
    - The prefix must be logically correct for what the project actually does — never randomly assigned
    - NEVER use generic prefixes: "System", "Platform", "App", "Web App", "Software" alone
    - The full name after the colon must still be 5-10 descriptive words (no single-word names)
    - Derive the prefix entirely from the JD domain and project functionality — nothing hardcoded

PROJECT TECH TAGS (CRITICAL - ABSOLUTE RULES):
  Each project MUST have EXACTLY 5-7 techTags. Fewer than 5 is a HARD FAILURE.
  ALL tags MUST be real named software tools/frameworks/platforms from the JD ecosystem.
  ABSOLUTELY BANNED from techTags (instant failure if any appear):
    'Cloud', 'Development', 'Web', 'Good', 'Strong', 'Hands', 'Ability', 'Remote', 'Setup',
    'APIs', 'REST', 'Data', 'Code', 'Net', 'App', 'Backend', 'Frontend', 'Database',
    any plain English adjective, noun, or verb - ONLY real product names are valid.
  VALID examples: 'ASP.NET Core', 'Angular', 'SQL Server', 'Entity Framework Core', 'Azure App Service'
  INVALID examples: 'Cloud', 'Development', 'Good', 'REST', 'Data', 'APIs'

PROJECT NAMING RULES (CRITICAL - enforce before writing anything else):

  *** SINGLE-WORD PROJECT NAMES ARE AN INSTANT HARD FAILURE ***
  Names like 'EcoCycle', 'HealthHub', 'SmartFarm', 'InvoiceFlow', 'ShiftSync', 'ClaimsPulse', 'VendorLink'
  on their own (without a dash and full description) are BANNED. A one-word or two-word blended name alone
  is NEVER acceptable as a project name under any circumstances.

  REQUIRED FORMAT (MANDATORY — no exceptions):
  "PREFIX: Full Descriptive Project Name"

  The PREFIX is a short system-class label (1-4 words) placed BEFORE the colon.
  The PREFIX must be determined by what the project actually DOES — not picked from a fixed list.
  The name AFTER the colon must be 5-10 descriptive words explaining what the system does.

  HOW TO DERIVE THE PREFIX (dynamic — do this for every project individually):
    Step 1: Describe what the project does in one sentence.
    Step 2: Ask "What class of system is this?" from a technical/business perspective.
    Step 3: Use that answer as the prefix.
    Examples of correct derivation:
      "This project ingests raw sales data, transforms it, and loads it into a data warehouse"
        → PREFIX: ETL
      "This project manages multi-level approval chains for purchase orders"
        → PREFIX: Workflow System
      "This project tracks fleet vehicles in real time and assigns delivery routes"
        → PREFIX: Dispatch System
      "This project handles insurance claim submissions and tracks resolution status"
        → PREFIX: Claims System
      "This project stores and manages employee records, payroll data, and leave balances"
        → PREFIX: ERP  (or HR System — whichever is more precise for the functionality)
      "This project generates weekly KPI reports for branch managers"
        → PREFIX: Reporting System
      "This project processes customer invoices and tracks payment status"
        → PREFIX: Billing System

  DOMAIN ALIGNMENT (CRITICAL — derived from JD and company, never hardcoded):
  All 4 projects MUST be in the SAME industry/domain as the target company.
  Derive the domain from: company name, company context, JD responsibilities, and industry keywords.
  All 4 projects must feel like they could all exist inside the same company at the same time.
  NEVER mix unrelated industries across the 4 projects.

PROJECT UNIQUENESS + RELEVANCE RULES (STRICT):
  All 4 projects MUST differ in: Purpose, Domain, and Functionality - no reworded duplicates.
  Each project MUST still be strongly aligned with the same job role/domain.
  Technology stacks MUST NOT be identical across all 4 projects - vary the specific tools used.
  Each project MUST have a unique business impact metric - ZERO metric formats repeated.

PROJECT COMPANY INTELLIGENCE RULE:
  For projects 3 & 4: Infer the company's domain from public knowledge of the company name and sector.
  Projects MUST reflect what the company realistically builds - no random or generic assumptions.

PROJECT DESCRIPTION RULES - ABSOLUTE (MOST IMPORTANT FOR PROJECTS):
Each project "overview" MUST tell a complete mini-story across 3-4 sentences covering ALL of:
  PROBLEM: What real-world BUSINESS problem or operational pain point did this system solve? Be specific about who suffers (operations team, billing department, field agents, warehouse staff) and what they lose (hours, revenue, accuracy) without this system. NEVER start with 'The company needed...' or 'We needed to build...'.
  SOLUTION: What specific architectural approach or technical solution was designed? Name the actual JD technologies used. Sound like a real engineer explaining their approach, not a sales brochure.
  FUNCTIONALITY: What does the system do in practice? Describe key features in concrete, operational terms — what users actually do with it (approve requests, track shipments, generate reports, assign routes).
  BUSINESS IMPACT: What measurable, believable outcome was achieved? (e.g. reduced approval time from 3 days to 4 hours, eliminated 200 manual data entry steps per week, served 150 internal users across 3 branches). UNIQUE per project — no metric format reused.

ENTERPRISE REALISM RULES (critical for authenticity):
  - Projects should solve real operational problems that a company of that domain actually faces
  - Feature descriptions should reflect realistic enterprise functionality: role-based access, approval workflows, audit trails, multi-branch support, export to Excel/PDF, notification systems, batch processing
  - Avoid grandiose claims: say 'reduced manual workload by 35%' not 'transformed operations revolutionarily'
  - Include realistic technical challenges: data migration from legacy systems, handling concurrent users, integrating with existing ERP/CRM, meeting compliance requirements
  
EXAMPLE STRUCTURE (adapt to actual JD domain):
  "Operations teams across 4 regional offices tracked vendor approvals via email threads and shared spreadsheets, causing version conflicts and 3-5 day delays on time-sensitive purchase orders. The team built a centralised procurement workflow engine using [JD-Tech1] and [JD-Tech2], with configurable approval chains, budget validation against ERP data, and email notifications at each stage. Role-based access controls ensured finance managers, department heads, and procurement officers each saw only their relevant queue. The platform reduced average approval cycle time from 4 days to under 6 hours and handled 300+ purchase orders monthly across all branches."

HARD RULES FOR OVERVIEWS:
- MINIMUM 3 sentences, ideally 4 - short 1-sentence overviews are a HARD FAILURE
- Each overview MUST name at least 2 specific JD technologies in the solution/functionality sentences
- Each overview MUST include ONE unique business impact metric (number + unit) not reused in other projects
- NO overview may reuse the same phrase or sentence structure as another project
- NEVER include any real company name (hiring company, candidate's companies, or any known brand) in any project name or overview.
- NEVER use generic platform names: "E-commerce Platform", "Social Media Platform", "Project Management Platform", "Business Intelligence Platform", "Web Application", "Mobile App" are all BANNED as project names.
- Each project name MUST be a coined, original enterprise-product-style name followed by a dash and a specific description of what it does.
- NEVER start overviews with 'The company' — start from the operational problem or the user's perspective.

{_company_proj_block}

R8 RELATED TECH: Exactly 5 boxes, specific real category names, 5-6 items each from JD ecosystem.

R9 COMPETENCIES: Exactly 10 domain-specific phrases separated by *.
STRUCTURE: The 10 phrases MUST span at least 4 different competency categories relevant to the JD:
  - Technical Practices: e.g. 'API Design', 'Test-Driven Development', 'Microservices Architecture', 'RESTful API Development'
  - Domain Expertise: e.g. 'ERP System Integration', 'Cloud-Native Development', 'Data Pipeline Engineering', 'Real-Time Analytics'
  - Engineering Process: e.g. 'CI/CD Pipeline Automation', 'Agile Sprint Delivery', 'Code Review Leadership', 'DevOps Practices'
  - Impact Areas: e.g. 'Performance Optimisation', 'System Scalability', 'Database Query Tuning', 'Security Hardening'
DERIVATION: Extract competencies from JD methodologies, tools, practices, and outcomes - NOT generic soft skills.
If fewer than 10 domain-specific competencies are in the JD, extend with closely adjacent engineering practices in the SAME tech domain.
Each phrase: 2-4 words, directly tied to JD tech domain. Fewer than 10 is a HARD FAILURE. Generic filler like "Problem Solving" or "Teamwork" is a HARD FAILURE.

R10 KEYWORDS: 18-20 ATS terms from JD.
ATS KEYWORD INTEGRATION RULES:
- Keywords must be naturally embedded in experience bullets and summary — not force-inserted
- Mix exact JD terms (for ATS parsing) with natural synonyms (for recruiter readability)
- Include role-level terms (e.g. 'mid-level', 'senior', 'full-stack') that match the candidate's level
- Include domain-specific industry terms from the JD (e.g. 'multi-tenant', 'SLA compliance', 'audit trail')
- NEVER repeat the same keyword more than 2-3 times across the entire CV

R12 TECHNOLOGIES (mandatory, 3 sub-arrays):
- "mustHave": All explicitly required/must-have tools from the JD + 1-3 closely related ecosystem tools (6-10 items total).
- "niceToHave": All nice-to-have/preferred tools from the JD + 1-3 related ecosystem additions (5-8 items total).
- "additional": 5-8 complementary tools NOT in the JD but standard in this role domain (e.g. testing, security, observability).
Every item must be a real named tool - no placeholders. No item repeated across sub-arrays.

R13 ARCHITECTURES (mandatory, 3-5 objects):
Produce 3-5 objects, each with "name" (concise pattern) and "description" (1-2 sentences, 25-40 words, concrete tech + outcome).
FIRST cover every pattern explicitly in the JD. THEN add closely related patterns that are standard in this role domain to reach 3-5 total.
Each description MUST name actual technologies from the JD and state a concrete outcome (metric, improvement, or capability gained).
Do NOT use plain strings. Every item must have both "name" and "description".

FINAL ATS + RECRUITER READABILITY CHECKLIST (apply before output):
  ✓ Summary reads naturally — no buzzwords, no AI-generated phrasing
  ✓ Summary is 7-8 full sentences and 120-150 words — shorter is a HARD FAILURE
  ✓ Summary embeds at least 4-6 different real JD technology names in natural prose (not a list)
  ✓ No technology name appears more than once in the summary
  ✓ All 12 experience bullets use different verbs and different sentence structures
  ✓ At least 3 bullets reflect realistic day-to-day work (maintenance, debugging, collaboration)
  ✓ Each company's tech tags include at least 2-3 technologies UNIQUE to that company only
  ✓ No single technology name appears in ALL 3 companies' tech tags
  ✓ All project names use 'PREFIX: Name' format with a COLON — never a dash ' - '
  ✓ All project names are 3-6 word descriptive names — NO single-word or blended-word names
  ✓ All 4 project names begin with "PREFIX: Name" format — unique prefix per project, logically correct for what it does
  ✓ All 4 project overviews tell a complete operational story starting from the business problem
  ✓ No metric format repeated across projects or bullets
  ✓ No ML/AI/predictive analytics injected unless JD explicitly requires it
  ✓ No fake percentages (99% uptime, 92% accuracy) — use realistic operational numbers
  ✓ Skills are cleanly separated — no tool appears in two categories
  ✓ Tech tags per company contain only real named tools from JD ecosystem
  ✓ Career progression feels realistic — seniority matches years of experience naturally
  ✓ Technologies used in bullets align with the tech tags shown for that company

JSON shape (totalYears="{total_years}", degree years={edu_start}-{edu_end}, EXACTLY {num_cos} companies):
{{"totalYears":"{total_years}","title":"Related Role Title - Tech1, Tech2, Tech3","summary":"[7-8 sentences, 120-150 words, human-written, no buzzwords. Sentence 1: {total_years} years + primary JD domain + 1-2 named techs. Sentences 2-6: each naturally embeds 1-2 DIFFERENT named JD technologies in context of what was built/done. Sentence 7: realistic business impact. Sentence 8 (optional): forward values tied to JD domain. MINIMUM 4-6 different real technology names spread across the summary — never listed, always embedded in prose.]","companies":[{json_companies}],"skills":["JD-Derived Domain 1: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10","JD-Derived Domain 2: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10","JD-Derived Domain 3: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10","JD-Derived Domain 4: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10","JD-Derived Domain 5: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10"],"education":{{"university":"QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY","degree":"Bachelor of Science in Computer Science (BSCS)","cgpa":"3.97/4.0","years":"{edu_start} - {edu_end}","achievement":"Gold Medalist for Academic Excellence"}},"projects":[{{"name":"PREFIX: Multi-Word Descriptive Project Name (e.g. ERP: Internal Procurement and Purchase Order System)","systemType":"same PREFIX used in name field","overview":"[OPERATIONAL PROBLEM: who is affected and how.] [SOLUTION with 2 named JD techs and architecture choice.] [FUNCTIONALITY: concrete user-facing features.] [BUSINESS IMPACT: unique believable metric.]","bullets":["Enterprise-realistic component built using JD-tech - what operational problem it solved (20-30 words)","Technical challenge encountered + how solved + unique concrete metric (20-30 words)","Business outcome with unique believable number (20-30 words)"]}},{{"name":"DIFFERENT_PREFIX: Multi-Word Descriptive Project Name","systemType":"DIFFERENT_PREFIX","overview":"[PROBLEM.] [SOLUTION with 2 JD techs.] [FUNCTIONALITY.] [BUSINESS IMPACT unique metric.]","bullets":["realistic enterprise feature+tech (20-30w)","technical challenge+UNIQUE metric (20-30w)","business outcome+UNIQUE number (20-30w)"]}},{{"name":"DIFFERENT_PREFIX: Multi-Word Descriptive Project Name","systemType":"DIFFERENT_PREFIX","overview":"[PROBLEM.] [SOLUTION with 2 JD techs.] [FUNCTIONALITY.] [BUSINESS IMPACT unique metric.]","bullets":["realistic feature+tech (20-30w)","challenge+UNIQUE metric (20-30w)","outcome+UNIQUE number (20-30w)"]}},{{"name":"DIFFERENT_PREFIX: Multi-Word Descriptive Project Name","systemType":"DIFFERENT_PREFIX","overview":"[PROBLEM.] [SOLUTION with 2 JD techs.] [FUNCTIONALITY.] [BUSINESS IMPACT unique metric.]","bullets":["realistic feature+tech (20-30w)","challenge+UNIQUE metric (20-30w)","outcome+UNIQUE number (20-30w)"]}}],"competencies":"TechPractice1 * DomainExpertise1 * EngineeringProcess1 * ImpactArea1 * TechPractice2 * DomainExpertise2 * EngineeringProcess2 * ImpactArea2 * TechPractice3 * DomainExpertise3","relatedTech":[{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}}],"keywords":"kw1, kw2, kw3, kw4, kw5, kw6, kw7, kw8, kw9, kw10, kw11, kw12, kw13, kw14, kw15, kw16, kw17, kw18","technologies":{{"mustHave":["tool1","tool2","tool3","tool4","tool5","tool6","tool7"],"niceToHave":["tool1","tool2","tool3","tool4","tool5","tool6"],"additional":["tool1","tool2","tool3","tool4","tool5","tool6"]}},"architectures":[{{"name":"Pattern Name 1","description":"How you applied this pattern with concrete tech and outcome metric."}},{{"name":"Pattern Name 2","description":"How you applied this pattern with concrete tech and outcome metric."}},{{"name":"Pattern Name 3","description":"How you applied this pattern with concrete tech and outcome metric."}}]}}\""""
    return system, user


# -- JSON extraction with truncation recovery ----------------------------------
def extract_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object in model response")

    end = raw.rfind("}")
    if end == -1:
        raw = _repair_truncated_json(raw[start:])
        j = raw
    else:
        j = raw[start:end + 1]

    j = re.sub(r",\s*([}\]])", r"\1", j)
    try:
        return json.loads(j)
    except json.JSONDecodeError:
        j = _repair_truncated_json(j)
        j = re.sub(r'[\x00-\x1f\x7f]', ' ', j)
        j = re.sub(r',\s*([}\]])', r'\1', j)
        return json.loads(j)


def _repair_truncated_json(j: str) -> str:
    j = j.rstrip()
    j = re.sub(r",\s*$", "", j)

    in_string   = False
    escape_next = False
    opens       = []

    for ch in j:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            opens.append(ch)
        elif ch in ('}', ']'):
            if opens:
                opens.pop()

    if in_string:
        j += '"'

    j = re.sub(r",\s*$", "", j.rstrip())

    for ch in reversed(opens):
        j += '}' if ch == '{' else ']'

    return j


# -- Universal CV sanitiser ----------------------------------------------------
def _to_str(val, fallback: str = "") -> str:
    if val is None:
        return fallback
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return ", ".join(_to_str(v) for v in val if v is not None and v != "")
    if isinstance(val, dict):
        for k in ("text", "value", "content", "name", "title"):
            if k in val:
                return _to_str(val[k])
        return str(val)
    return str(val).strip()


def _to_str_list(val, fallback=None) -> list:
    if fallback is None:
        fallback = []
    if val is None:
        return fallback
    if isinstance(val, str):
        return [val.strip()] if val.strip() else fallback
    if isinstance(val, list):
        result = []
        for item in val:
            if isinstance(item, list):
                result.extend(_to_str(i) for i in item if i is not None and str(i).strip())
            elif isinstance(item, dict):
                result.append(_to_str(item))
            elif item is not None and str(item).strip():
                result.append(str(item).strip())
        return result if result else fallback
    return [_to_str(val)] if val else fallback


def sanitise_cv(cv: dict) -> dict:
    if not isinstance(cv, dict):
        return {}

    for field in ("totalYears", "title", "summary", "competencies", "keywords"):
        cv[field] = _to_str(cv.get(field), "")

    # Normalize title - remove any duplicated words (e.g. "Senior Senior Angular Dev")
    if cv.get("title"):
        cv["title"] = _normalize_job_title(cv["title"])

    # Always override totalYears with the real calculated value
    # Note: years_exp is passed via final_polish; sanitise_cv keeps the AI value here
    pass  # totalYears override happens in final_polish with the correct years_exp

    companies = cv.get("companies")
    if not isinstance(companies, list):
        companies = []
    clean_companies = []
    for co in companies:
        if not isinstance(co, dict):
            continue
        raw_bullets = co.get("bullets") or []
        if not isinstance(raw_bullets, list):
            raw_bullets = [_to_str(raw_bullets)]
        bullets = [_to_str(b) for b in raw_bullets if b and _to_str(b)]

        tech = co.get("tech") or co.get("technologies") or co.get("stack") or ""
        if isinstance(tech, list):
            tech = " | ".join(_to_str(t) for t in tech if t)
        else:
            tech = _to_str(tech)

        clean_companies.append({
            "company":   _to_str(co.get("company") or co.get("name") or ""),
            "role":      _to_str(co.get("role") or co.get("title") or ""),
            "dateRange": _to_str(co.get("dateRange") or co.get("dates") or co.get("date") or ""),
            "bullets":   bullets,
            "tech":      tech,
        })
    cv["companies"] = clean_companies

    if not isinstance(cv.get("skills"), list):
        cv["skills"] = []

    projects = cv.get("projects")
    if not isinstance(projects, list):
        projects = []

    # -- Generic project name detection ----------------------------------------
    _GENERIC_PROJECT_PATTERNS = re.compile(
        r'^(azure app|angular application|react (app|application|dashboard)|'
        r'web (platform|application|app)|mobile app|laravel (app|application)|'
        r'python (app|pipeline|application)|node\.?js app|django app|'
        r'erp (platform|system)|social media platform|e-?commerce platform|'
        r'project management platform|business intelligence platform)$',
        re.IGNORECASE
    )

    clean_projects = []
    for p in projects:
        if isinstance(p, dict):
            raw_overview = _to_str(p.get("overview") or p.get("desc") or p.get("description") or "")
            raw_bullets  = p.get("bullets") or []
            proj_bullets = _to_str_list(raw_bullets, [])

            overview = raw_overview
            if not proj_bullets and len(raw_overview) > 80:
                import re as _re
                sentences = _re.split(r'(?<=[.!?])\s+', raw_overview.strip())
                if len(sentences) >= 2:
                    overview      = " ".join(sentences[:2])
                    extra_bullets = [s for s in sentences[2:] if len(s) > 20]
                    proj_bullets  = extra_bullets[:3] if extra_bullets else proj_bullets

            # Extract techTags from project
            tech_tags = p.get("techTags", [])
            if not tech_tags and p.get("tech"):
                tech_tags = p.get("tech")
            if isinstance(tech_tags, str):
                tech_tags = [t.strip() for t in re.split(r'[|,]', tech_tags) if t.strip()]

            proj_name = _to_str(p.get("name") or p.get("title") or "")
            # Normalize separator: "PREFIX - Description" → "PREFIX: Description"
            if ":" not in proj_name and " - " in proj_name:
                _dash_parts = proj_name.split(" - ", 1)
                if len(_dash_parts) == 2 and len(_dash_parts[0].split()) <= 5:
                    proj_name = _dash_parts[0].strip() + ": " + _dash_parts[1].strip()
            # Strip any real company name from project name if it slips through
            # (project names must be coined product names, not generic labels)
            bare_name = re.sub(r'\s*\[.*?\]', '', proj_name).split('-')[0].strip()
            if _GENERIC_PROJECT_PATTERNS.match(bare_name):
                # Flag it but still include - the prompt should prevent this
                proj_name = proj_name  # Keep as-is; log warning could go here

            # Extract systemType (dynamically generated by AI from JD)
            system_type = _to_str(p.get("systemType") or p.get("system_type") or p.get("type") or "")

            clean_projects.append({
                "name":       proj_name,
                "systemType": system_type,
                "overview":   overview,
                "bullets":    proj_bullets,
                "desc":       overview,
                "techTags":   tech_tags[:7] if tech_tags else [],
            })
        elif isinstance(p, str):
            clean_projects.append({"name": p, "overview": "", "bullets": [], "desc": ""})
    cv["projects"] = clean_projects[:4]

    # -- Education block ---------------------------------------------------------
    edu_raw = cv.get("education")
    if isinstance(edu_raw, dict):
        cv["education"] = {
            "university":  _to_str(edu_raw.get("university",  "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY")),
            "degree":      _to_str(edu_raw.get("degree",      "Bachelor of Science in Computer Science (BSCS)")),
            "cgpa":        _to_str(edu_raw.get("cgpa",        "3.97/4.0")),
            "years":       _to_str(edu_raw.get("years",       "2017 - 2021")),
            "achievement": _to_str(edu_raw.get("achievement", "Gold Medalist for Academic Excellence")),
        }
    elif not isinstance(cv.get("education"), dict):
        cv["education"] = {
            "university":  "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
            "degree":      "Bachelor of Science in Computer Science (BSCS)",
            "cgpa":        "3.97/4.0",
            "years":       "2017 - 2021",
            "achievement": "Gold Medalist for Academic Excellence",
        }

    related = cv.get("relatedTech")
    if not isinstance(related, list):
        related = []
    clean_related = []
    for cat in related:
        if isinstance(cat, dict):
            items_raw = (cat.get("items") or cat.get("technologies") or
                         cat.get("tools") or cat.get("values") or [])
            clean_related.append({
                "category": _to_str(cat.get("category") or cat.get("name") or ""),
                "items":    _to_str_list(items_raw, []),
            })
        elif isinstance(cat, list) and len(cat) >= 2:
            clean_related.append({
                "category": _to_str(cat[0]),
                "items":    _to_str_list(cat[1:], []),
            })
    # Deduplicate items within and across relatedTech boxes
    global_rt = set()
    for cat in clean_related:
        unique = []
        for it in cat.get("items", []):
            k = str(it).lower().strip()
            if k and k not in global_rt:
                global_rt.add(k)
                unique.append(it)
        cat["items"] = unique[:13]
    cv["relatedTech"] = [c for c in clean_related if c.get("items")][:6]

    # -- Technologies block (mustHave / niceToHave / additional) ----------------
    tech_raw = cv.get("technologies")
    if isinstance(tech_raw, dict):
        _seen_tech = set()
        def _dedup(lst):
            out = []
            for t in _to_str_list(lst, []):
                k = str(t).lower().strip()
                if k and k not in _seen_tech:
                    _seen_tech.add(k)
                    out.append(t)
            return out
        cv["technologies"] = {
            "mustHave":   _dedup(tech_raw.get("mustHave")   or tech_raw.get("must_have")   or []),
            "niceToHave": _dedup(tech_raw.get("niceToHave") or tech_raw.get("nice_to_have") or []),
            "additional": _dedup(tech_raw.get("additional") or tech_raw.get("other")        or []),
        }
    elif not isinstance(cv.get("technologies"), dict):
        cv["technologies"] = {"mustHave": [], "niceToHave": [], "additional": []}

    # -- Architectures block ----------------------------------------------------
    arch_raw = cv.get("architectures")
    clean_arch = []
    if isinstance(arch_raw, list):
        for a in arch_raw:
            if isinstance(a, dict):
                name = _to_str(a.get("name") or a.get("pattern") or a.get("title") or "")
                desc = _to_str(a.get("description") or a.get("desc") or a.get("detail") or "")
                if name:
                    clean_arch.append({"name": name, "description": desc})
            elif isinstance(a, str) and a.strip():
                # Legacy plain-string fallback: promote to object with empty description
                clean_arch.append({"name": a.strip(), "description": ""})
    elif isinstance(arch_raw, str) and arch_raw.strip():
        for s in re.split(r"[,*|]", arch_raw):
            if s.strip():
                clean_arch.append({"name": s.strip(), "description": ""})
    # Clamp to 3-5 items
    cv["architectures"] = clean_arch[:5] if len(clean_arch) >= 3 else clean_arch

    return cv


def _is_real_tech(token: str) -> bool:
    """Return False if the token looks like a common English word rather than a real tool/technology."""
    token = token.strip()
    if len(token) < 2:
        return False
    # Blocklist of non-tech English words that slip through
    # NOTE: "web", "hands", "good", "ability", "strong", "dev", "net" (alone) are the
    # classic thin-JD hallucinations - the AI picks up plain-English prose words as tech tags.
    FAKE_TECH = {
        # Thin-JD culprits - single plain-English words that appear in bullet-point JDs
        "web", "hands", "good", "ability", "strong", "dev", "remote", "setup",
        "mindset", "detail", "attention", "focus", "solid", "working", "independent",
        "effectively", "efficiently", "proficiency", "familiarity",
        # Resumes / HR words
        "english", "resumes", "resume", "manager", "director", "applicants", "applicant",
        "position", "overview", "key", "responsibilities", "responsibility", "role", "roles",
        # Action verbs (the main culprits: "Write", "Troubleshoot", "Test", etc.)
        "optimize", "optimise", "implement", "conduct", "manage", "determine", "develop",
        "create", "build", "deliver", "provide", "support", "maintain", "ensure", "execute",
        "write", "writing", "read", "reading", "test", "testing", "debug", "debugging",
        "troubleshoot", "troubleshooting", "diagnose", "diagnoses", "deploy", "deploying",
        "design", "designing", "architect", "architecting", "configure", "configuring",
        "install", "installing", "operate", "operating", "run", "running", "use", "using",
        "apply", "applying", "review", "reviewing", "audit", "auditing", "document",
        "documenting", "report", "reporting", "monitor", "monitoring", "track", "tracking",
        "analyze", "analyse", "analyzing", "analysing", "integrate", "integrating",
        "migrate", "migrating", "scale", "scaling", "secure", "securing", "automate",
        "automating", "orchestrate", "orchestrating", "plan", "planning", "lead", "leading",
        "collaborate", "collaborating", "coordinate", "coordinating", "communicate",
        # Requirement / spec words
        "requirements", "requirement", "specifications", "specification", "criteria",
        "standards", "standard", "guidelines", "guideline", "procedures", "procedure",
        "protocols", "protocol", "policies", "policy",
        # Seniority / title words
        "senior", "junior", "lead", "principal", "staff", "associate", "specialist",
        "engineer", "developer", "programmer", "analyst", "architect",
        # Generic tech-adjacent nouns that are NOT tool names
        "skills", "tools", "technologies", "experience", "knowledge", "understanding",
        "ability", "proficiency", "expertise", "background", "strong", "excellent", "good",
        "required", "preferred", "must", "nice", "have", "with", "and", "or",
        "team", "work", "working", "communication", "written", "verbal",
        "problem", "solving", "analytical", "detail", "oriented", "self", "motivated",
        "deadline", "driven", "results", "client", "customers", "business", "company",
        "organization", "department", "projects", "project", "tasks", "task",
        # Infrastructure generic words (not product names)
        "server", "servers", "network", "networks", "system", "systems", "platform",
        "platforms", "service", "services", "solution", "solutions", "application",
        "applications", "module", "modules", "component", "components", "interface",
        "interfaces", "integration", "integrations", "environment", "environments",
        "infrastructure", "architecture", "framework", "frameworks", "library", "libraries",
        # Net/NET alone is not a technology name (.NET is caught separately via dot prefix)
        "net",
    }
    # Additional pattern-based checks
    t_lower = token.lower()
    if t_lower in FAKE_TECH:
        return False
    # Reject pure verbs ending in common verb suffixes (catch "Troubleshoot", "Configure" etc.)
    verb_suffixes = ("ise", "ize", "ify", "ate", "ect", "oot", "ure")
    if len(token) > 6 and t_lower.endswith(verb_suffixes) and token[0].isupper():
        # Allow known real tech names that happen to end in these (e.g. "Hibernate", "Normalize")
        REAL_TECH_EXCEPTIONS = {
            "hibernate", "normalize", "virtualise", "virtualise", "containerize",
            "accelerate",  # used rarely but legitimately
        }
        if t_lower not in REAL_TECH_EXCEPTIONS:
            return False
    return True


def _sanitize_tech_string(tech: str) -> str:
    """Remove non-technology words from a pipe/comma-separated tech string."""
    if not tech:
        return ""
    sep = "|" if "|" in tech else ","
    parts = [t.strip() for t in tech.split(sep) if t.strip()]
    real = [t for t in parts if _is_real_tech(t)]
    return " | ".join(real) if real else ""


def _sanitize_skills_list(skills: list) -> list:
    """
    Remove fake-tech items from skills category strings.
    Each item in `skills` is expected to be: "Category Name: item1, item2, item3, ..."
    Rules applied:
      - Split on comma; strip each token; reject tokens that fail _is_real_tech
      - Also reject any token that is a single generic English word (first-letter-cap check)
      - Keep the row only if >=3 real items remain (rows with fewer are dropped)
    """
    cleaned = []
    for row in skills:
        if not row:
            continue
        if ":" not in row:
            # No colon -> treat the whole row as a category name without items; keep as-is
            cleaned.append(row)
            continue
        colon = row.index(":")
        cat = row[:colon].strip()
        items_raw = row[colon + 1:].strip()
        # Split on comma (skills are always comma-separated inside each category)
        items = [t.strip() for t in re.split(r"[,|;?·•]", items_raw) if t.strip()]
        real_items = []
        for t in items:
            if not _is_real_tech(t):
                continue
            # Extra guard: reject single-word all-caps-first tokens that look like verbs
            # (e.g. "Write", "Troubleshoot", "Requirements")
            # Real tech names are usually: acronyms (SQL, AWS), mixed-case (PostgreSQL),
            # or multi-word ("Azure DevOps", "Entity Framework")
            # Allowlist: real named tools whose names happen to end in a bad suffix.
            # These must never be rejected by the bad-endings heuristic.
            _REAL_TECH_SAFE_WORDS = {
                "firebase", "confluence", "lighthouse", "clearscope", "semrush",
                "discourse", "hibernate", "normalize", "virtualise", "containerize",
                "accelerate", "surfer", "majestic", "moz", "compose",
                "frase", "sitebulb", "screaming", "ubersuggest", "kwfinder",
                "serpstat", "spyfu", "clearance", "airtable", "base",
                "stripe", "combine", "coverage", "distance", "interface",
                "presence", "instance", "variance", "response", "sequence",
                "closure", "exposure", "dispose", "azure", "vercel",
                "grafana", "rance", "confluence",
            }
            words = t.split()
            if len(words) == 1:
                w = words[0]
                # Allow all-caps acronyms (SQL, AWS, HTML, CSS, PHP, etc.)
                if w.isupper():
                    real_items.append(t)
                    continue
                # Always allow known real tool names regardless of suffix
                if w.lower() in _REAL_TECH_SAFE_WORDS:
                    real_items.append(t)
                    continue
                # Allow known camelCase / PascalCase tech names (React, Vue, Django, etc.)
                # by checking: starts with uppercase but contains lowercase -> likely a proper noun/tech name
                # Reject if it looks like a plain English verb or noun
                if w[0].isupper() and any(c.islower() for c in w[1:]):
                    # Additional verb pattern rejection: common -ing, -ed, -ment, -tion, -ment endings
                    _bad_endings = (
                        "ing", "ed", "ment", "tion", "sion", "ure", "oot",
                        "ise", "ize", "ify", "age", "ance", "ence",
                    )
                    if w.lower().endswith(_bad_endings):
                        continue  # skip - looks like a verb/gerund/noun
                    real_items.append(t)
                elif w[0].isupper() and w[1:].isupper():
                    # All-caps after first letter -> acronym variant (e.g. "ASP.NET" handled separately)
                    real_items.append(t)
                else:
                    real_items.append(t)
            else:
                # Multi-word items (e.g. "Azure DevOps", "Entity Framework") - keep if not all-fake
                real_items.append(t)
        if len(real_items) >= 3:
            cleaned.append(f"{cat}: {', '.join(real_items)}")
        # Rows with fewer than 3 real items are dropped entirely - do NOT keep bad originals
    return cleaned



# ══════════════════════════════════════════════════════════════════════════════
# DEDICATED TECHNICAL SKILLS EXTRACTION — Second LLM request
# Runs AFTER the full CV is generated. The full CV JSON is shown to the model
# so it knows exactly what technologies are already being used, and can output
# a perfectly consistent, non-duplicate skills section.
# ══════════════════════════════════════════════════════════════════════════════

def build_dedicated_skills_prompt(req: CVRequest, cv: dict, techs: dict) -> tuple:
    """
    Build a standalone second-request prompt that extracts Technical Skills.
    Input: the full generated CV + the raw JD + the extracted tech list.
    Output: {"skills": ["Category: item1, item2, ...", ...]}
    
    This guarantees:
    - Every technology explicitly named in the JD appears somewhere in the skills
    - Sub-technologies are strictly aligned to their parent category
    - No ecosystem mixing (no React in a .NET CV unless JD says so)
    - Each category has 10-13 items from ONE coherent domain
    - Zero items repeated across categories
    """
    jd = req.job_description.strip()[:1400]
    job_title = req.job_title.strip()

    # Build the allowed tech list from extracted techs + CV companies tech tags
    core      = techs.get("core",      techs.get("mustHave",   []))
    preferred = techs.get("preferred", techs.get("niceToHave", []))
    ecosystem = techs.get("ecosystem", techs.get("additional", []))
    
    all_allowed = list(dict.fromkeys(core + preferred + ecosystem))[:50]
    
    # Also add any tech tags from the CV companies (they're already validated)
    company_techs = set()
    for co in cv.get("companies", []):
        tech_str = co.get("tech", "")
        if tech_str:
            for t in re.split(r"[|,]", tech_str):
                t = t.strip()
                if t and _is_real_tech(t):
                    company_techs.add(t)
    
    # Merge: JD techs first, then company techs
    for t in company_techs:
        if t not in all_allowed:
            all_allowed.append(t)
    
    allowed_str = ", ".join(all_allowed[:60]) if all_allowed else "technologies from the JD"
    
    # Extract what's already used in experience bullets for context
    used_in_bullets = []
    for co in cv.get("companies", []):
        for b in co.get("bullets", []):
            # Extract capitalized words that look like tech names from bullets
            words = re.findall(r'\b([A-Z][a-zA-Z0-9#\.\+]+)\b', b)
            for w in words:
                if len(w) > 2 and w not in {"The", "A", "An", "In", "By", "To"}:
                    used_in_bullets.append(w)
    
    used_in_bullets_str = ", ".join(list(dict.fromkeys(used_in_bullets))[:20]) if used_in_bullets else "(see JD)"
    
    # Detect role domain for category hints
    title_lower = job_title.lower()
    jd_lower    = jd.lower()
    
    if ".net" in title_lower or "c#" in title_lower or "asp" in title_lower:
        domain       = "DOTNET"
        cat_guidance = (
            "Category 1 (Backend & API): ASP.NET Core, C#, .NET 8, Entity Framework Core, Dapper, MediatR, SignalR, REST APIs, gRPC, Swagger/OpenAPI\n"
            "Category 2 (Database & Storage): SQL Server, PostgreSQL, Redis, T-SQL, Azure SQL, Elasticsearch, Entity Framework Migrations, Dapper, PgBouncer, Azure Cache for Redis\n"
            "Category 3 (Azure Cloud Services): Azure App Service, Azure Functions, Azure DevOps, Azure Service Bus, Azure Blob Storage, Azure AD, Azure Key Vault, Azure Monitor, Azure CDN, Azure Container Registry\n"
            "Category 4 (DevOps & CI/CD): Docker, Kubernetes, Helm, GitHub Actions, Azure Pipelines, Terraform, Git, SonarQube, ArgoCD, Dockerfile\n"
            "Category 5 (Testing & Quality): xUnit, NUnit, MSTest, Moq, Postman, Playwright, SonarQube, OWASP ZAP, Serilog, OpenTelemetry"
        )
        banned_from_skills = "React, Vue.js, Next.js, Node.js, Express.js, Django, Flask, MongoDB, DynamoDB (unless in JD)"
    elif "angular" in title_lower or "angular" in jd_lower[:300]:
        domain       = "ANGULAR"
        cat_guidance = (
            "Category 1 (Angular & Frontend): Angular 17, TypeScript, RxJS, NgRx, Angular Material, Angular Router, Angular Forms, Angular CLI, Angular CDK, Standalone Components\n"
            "Category 2 (UI & Styling): HTML5, CSS3, SCSS, Tailwind CSS, Bootstrap, Storybook, Figma, Webpack, Vite, Nx Monorepo\n"
            "Category 3 (API & Integration): REST APIs, GraphQL, Apollo Client, HTTP Client, JWT, OAuth 2.0, WebSockets, OpenAPI, Swagger, Axios\n"
            "Category 4 (Testing & Quality): Jasmine, Karma, Jest, Cypress, Playwright, ESLint, Prettier, SonarQube, TestBed, Protractor\n"
            "Category 5 (DevOps & Tooling): Git, GitHub Actions, Docker, npm, Yarn, Jenkins, GitLab CI, Webpack, Vercel, Firebase Hosting"
        )
        banned_from_skills = "React, Vue.js, Next.js, Redux, Spring Boot, Django, Laravel (unless in JD)"
    elif "react" in title_lower or "react" in jd_lower[:300]:
        domain       = "REACT"
        cat_guidance = (
            "Category 1 (React & Frontend): React 18, TypeScript, Redux Toolkit, React Query, React Router, React Hook Form, Zustand, Context API, Custom Hooks, Suspense\n"
            "Category 2 (UI & Styling): HTML5, CSS3, Tailwind CSS, SCSS, Material UI, shadcn/ui, Radix UI, Framer Motion, Storybook, Figma\n"
            "Category 3 (API & Backend Integration): REST APIs, GraphQL, Apollo Client, Axios, SWR, tRPC, JWT, OAuth 2.0, WebSockets, Next.js\n"
            "Category 4 (Testing & Quality): Jest, React Testing Library, Cypress, Playwright, ESLint, Prettier, Storybook, SonarQube, Vitest, MSW\n"
            "Category 5 (Build & DevOps): Vite, Webpack, npm, Yarn, GitHub Actions, Docker, Vercel, Netlify, Git, CI/CD"
        )
        banned_from_skills = "Angular, NgRx, Vue.js, Spring Boot, Django (unless in JD)"
    elif "node" in title_lower or "express" in title_lower:
        domain       = "NODE"
        cat_guidance = (
            "Category 1 (Node.js & Backend): Node.js, Express.js, NestJS, TypeScript, REST APIs, GraphQL, JWT, OAuth 2.0, Socket.IO, Fastify\n"
            "Category 2 (Database & ORM): PostgreSQL, MongoDB, MySQL, Redis, Mongoose, Prisma, Sequelize, TypeORM, DynamoDB, Elasticsearch\n"
            "Category 3 (API & Messaging): REST APIs, GraphQL, Apollo Server, gRPC, RabbitMQ, Kafka, AWS SQS, Swagger, OpenAPI, WebSockets\n"
            "Category 4 (Cloud & DevOps): AWS Lambda, AWS EC2, AWS S3, Docker, Kubernetes, GitHub Actions, Terraform, PM2, Nginx, Heroku\n"
            "Category 5 (Testing & Quality): Jest, Mocha, Supertest, Chai, ESLint, Prettier, SonarQube, Husky, Artillery, Postman"
        )
        banned_from_skills = "Angular, React, Vue.js, Spring Boot, Django (unless in JD)"
    elif "python" in title_lower or "django" in title_lower or "fastapi" in title_lower:
        domain       = "PYTHON"
        cat_guidance = (
            "Category 1 (Python & Backend): Python 3.x, Django, FastAPI, Flask, SQLAlchemy, Alembic, Pydantic, Celery, asyncio, Gunicorn\n"
            "Category 2 (Database & Storage): PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, DynamoDB, Cassandra, SQLite, PgBouncer, MinIO\n"
            "Category 3 (Cloud & Infrastructure): AWS Lambda, AWS EC2, AWS S3, GCP Cloud Run, Docker, Kubernetes, Terraform, GitHub Actions, Nginx, Ansible\n"
            "Category 4 (Data & Analytics): Pandas, NumPy, Jupyter, Matplotlib, Seaborn, Apache Airflow, dbt, PySpark, Scikit-learn, Plotly\n"
            "Category 5 (Testing & Quality): Pytest, Unittest, Mypy, Flake8, Black, Bandit, Hypothesis, Coverage.py, Locust, Postman"
        )
        banned_from_skills = "React, Angular, Vue.js, Spring Boot, Laravel (unless in JD)"
    elif "java" in title_lower and "javascript" not in title_lower:
        domain       = "JAVA"
        cat_guidance = (
            "Category 1 (Java & Spring): Java 17, Spring Boot, Spring MVC, Spring Security, Hibernate, JPA, Lombok, MapStruct, Jackson, Maven\n"
            "Category 2 (Database & Messaging): PostgreSQL, MySQL, MongoDB, Redis, Kafka, RabbitMQ, Elasticsearch, Oracle DB, Liquibase, HikariCP\n"
            "Category 3 (Cloud & DevOps): AWS EC2, AWS S3, AWS RDS, Docker, Kubernetes, Helm, GitHub Actions, Jenkins, Terraform, Ansible\n"
            "Category 4 (API & Integration): REST APIs, GraphQL, gRPC, JWT, OAuth 2.0, OpenAPI, Swagger, WebSockets, Spring Integration, Apache Camel\n"
            "Category 5 (Testing & Quality): JUnit 5, Mockito, TestContainers, Postman, SonarQube, Jacoco, Gatling, AssertJ, WireMock, Selenium"
        )
        banned_from_skills = "React, Angular, Vue.js, Node.js, Django (unless in JD)"
    elif "seo" in title_lower or "digital marketing" in title_lower or "marketing" in title_lower:
        domain       = "DIGITAL_MARKETING"
        cat_guidance = (
            "Category 1 (SEO & Analytics): Google Analytics 4, Google Search Console, SEMrush, Ahrefs, Moz, Screaming Frog, Sitebulb, Majestic, SurferSEO, Looker Studio\n"
            "Category 2 (Content & CMS): WordPress, Yoast SEO, Rank Math, Elementor, Contentful, HubSpot CMS, Webflow, WP Rocket, Clearscope, Frase\n"
            "Category 3 (Paid & Social): Google Ads, Facebook Ads Manager, LinkedIn Ads, TikTok Ads, Microsoft Ads, Google Tag Manager, Pixel, Hootsuite, Buffer, Sprout Social\n"
            "Category 4 (Email & CRM): Mailchimp, HubSpot, Klaviyo, ActiveCampaign, Salesforce, Pardot, Marketo, Constant Contact, SendGrid, Drip\n"
            "Category 5 (Reporting & BI): Google Data Studio, Looker Studio, Tableau, Power BI, Supermetrics, DataBox, Hotjar, Crazy Egg, Heap, Mixpanel"
        )
        banned_from_skills = "Docker, Kubernetes, React, Angular, Spring Boot, Django (unless in JD)"
    else:
        domain       = "FULLSTACK"
        cat_guidance = (
            "Category 1 (Backend & API): Pick the primary backend tech from JD - add its framework, ORM, auth lib, test framework\n"
            "Category 2 (Frontend & UI): Pick the primary frontend tech from JD - add its state manager, UI lib, CSS framework, build tool\n"
            "Category 3 (Database & Storage): Pick databases from JD - add migration tool, caching layer, search engine\n"
            "Category 4 (Cloud & DevOps): Pick cloud/DevOps tools from JD - add CI/CD, containers, monitoring\n"
            "Category 5 (Testing & Quality): Pick testing tools from JD - add quality gates, mocking, API testing"
        )
        banned_from_skills = "Tools from completely unrelated ecosystems"

    system = (
        "You are a senior technical CV writer and technology classifier. Output ONLY valid JSON. "
        "No explanations, no markdown, no backticks. Start { end }.\n\n"
        
        "YOUR SOLE TASK: Generate the TECHNICAL SKILLS section for a CV.\n\n"
        
        f"=== DETECTED ECOSYSTEM: {domain} ===\n"
        f"BANNED from skills (ecosystem gate): {banned_from_skills}\n\n"
        
        "=== CRITICAL RULES ===\n"
        "1. EXACTLY 5 categories.\n"
        "2. Each category: EXACTLY 10-13 items. Fewer than 10 = HARD FAILURE.\n"
        "3. ZERO items repeated across any two categories.\n"
        "4. Each category name must precisely describe its ACTUAL contents.\n"
        "   - 'Backend & API' → ONLY server-side frameworks, APIs, ORMs, middleware\n"
        "   - 'Frontend & UI' → ONLY browser-side frameworks, CSS, state managers\n"
        "   - 'Database & Storage' → ONLY databases, caches, search engines\n"
        "   - 'Cloud Services' → ONLY cloud platform services (not Docker/K8s)\n"
        "   - 'DevOps & CI/CD' → ONLY containers, pipelines, IaC\n"
        "   - 'Testing & Quality' → ONLY test frameworks, mocking, quality tools\n"
        "5. SUB-TECHNOLOGY ALIGNMENT: Every item in a category MUST belong to that category's domain.\n"
        "   HARD FAILURE examples:\n"
        "   - React in 'Database & Storage' category\n"
        "   - PostgreSQL in 'Frontend & UI' category\n"
        "   - Docker in 'Backend & API' category\n"
        "6. STACK CONSISTENCY: Never mix competing ecosystems.\n"
        "   - If detected ecosystem is DOTNET → no React, Vue, Django\n"
        "   - If detected ecosystem is ANGULAR → no React, Next.js, Redux\n"
        "   - If detected ecosystem is REACT → no Angular, NgRx\n"
        "7. JD PRIORITY: Technologies EXPLICITLY named in the JD MUST appear.\n"
        "   Do not drop any technology from the JD. It must be placed in the correct category.\n"
        "8. Every item must be a real named tool (e.g. 'ASP.NET Core', 'PostgreSQL', 'GitHub Actions').\n"
        "   NEVER use: verbs (Deploy, Build, Test), nouns (Platform, System), adjectives (Good, Strong)\n\n"
        
        f"=== CATEGORY GUIDANCE FOR {domain} ECOSYSTEM ===\n"
        f"{cat_guidance}\n\n"
        
        "=== MANDATORY JD KEYWORDS — EVERY SINGLE ONE MUST APPEAR IN OUTPUT ===\n"
        "The following technologies come directly from the job description.\n"
        "EVERY item in this list MUST appear in exactly one skill category.\n"
        "Skipping even one of these is a HARD FAILURE.\n"
        f"MANDATORY LIST: {', '.join(core[:40]) if core else 'see JD'}\n\n"

        "=== PREFERRED / NICE-TO-HAVE FROM JD — ALL MUST ALSO APPEAR ===\n"
        "These are explicitly listed as preferred, nice-to-have, good-to-have, or bonus.\n"
        "They are NOT optional — include ALL of them in the skills section.\n"
        f"PREFERRED LIST: {', '.join(preferred[:30]) if preferred else '(none listed separately)'}\n\n"

        "=== ECOSYSTEM COMPANIONS (standard tools for this stack) ===\n"
        "Add closely related tools to fill each category to 10-13 items.\n"
        "Only add companions that STRICTLY belong to the same ecosystem.\n"
        f"Ecosystem pool: {', '.join(ecosystem[:40]) if ecosystem else '(derive from JD)'}\n\n"

        "Technologies already in experience bullets (for reference — include in skills too): "
        f"{used_in_bullets_str}\n\n"

        "COMPLETENESS SELF-CHECK (run before outputting):\n"
        "1. Count items in MANDATORY LIST above.\n"
        "2. Confirm each one appears in your output.\n"
        "3. Count items in PREFERRED LIST above.\n"
        "4. Confirm each one appears in your output.\n"
        "5. If ANY mandatory or preferred item is missing, add it now before outputting.\n\n"

        "OUTPUT FORMAT:\n"
        '{"skills": [\n'
        '  "Category Name: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7, Tool8, Tool9, Tool10",\n'
        '  "Category Name: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7, Tool8, Tool9, Tool10",\n'
        '  "Category Name: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7, Tool8, Tool9, Tool10",\n'
        '  "Category Name: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7, Tool8, Tool9, Tool10",\n'
        '  "Category Name: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7, Tool8, Tool9, Tool10"\n'
        "]}"
    )

    # Build a short mandatory keywords string for the user prompt
    _mandatory_str = ", ".join(list(dict.fromkeys(core + preferred))[:50]) if (core or preferred) else allowed_str

    user = (
        f"Job Title: {job_title}\n\n"
        f"FULL JOB DESCRIPTION (read every word — extract ALL tools):\n{jd}\n\n"
        f"Detected Ecosystem: {domain}\n\n"
        f"MANDATORY KEYWORDS (from JD required + preferred sections — ALL must be in output):\n"
        f"{_mandatory_str}\n\n"
        f"Full technology pool (mandatory + ecosystem companions):\n{allowed_str}\n\n"
        "TASK: Generate EXACTLY 5 skill categories (10-13 items each).\n"
        "REQUIREMENTS:\n"
        "1. Every item in the MANDATORY KEYWORDS list MUST appear in a category.\n"
        "2. Every item in the PREFERRED list MUST appear in a category.\n"
        "3. Sub-technologies MUST align to their parent category's domain.\n"
        "4. Zero items repeated across categories.\n"
        "5. Minimum 10 items per category — HARD REQUIREMENT.\n"
        "6. Run the COMPLETENESS SELF-CHECK before outputting.\n\n"
        "Output JSON only:"
    )

    return system, user


async def extract_dedicated_skills(
    client: httpx.AsyncClient,
    key: str,
    model: str,
    url: str,
    headers: dict,
    req: CVRequest,
    cv: dict,
    techs: dict,
    max_tokens: int = 1800,
    provider: str = "gemini",
    _deadline: float = 0.0
) -> list:
    """
    Make a dedicated second LLM request solely to extract Technical Skills.
    Returns a list of skill strings: ["Category: item1, item2, ...", ...]
    Console-logs every step so issues are visible in the server terminal.
    """
    import time as _t

    print(f"\n{'='*60}")
    print(f"[SKILLS-EXTRACT] Starting dedicated skills extraction")
    print(f"[SKILLS-EXTRACT] Job Title: {req.job_title}")
    print(f"[SKILLS-EXTRACT] Provider: {provider} | Model: {model}")
    
    # Log what we have BEFORE the dedicated call
    existing_skills = cv.get("skills", [])
    print(f"[SKILLS-EXTRACT] Skills from main CV: {len(existing_skills)} categories")
    for i, s in enumerate(existing_skills):
        colon = s.find(":")
        if colon > 0:
            cat   = s[:colon].strip()
            items = [t.strip() for t in s[colon+1:].split(",") if t.strip()]
            print(f"  [{i+1}] {cat}: {len(items)} items → {items[:5]}{'...' if len(items)>5 else ''}")
        else:
            print(f"  [{i+1}] (no colon): {s[:80]}")
    
    # Log technology pool available
    core      = techs.get("core",      techs.get("mustHave",   []))
    preferred = techs.get("preferred", techs.get("niceToHave", []))
    ecosystem = techs.get("ecosystem", techs.get("additional", []))
    print(f"[SKILLS-EXTRACT] Tech pool — core:{len(core)} preferred:{len(preferred)} ecosystem:{len(ecosystem)}")
    print(f"[SKILLS-EXTRACT] Core techs: {core[:10]}")
    print(f"[SKILLS-EXTRACT] Preferred:  {preferred[:10]}")
    print(f"[SKILLS-EXTRACT] Ecosystem:  {ecosystem[:10]}")

    sys_p, usr_p = build_dedicated_skills_prompt(req, cv, techs)

    print(f"[SKILLS-EXTRACT] Sending dedicated skills request (max_tokens={max_tokens})")
    t0 = _t.time()

    # Compute per-call timeout respecting deadline
    _skills_timeout = 60.0
    if _deadline:
        remaining = _deadline - _t.time()
        if remaining < 10:
            print(f"[SKILLS-EXTRACT] Skipping — only {remaining:.0f}s left on deadline")
            return existing_skills
        _skills_timeout = min(_skills_timeout, remaining - 5)
    
    try:
        if provider == "gemini":
            # Gemini uses a different API structure
            payload = {
                "contents": [{"parts": [{"text": sys_p + "\n\n" + usr_p}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json"
                }
            }
            r = await client.post(url, headers=headers, json=payload, timeout=_skills_timeout)
        else:
            # OpenAI-compatible (Groq, Cerebras, DeepSeek, OpenAI)
            r = await client.post(
                url,
                headers=headers,
                json={
                    "model":    model,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user",   "content": usr_p}
                    ],
                    "temperature": 0.1,
                    "max_tokens":  max_tokens,
                },
                timeout=_skills_timeout
            )
        
        elapsed = _t.time() - t0
        print(f"[SKILLS-EXTRACT] Response received in {elapsed:.1f}s — HTTP {r.status_code}")
        
        if r.status_code != 200:
            print(f"[SKILLS-EXTRACT] ERROR: HTTP {r.status_code} — {r.text[:300]}")
            print(f"[SKILLS-EXTRACT] Falling back to existing CV skills")
            return existing_skills
        
        # Parse response
        try:
            resp_json = r.json()
            if provider == "gemini":
                raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raw_text = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            print(f"[SKILLS-EXTRACT] Parse error extracting text: {e}")
            return existing_skills
        
        raw_text = raw_text.strip()
        raw_text = re.sub(r"```json\s*", "", raw_text)
        raw_text = re.sub(r"```\s*$", "", raw_text)
        
        start = raw_text.find("{")
        end   = raw_text.rfind("}")
        if start == -1 or end == -1:
            print(f"[SKILLS-EXTRACT] No JSON found in response: {raw_text[:200]}")
            return existing_skills
        
        raw_text = raw_text[start:end+1]
        raw_text = re.sub(r",\s*}", "}", raw_text)
        raw_text = re.sub(r",\s*]", "]", raw_text)
        
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"[SKILLS-EXTRACT] JSON decode error: {e}")
            print(f"[SKILLS-EXTRACT] Raw (first 500): {raw_text[:500]}")
            return existing_skills
        
        new_skills = result.get("skills", [])
        print(f"[SKILLS-EXTRACT] Dedicated skills extracted: {len(new_skills)} categories")
        
        if not new_skills or not isinstance(new_skills, list):
            print(f"[SKILLS-EXTRACT] Empty/invalid skills returned — keeping existing")
            return existing_skills
        
        # Validate and log the new skills
        valid_skills = []
        for i, s in enumerate(new_skills):
            if not isinstance(s, str):
                print(f"[SKILLS-EXTRACT]   [{i+1}] SKIP — not a string: {type(s)}")
                continue
            colon = s.find(":")
            if colon <= 0:
                print(f"[SKILLS-EXTRACT]   [{i+1}] SKIP — no colon: {s[:80]}")
                continue
            cat   = s[:colon].strip()
            items = [t.strip() for t in s[colon+1:].split(",") if t.strip()]
            n     = len(items)
            print(f"[SKILLS-EXTRACT]   [{i+1}] {cat}: {n} items → {items[:5]}{'...' if n>5 else ''}")
            if n < 5:
                print(f"[SKILLS-EXTRACT]   [{i+1}] WARN — only {n} items (expected ≥10), keeping but flagging")
            valid_skills.append(s)
        
        if len(valid_skills) < 3:
            print(f"[SKILLS-EXTRACT] Too few valid categories ({len(valid_skills)}) — keeping existing")
            return existing_skills

        # ── Python-level mandatory keyword enforcement ─────────────────────
        # Even if the LLM missed some JD keywords, we catch and fix it here.
        mandatory = list(dict.fromkeys(core + preferred))  # required + nice-to-have
        skills_flat_lower = ", ".join(valid_skills).lower()

        missing_mandatory = []
        for t in mandatory:
            if len(t) > 2 and t.lower() not in skills_flat_lower:
                missing_mandatory.append(t)

        if missing_mandatory:
            print(f"[SKILLS-EXTRACT] ENFORCE — injecting {len(missing_mandatory)} missing JD keywords: {missing_mandatory}")
            # Find the largest category and append the missing tools there
            # (they'll be deduplicated by fix_skills later)
            best_idx  = 0
            best_len  = 0
            for idx, s in enumerate(valid_skills):
                colon = s.find(":")
                if colon > 0:
                    n = len([t.strip() for t in s[colon+1:].split(",") if t.strip()])
                    if n > best_len:
                        best_len = n
                        best_idx = idx
            # Append missing tools to the best category (up to 13 items per category)
            # Spread overflow into other categories
            overflow = []
            colon = valid_skills[best_idx].find(":")
            cat_name = valid_skills[best_idx][:colon].strip()
            existing_items = [t.strip() for t in valid_skills[best_idx][colon+1:].split(",") if t.strip()]
            space = max(0, 13 - len(existing_items))
            to_add_here = missing_mandatory[:space]
            overflow    = missing_mandatory[space:]
            if to_add_here:
                existing_items.extend(to_add_here)
                valid_skills[best_idx] = f"{cat_name}: {', '.join(existing_items)}"
            # Put overflow in the next largest category
            if overflow:
                for idx2, s2 in enumerate(valid_skills):
                    if idx2 == best_idx:
                        continue
                    colon2 = s2.find(":")
                    if colon2 > 0:
                        cat2   = s2[:colon2].strip()
                        items2 = [t.strip() for t in s2[colon2+1:].split(",") if t.strip()]
                        space2 = max(0, 13 - len(items2))
                        items2.extend(overflow[:space2])
                        overflow = overflow[space2:]
                        valid_skills[idx2] = f"{cat2}: {', '.join(items2)}"
                    if not overflow:
                        break
        else:
            print(f"[SKILLS-EXTRACT] ✓ All mandatory JD keywords present in skills")

        # Final check
        skills_flat_lower = ", ".join(valid_skills).lower()
        still_missing = [t for t in mandatory if len(t) > 2 and t.lower() not in skills_flat_lower]
        if still_missing:
            print(f"[SKILLS-EXTRACT] WARN — {len(still_missing)} still missing after enforcement: {still_missing}")
        else:
            print(f"[SKILLS-EXTRACT] ✓ All {len(mandatory)} mandatory JD keywords confirmed in output")

        print(f"[SKILLS-EXTRACT] ✓ Dedicated skills extraction complete — replacing CV skills")
        print(f"{'='*60}\n")
        return valid_skills
        
    except httpx.TimeoutException:
        print(f"[SKILLS-EXTRACT] TIMEOUT after {_t.time()-t0:.1f}s — keeping existing skills")
        return existing_skills
    except Exception as e:
        print(f"[SKILLS-EXTRACT] EXCEPTION: {type(e).__name__}: {e}")
        return existing_skills


def fix_companies(cv: dict) -> dict:
    """Fix placeholder names and enforce appropriate role titles based on real years."""
    companies = cv.get("companies", [])
    real_years_str = cv.get("totalYears", _calc_total_years())
    try:
        real_years = float(real_years_str.replace("+", "").strip())
    except Exception:
        real_years = 3.0

    # Seniority tiers - strictly based on number of companies present:
    # 1 company  (<=1 yr)  -> Junior only
    # 2 companies (<=2 yr)  -> Co1: plain (no prefix), Co2: Junior
    # 3 companies (3+ yr)  -> Co1: Senior, Co2: plain (no prefix), Co3: Junior
    num_cos = len(companies)
    if num_cos == 1:
        tier_labels = ["Junior"]
    elif num_cos == 2:
        tier_labels = ["", "Junior"]
    else:
        tier_labels = ["Senior", "", "Junior"]

    # Only use Architect if the JD explicitly calls for it
    cv_title_lower = (cv.get("title") or "").lower()
    jd_has_architect = "architect" in cv_title_lower

    for i, co in enumerate(companies):
        if i >= len(CANDIDATE_COMPANIES): break
        real = CANDIDATE_COMPANIES[i]

        # Fix placeholder company names
        name = (co.get("company") or co.get("name") or "").strip()
        if not name or re.match(r"(?i)(company\s*\d|placeholder|example)", name):
            co["company"]   = real["name"]
            co["dateRange"] = f"{real['start']} - {real['end']}"
        co.setdefault("company", real["name"])

        # Strip seniority words AND "Intern/Internship" from role
        role = co.get("role", "").strip()
        _strip_pat = r"(?i)\b(lead|principal|staff|senior|mid[\s\-]?level|junior|associate|graduate|entry[\s\-]?level|intern(?:ship)?)(/\w+)?\b\s*"
        domain = re.sub(_strip_pat, "", role).strip()
        domain = re.sub(r"\s+", " ", domain).strip()

        # Fall back to title-derived domain if empty or generic
        if not domain or re.match(r"(?i)^(role|engineer|developer|specialist)$", domain):
            raw_title = (cv.get("title") or "Software Engineer").split("|")[0].strip()
            domain = re.sub(_strip_pat, "", raw_title).strip()
            domain = re.sub(r"\s+", " ", domain).strip() or "Software Engineer"

        # Remove "Architect" from domain unless JD explicitly asks for it
        if not jd_has_architect:
            domain = re.sub(r"(?i)\s*\barchitect\b", " Engineer", domain).strip()
            domain = re.sub(r"\s+", " ", domain).strip()

        tier = tier_labels[min(i, len(tier_labels) - 1)]
        # Avoid double-prefix: "Senior Senior Engineer" -> just set the role cleanly
        # When tier is empty (plain, no prefix), just use the domain directly
        co["role"] = f"{tier} {domain}".strip() if tier else domain

    # Ensure all 3 roles are distinct
    if len(companies) >= 3:
        roles = [companies[j].get("role", "") for j in range(3)]
        if len(set(r.lower() for r in roles)) < len(roles):
            domain = re.sub(
                r"(?i)\b(lead|principal|staff|senior|mid[\s\-]?level|junior|associate)(/\w+)?\b\s*",
                "", roles[0]
            ).strip()
            domain = re.sub(r"\s+", " ", domain).strip() or "Software Engineer"
            for idx in range(min(3, len(companies))):
                t = tier_labels[idx]
                companies[idx]["role"] = f"{t} {domain}".strip() if t else domain

    return cv


def _clean_tech_string(tech: str) -> str:
    if not tech:
        return ""
    parts = re.split(r"[|,]", tech)
    seen  = set()
    clean = []
    for p in parts:
        p = p.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            clean.append(p)
    return " | ".join(clean)


def _clean_skill_row(row: str) -> str:
    colon = row.find(":")
    if colon <= 0:
        return row
    cat   = row[:colon].strip()
    val   = row[colon + 1:].strip()
    items = [p.strip() for p in re.split(r"[|,]", val) if p.strip()]
    seen  = set()
    unique = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return f"{cat}: {', '.join(unique)}"


def final_polish(cv: dict, years_exp: str = "") -> dict:
    # 1. Clean tech strings per company - remove fake/non-tech words
    for co in cv.get("companies", []):
        co["tech"] = _sanitize_tech_string(co.get("tech", ""))

    # 2. Clean + deduplicate skill rows - also remove fake tech words
    skills_before = cv.get("skills", [])
    print(f"[FINAL-POLISH] Skills before clean: {len(skills_before)} categories")
    cv["skills"] = [_clean_skill_row(s) for s in skills_before if s]
    after_clean = len(cv["skills"])
    cv["skills"] = _sanitize_skills_list(cv["skills"])
    after_sanitize = len(cv["skills"])
    print(f"[FINAL-POLISH] Skills after clean_row: {after_clean} | after sanitize: {after_sanitize}")
    for i, s in enumerate(cv["skills"]):
        colon = s.find(":")
        cat   = s[:colon].strip() if colon > 0 else "?"
        items = [t.strip() for t in s[colon+1:].split(",")] if colon > 0 else []
        print(f"[FINAL-POLISH]   [{i+1}] {cat}: {len(items)} items")

    # 2b. HUMANIZATION: Strip AI buzzwords from summary and bullets
    _AI_BUZZWORDS = [
        (r'\bHighly motivated\b', 'Experienced'),
        (r'\bResults-driven\b', 'Delivery-focused'),
        (r'\bDynamic professional\b', 'Experienced professional'),
        (r'\bPassionate about\b', 'Focused on'),
        (r'\bLeveraged\b', 'Used'),
        (r'\bUtilized\b', 'Used'),
        (r'\bRevolutionary\b', 'Effective'),
        (r'\bCutting-edge\b', 'Modern'),
        (r'\bNext-generation\b', 'Modern'),
        (r'\bAI-powered\b', 'Automated'),
        (r'\bSmart ecosystem\b', 'integrated system'),
        (r'\bInnovative solution\b', 'technical solution'),
        (r'\bSynergy\b', 'collaboration'),
        (r'\bSeamlessly\b', 'effectively'),
        (r'\bRobust solution\b', 'reliable system'),
        (r'\bTransformed\b', 'Improved'),
        (r'\bGamechanger\b', 'significant improvement'),
        (r'\b10x\b', ''),
    ]
    summary = cv.get("summary", "")
    if summary:
        for pattern, replacement in _AI_BUZZWORDS:
            summary = re.sub(pattern, replacement, summary, flags=re.IGNORECASE)
        cv["summary"] = summary

    # Also clean bullets in each company
    for co in cv.get("companies", []):
        cleaned_bullets = []
        for bullet in (co.get("bullets") or []):
            for pattern, replacement in _AI_BUZZWORDS:
                bullet = re.sub(pattern, replacement, bullet, flags=re.IGNORECASE)
            cleaned_bullets.append(bullet)
        co["bullets"] = cleaned_bullets

    # 2c. PROJECT NAME VALIDATOR: enforce "PREFIX: Full Name" format
    # Catches blended single-word names like "EcoCycle - ..." and converts them
    # Also ensures the prefix-colon format is present

    def _infer_system_type_prefix(text: str) -> str:
        """
        Derive the precise system-type prefix from a project description string.
        Rules mirror the AI prompt's system-type taxonomy exactly — ordered from
        most-specific to least-specific so the first match wins.
        Returns a concrete class label; never a generic fallback.
        """
        t = text.lower()

        # --- Data & Integration ---
        if any(w in t for w in ['etl', 'data ingestion', 'ingestion', 'transformation pipeline',
                                  'batch load', 'data pipeline', 'data sync']):
            return "ETL"
        if any(w in t for w in ['integration', 'middleware', 'webhook', 'third-party api',
                                  'data sync', 'api gateway', 'connector']):
            return "Integration System"

        # --- Analytics & Reporting ---
        if any(w in t for w in ['analytics', 'olap', 'business intelligence', 'bi platform',
                                  'data visualis', 'data visualiz', 'aggregated metric']):
            return "Analytics System"
        if any(w in t for w in ['report', 'kpi', 'dashboard', 'business report',
                                  'scheduled report', 'export report']):
            return "Reporting System"

        # --- Finance & Commerce ---
        if any(w in t for w in ['invoice', 'billing', 'subscription', 'payment', 'reconciliation',
                                  'charge', 'revenue management', 'fee management']):
            return "Billing System"
        if any(w in t for w in ['payroll', 'salary', 'compensation', 'wage']):
            return "Payroll System"

        # --- Enterprise Resource & Operations ---
        if any(w in t for w in ['erp', 'enterprise resource', 'procurement', 'purchase order',
                                  'finance module', 'operations management', 'fulfilment',
                                  'supply chain', 'vendor management', 'asset management']):
            return "ERP"
        if any(w in t for w in ['inventory', 'warehouse', 'stock', 'stockroom', 'goods',
                                  'shelf', 'bin location']):
            return "Inventory System"

        # --- Customer & Sales ---
        if any(w in t for w in ['crm', 'customer relationship', 'client record', 'sales pipeline',
                                  'lead management', 'opportunity tracking', 'customer management',
                                  'sales tracking', 'client management', 'account management']):
            return "CRM"

        # --- Logistics ---
        if any(w in t for w in ['dispatch', 'fleet', 'driver', 'delivery tracking', 'route',
                                  'freight', 'courier', 'logistics tracking', 'last mile']):
            return "Dispatch System"

        # --- Support & Helpdesk ---
        if any(w in t for w in ['ticket', 'support ticket', 'helpdesk', 'help desk',
                                  'service desk', 'issue tracking', 'bug tracker']):
            return "Ticketing System"

        # --- Case / Claims ---
        if any(w in t for w in ['claim', 'insurance claim', 'case management', 'legal case',
                                  'adjudication', 'settlement']):
            return "Claims System"

        # --- Process & Approval ---
        if any(w in t for w in ['workflow', 'approval', 'approval chain', 'routing', 'process automation',
                                  'multi-step', 'task routing', 'sign-off', 'signoff']):
            return "Workflow System"

        # --- Scheduling & Bookings ---
        if any(w in t for w in ['schedule', 'scheduling', 'appointment', 'booking', 'calendar',
                                  'reservation', 'slot', 'queue management']):
            return "Scheduling System"

        # --- Monitoring & Observability ---
        if any(w in t for w in ['monitor', 'monitoring', 'observability', 'alerting', 'health check',
                                  'sla tracking', 'uptime', 'alert', 'incident', 'log aggregat',
                                  'metrics collect', 'apm', 'tracing']):
            return "Monitoring System"

        # --- Compliance & Risk ---
        if any(w in t for w in ['compliance', 'audit', 'regulatory', 'risk control', 'gdpr',
                                  'kyc', 'aml', 'policy enforcement', 'governance']):
            return "Compliance System"

        # --- Notifications & Messaging ---
        if any(w in t for w in ['notification', 'alert system', 'email alert', 'sms', 'push notification',
                                  'event-driven message', 'messaging platform', 'broadcast']):
            return "Notification System"

        # --- Documents & Content ---
        if any(w in t for w in ['document', 'file management', 'digital archive', 'version control',
                                  'file storage', 'document management', 'dms']):
            return "Document System"
        if any(w in t for w in ['cms', 'content management', 'content publishing', 'editorial',
                                  'web content', 'blog platform']):
            return "CMS"

        # --- Identity & Access ---
        if any(w in t for w in ['identity', 'authentication', 'authorisation', 'authorization',
                                  'sso', 'rbac', 'oauth', 'access control', 'user management',
                                  'login', 'iam', 'permission']):
            return "Identity System"

        # --- Learning ---
        if any(w in t for w in ['lms', 'learning management', 'course', 'e-learning', 'elearning',
                                  'training platform', 'assessment', 'learner', 'quiz']):
            return "LMS"

        # --- DevOps / CI/CD ---
        if any(w in t for w in ['ci/cd', 'cicd', 'pipeline deployment', 'build pipeline',
                                  'devops', 'release management', 'deployment automation',
                                  'infrastructure as code', 'iac', 'gitops']):
            return "CI/CD System"

        # --- API / Backend services — now a conscious, accurate choice, not a lazy fallback ---
        if any(w in t for w in ['api', 'rest api', 'graphql', 'microservice', 'service mesh',
                                  'backend service', 'server-side', 'endpoint', 'rpc']):
            return "API Gateway"

        # --- Absolute last resort: derive from the most prominent noun in the text ---
        # Try to extract the first meaningful noun phrase rather than printing "Backend System"
        _noun_candidates = [
            ('hr', 'HR System'), ('human resource', 'HR System'),
            ('recruit', 'Recruitment System'), ('onboard', 'Onboarding System'),
            ('asset', 'Asset Management System'), ('project', 'Project Management System'),
            ('product', 'Product Management System'), ('catalog', 'Product Catalog System'),
            ('ecommerce', 'E-Commerce Platform'), ('e-commerce', 'E-Commerce Platform'),
            ('marketplace', 'Marketplace Platform'), ('portal', 'Portal System'),
            ('mobile', 'Mobile Platform'), ('patient', 'Patient Management System'),
            ('health', 'Healthcare System'), ('medical', 'Medical Records System'),
            ('fraud', 'Fraud Detection System'), ('recommend', 'Recommendation Engine'),
            ('search', 'Search Platform'), ('geo', 'Geospatial System'),
            ('real-time', 'Real-Time Processing System'), ('event', 'Event Management System'),
            ('feedback', 'Feedback System'), ('survey', 'Survey Platform'),
            ('chat', 'Messaging Platform'), ('social', 'Social Platform'),
        ]
        for keyword, label in _noun_candidates:
            if keyword in t:
                return label

        # Truly unknown — use the most prominent capitalized word from the description
        # to construct a specific label rather than "Backend System"
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        if words:
            return f"{words[0]} System"

        return "Application System"

    _BLENDED_WORD_DASH = re.compile(
        r'^([A-Z][a-z]+[A-Z][a-zA-Z]*)\s*[-–]\s*(.+)$'  # e.g. "EcoCycle - Environmental Data Platform"
    )
    _VALID_PREFIX_FORMAT = re.compile(r'^[A-Za-z&/\s]{2,25}:\s*.{10,}$')  # "ETL: Real-Time Pipeline..."

    for proj in cv.get("projects", []):
        raw_name = proj.get("name", "")
        clean_name = re.sub(r'\[.*?\]', '', raw_name).strip()

        # Case 1: already has "PREFIX: Name" format — keep as-is
        if _VALID_PREFIX_FORMAT.match(clean_name):
            continue  # good format

        # Case 2: blended CamelCase word before dash — "EcoCycle - Description"
        m = _BLENDED_WORD_DASH.match(clean_name)
        if m:
            description = m.group(2).strip().rstrip('.')
            prefix = _infer_system_type_prefix(description)
            proj["name"] = f"{prefix}: {description}"
            tag_match = re.search(r'\[.*?\]', raw_name)
            if tag_match:
                proj["name"] += f" {tag_match.group(0)}"

    # 3. Fix competencies separator + detect and replace placeholders
    comp = cv.get("competencies", "")
    if comp:
        if "*" not in comp:
            comp = re.sub(r"\s*[|,]\s*", " * ", comp)
        comp = comp.strip()

        # Detect placeholder competencies like "Competency * Competency * ..."
        parts = [p.strip() for p in comp.split("*") if p.strip()]
        placeholder_count = sum(1 for p in parts
                                if re.match(r"(?i)^competency\s*\d*$", p.strip()))
        if placeholder_count >= 3 or not parts:
            # Build real competencies from skills and title
            title_words = (cv.get("title") or "").split("|")[0].strip()
            skill_cats  = []
            for row in cv.get("skills", []):
                colon = row.find(":")
                if colon > 0:
                    skill_cats.append(row[:colon].strip())

            # Generic but real competency pool - better than placeholders
            domain_comp = [title_words] if title_words else []
            domain_comp += skill_cats[:5]
            domain_comp += [
                "Agile Development", "Code Review", "System Design",
                "Cross-functional Collaboration", "CI/CD Pipelines",
                "Performance Optimisation", "Technical Documentation",
                "Problem Solving", "API Design", "Test-Driven Development"
            ]
            # Deduplicate and take exactly 10
            seen_c = set()
            unique_c = []
            for c in domain_comp:
                if c.lower() not in seen_c and c:
                    seen_c.add(c.lower())
                    unique_c.append(c)
            comp = " * ".join(unique_c[:10])
        elif len(parts) < 10:
            # AI gave fewer than 10 - pad with generic competencies
            generic_pool = [
                "Agile Development", "Code Review", "System Design",
                "Cross-functional Collaboration", "CI/CD Pipelines",
                "Performance Optimisation", "Technical Documentation",
                "Problem Solving", "API Design", "Test-Driven Development"
            ]
            seen_c = set(p.lower() for p in parts)
            for g in generic_pool:
                if len(parts) >= 10:
                    break
                if g.lower() not in seen_c:
                    parts.append(g)
                    seen_c.add(g.lower())
            comp = " * ".join(parts[:10])

        cv["competencies"] = comp

    # 4. Remove company names from summary
    summary = cv.get("summary", "")
    if summary:
        summary = re.sub(
            r"\b[A-Z][a-zA-Z]+(?:'s)?\s+(?:AI-first\s+)?(?:recruitment advertising |hiring |job |talent )?(?:platform|product|system|service)\b",
            "this platform",
            summary
        )
        cv["summary"] = summary

    # 5. Fix totalYears and summary year
    real_years = _calc_total_years(years_exp)
    cv["totalYears"] = real_years

    summary = cv.get("summary", "")
    if summary:
        summary = re.sub(
            r'(?:over\s+|more\s+than\s+|approximately\s+)?\b\d+\+?\s+years?\b',
            f"{real_years} years",
            summary,
            count=1
        )
        cv["summary"] = summary

    # 7. Override company date ranges with dynamically calculated ones
    if years_exp:
        dynamic_cos = _build_dynamic_companies(years_exp)
        for i, co in enumerate(cv.get("companies", [])):
            if i < len(dynamic_cos):
                co["dateRange"] = f"{dynamic_cos[i]['start']} - {dynamic_cos[i]['end']}"

    # 8. Override education years with calculated values
    edu_years = _build_education_year(years_exp)
    edu = cv.get("education", {})
    if isinstance(edu, dict):
        edu["years"] = f"{edu_years['start']} - {edu_years['end']}"
        cv["education"] = edu

    return cv


# -- _rebuild_skills_from_techs: AI-powered fallback ---------------------------
def _infer_category_name(items: list, job_title: str = "", slot_index: int = 0) -> str:
    """
    Derive a meaningful skill category heading purely from the items in that group
    and the job title — no AI call, no hardcoded positions.

    Logic: score each candidate domain label by how many of its signature keywords
    appear (as substrings) in the lowercased items list.  The highest-scoring label
    wins.  Ties are broken by slot_index so consecutive groups never share a name.

    This is used only when the AI call fails and we must name a group ourselves.
    The result is ALWAYS a real domain phrase, never "Technical Skills N".
    """
    items_str = " ".join(items).lower()
    jt = job_title.lower()

    # Candidate domain labels with their signature token sets.
    # Order matters only for tie-breaking (prefer earlier entries).
    DOMAIN_SIGNALS: list = [
        # (label, [tokens_that_signal_this_domain])
        ("Languages & Frameworks",      ["python", "javascript", "typescript", "java", "c#", "php", "ruby", "go", "rust", "kotlin", "swift", "scala", "react", "angular", "vue", "django", "flask", "spring", "laravel", "rails", "express", "fastapi", "next.js", "nuxt"]),
        ("Frontend & UI",               ["react", "angular", "vue", "html", "css", "sass", "tailwind", "bootstrap", "webpack", "vite", "next.js", "nuxt", "svelte", "figma", "ui", "ux", "responsive"]),
        ("Backend & API",               ["node.js", "express", "fastapi", "django", "flask", "spring boot", "laravel", "rest", "graphql", "grpc", "api", "microservices", "rabbitmq", "kafka", "celery", "nginx", "gunicorn"]),
        ("Database & Storage",          ["postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle", "sql server", "cassandra", "dynamodb", "elasticsearch", "firestore", "prisma", "sequelize", "hibernate", "entity framework", "supabase"]),
        ("Cloud & Infrastructure",      ["aws", "azure", "gcp", "google cloud", "ec2", "s3", "lambda", "rds", "ecs", "eks", "app service", "cloud run", "gke", "firebase", "heroku", "digitalocean", "cloudflare", "vercel", "netlify"]),
        ("DevOps & CI/CD",              ["docker", "kubernetes", "jenkins", "github actions", "gitlab ci", "circleci", "terraform", "ansible", "helm", "argocd", "ci/cd", "pipeline", "vagrant", "packer", "pulumi"]),
        ("Testing & Quality",           ["jest", "pytest", "junit", "cypress", "selenium", "playwright", "mocha", "jasmine", "xunit", "nunit", "testng", "postman", "soapui", "k6", "locust", "sonarqube", "eslint"]),
        ("Monitoring & Observability",  ["prometheus", "grafana", "datadog", "new relic", "elk", "elasticsearch", "kibana", "logstash", "sentry", "pagerduty", "cloudwatch", "splunk", "jaeger", "opentelemetry", "zipkin"]),
        ("Security & Compliance",       ["owasp", "oauth", "jwt", "ssl", "tls", "iam", "vault", "keycloak", "snyk", "trivy", "cve", "penetration", "firewall", "waf", "encryption", "sso", "ldap", "saml"]),
        ("Data & Analytics",            ["spark", "hadoop", "airflow", "dbt", "pandas", "numpy", "bigquery", "redshift", "snowflake", "tableau", "power bi", "looker", "metabase", "etl", "datalake", "kafka", "flink"]),
        ("Machine Learning & AI",       ["tensorflow", "pytorch", "scikit-learn", "keras", "xgboost", "hugging face", "langchain", "openai", "bert", "gpt", "llm", "mlflow", "vertex ai", "sagemaker", "opencv", "nltk"]),
        ("Mobile Development",          ["swift", "kotlin", "flutter", "react native", "ionic", "xcode", "android studio", "expo", "fastlane", "testflight", "play console", "firebase", "push notifications", "core data", "realm"]),
        ("SEO & Digital Marketing",     ["semrush", "ahrefs", "google analytics", "google search console", "moz", "screaming frog", "keyword planner", "google ads", "facebook ads", "hubspot", "mailchimp", "hotjar", "tag manager"]),
        ("Version Control & Tooling",   ["git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello", "asana", "slack", "notion", "linear", "figma", "miro", "postman", "insomnia"]),
        ("Scripting & Automation",      ["bash", "powershell", "python", "makefile", "ansible", "terraform", "chef", "puppet", "cron", "airflow", "luigi", "prefect", "shell"]),
        ("Content & CMS",               ["wordpress", "drupal", "contentful", "strapi", "sanity", "ghost", "shopify", "magento", "woocommerce", "webflow", "squarespace"]),
        ("Networking & Systems",        ["tcp/ip", "dns", "http", "https", "load balancer", "cdn", "vpn", "linux", "ubuntu", "centos", "windows server", "active directory", "nginx", "apache", "haproxy"]),
    ]

    scores: list = []
    for label, signals in DOMAIN_SIGNALS:
        score = sum(1 for sig in signals if sig in items_str)
        # Bonus if job title reinforces domain
        jt_bonus = sum(0.5 for sig in signals if sig in jt)
        scores.append((score + jt_bonus, label))

    # Sort descending by score; for equal scores preserve list order
    scores.sort(key=lambda x: -x[0])

    # Collect used labels so we don't repeat within one CV's skills section
    # (slot_index is used to skip already-used positions deterministically)
    used_in_run = getattr(_infer_category_name, "_used", set())
    _infer_category_name._used = used_in_run

    for score_val, label in scores:
        if label not in used_in_run:
            used_in_run.add(label)
            return label

    # Absolute last resort: clear used set and return top scorer
    _infer_category_name._used = set()
    return scores[0][1] if scores else "Core Technologies"


def _reset_infer_category_name():
    """Call at the start of each CV generation to reset the used-label cache."""
    _infer_category_name._used = set()


def _rebuild_skills_from_techs(techs: dict, job_title: str = "", jd_text: str = "") -> list:
    """
    Emergency fallback: ask the AI to group available JD technologies into 5
    named skill categories that match this specific job title and JD.
    No hardcoded keyword buckets, no hardcoded slot names.
    """
    import json as _json

    core      = [t for t in techs.get("core",      []) if _is_real_tech(t)]
    preferred = [t for t in techs.get("preferred", []) if _is_real_tech(t)]
    ecosystem = [t for t in techs.get("ecosystem", []) if _is_real_tech(t)]
    all_techs = list(dict.fromkeys(core + preferred + ecosystem))

    if not all_techs:
        return []

    jd_hint  = f" JD context: {jd_text[:400]}." if jd_text else ""
    tech_str = ", ".join(all_techs)

    prompt = (
        f"Job title: {job_title}.{jd_hint}\n"
        f"Available technologies from this JD: {tech_str}\n\n"
        "Group these technologies into EXACTLY 5 skill categories matching this job's "
        "actual technology domains. Rules:\n"
        "1. Derive category names from this JD — not generic labels.\n"
        "   For DevOps/K8s: 'Container Orchestration', 'CI/CD & Automation', "
        "'IaC & Config Management', 'Monitoring & Observability', 'Scripting & Version Control'.\n"
        "   For backend: 'Backend & Frameworks', 'Database & Storage', etc.\n"
        "2. Each tool goes under the heading it logically belongs to — no mixing.\n"
        "3. Minimum 5 tools per category.\n"
        "4. Zero duplicates across categories.\n"
        "5. Output ONLY a JSON array of 5 strings: 'Category Name: tool1, tool2, ...'\n"
        "No markdown, no extra text."
    )

    try:
        import urllib.request as _ur
        payload = _json.dumps({
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 600,
        }).encode()
        req = _ur.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {GROQ_API_KEY}"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=15) as resp:
            body = _json.loads(resp.read())
        raw = body["choices"][0]["message"]["content"].strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        rows = _json.loads(raw)
        if isinstance(rows, list):
            valid = [r for r in rows
                     if isinstance(r, str) and ":" in r
                     and len(r[r.index(":")+1:].split(",")) >= 3]
            if len(valid) >= 3:
                return valid[:5]
    except Exception:
        pass

    # Last resort: split evenly into 5 chunks, label each by item content
    chunk = max(5, len(all_techs) // 5)
    result = []
    for i in range(5):
        group = all_techs[i*chunk:(i+1)*chunk] or all_techs[-chunk:]
        label = _infer_category_name(group, job_title, i)
        result.append(f"{label}: {', '.join(group[:13])}")
    return result


def fix_skills(cv: dict) -> dict:
    """
    Normalise skills into 'Category: item1, item2, ...' strings.
    Guarantees at least 5 categories with >=5 real-tech items each.
    If the LLM output is thin, rebuilds from cv['_techs'] (injected by generate_cv_atomic)
    or from a best-effort bucket split of the technologies block.
    """
    print(f"[FIX-SKILLS] Starting fix_skills()")
    cleaned    = []
    raw_skills = cv.get("skills", [])
    if not isinstance(raw_skills, list):
        print(f"[FIX-SKILLS] WARN — skills is not a list ({type(raw_skills)}), resetting to []")
        raw_skills = []
    else:
        print(f"[FIX-SKILLS] Raw skills input: {len(raw_skills)} entries")

    normalised = []
    for row in raw_skills:
        if row is None:
            continue

        if isinstance(row, dict):
            cat = _to_str(
                row.get("category") or row.get("name") or
                row.get("cat") or row.get("label") or "Skills"
            )
            items_raw = (row.get("items") or row.get("technologies") or
                         row.get("tools") or row.get("values") or [])
            val = _to_str(items_raw)
            normalised.append(f"{cat}: {val}")

        elif isinstance(row, list):
            if len(row) >= 2 and isinstance(row[0], str) and not row[0].strip().startswith("gsk_"):
                cat  = row[0].strip()
                rest = row[1:]
                val  = _to_str(rest)
                normalised.append(f"{cat}: {val}")
            else:
                val = _to_str(row)
                if val:
                    normalised.append(f"Skills: {val}")

        elif isinstance(row, str):
            normalised.append(row.strip())

    for row in normalised:
        if not row:
            continue

        colon = row.find(":")
        if colon <= 0:
            items = [t.strip() for t in row.split(",") if t.strip()]
            if items:
                cleaned.append(f"Skills: {', '.join(items)}")
            continue

        cat_raw = row[:colon].strip()
        val_raw = row[colon + 1:].strip()

        if not val_raw or val_raw.lower() in ("none", "n/a", "null", "-", "-", "[]", "{}"):
            continue  # skip empty rows

        items = [
            t.strip() for t in val_raw.split(",")
            if t.strip() and t.strip().lower() not in ("none", "n/a", "null", "-", "[]", "{}")
        ]

        if items:
            # Deduplicate within category then enforce 11-13 cap
            seen_i = set()
            deduped = []
            for it in items:
                if it.lower().strip() not in seen_i:
                    seen_i.add(it.lower().strip())
                    deduped.append(it)
            if len(deduped) > 13:
                deduped = deduped[:13]
            if deduped:
                cleaned.append(f"{cat_raw}: {', '.join(deduped)}")

    cv["skills"] = cleaned
    print(f"[FIX-SKILLS] After normalisation: {len(cleaned)} valid categories")
    for i, s in enumerate(cleaned):
        colon = s.find(":")
        if colon > 0:
            items = [t.strip() for t in s[colon+1:].split(",") if t.strip()]
            print(f"[FIX-SKILLS]   [{i+1}] {s[:colon].strip()}: {len(items)} items")

    # -- FALLBACK: if fewer than 5 categories or any has <5 items, rebuild -----
    def _count_real_items(row: str) -> int:
        ci = row.find(":")
        if ci <= 0:
            return 0
        items = [t.strip() for t in row[ci+1:].split(",") if t.strip() and _is_real_tech(t)]
        return len(items)

    needs_rebuild = (
        len(cv["skills"]) < 5 or
        any(_count_real_items(r) < 5 for r in cv["skills"])
    )
    
    if needs_rebuild:
        thin_cats = [s for s in cv["skills"] if _count_real_items(s) < 5]
        print(f"[FIX-SKILLS] WARN — needs rebuild: {len(cv['skills'])} cats, {len(thin_cats)} thin")
        for s in thin_cats:
            colon = s.find(":")
            if colon > 0:
                items = [t.strip() for t in s[colon+1:].split(",") if t.strip()]
                print(f"[FIX-SKILLS]   THIN: {s[:colon].strip()}: {len(items)} real items")

    if needs_rebuild:
        # Prefer the _techs dict injected by the pipeline; else reconstruct from technologies block
        techs = cv.get("_techs")
        if not techs:
            tech_block = cv.get("technologies", {})
            if isinstance(tech_block, dict):
                techs = {
                    "core":      tech_block.get("mustHave",   []),
                    "preferred": tech_block.get("niceToHave", []),
                    "ecosystem": tech_block.get("additional", []),
                }
            print(f"[FIX-SKILLS] Rebuilding from technologies block: {len(techs.get('core',[]))} core items")
        if techs:
            rebuilt = _rebuild_skills_from_techs(techs, cv.get("title", "") or cv.get("_job_title", ""))
            if rebuilt:
                print(f"[FIX-SKILLS] Rebuilt {len(rebuilt)} categories from tech pool")
                # Merge: keep any good existing rows, replace thin/missing ones
                existing_cats = {}
                for row in cv["skills"]:
                    ci = row.find(":")
                    if ci > 0 and _count_real_items(row) >= 5:
                        existing_cats[row[:ci].strip().lower()] = row
                for row in rebuilt:
                    ci = row.find(":")
                    cat_key = row[:ci].strip().lower() if ci > 0 else ""
                    if cat_key not in existing_cats:
                        existing_cats[cat_key] = row
                merged = list(existing_cats.values())
                cv["skills"] = merged[:5] if len(merged) >= 5 else (merged + rebuilt)[:5]
                print(f"[FIX-SKILLS] After rebuild+merge: {len(cv['skills'])} categories")

    # Clean up internal pipeline hints so they don't reach the PDF
    cv.pop("_techs", None)
    cv.pop("_job_title", None)

    return cv


# -- fix_projects: strip company names from project names/overviews ------------
def fix_projects(cv: dict) -> dict:
    """
    Post-processing guard: remove any real company name or known brand
    that leaked into project names or overviews.
    Also flags and rewrites robotically identical project overview structures.
    """
    # Collect all known company names to scrub
    scrub_names = [c["name"] for c in CANDIDATE_COMPANIES]
    # Also scrub from the CV title domain words that look like company names
    title = cv.get("title", "")

    projects = cv.get("projects", [])
    seen_overviews: list[str] = []

    for p in projects:
        if not isinstance(p, dict):
            continue

        name = p.get("name", "")
        overview = p.get("overview", "")

        # 1. Strip any real company name from project name
        for co_name in scrub_names:
            # Match full company name and common abbreviations
            words = co_name.split()
            for w in words:
                if len(w) > 3:  # skip short words like "AND", "NOW", "THE"
                    # Only remove if it looks like a proper noun (starts uppercase)
                    pattern = r'\b' + re.escape(w) + r'\b'
                    if re.search(pattern, name, re.IGNORECASE):
                        name = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
                        name = re.sub(r'\s{2,}', ' ', name).strip(' ---')

        # 2. Strip "Project " prefix if AI added it (e.g. "Project LeoConnect")
        name = re.sub(r'^Project\s+', '', name, flags=re.IGNORECASE).strip()

        # 3. If name became empty or too short after scrubbing, keep original minus "Project "
        original_name = p.get("name", "")
        if len(name) < 4:
            name = re.sub(r'^Project\s+', '', original_name, flags=re.IGNORECASE).strip()

        # 5. Sanitize techTags - remove fake words that are not real technology names
        for tag_field in ("techTags", "tech"):
            raw_tags = p.get(tag_field)
            if raw_tags is None:
                continue
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in re.split(r"[,|;]", raw_tags) if t.strip()]
            if isinstance(raw_tags, list):
                clean_tags = [t for t in raw_tags if _is_real_tech(t)]
                # Also reject bare "NET" (must be ".NET 8" or "ASP.NET Core")
                clean_tags = [t for t in clean_tags if t.upper() != "NET"]
                p[tag_field] = clean_tags

        p["name"] = name

        # 4. Strip company names from overview too
        for co_name in scrub_names:
            words = co_name.split()
            for w in words:
                if len(w) > 4:
                    pattern = r'\b' + re.escape(w) + r'\b'
                    overview = re.sub(pattern, 'this company', overview, flags=re.IGNORECASE)
        p["overview"] = overview

    cv["projects"] = projects
    return cv


# -- fix_skills: detect and remove duplicate-tool categories -------------------

def _repair_project_tech_tags(cv: dict, techs: dict) -> dict:
    """
    Hard post-processor: guarantees every project has 5-7 unique, real-tech
    Stack tags derived from the JD. Runs AFTER all other sanitisation.

    Problems it fixes:
      - Tags like "Cloud", "Development", "Good", "Strong", "Hands" from thin JDs
      - Tags that are plain English words rather than named software tools
      - Projects with fewer than 5 Stack entries
      - Tags not from the JD ecosystem at all
    """
    projects = cv.get("projects", [])
    if not projects:
        return cv

    # Build the full JD tech pool (core > preferred > ecosystem)
    pool: list = []
    seen_pool: set = set()
    for arr in ("core", "preferred", "ecosystem"):
        for t in techs.get(arr, []):
            t = (t or "").strip()
            if t and t.lower() not in seen_pool and _is_real_tech(t):
                pool.append(t)
                seen_pool.add(t.lower())

    # Also pull real techs from cv["technologies"] if pool is thin
    cv_tech = cv.get("technologies", {})
    if isinstance(cv_tech, dict):
        for arr in ("mustHave", "niceToHave", "additional"):
            for t in cv_tech.get(arr, []):
                t = (t or "").strip()
                if t and t.lower() not in seen_pool and _is_real_tech(t):
                    pool.append(t)
                    seen_pool.add(t.lower())

    # Pull real techs from skills section too (last resort)
    for row in cv.get("skills", []):
        if isinstance(row, str) and ":" in row:
            colon = row.index(":")
            for t in row[colon+1:].split(","):
                t = t.strip()
                if t and t.lower() not in seen_pool and _is_real_tech(t):
                    pool.append(t)
                    seen_pool.add(t.lower())

    _GENERIC_TAG_WORDS = {
        "cloud", "development", "web", "good", "strong", "hands",
        "ability", "remote", "setup", "mindset", "rest", "api",
        "apis", "app", "apps", "core", "net", "sdk", "http",
        "json", "xml", "yaml", "data", "code", "base", "stack",
        "work", "tool", "tools", "service", "platform", "system",
        "backend", "frontend", "database", "server", "client",
        "framework", "library", "language", "testing", "security",
        "solution", "solutions", "module", "integration", "environment",
        "sql", "nosql",  # bare "SQL"/"NoSQL" should be "SQL Server", "MySQL", "PostgreSQL" etc.
    }

    used_global: set = set()

    for proj in projects:
        # Get raw tags from techTags OR tech field OR extract from project name
        raw: list = []
        for field in ("techTags", "tech"):
            val = proj.get(field)
            if val:
                if isinstance(val, list):
                    raw = val
                elif isinstance(val, str):
                    raw = [t.strip() for t in re.split(r"[|,;]", val) if t.strip()]
                if raw:
                    break

        # If still empty, try extracting from project name brackets: "Name [T1, T2, T3]"
        if not raw:
            name = proj.get("name", "")
            m = re.search(r"\[([^\]]+)\]", name)
            if m:
                raw = [t.strip() for t in m.group(1).split(",") if t.strip()]

        # Filter raw to only real, non-generic named tools
        seen_local: set = set()
        clean: list = []
        for t in raw:
            t = (t or "").strip()
            tl = t.lower()
            if not t or tl in seen_local:
                continue
            if not _is_real_tech(t):
                continue
            if tl in _GENERIC_TAG_WORDS:
                continue
            clean.append(t)
            seen_local.add(tl)

        # Backfill to reach 5 from pool
        if len(clean) < 5:
            for t in pool:
                if len(clean) >= 7:
                    break
                tl = t.lower()
                if tl not in seen_local:
                    clean.append(t)
                    seen_local.add(tl)

        # Still under 5? Use pool without global uniqueness restriction
        if len(clean) < 5:
            for t in pool:
                if len(clean) >= 5:
                    break
                tl = t.lower()
                if tl not in seen_local:
                    clean.append(t)
                    seen_local.add(tl)

        # Absolute last resort: use company tech tags from experience
        if len(clean) < 3:
            for co in cv.get("companies", []):
                tech_str = co.get("tech", "")
                if tech_str:
                    for t in re.split(r"[|,]", tech_str):
                        t = t.strip()
                        if t and _is_real_tech(t) and t.lower() not in seen_local:
                            clean.append(t)
                            seen_local.add(t.lower())
                        if len(clean) >= 5:
                            break
                if len(clean) >= 5:
                    break

        # Deduplicate and cap
        final: list = []
        seen_final: set = set()
        for t in clean:
            if t.lower() not in seen_final:
                final.append(t)
                seen_final.add(t.lower())

        proj["techTags"] = final[:7]
        for t in final[:7]:
            used_global.add(t.lower())

    cv["projects"] = projects
    return cv


def fix_skills_dedup(cv: dict) -> dict:
    """
    If two or more skill categories share the same set of tools (or differ by only 1 tool),
    merge or rename them to avoid the visual of 6 rows all listing the exact same 5 tools.
    """
    skills = cv.get("skills", [])
    if not skills:
        return cv

    parsed = []
    for row in skills:
        colon = row.find(":")
        if colon > 0:
            cat = row[:colon].strip()
            items = [t.strip() for t in row[colon+1:].split(",") if t.strip()]
            parsed.append((cat, items))
        else:
            parsed.append((row, []))

    # Detect categories whose item sets overlap by more than 60%
    def overlap(a: list, b: list) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(i.lower() for i in a), set(i.lower() for i in b)
        return len(sa & sb) / min(len(sa), len(sb))

    keep_indices = list(range(len(parsed)))
    to_remove = set()
    for i in range(len(parsed)):
        for j in range(i+1, len(parsed)):
            if overlap(parsed[i][1], parsed[j][1]) > 0.6:
                # Keep the one with more items; mark the other for removal
                if len(parsed[i][1]) >= len(parsed[j][1]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    cleaned = [
        f"{cat}: {', '.join(items)}" if items else cat
        for idx, (cat, items) in enumerate(parsed)
        if idx not in to_remove
    ]

    cv["skills"] = cleaned if cleaned else skills
    return cv




# -- Skills post-processing: trust the AI completely ---------------------------
# The AI decides every category name, every technology, and their placement.
# Post-processing ONLY removes provably fake English words (verbs, HR nouns, etc.)
# It never moves items between categories, never reclassifies, never adds anything.

def _enforce_skill_domains(cv: dict, techs: dict = None, job_title: str = "") -> dict:
    """
    Passthrough enforcer — preserves the AI's skill layout exactly as generated.
    Only strips items that are provably not real technology names (caught by
    _is_real_tech and _sanitize_skills_list which are called in final_polish).
    No re-bucketing, no cross-category moves, no stack filtering.
    """
    return cv


def _is_token_error(body: dict) -> bool:
    msg  = str(body.get("error", {}).get("message", "")).lower()
    code = str(body.get("error", {}).get("code",    "")).lower()
    return (
        "request too large" in msg or
        ("token" in msg and "limit" in msg) or
        "context_length_exceeded" in code or
        "request_too_large" in code
    )


# -- Smart key rotation --------------------------------------------------------
def _prioritised_keys(valid_keys: list) -> list:
    import time
    now = time.time()
    def _sort_key(k):
        mk = mask(k)
        cooldown_until = _key_rate_limited_until.get(mk, 0)
        is_limited = 1 if cooldown_until > now else 0
        return (is_limited, _key_usage.get(mk, 0))
    return sorted(valid_keys, key=_sort_key)


# -- Groq caller ---------------------------------------------------------------
# ===================================================================
# GROQ 6-PIPELINE ATOMIC GENERATION - mirrors Cerebras exactly
# Each pipeline is a short, focused prompt - no truncation possible.
# Pipeline 1  -> technology extraction
# Pipeline 2A -> title only
# Pipeline 2B -> summary only
# Pipeline 3A -> Technical Skills  (5 categories x 10 items)
# Pipeline 3B -> Core Competencies (exactly 10 domain phrases)
# Pipeline 4  -> experience bullets + roles
# Pipeline 5A -> 4 projects (natural descriptions)
# Pipeline 5B -> Related Tech (5 boxes x 5 items)
# ===================================================================

async def call_groq(req: CVRequest) -> tuple:
    raw_keys = req.groq_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Groq API keys provided.")

    valid_keys = [k.strip() for k in raw_keys if k and k.strip().startswith("gsk_")]
    if not valid_keys:
        raise HTTPException(400, "No valid Groq keys (must start with gsk_).")

    model        = _normalize_model_name(req.model or "llama-3.1-8b-instant", "groq")
    sorted_keys  = _prioritised_keys(valid_keys)
    last_error   = "Unknown error"
    errors_by_key: list = []

    # Per-call timeouts inside call_llm_atomic (60s). Session must not interfere.
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=180, write=15, pool=10)) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

            # Quick probe before spending 7 pipeline calls on a bad key
            try:
                probe = await client.post(
                    GROQ_URL,
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=15,
                )
                if probe.status_code == 429:
                    errors_by_key.append(f"Key {i+1} ({mk}): rate limited (429) - daily limit hit")
                    last_error = "rate limited"
                    continue
                if probe.status_code in (401, 403):
                    body = probe.text[:120] if probe.text else ""
                    errors_by_key.append(f"Key {i+1} ({mk}): invalid key ({probe.status_code}) - {body}")
                    last_error = f"invalid key ({probe.status_code})"
                    continue
                if probe.status_code == 404:
                    errors_by_key.append(
                        f"Key {i+1} ({mk}): model '{model}' not found (404) - "
                        "try llama-3.1-8b-instant in Keys & Model tab"
                    )
                    last_error = f"model '{model}' not found (404)"
                    break
                if probe.status_code not in (200, 201):
                    body = probe.text[:150] if probe.text else "empty"
                    errors_by_key.append(f"Key {i+1} ({mk}): HTTP {probe.status_code} - {body}")
                    last_error = f"HTTP {probe.status_code}"
                    if probe.status_code < 500:
                        continue
            except httpx.TimeoutException:
                errors_by_key.append(f"Key {i+1} ({mk}): probe timed out")
                last_error = "probe timeout"
                continue
            except httpx.ConnectError as ce:
                errors_by_key.append(f"Key {i+1} ({mk}): cannot connect to api.groq.com - {ce}")
                last_error = "connection error"
                break

            # Key is live - run full atomic pipeline generation
            # call_llm_atomic already retries on 429 with backoff, so a
            # rate-limit exception here means all retries were exhausted.
            try:
                cv = await generate_cv_atomic(req, client, key, model, GROQ_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                return cv, mk, i
            except ValueError as e:
                err_str = str(e).lower()
                if "rate limited" in err_str or "429" in err_str:
                    errors_by_key.append(
                        f"Key {i+1} ({mk}): rate limited during generation (retries exhausted) - "
                        "try switching to llama-3.1-8b-instant (lower token usage) or add more keys"
                    )
                    last_error = "rate limited during generation"
                    continue
                if "invalid key" in err_str or "401" in err_str or "403" in err_str:
                    errors_by_key.append(f"Key {i+1} ({mk}): key rejected during generation")
                    last_error = "key rejected"
                    continue
                _log_generation(req.job_title, mk, i, 0, model, False, str(e))
                raise HTTPException(502, f"Groq pipeline error: {str(e)}")
            except HTTPException:
                raise
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {type(e).__name__} - {str(e)[:120]}")
                last_error = str(e)
                _log_generation(req.job_title, mk, i, 0, model, False, str(e))
                continue

    bullet_list = "\n".join(f"  * {e}" for e in errors_by_key)
    model_used  = model.lower()
    if "deepseek-r1" in model_used:
        fix_hint = (
            "FIX for deepseek-r1: This model outputs extra reasoning tokens and hits Groq's 6K TPM limit.\n"
            "Best options:\n"
            "  1. Switch to llama-3.3-70b-versatile (Free + Smart, no reasoning overhead)\n"
            "  2. Switch to llama-3.1-8b-instant (fastest, lowest token usage)\n"
            "  3. Add more Groq keys from console.groq.com - each key has its own 6K TPM quota"
        )
    else:
        fix_hint = (
            "FIX: Switch model to llama-3.1-8b-instant in Keys & Model tab (uses ~4x fewer tokens), "
            "or add more keys from console.groq.com - free & fast."
        )
    raise HTTPException(
        502,
        f"All Groq keys failed: {last_error}\n{bullet_list}\n{fix_hint}"
    )


# -- Cerebras caller -----------------------------------------------------------
# ===================================================================
# CEREBRAS 3-STAGE PIPELINE - Fully Dynamic, Zero Hardcoded Tech
# Every output is derived 100% from the JD submitted at runtime.
# Stage 1: tech extraction + title + summary + skills + competencies
# Stage 2: experience (companies, bullets, tech tags)
# Stage 3: 4 projects + 6 relatedTech boxes
# ===================================================================

def build_stage1_prompt(req: CVRequest) -> tuple:
    jd           = req.job_description.strip()[:2000]
    years_exp    = (req.years_exp or "").strip()
    total_years  = _calc_total_years(years_exp)
    company_name = (req.company_name or "").strip()

    system = (
        "You are an expert ATS-focused CV writer. Output ONLY valid JSON - no markdown, no backticks. Start { end }.\n\n"

        "=== STEP 1: EXTRACT TECHNOLOGIES FROM THIS SPECIFIC JD ===\n"
        "Read the JD carefully. Extract every named technology:\n"
        "  CORE: every tool/language/framework in Requirements, Responsibilities, Must Have\n"
        "  PREFERRED: every tool in Nice-to-Have, Preferred, Bonus sections\n"
        "  ECOSYSTEM: for each CORE and PREFERRED tool, identify its standard production companions\n"
        "    - if JD names a backend framework: add its ORM, auth lib, test framework, logger, serialiser\n"
        "    - if JD names a cloud platform: add the specific services named + closely related managed services\n"
        "    - if JD names a frontend framework: add its state manager, router, UI library, build tool, test runner\n"
        "    - if JD names a database: add its migration tool, admin tool, ORM/query builder, connection pooler\n"
        "    - if JD names a DevOps/infra tool: add its CI/CD companion, registry, orchestration tool, IaC tool\n"
        "    - apply this expansion to every technology in the JD, regardless of domain\n"
        "ABSOLUTE: Every technology in the output MUST come from CORE, PREFERRED, or ECOSYSTEM of THIS JD.\n"
        "ABSOLUTE: Technologies not in the JD are BANNED - do not add anything from your training knowledge.\n"
        "ABSOLUTE: Output changes completely for every different JD - nothing is ever hardcoded.\n\n"

        "=== SKILLS (5 categories, 10-13 items each) ===\n"
        "Group the extracted technologies into 5 natural technical domains that reflect how THIS specific role works.\n"
        "The category names, which tools go where, and the grouping logic are all decided by you based on the JD.\n"
        "No preset category names. No fixed bucket structure. Each JD produces a completely different skills section.\n"
        "Assign each technology to exactly ONE category — zero repeated items across categories.\n"
        "Name each category after the actual technology area it represents in this JD.\n"
        "MINIMUM 10 items per category. MAXIMUM 13. Fewer than 10 is a HARD FAILURE.\n"
        "Every item must be a real named tool — no verbs, no generic nouns, no soft skills.\n\n"

        "=== TECHNOLOGIES OBJECT ===\n"
        "mustHave: 10-14 items - CORE technologies + top ECOSYSTEM companions\n"
        "niceToHave: 8-12 items - PREFERRED technologies + ECOSYSTEM companions (zero overlap with mustHave)\n"
        "additional: 8-12 items - standard tools for this role domain not already listed above (zero overlap)\n\n"

        "=== SUMMARY (4 sentences, 70+ words) ===\n"
        f"S1 MUST start: '{total_years} years of experience in [domain from JD]...'\n"
        "S2: name 4-5 technologies FROM THIS JD + the system types built\n"
        "S3: scale and complexity metrics derived from the JD\n"
        "S4: methodology and business outcome relevant to the target domain\n"
        "Use ONLY technologies from this JD - never add outside technologies.\n\n"

        "=== COMPETENCIES (exactly 10, separated by ' * ') ===\n"
        "Each phrase 2-4 words. Span: Technical Practices, Domain Expertise, Engineering Process, Impact Areas.\n"
        "Derive ALL from this JD's domain - zero generic phrases.\n\n"

        "=== ARCHITECTURES (3-5 objects) ===\n"
        "Each: {name, description (25-40 words naming JD technologies + concrete outcome)}.\n"
        "Derive patterns from what THIS JD says - not generic patterns.\n\n"

        "BANNED: percentage bars, wrong technologies, <11 skill items per category, "
        "duplicate tools across categories, generic competencies."
    )

    user = f"""Job Title: {req.job_title}
{"Target Company: " + company_name if company_name else ""}
Total Experience: {total_years} years

Job Description:
{jd}

Output JSON with EXACTLY these keys - no companies, no projects yet:
{{
  "  \"title\": \"[MUST BE DERIVED FROM JD: Change the job title by swapping function word and using a DIFFERENT core technology from the JD] | [Tech1 from JD], [Tech2 from JD], [Tech3 from JD]\",\n"
  "summary": "{total_years} years of experience in [JD domain]... (4 sentences, 70+ words, JD technologies only)",
  "skills": [
    "JD Domain Category: item1, item2, ...(11-13 items, all from JD ecosystem)",
    "... (6 categories total, zero items repeated across categories)"
  ],
  "competencies": "JD-Phrase1 * JD-Phrase2 * ... (exactly 10, all from JD domain)",
  "keywords": "jd-term1, jd-term2, ... (18-20 ATS terms from JD)",
  "technologies": {{
    "mustHave": ["JD-core-tool", ... (10-14 items)],
    "niceToHave": ["JD-preferred-tool", ... (8-12 items, zero overlap with mustHave)],
    "additional": ["domain-standard-tool", ... (8-12 items, zero overlap above)]
  }},
  "architectures": [{{"name": "Pattern from JD", "description": "25-40 words with JD tech + outcome"}}]
}}"""

    return system, user


def build_stage2_prompt(req: CVRequest, stage1: dict) -> tuple:
    jd          = req.job_description.strip()[:1400]
    years_exp   = (req.years_exp or "").strip()
    total_years = _calc_total_years(years_exp)
    companies   = _build_dynamic_companies(years_exp)
    num_cos     = len(companies)

    # Build locked tech list from stage1 output - only what the model extracted from the JD
    allowed = []
    seen_a  = set()
    for row in stage1.get("skills", []):
        colon = row.find(":")
        if colon > 0:
            for x in row[colon+1:].split(","):
                x = x.strip()
                if x and x.lower() not in seen_a:
                    seen_a.add(x.lower()); allowed.append(x)
    for arr_key in ("mustHave", "niceToHave", "additional"):
        for t in stage1.get("technologies", {}).get(arr_key, []):
            if t and t.lower() not in seen_a:
                seen_a.add(t.lower()); allowed.append(t)
    allowed_str = ", ".join(allowed[:60]) if allowed else "technologies from the JD"

    if num_cos == 1:
        sen = [("Junior", "only company")]
        verbs = "Co1 (Junior - only role): Implemented, Built, Developed, Deployed, Configured, Automated."
    elif num_cos == 2:
        sen = [("", "current - NO prefix"), ("Junior", "oldest")]
        verbs = ("Co1 (current): Developed, Engineered, Designed, Integrated, Streamlined, Built.\n"
                 "Co2 (Junior): Implemented, Configured, Deployed, Automated, Built, Assisted.")
    else:
        sen = [("Senior", "current"), ("", "mid - NO prefix"), ("Junior", "oldest")]
        verbs = ("Co1 (Senior): Architected, Engineered, Led, Spearheaded, Established, Designed, Launched.\n"
                 "Co2 (mid): Optimised, Refactored, Scaled, Migrated, Integrated, Streamlined, Unified.\n"
                 "Co3 (Junior): Implemented, Built, Developed, Deployed, Configured, Automated, Instrumented.")

    co_lines = []
    for i, co in enumerate(companies):
        prefix, label = sen[i]
        co_lines.append(
            f'Co{i+1} ({label}): name="{co["name"]}", dates="{co["start"]} - {co["end"]}"'
            + (f', role MUST start with "{prefix}"' if prefix else ', role has NO seniority prefix')
        )
    co_block = "\n".join(co_lines)

    system = (
        "You are an expert ATS-focused CV writer. Output ONLY valid JSON. No markdown, no backticks. Start { end }.\n\n"

        "=== LOCKED TECHNOLOGY LIST (use ONLY these - nothing else) ===\n"
        f"{allowed_str}\n\n"

        "=== EXPERIENCE RULES ===\n"
        f"Produce EXACTLY {num_cos} company objects. Use exact company names and date ranges.\n\n"

        "ROLE TITLES:\n"
        "Each company: unique role title derived from the JD domain.\n"
        "Format: '[Seniority] [JD Domain] [UniqueFunction]'\n"
        "Function word pool (one per company, no repeats): Engineer, Developer, Specialist, Analyst, Consultant, Programmer, Architect, Designer\n\n"

        "TECH TAGS:\n"
        "Each company 'tech': exactly 6-8 pipe-separated tools from the locked list above.\n"
        "Co1 gets architectural/advanced tools. Co2 gets mid-level. Co3 gets foundational.\n"
        "NEVER fewer than 6 - hard failure.\n\n"

        "BULLETS (4 per company, 20-30 words each):\n"
        "ALL 12 bullets across all companies describe 12 DIFFERENT systems or features.\n"
        "Before each bullet - ask: have I described this system type before? If yes, choose different.\n"
        "Each bullet: 1+ technology from the locked list + specific named system + unique metric.\n"
        "ZERO verbs repeated across all 12 bullets.\n"
        "ZERO metrics repeated (no two bullets share the same number/percentage).\n"
        "ZERO sentence structures repeated.\n"
        "NEVER 'web application' or 'full stack app' as deliverable - name the specific system.\n\n"

        f"VERB GUIDE:\n{verbs}\n\n"

        "BANNED: tools outside locked list, <6 tech tags, repeated bullets, repeated verbs, repeated metrics."
    )

    user = f"""Job: {req.job_title}
Experience: {total_years} years

COMPANIES (exact names and dates):
{co_block}

JD context:
{jd}

Output JSON:
{{"companies":[
  {{"company":"EXACT NAME","role":"Seniority JD-Domain UniqueFunction",
    "dateRange":"Start - End",
    "bullets":["20-30w bullet 1","20-30w bullet 2","20-30w bullet 3","20-30w bullet 4"],
    "tech":"JD-Tech1 | JD-Tech2 | JD-Tech3 | JD-Tech4 | JD-Tech5 | JD-Tech6"}},
  ...({num_cos} total)
]}}"""

    return system, user


def build_stage3_prompt(req: CVRequest, stage1: dict, stage2: dict) -> tuple:
    jd           = req.job_description.strip()[:1400]
    company_name = (req.company_name    or "").strip()
    company_ctx  = (req.company_context or "").strip()[:1500]

    must   = stage1.get("technologies", {}).get("mustHave", [])
    nice   = stage1.get("technologies", {}).get("niceToHave", [])
    seen_p = set()
    allowed = []
    for t in must + nice:
        if t and t.lower() not in seen_p:
            seen_p.add(t.lower()); allowed.append(t)
    allowed_str = ", ".join(allowed[:30]) if allowed else "JD technologies"

    used = []
    for co in stage2.get("companies", []):
        for b in co.get("bullets", []):
            used.append(b[:80])
    used_str = "\n".join(f"  - {s}" for s in used[:12]) if used else "  (none yet)"

    if company_ctx and company_name:
        co_intel = (
            f"TARGET COMPANY: {company_name}\n"
            f"COMPANY DATA (from website/search):\n{company_ctx[:800]}\n\n"
            "PROJECTS 3 & 4 - COMPANY-DOMAIN ALIGNMENT:\n"
            f"Read the company data above. Identify their CORE BUSINESS DOMAIN and PRODUCT PORTFOLIO.\n"
            "Project 3: Build a project in an ADJACENT SECTOR using the SAME underlying tech pattern as the company's core product - using JD technologies only. Do NOT copy their product.\n"
            "Project 4: Build a project inspired by the company's SECONDARY services or industry reputation - different adjacent domain, JD technologies only.\n"
            "ANALOGY RULE: Abstract the company's business pattern (e.g. multi-tenant SaaS, marketplace, fleet management, billing engine) then apply it to a different sector.\n"
            "All 4 projects must feel like real-world systems a developer at this company would build.\n"
        )
    elif company_name:
        co_intel = (
            f"TARGET COMPANY: {company_name}\n"
            f"Use your knowledge of {company_name}'s products, services, and industry positioning.\n"
            "Project 3: Adjacent-sector project mirroring their core product pattern, using JD technologies.\n"
            "Project 4: Project inspired by their secondary offerings or market reputation, different sector, JD technologies.\n"
            f"NEVER copy {company_name}'s exact product - build analogous systems in parallel domains.\n"
        )
    else:
        co_intel = (
            "No company provided - all 4 projects are JD-driven.\n"
            "Derive 4 different system types from the JD's tech stack, responsibilities, and domain signals.\n"
        )

    system = (
        "You are an expert ATS-focused CV writer. Output ONLY valid JSON. No markdown, no backticks. Start { end }.\n\n"

        f"=== LOCKED TECHNOLOGIES (use ONLY these in projects and relatedTech) ===\n"
        f"{allowed_str}\n\n"

        "=== 4 PROJECTS ===\n"
        "Each project: a coined SaaS-style product name + 3-4 sentence overview + 3 bullets.\n\n"

        "NAMING: '[CoinedOriginalWord] - [what it does using JD tech]'\n"
        "Product name: an original coined word (e.g. NexaFlow, VaultIQ, ShiftSync, PulseGrid).\n"
        "BANNED names: 'ERP Platform', 'Web App', 'Social Media Platform', 'Business Intelligence Platform', real company names.\n\n"

        "OVERVIEW (3-4 sentences, MANDATORY structure per project):\n"
        "  Sentence 1 - PROBLEM: who is affected, what they lose without this system\n"
        "  Sentence 2 - SOLUTION: architecture designed + 2 specific JD technologies used\n"
        "  Sentence 3 - FUNCTIONALITY: what the system does, how users interact\n"
        "  Sentence 4 - BUSINESS IMPACT: one unique measurable outcome (number unique per project)\n"
        "Each overview opens with a DIFFERENT sentence structure across the 4 projects.\n"
        "ZERO repeated metrics across all 4 projects.\n\n"

        "BULLETS (3 per project, 20-30 words each):\n"
        "  Bullet 1: specific component built + JD technology\n"
        "  Bullet 2: hardest challenge + unique metric (different from all other project metrics)\n"
        "  Bullet 3: business outcome + unique number\n"
        "Zero repeated bullet structures across the 4 projects.\n\n"

        "=== 6 RELATED TECH BOXES ===\n"
        "6 category boxes. Each box: EXACTLY 10-13 items from the JD ecosystem. Fewer than 10 items per box is a HARD FAILURE.\\n"
        "ZERO items repeated across any of the 6 boxes.\n"
        "Category names reflect actual JD technical domains - not generic labels.\n\n"

        "BANNED: generic project names, <3 sentence overviews, repeated metrics, "
        "real company names in projects, technologies outside locked list, <10 relatedTech items per box, "
        "fewer than 5 techTags per project, fake tech words in techTags (Cloud, Development, APIs, Web, Good, Strong)."

    )

    user = f"""Job: {req.job_title}

{co_intel}

JD context:
{jd}

SYSTEMS ALREADY IN EXPERIENCE (do NOT repeat in projects):
{used_str}

Output JSON:
{{"projects":[
  {{"name":"CoinedWord - what it does [JD-Tech1, JD-Tech2]",
    "overview":"PROBLEM sentence. SOLUTION sentence (2 JD techs). FUNCTIONALITY sentence. BUSINESS IMPACT (unique number).",
    "bullets":["component + JD tech (20-30w)","challenge + unique metric (20-30w)","outcome + unique number (20-30w)"]}},
  ...(4 projects: 1&2 JD-driven, 3&4 company-domain-driven per above rules)
],
"relatedTech":[
  {{"category":"JD Technical Domain","items":["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11"]}},
  ...(6 boxes, 10-13 items each, ZERO duplicates across boxes)
]}}"""

    return system, user


async def _cerebras_call_single(client, key: str, model: str,
                                 system: str, user: str, stage_name: str) -> dict:
    r = await client.post(
        CEREBRAS_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model":       model,
            "messages":    [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.2,
            "max_tokens":  4096,
        }
    )
    if r.status_code == 429:
        raise ValueError(f"429 rate limited on stage {stage_name}")
    if r.status_code in (401, 403):
        raise ValueError(f"401/403 invalid key on stage {stage_name}")
    if r.status_code != 200:
        body = {}
        try: body = r.json()
        except: pass
        msg = (body.get("error") or {}).get("message", f"HTTP {r.status_code}")
        raise ValueError(f"Stage {stage_name}: {msg}")
    raw = r.json()["choices"][0]["message"]["content"]
    return extract_json(raw)


def _merge_stages(stage1: dict, stage2: dict, stage3: dict, req: CVRequest) -> dict:
    years_exp   = (req.years_exp or "").strip()
    total_years = _calc_total_years(years_exp)
    edu         = _build_education_year(years_exp)
    cv = {
        "totalYears":    total_years,
        "title":         stage1.get("title", ""),
        "summary":       stage1.get("summary", ""),
        "skills":        stage1.get("skills", []),
        "competencies":  stage1.get("competencies", ""),
        "keywords":      stage1.get("keywords", ""),
        "technologies":  stage1.get("technologies", {"mustHave":[],"niceToHave":[],"additional":[]}),
        "architectures": stage1.get("architectures", []),
        "companies":     stage2.get("companies", []),
        "projects":      stage3.get("projects", []),
        "relatedTech":   stage3.get("relatedTech", []),
        "education": {
            "university":  "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
            "degree":      "Bachelor of Science in Computer Science (BSCS)",
            "cgpa":        "3.97/4.0",
            "years":       f"{edu['start']} - {edu['end']}",
            "achievement": "Gold Medalist for Academic Excellence",
        },
    }
    _techs_for_enforce = {
        "core":      stage1.get("technologies", {}).get("mustHave", []),
        "preferred": stage1.get("technologies", {}).get("niceToHave", []),
        "ecosystem": stage1.get("technologies", {}).get("additional", []),
    }
    cv_sanitised  = sanitise_cv(cv)
    cv_companies  = fix_companies(cv_sanitised)
    cv_skills     = fix_skills(cv_companies)
    cv_enforced   = _enforce_skill_domains(cv_skills, _techs_for_enforce, req.job_title)
    cv_projtags   = _repair_project_tech_tags(cv_enforced, _techs_for_enforce)
    cv_polished   = final_polish(fix_skills_dedup(fix_projects(cv_projtags)), years_exp=years_exp)
    return run_validation_pipeline(cv_polished, req.job_description, req.job_title, _techs_for_enforce)


# ===================================================================
# 5-PIPELINE ATOMIC GENERATION - No prompt ever truncated
# Each pipeline is <1500 tokens, well under all model limits
# ===================================================================

def build_pipeline1_tech(req: CVRequest) -> tuple:
    jd = req.job_description.strip()[:1000]

    system = (
        "You are a senior software architect. Output ONLY valid JSON. No explanations.\n\n"
        "MISSION: Extract every technology from the JD, then for each one, "
        "add its standard production companion tools.\n"
        "For example:\n"
        "  - A real-time protocol -> add its broker, client library, message format, monitoring tool\n"
        "  - A frontend language -> add its build tool, bundler, linter, test runner, UI library\n"
        "  - A backend language -> add its framework, ORM, auth lib, task queue, test framework\n"
        "  - A database -> add its migration tool, admin UI, caching layer, connection pooler\n"
        "  - A cloud service -> add related managed services from the same provider\n"
        "Apply companion expansion to EVERY technology found in the JD.\n"
        "Target: at least 30 distinct named tools across all three arrays.\n"
        "Every item must be a real named tool - no generic words."
    )

    user = f"""Job Title: {req.job_title}

Job Description:
{jd}

Step 1: List every technology explicitly named in the JD -> these are 'core' or 'preferred'.
Step 2: For each technology found, add 3-5 standard companion tools -> these are 'ecosystem'.
Step 3: Output the result.

Output JSON (aim for 30+ total items):
{{"core": ["tech1", "tech2", ...], "preferred": ["tech1", ...], "ecosystem": ["companion1", "companion2", ...]}}

Rules:
- core: required/mandatory technologies from the JD
- preferred: nice-to-have technologies from the JD  
- ecosystem: companion tools for each core/preferred tech (3-5 companions per tech)
- Every item must be a real named technology
- Minimum 30 total items across all three arrays
- No generic words like 'platform', 'system', 'tool', 'framework' alone"""

    return system, user

def build_pipeline2_title_summary(req: CVRequest, techs: dict) -> tuple:
    """Pipeline 2: Generate title + summary - 600 tokens max"""
    jd = req.job_description.strip()[:800]
    total_years = _calc_total_years(req.years_exp or "")
    core_techs = techs.get("core", [])[:6]
    techs_str = ", ".join(core_techs) if core_techs else "technologies from the job description"
    
    system = "You are an expert CV writer. Output ONLY valid JSON. No examples."
    
    user = f"""Job Title: {req.job_title}
Experience: {total_years} years
Technologies from this JD: {techs_str}

Title rules (STRICT - MUST FOLLOW):
- MUST be DIFFERENT from "{req.job_title}" - never copy it directly
- Change at least 2 words or restructure completely
- Use a DIFFERENT function word (choices: Engineer, Developer, Specialist, Architect, Analyst, Programmer, Consultant, Designer)
- Use 3 different technologies from the list above
- Format: "Transformed Title | Tech1, Tech2, Tech3"

Examples of acceptable transformations (these are FORMAT examples only - use technologies from the JD, not these):
- "[Role A]" -> "[Different Function Word] | [JD-Tech1], [JD-Tech2], [JD-Tech3]"
- "[Role B]" -> "[Domain]-Focused [Function] | [JD-Tech1], [JD-Tech2], [JD-Tech3]"
- "[Role C]" -> "[Scope-shifted Title] | [JD-Tech1], [JD-Tech2], [JD-Tech3]"

NEVER output the exact job title. ALWAYS transform it using the rules above.

Summary rules:
- 4 sentences, minimum 70 words
- Start with "{total_years} years of experience in [domain from JD]..."
- Name 4-5 technologies from the list above
- Include metrics and outcomes

Output JSON:
{{"title": "...", "summary": "..."}}"""
    
    return system, user

def build_pipeline2_summary_only(req: CVRequest, techs: dict) -> tuple:
    """Pipeline 2B: Generate ONLY the summary - 600 tokens max"""
    jd = req.job_description.strip()[:800]
    total_years = _calc_total_years(req.years_exp or "")
    core_techs = techs.get("core", [])[:6]
    techs_str = ", ".join(core_techs) if core_techs else "technologies from the job description"
    
    system = "You are an expert CV writer. Output ONLY valid JSON. No examples."
    
    user = f"""Job Title: {req.job_title}
Experience: {total_years} years
Technologies from this JD: {techs_str}

Summary rules:
- 4 sentences, minimum 70 words
- Start with "{total_years} years of experience in [domain from JD]..."
- Name 4-5 technologies from the list above
- Include metrics and outcomes
- NO location words, NO company names in summary

Output JSON:
{{"summary": "..."}}"""
    
    return system, user

def build_pipeline_title_only(req: CVRequest, techs: dict) -> tuple:
    """Dedicated pipeline for generating ONLY the CV title - no location, no company names, pure role focus"""
    jd = req.job_description.strip()[:600]
    total_years = _calc_total_years(req.years_exp or "")
    core_techs = techs.get("core", [])[:6]
    techs_str = ", ".join(core_techs) if core_techs else "technologies from the job description"
    job_title_lower = req.job_title.lower()
    
    # Determine if it's senior/junior based on years
    seniority_prefix = ""
    try:
        yrs = float(total_years.replace('+', '').strip())
        if yrs >= 5:
            seniority_prefix = "Senior "
        elif yrs <= 2:
            seniority_prefix = "Junior "
    except:
        pass
    
    system = """You are an expert CV title writer. Output ONLY valid JSON. No explanations, no markdown, no backticks.

CRITICAL RULES:
- NEVER include location words (Dallas, Texas, Pakistan, Remote, USA, UK, London, etc.)
- NEVER include company names
- NEVER include words like "Based", "Located", "Remote" in the title
- Title must be PURELY about the role and technology domain
- Use simple format: "[Optional Seniority][Domain] [Function] | [Tech1], [Tech2], [Tech3]"
- Function words allowed: Engineer, Developer, Specialist, Architect (only if in JD), Analyst, Programmer
- Keep title under 60 characters before the pipe

BAD examples (REJECT):
- "Dallas-Based Software Architect" -> has location
- "Remote [Role]" -> has location word
- "Pakistan [Role]" -> has location word
- "Company X Lead Developer" -> has company name

GOOD examples (FORMAT only - use JD technologies, not these):
- "[Domain] [Function] | [JD-Tech1], [JD-Tech2], [JD-Tech3]"
- "[Seniority] [Domain] [Function] | [JD-Tech1], [JD-Tech2], [JD-Tech3]"
"""

    user = f"""Job Title: {req.job_title}
Experience: {total_years} years
Technologies from JD: {techs_str}
Suggested seniority: {seniority_prefix}

TASK: Generate a clean, professional CV title following ALL rules above.

Output JSON:
{{"title": "{seniority_prefix}{req.job_title.split('|')[0].strip().split('(')[0].strip()} | {core_techs[0] if core_techs else 'Technology'}, {core_techs[1] if len(core_techs) > 1 else 'Technology'}, {core_techs[2] if len(core_techs) > 2 else 'Technology'}"}}"""

    return system, user

def build_pipeline3a_skills(req: CVRequest, techs: dict) -> tuple:
    """Pipeline 3A: Generate ONLY Technical Skills - dedicated pipeline.
    Produces exactly 5 categories, each with exactly 10-12 items,
    all strictly derived from the JD's technology stack.
    """
    jd = req.job_description.strip()[:800]
    core     = techs.get("core", [])
    preferred = techs.get("preferred", [])
    ecosystem = techs.get("ecosystem", [])

    # Build a rich allowed list: core first, then preferred, then ecosystem
    all_techs = list(dict.fromkeys(core + preferred + ecosystem))[:50]
    techs_str = ", ".join(all_techs) if all_techs else "(derive from JD below)"

    system = (
        "You are an expert CV technical skills writer. Output ONLY valid JSON. "
        "No markdown, no backticks, no explanations. Start { end }.\n\n"
        "MISSION: Produce exactly 5 skill categories for a CV 'Technical Skills' section.\n"
        "EVERY item must come from the technologies list provided - nothing invented.\n"
        "Each category MUST have exactly 10 items (hard minimum and maximum).\n"
        "Assign each technology to ONE category only - zero items repeated across categories.\n"
        "Category names MUST reflect the JD's actual technical domains "
        "(e.g. 'Backend Frameworks', 'Cloud & DevOps', 'Frontend Technologies', "
        "'Database & Storage', 'Testing & Quality'). "
        "Never use generic names like 'Skills' or 'Tools'."
    )

    user = f"""Job Title: {req.job_title}

Job Description:
{jd}

Allowed Technologies (use ONLY technologies from this list - nothing else):
{techs_str}

TASK: Create exactly 5 skill categories. Each category must have EXACTLY 10 items.

RULES:
1. Category names come from the JD's actual technical domains - infer from the job title and JD.
2. Populate each category by grouping related technologies from the allowed list.
3. If a domain has fewer than 10 items in the allowed list, include the closest ecosystem companions standard to that domain.
4. ZERO items repeated across any two categories.
5. EXACTLY 5 categories - no more, no less.
6. EXACTLY 10 items per category - no more, no less.

Output JSON (exactly this shape):
{{"skills": [
  "Category Name 1: item1, item2, item3, item4, item5, item6, item7, item8, item9, item10",
  "Category Name 2: item1, item2, item3, item4, item5, item6, item7, item8, item9, item10",
  "Category Name 3: item1, item2, item3, item4, item5, item6, item7, item8, item9, item10",
  "Category Name 4: item1, item2, item3, item4, item5, item6, item7, item8, item9, item10",
  "Category Name 5: item1, item2, item3, item4, item5, item6, item7, item8, item9, item10"
]}}"""

    return system, user


def build_pipeline3b_competencies(req: CVRequest, techs: dict) -> tuple:
    """Pipeline 3B: Generate ONLY Core Competencies - dedicated pipeline.
    Produces exactly 10 domain-specific 2-4 word phrases from the JD.
    """
    jd = req.job_description.strip()[:700]
    core_str = ", ".join(techs.get("core", [])[:12])

    system = (
        "You are an expert CV writer. Output ONLY valid JSON. "
        "No markdown, no backticks, no explanations. Start { end }.\n\n"
        "MISSION: Produce exactly 10 core competency phrases for a CV.\n"
        "ALL phrases must be derived from the job description's domain - zero generic soft skills.\n"
        "Each phrase: 2-4 words. Format: separated by ' * '.\n"
        "Span these 4 areas:\n"
        "  Technical Practices (e.g. 'API Design', 'Test-Driven Development')\n"
        "  Domain Expertise (e.g. 'Cloud-Native Architecture', 'Real-Time Analytics')\n"
        "  Engineering Process (e.g. 'CI/CD Pipeline Automation', 'Agile Sprint Delivery')\n"
        "  Impact Areas (e.g. 'Performance Optimisation', 'System Scalability')\n"
        "BANNED: 'Problem Solving', 'Teamwork', 'Communication', 'Leadership', "
        "any phrase not derived from this specific JD's tech domain."
    )

    user = f"""Job Title: {req.job_title}
Core Technologies from JD: {core_str}

Job Description:
{jd}

TASK: Generate exactly 10 core competency phrases.

RULES:
1. Each phrase: 2-4 words, directly tied to the JD's technical domain.
2. Must span all 4 areas: Technical Practices, Domain Expertise, Engineering Process, Impact Areas.
3. Derive ALL from the JD - read the responsibilities, tech stack, and outcomes mentioned.
4. Zero generic phrases (no 'Problem Solving', 'Teamwork', 'Communication').
5. Exactly 10 phrases - no more, no less.

Output JSON:
{{"competencies": "Phrase1 * Phrase2 * Phrase3 * Phrase4 * Phrase5 * Phrase6 * Phrase7 * Phrase8 * Phrase9 * Phrase10"}}"""

    return system, user


def build_pipeline4_experience(req: CVRequest, techs: dict) -> tuple:
    years_exp   = (req.years_exp or "").strip()
    companies   = _build_dynamic_companies(years_exp)
    total_years = _calc_total_years(years_exp)
    num_cos     = len(companies)

    # Build full deduped tech list
    core      = techs.get("core", [])
    preferred = techs.get("preferred", [])
    ecosystem = techs.get("ecosystem", [])
    all_techs = list(dict.fromkeys(core + preferred + ecosystem))

    # Slice into non-overlapping pools - one per company
    # Co1 gets first slice (most advanced), Co2 middle, Co3 last (foundational)
    chunk = max(6, len(all_techs) // num_cos) if num_cos > 0 else len(all_techs)
    co_tech_pools = []
    for i in range(num_cos):
        start = i * chunk
        end   = start + chunk if i < num_cos - 1 else len(all_techs)
        pool  = all_techs[start:end]
        # Guarantee minimum 6 items by padding from core (without duplicating)
        if len(pool) < 6:
            for t in core:
                if t not in pool:
                    pool.append(t)
                if len(pool) >= 6:
                    break
        co_tech_pools.append(pool[:10])  # cap at 10 per company

    if num_cos == 1:
        seniority  = ["Junior"]
        verb_guide = "Co1 (Junior): Implemented, Built, Developed, Deployed, Configured, Automated."
    elif num_cos == 2:
        seniority  = ["", "Junior"]
        verb_guide = (
            "Co1 (current, no prefix): Developed, Engineered, Designed, Integrated, Streamlined, Built.\n"
            "Co2 (Junior): Implemented, Configured, Deployed, Automated, Assisted, Instrumented."
        )
    else:
        seniority  = ["Senior", "", "Junior"]
        verb_guide = (
            "Co1 (Senior): Architected, Engineered, Led, Spearheaded, Established, Launched, Designed.\n"
            "Co2 (mid, no prefix): Optimised, Refactored, Scaled, Migrated, Integrated, Streamlined, Unified.\n"
            "Co3 (Junior): Implemented, Built, Developed, Deployed, Configured, Automated, Instrumented."
        )

    # Build co_block with explicit locked pool per company
    co_lines = []
    for i, co in enumerate(companies):
        prefix = seniority[i] if i < len(seniority) else ""
        pool_str = " | ".join(co_tech_pools[i]) if i < len(co_tech_pools) else ""
        co_lines.append(
            f'Co{i+1}: company="{co["name"]}", dates="{co["start"]} - {co["end"]}"'
            + (f', role MUST start with "{prefix}"' if prefix else ", role has NO seniority prefix")
            + f'\n  *** LOCKED TECH POOL (use ONLY these for Co{i+1} tech tags AND bullets): {pool_str} ***'
            + f'\n  Tech tags: pick exactly 6-8 from the locked pool above - DIFFERENT from every other company.'
        )
    co_block = "\n\n".join(co_lines)

    system = (
        "You are an expert ATS-focused CV writer. Output ONLY valid JSON. No markdown, no backticks. Start { end }.\n\n"

        "=== TECH TAG UNIQUENESS - THIS IS THE #1 RULE ===\n"
        "Each company has a LOCKED tech pool printed below. "
        "You MUST pick tech tags ONLY from that company's locked pool.\n"
        "HARD RULE: A technology that appears in Co1's tags MUST NOT appear in Co2's or Co3's tags.\n"
        "HARD RULE: A technology that appears in Co2's tags MUST NOT appear in Co3's tags.\n"
        "HARD RULE: Zero tech overlap across any two companies - every company's tag set is 100% unique.\n"
        "HARD RULE: Bullets for each company must reference technologies from THAT company's locked pool.\n\n"

        "=== BULLET RULES ===\n"
        "4 bullets per company. Each bullet 20-30 words.\n"
        "All 12 bullets describe 12 COMPLETELY DIFFERENT systems or features.\n"
        "ZERO verbs repeated across all 12 bullets.\n"
        "ZERO metrics repeated - every number/percentage is unique.\n"
        "ZERO sentence structures repeated.\n"
        "Name the SPECIFIC system type in every bullet - never say 'web application' or 'full stack app'.\n\n"

        f"VERB GUIDE:\n{verb_guide}\n\n"

        "ROLE TITLES:\n"
        "Each company gets a UNIQUE role title derived from the JD domain.\n"
        "Format: '[Seniority] [JD Domain] [Function]'\n"
        "Function pool (no repeats): Engineer, Developer, Specialist, Analyst, Consultant, Programmer, Architect, Designer\n\n"

        "BANNED: tech tags shared across companies, technologies outside each company's locked pool, "
        "repeated verbs, repeated metrics, repeated sentence structures."
    )

    user = f"""Job: {req.job_title}
Experience: {total_years} years

COMPANIES WITH LOCKED TECH POOLS:
{co_block}

Output JSON:
{{"companies":[
  {{"company":"EXACT NAME","role":"Seniority JD-Domain Function",
    "dateRange":"Start - End",
    "bullets":["20-30w bullet using Co1 pool tech","20-30w bullet","20-30w bullet","20-30w bullet"],
    "tech":"Co1-Pool-Tech1 | Co1-Pool-Tech2 | Co1-Pool-Tech3 | Co1-Pool-Tech4 | Co1-Pool-Tech5 | Co1-Pool-Tech6"}},
  ...({num_cos} total, each using ONLY its own locked pool)
]}}"""

    return system, user

def build_pipeline4_roles_only(req: CVRequest, techs: dict, companies_list: list) -> tuple:
    """Dedicated pipeline for generating ONLY company roles and details - no location words, pure role focus"""
    jd = req.job_description.strip()[:1000]
    total_years = _calc_total_years(req.years_exp or "")
    num_cos = len(companies_list)
    
    # Build company list with their years ranges
    company_details = []
    for i, co in enumerate(companies_list):
        company_details.append(f"Company {i+1}: {co['name']} ({co['start']} - {co['end']})")
    companies_str = "\n".join(company_details)
    
    # Determine seniority based on years and position
    try:
        yrs = float(total_years.replace('+', '').strip())
        if yrs >= 5:
            seniority_pattern = "Company 1 (most recent): Senior, Company 2: (no prefix), Company 3 (oldest): Junior"
        elif yrs >= 3:
            seniority_pattern = "Company 1 (most recent): Senior, Company 2: (no prefix), Company 3 (oldest): Junior"
        elif yrs >= 2:
            seniority_pattern = "Company 1 (most recent): (no prefix), Company 2 (oldest): Junior"
        else:
            seniority_pattern = "Company 1 (only company): Junior"
    except:
        seniority_pattern = "Company 1: Senior, Company 2: (no prefix), Company 3: Junior"
    
    # Get technologies and function words
    all_techs = techs.get("core", []) + techs.get("preferred", [])
    all_techs = list(dict.fromkeys(all_techs))[:20]
    techs_for_roles = ", ".join(all_techs[:12])
    
    system = """You are an expert CV writer specializing in professional role titles. Output ONLY valid JSON. No markdown, no backticks.

CRITICAL RULES FOR ROLE TITLES:
- ABSOLUTELY NO location words (Dallas, Texas, Pakistan, Remote, USA, London, etc.) in any role title
- ABSOLUTELY NO company names in role titles
- NO words like "Based", "Located", "Remote" in role titles
- Each role title MUST be unique within the CV
- Format: "[Seniority] [Domain Focus] [Function Word]"
- Function words pool (no repeats): Engineer, Developer, Specialist, Analyst, Programmer

Domain Focus examples (pick from JD):
- Backend, Frontend, Full-Stack, Data, Cloud, Security, API, Database, DevOps, ML, IoT

Seniority rules based on company position:
- Most recent company (Company 1): Use Senior if experience >= 3 years, otherwise no prefix
- Middle company (Company 2): No seniority prefix
- Oldest company (Company 3): Use Junior prefix
- If only 2 companies: Current has no prefix, Previous has Junior
- If only 1 company: Use Junior for entry-level (1-2 years), no prefix for mid-level (3+ years)

BULLETS RULES (4 per company, 20-30 words each):
- Each bullet MUST name at least 2 specific technologies from the JD
- Each bullet MUST describe a DIFFERENT system or feature
- Each bullet MUST have a UNIQUE metric (number, percentage, user count)
- NEVER start two bullets with the same verb
- NEVER use 'web application' or 'full stack app' - name specific system type

TECH TAGS (6-8 technologies per company):
- Use technologies actually present in the JD
- Company 1: most advanced/architectural technologies
- Company 2: mid-level implementation technologies
- Company 3: foundational technologies
- NO technology can appear in more than one company's tech tags

GOOD role title examples FORMAT (ACCEPT - use JD domain, not these specific words):
- "[Seniority] [JD-Domain] [Function]"   e.g. Senior [JD domain word] Engineer
- "[JD-Domain] [Function]"               e.g. [JD domain word] Developer
- "Junior [JD-Domain] [Function]"        e.g. Junior [JD domain word] Specialist

BAD role title examples (REJECT):
- "[Role] Dallas" -> has location
- "Remote [Role]" -> has location
- "Pakistan [Role]" -> has location"""

    user = f"""Job Title from JD: {req.job_title}
Total Experience: {total_years} years
Seniority Pattern: {seniority_pattern}

COMPANIES (exact names and dates - DO NOT change these):
{companies_str}

Technologies available from JD (use ONLY these):
{techs_for_roles}

JD Summary (for context on domain focus):
{jd[:400]}

TASK: Generate professional role titles and work details for each company following ALL rules above.

Output JSON:
{{
    "companies": [
        {{
            "role": "Senior [JD-Domain] [Function]",
            "bullets": [
                "[Senior-level achievement verb] [specific system] using [JD-Tech1] and [JD-Tech2], achieving [metric].",
                "[Senior-level verb] [specific deliverable] with [JD-Tech3], resulting in [unique metric].",
                "[Senior-level verb] [specific outcome] using [JD-Tech4], improving [measure] by [unique %].",
                "[Senior-level verb] [specific system] with [JD-Tech5], enabling [business outcome] for [scale]."
            ],
            "tech": "[JD-Tech1] | [JD-Tech2] | [JD-Tech3] | [JD-Tech4] | [JD-Tech5] | [JD-Tech6] | [JD-Tech7]"
        }},
        {{
            "role": "[JD-Domain] [Function]",
            "bullets": [
                "[Mid-level verb] [specific system] using [JD-Tech1] and [JD-Tech2], delivering [unique metric].",
                "[Mid-level verb] [specific component] with [JD-Tech3], improving [measure] by [unique %].",
                "[Mid-level verb] [specific deliverable] using [JD-Tech4], reducing [measure] from X to Y.",
                "[Mid-level verb] [specific feature] with [JD-Tech5], serving [scale] with [unique outcome]."
            ],
            "tech": "[JD-Tech1] | [JD-Tech2] | [JD-Tech3] | [JD-Tech4] | [JD-Tech5] | [JD-Tech6] | [JD-Tech7]"
        }},
        {{
            "role": "Junior [JD-Domain] [Function]",
            "bullets": [
                "[Junior verb] [specific system] using [JD-Tech1], contributing to [outcome] for [scale].",
                "[Junior verb] [specific component] with [JD-Tech2] and [JD-Tech3], improving [measure] by [unique %].",
                "[Junior verb] [specific deliverable] using [JD-Tech4], reducing [measure] by [unique number].",
                "[Junior verb] [specific feature] with [JD-Tech5], increasing [measure] by [unique %]."
            ],
            "tech": "[JD-Tech1] | [JD-Tech2] | [JD-Tech3] | [JD-Tech4] | [JD-Tech5] | [JD-Tech6] | [JD-Tech7]"
        }}
    ]
}}

IMPORTANT: 
- Adjust number of companies based on the actual companies provided ({num_cos} companies)
- If 2 companies, provide only 2 objects
- If 1 company, provide only 1 object
- Adjust seniority prefix (Senior/ no prefix/ Junior) based on company position
- Each company's role title MUST be unique and from a different function word
- Bullet metrics must be realistic and varied (not all percentages, not all user counts)
- Tech tags must be relevant to each company's seniority level"""

    return system, user

def assign_project_techs(techs: dict, num_projects: int = 4) -> list:
    """
    Simply slice the full tech list into 4 non-overlapping groups.
    Each group becomes one project's tech pool - guaranteeing zero overlap.
    """
    core      = techs.get("core", [])
    preferred = techs.get("preferred", [])
    ecosystem = techs.get("ecosystem", [])
    all_techs = list(dict.fromkeys(core + preferred + ecosystem))

    if len(all_techs) < num_projects:
        return []

    # Split into num_projects roughly equal slices
    chunk = max(1, len(all_techs) // num_projects)
    assignments = []
    for i in range(num_projects):
        start   = i * chunk
        end     = start + chunk if i < num_projects - 1 else len(all_techs)
        group   = all_techs[start:end]
        primary = group[0] if group else None
        if not primary:
            continue
        assignments.append({
            "primary":    primary,
            "supporting": group[1:6],
            "cluster":    f"group_{i+1}",
        })
    return assignments

def build_pipeline5a_projects(req: CVRequest, techs: dict, experience: dict) -> tuple:
    """Pipeline 5A: Generate ONLY 4 projects - each locked to a unique primary technology."""
    company_name = (req.company_name or "").strip()
    company_ctx  = (req.company_context or "").strip()[:700]
    all_techs    = techs.get("core", []) + techs.get("preferred", []) + techs.get("ecosystem", [])
    all_techs    = list(dict.fromkeys(all_techs))[:30]
    techs_str    = ", ".join(all_techs) if all_techs else "technologies from the job description"

    used_systems = []
    for co in experience.get("companies", []):
        for bullet in co.get("bullets", []):
            if len(bullet) > 20:
                used_systems.append(bullet[:60])
    used_str = "\n".join([f"  - {s}" for s in used_systems[:6]]) if used_systems else "  (none)"

    company_intel = ""
    if company_ctx and company_name:
        company_intel = (
            f"Target Company: {company_name}\n"
            f"Company Context: {company_ctx[:400]}\n"
            "Projects 3 & 4 should be analogous to this company's domain - "
            "use the same underlying tech pattern but in a different adjacent sector.\n"
        )
    elif company_name:
        company_intel = (
            f"Target Company: {company_name}\n"
            f"Projects 3 & 4: Use your knowledge of {company_name}'s industry "
            "to build analogous projects in adjacent sectors. "
            f"Do NOT copy {company_name}'s exact product.\n"
        )

    # Pre-assign unique primary tech per project
    assignments = assign_project_techs(techs, num_projects=4)

    if assignments:
        tech_assignments_block = (
            "=== MANDATORY TECHNOLOGY ASSIGNMENT PER PROJECT ===\n"
            "Each project is PRE-ASSIGNED a unique primary technology. "
            "You MUST use that technology as the core of that project. "
            "NEVER use another project's primary technology as the core of a different project.\n\n"
        )
        for i, a in enumerate(assignments):
            supporting_str = ", ".join(a["supporting"][:4])
            tech_assignments_block += (
                f"Project {i+1} PRIMARY TECHNOLOGY: {a['primary']}\n"
                f"Project {i+1} SUPPORTING TECHNOLOGIES (pick 3-4): {supporting_str}\n"
                f"Project {i+1} TECH TAGS must include '{a['primary']}' and 3-4 from supporting list above.\n\n"
            )
        tech_assignments_block += (
            "HARD RULE: Project 1's primary tech must NOT appear as the primary tech in Projects 2, 3, or 4.\n"
            "HARD RULE: No two projects may share the same primary technology.\n"
            "HARD RULE: Each project's bullet B1 must name that project's assigned primary technology.\n"
        )
    else:
        tech_assignments_block = (
            f"Allowed Technologies (use ONLY these): {techs_str}\n"
            "Each project must use a DIFFERENT primary technology as its core.\n"
        )

    system = (
        "You are an expert CV writer. Output ONLY valid JSON. "
        "No markdown, no backticks. Start { end }.\n\n"
        "MISSION: Generate exactly 4 COMPLETELY DIFFERENT projects. "
        "Each project must be in a DIFFERENT problem domain with a DIFFERENT primary technology.\n\n"

        "=== DIVERSITY HARD RULES ===\n"
        "- All 12 bullets (3 per project x 4 projects) must open with a DIFFERENT verb - zero repeats.\n"
        "- All 4 overviews must describe DIFFERENT user personas (not all 'enterprise administrators').\n"
        "- All 4 business impact metrics must use DIFFERENT units or formats.\n"
        "- All 4 system domains must be DIFFERENT (not all 'real-time dashboards').\n"
        "- NEVER use 'JavaScript and WebSocket' as the solution in more than ONE project.\n\n"

        "PROJECT NAMING:\n"
        "- Each name: a coined original SaaS-style word followed by ' - ' and what it does.\n"
        "- BANNED: 'ERP Platform', 'Web App', 'Social Media Platform', real company names.\n\n"

        "OVERVIEW (3-4 sentences, MANDATORY structure):\n"
        "  S1 PROBLEM: specific user persona + what they lose - DIFFERENT persona each project.\n"
        "  S2 SOLUTION: architecture using the project's ASSIGNED primary technology + 1 supporting tech.\n"
        "  S3 FUNCTIONALITY: key features of this specific system type.\n"
        "  S4 BUSINESS IMPACT: unique measurable outcome - number and unit differ across all 4 projects.\n\n"

        "BULLETS (3 per project):\n"
        "  B1: component built using the project's ASSIGNED primary technology (20-30 words).\n"
        "  B2: hardest technical challenge + concrete metric - metric format unique across all projects (20-30 words).\n"
        "  B3: business outcome + unique number - unit must differ from all other B3s (20-30 words).\n"
        "ZERO repeated verbs across all 12 bullets.\n\n"

        "TECH TAGS: Each project gets 5-7 tags. "
        "Tags must start with the project's ASSIGNED primary technology. "
        "Remaining tags come from that project's supporting list. "
        "NEVER use the same tag set across two projects.\n\n"

        "BANNED: repeated primary tech across projects, repeated verbs, repeated metric formats, "
        "all projects being 'real-time dashboards', generic names, <3-sentence overviews, "
        "real company names, technologies outside the allowed list."
    )

    user = f"""Job Title: {req.job_title}
Job Description: {req.job_description.strip()[:500]}

{tech_assignments_block}
{company_intel}
Systems already in experience (do NOT repeat these system types):
{used_str}

Generate EXACTLY 4 projects. Each must be in a different domain with its assigned primary technology.

Output JSON:
{{"projects":[
  {{"name":"CoinedWord - what it does","overview":"PROBLEM (specific persona). SOLUTION (assigned primary tech + 1 supporting). FUNCTIONALITY. BUSINESS IMPACT (unique metric).","bullets":["component + assigned primary tech (20-30w)","challenge + unique metric format (20-30w)","outcome + unique number+unit (20-30w)"],"techTags":["AssignedPrimaryTech","supporting1","supporting2","supporting3","supporting4"]}},
  {{"name":"CoinedWord - what it does","overview":"...","bullets":["b1","b2","b3"],"techTags":["AssignedPrimaryTech","s1","s2","s3","s4"]}},
  {{"name":"CoinedWord - what it does","overview":"...","bullets":["b1","b2","b3"],"techTags":["AssignedPrimaryTech","s1","s2","s3","s4"]}},
  {{"name":"CoinedWord - what it does","overview":"...","bullets":["b1","b2","b3"],"techTags":["AssignedPrimaryTech","s1","s2","s3","s4"]}}
]}}"""

    return system, user

def build_pipeline5a_projects_natural(req: CVRequest, techs: dict, experience: dict) -> tuple:
    """Pipeline 5A: Generate 4 projects with NATURAL flowing descriptions - NO S1, S2, S3, S4 labels"""
    company_name = (req.company_name or "").strip()
    company_ctx = (req.company_context or "").strip()[:700]
    all_techs = techs.get("core", []) + techs.get("preferred", []) + techs.get("ecosystem", [])
    all_techs = list(dict.fromkeys(all_techs))[:30]
    techs_str = ", ".join(all_techs) if all_techs else "technologies from the job description"

    used_systems = []
    for co in experience.get("companies", []):
        for bullet in co.get("bullets", []):
            if len(bullet) > 20:
                used_systems.append(bullet[:60])
    used_str = "\n".join([f"  - {s}" for s in used_systems[:6]]) if used_systems else "  (none)"

    company_intel = ""
    if company_ctx and company_name:
        company_intel = f"""
Target Company: {company_name}
Company Context: {company_ctx[:500]}

Projects 3 & 4 should relate to this company's domain. Project 3: adjacent sector with same tech pattern. Project 4: inspired by secondary offerings.
"""
    elif company_name:
        company_intel = f"""
Target Company: {company_name}
Projects 3 & 4: Use your knowledge of {company_name}'s industry for analogous projects.
"""

    system = """You are an expert technical writer for CVs. Output ONLY valid JSON. No markdown, no backticks.

CRITICAL RULES FOR PROJECT DESCRIPTIONS:
- Write NATURAL, flowing sentences - NO labels like "S1:", "PROBLEM:", "SOLUTION:", etc.
- Each overview MUST tell a complete story: what problem existed -> what you built -> how it worked -> what result achieved
- Write 3-4 complete, professional sentences that flow naturally
- Each sentence must be substantive and technical

TECH TAGS (CRITICAL - MINIMUM 5, MAXIMUM 7 per project):
- Each project MUST have EXACTLY 5-7 technologies in techTags
- ALL technologies MUST come from the JD's extracted list
- NEVER repeat the same technology across different projects
- NEVER use generic placeholders like "Technology1", "Tech2"
- Format: ["Tech1", "Tech2", "Tech3", "Tech4", "Tech5", "Tech6", "Tech7"]

BULLETS (3 per project):
- Bullet 1: specific component built + key technology (20-30 words)
- Bullet 2: hardest challenge + concrete metric (20-30 words)
- Bullet 3: business outcome + unique number (20-30 words)
- All 12 bullets across 4 projects must open with DIFFERENT verbs

PROJECT NAMING:
- Format: "ProductName - What it does [Tech1, Tech2, Tech3]" - show 3 key techs in name

BANNED: S1/S2/S3/S4 labels, "Problem:" "Solution:" labels, generic project names, real company names in projects, repeated verbs across bullets, fewer than 5 tech tags per project."""
    user = f"""Job Title: {req.job_title}
Job Description Summary: {req.job_description.strip()[:500]}
{company_intel}

Allowed Technologies (use ONLY these - you MUST pick 5-7 per project):
{techs_str}

Systems already in experience (do NOT repeat these system types):
{used_str}

Generate EXACTLY 4 projects. Each project MUST have 5-7 UNIQUE technologies in techTags.

Output JSON:
{{"projects": [
    {{
        "name": "Descriptive system name (no colons, no prefixes, just what it does)",
        "overview": "[PROBLEM: who suffers and what they lose.] [SOLUTION: architecture using JD-Tech1 and JD-Tech2.] [FUNCTIONALITY: key features.] [BUSINESS IMPACT: unique metric.]",
        "bullets": [
            "[Verb] [specific component] using [JD-Tech1] and [JD-Tech2], enabling [outcome] for [scale].",
            "[Verb] [specific challenge] using [JD-Tech3], achieving [unique concrete metric].",
            "[Business outcome verb] [measure] from X to Y, saving [unique number] [units] monthly."
        ],
        "techTags": ["[JD-Tech1]", "[JD-Tech2]", "[JD-Tech3]", "[JD-Tech4]", "[JD-Tech5]", "[JD-Tech6]", "[JD-Tech7]"]
    }},
    {{
        "name": "[CoinedName] - [Specific system description using JD-Tech1, JD-Tech2]",
        "overview": "[PROBLEM.] [SOLUTION with JD-Tech1 and JD-Tech2.] [FUNCTIONALITY.] [BUSINESS IMPACT: unique metric.]",
        "bullets": [
            "[Verb] [specific component] using [JD-Tech1], enabling [outcome] for [scale].",
            "[Verb] [specific challenge] with [JD-Tech2], improving [measure] by [unique %].",
            "[Outcome verb] [measure], reducing [issue] from X to Y for [scale] users."
        ],
        "techTags": ["[JD-Tech1]", "[JD-Tech2]", "[JD-Tech3]", "[JD-Tech4]", "[JD-Tech5]", "[JD-Tech6]", "[JD-Tech7]"]
    }},
    {{
        "name": "[CoinedName] - [Specific system description using JD-Tech1, JD-Tech2]",
        "overview": "[PROBLEM.] [SOLUTION with JD-Tech1 and JD-Tech2.] [FUNCTIONALITY.] [BUSINESS IMPACT: unique metric.]",
        "bullets": [
            "[Verb] [specific component] using [JD-Tech1] and [JD-Tech2], serving [scale] with [outcome].",
            "[Verb] [specific challenge], cutting [measure] from X to Y using [JD-Tech3].",
            "[Outcome verb] [measure] by [unique %], enabling [business result] for [scale]."
        ],
        "techTags": ["[JD-Tech1]", "[JD-Tech2]", "[JD-Tech3]", "[JD-Tech4]", "[JD-Tech5]", "[JD-Tech6]", "[JD-Tech7]"]
    }},
    {{
        "name": "[CoinedName] - [Specific system description using JD-Tech1, JD-Tech2]",
        "overview": "[PROBLEM.] [SOLUTION with JD-Tech1 and JD-Tech2.] [FUNCTIONALITY.] [BUSINESS IMPACT: unique metric.]",
        "bullets": [
            "[Verb] [specific component] using [JD-Tech1], classifying [scale] items on [schedule].",
            "[Verb] [specific challenge] using [JD-Tech2] and [JD-Tech3], resolving [unique number] issues.",
            "[Outcome verb] [measure] from X to Y, improving [business indicator] by [unique %]."
        ],
        "techTags": ["[JD-Tech1]", "[JD-Tech2]", "[JD-Tech3]", "[JD-Tech4]", "[JD-Tech5]", "[JD-Tech6]", "[JD-Tech7]"]
    }}
]}}

IMPORTANT:
- Each project's techTags MUST have 5-7 technologies (not 3)
- All technologies must come from the allowed list above
- No technology can be repeated across different projects
- Project names can show 3 key technologies in brackets, but techTags must have 5-7
- Write natural overviews without S1/S2/S3/S4 labels
- Each overview must be 3-4 sentences
- All metrics must be unique across all 4 projects"""

    return system, user

def validate_project_techs(projects_result: dict, techs: dict) -> dict:
    """
    Ensure each project has exactly 5-7 unique, real-tech tags from the JD.
    Hard rules:
      - REJECT any tag that is a plain English word (fails _is_real_tech)
      - REJECT tags that are single generic nouns: Cloud, Development, Web, Good, Strong, etc.
      - Backfill from the JD tech pool if fewer than 5 valid tags remain
      - Never allow duplicate tags across or within projects
      - Cap at 7 tags per project
    """
    if not projects_result or not projects_result.get("projects"):
        return projects_result

    # Build ordered pool: core first, then preferred, then ecosystem
    all_allowed = []
    seen_pool = set()
    for arr in ("core", "preferred", "ecosystem"):
        for t in techs.get(arr, []):
            t = (t or "").strip()
            if t and t.lower() not in seen_pool and _is_real_tech(t):
                all_allowed.append(t)
                seen_pool.add(t.lower())

    used_global: set = set()

    for project in projects_result["projects"]:
        raw_tags = project.get("techTags") or project.get("tech") or []
        if isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in re.split(r"[|,;]", raw_tags) if t.strip()]

        # Step 1 - keep only real, non-duplicate tags
        seen_local: set = set()
        valid_tags: list = []
        for t in raw_tags:
            t = (t or "").strip()
            if not t:
                continue
            tl = t.lower()
            # Must pass _is_real_tech AND must not be a bare generic noun
            if _is_real_tech(t) and tl not in seen_local:
                # Extra guard: reject single-word all-lowercase common nouns
                _GENERIC_TAG_WORDS = {
                    "cloud", "development", "web", "good", "strong", "hands",
                    "ability", "remote", "setup", "mindset", "rest", "api",
                    "apis", "app", "apps", "core", "net", "sdk", "http",
                    "json", "xml", "yaml", "data", "code", "base", "stack",
                    "work", "tool", "tools", "service", "platform", "system",
                    "backend", "frontend", "database", "server", "client",
                    "framework", "library", "language", "testing", "security",
                    "sql", "nosql",
                }
                if tl in _GENERIC_TAG_WORDS:
                    continue
                valid_tags.append(t)
                seen_local.add(tl)

        # Step 2 - backfill from JD pool if fewer than 5
        if len(valid_tags) < 5:
            for t in all_allowed:
                if len(valid_tags) >= 7:
                    break
                tl = t.lower()
                if tl not in seen_local:
                    valid_tags.append(t)
                    seen_local.add(tl)

        # Step 3 - still under 5? relax global uniqueness and pull from pool again
        if len(valid_tags) < 5:
            for t in all_allowed:
                if len(valid_tags) >= 5:
                    break
                tl = t.lower()
                if tl not in seen_local:
                    valid_tags.append(t)
                    seen_local.add(tl)

        # Cap at 7
        valid_tags = valid_tags[:7]

        # Track globally (soft - we allow some overlap between projects if pool is small)
        for t in valid_tags:
            used_global.add(t.lower())

        project["techTags"] = valid_tags

    return projects_result


def build_pipeline5b_related_tech(req: CVRequest, techs: dict) -> tuple:
    """Pipeline 5B: Generate ONLY Related Technologies & Tools - dedicated pipeline.
    Produces exactly 5 category boxes, each with exactly 5 items.
    NO raw JD text is passed - only the pre-extracted technology list,
    preventing the model from grabbing non-tech words from the JD body.
    """
    core      = techs.get("core", [])
    preferred = techs.get("preferred", [])
    ecosystem = techs.get("ecosystem", [])
    all_techs = list(dict.fromkeys(core + preferred + ecosystem))[:50]

    # Hard guard: if extraction yielded fewer than 10 real techs, the model
    # has nothing useful to group - return empty so the sanitiser skips it.
    if len(all_techs) < 5:
        # Return a dummy prompt that will produce an empty relatedTech list.
        return ("Output ONLY: {}", '{{"relatedTech":[]}}')

    techs_str = ", ".join(all_techs)

    system = (
        "You are a technical CV editor. Output ONLY valid JSON. "
        "No markdown, no backticks, no prose. Start { end }.\n\n"
        "CRITICAL RULE: Every item in every 'items' array MUST be a real software technology, "
        "framework, language, tool, platform, or library.\n"
        "BANNED from items: company names, salary terms, job benefits, location words, "
        "soft skills, any English word that is not a named technology.\n"
        "If an item from the allowed list is not a real technology name, skip it entirely.\n"
        "Category names: concise technical domain labels only "
        "(e.g. 'Backend Frameworks', 'Cloud Platforms', 'Database Systems', "
        "'Frontend Libraries', 'DevOps & CI/CD', 'Testing Tools', 'API & Integration').\n"
        "BANNED category names: anything containing 'Insurance', 'Market', 'Business', "
        "'Global', 'Remote', 'Salary', 'Benefits', 'Work', 'Reasons', 'Join', 'About'."
    )

    user = f"""Job Role: {req.job_title}

This is the COMPLETE list of technologies extracted from the job description.
Use ONLY items from this list - copy them exactly as written:

{techs_str}

TASK: Group these technologies into exactly 5 category boxes.

STRICT RULES:
1. Exactly 5 categories.
2. Exactly 5 items per category - pick from the list above only.
3. Each item must be a real named technology (framework, language, tool, platform, library).
   If an entry in the list above looks like a non-tech word (e.g. a company name, benefit,
   location), skip it and pick something else from the list.
4. Zero duplicates across all 5 boxes.
5. Category name = a concise technical domain (e.g. "Backend Frameworks", "Cloud Services").
   Never name a category after a company, benefit, or business concept.

Output JSON:
{{"relatedTech": [
  {{"category": "Technical Domain 1", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
  {{"category": "Technical Domain 2", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
  {{"category": "Technical Domain 3", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
  {{"category": "Technical Domain 4", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}},
  {{"category": "Technical Domain 5", "items": ["tech1", "tech2", "tech3", "tech4", "tech5"]}}
]}}"""

    return system, user
# ===================================================================
# UNIVERSAL LLM CALLER - Works for Groq, Cerebras, OpenAI, DeepSeek
# ===================================================================

async def call_llm_atomic(client, key: str, model: str, url: str,
                          system: str, user: str, stage: str,
                          headers: dict, max_tokens: int = 1200,
                          _deadline: float = 0.0) -> dict:
    """Universal atomic LLM call - honours Retry-After on 429, one retry.

    Per-call timeout strategy (tight to enforce 2-minute total budget):
      - Cerebras: 25 s  (fast inference; if slow, fail fast and report)
      - Groq:     25 s  (very fast; 25 s is generous)
      - Others:   30 s  (conservative default)
    If _deadline is set (epoch seconds), the call is aborted before it even
    starts if there is not enough time left.
    """
    import time as _t
    # Choose per-call timeout based on provider URL
    if url == CEREBRAS_URL:
        per_call_timeout = 40
    elif url == GROQ_URL:
        per_call_timeout = 60
    else:
        per_call_timeout = 60

    # Hard deadline check — skip this call if we're already over budget
    if _deadline and _t.time() >= _deadline:
        raise ValueError(f"Stage {stage} skipped — generation deadline exceeded")

    # If very little time remains, skip rather than run with a tiny timeout
    if _deadline:
        remaining = _deadline - _t.time()
        if remaining < 15:
            raise ValueError(f"Stage {stage} skipped — only {remaining:.0f}s left, need at least 15s")
        # Don't cap per_call_timeout unless we're genuinely close to the deadline
        if remaining < per_call_timeout + 10:
            per_call_timeout = max(15, int(remaining) - 5)

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
            raise ValueError(f"Stage {stage} timed out after {per_call_timeout}s — server slow, try again")
        except Exception as e:
            raise ValueError(f"Stage {stage} failed: {str(e)}")

        if r.status_code == 200:
            raw = r.json()["choices"][0]["message"]["content"]
            raw = raw.strip()
            raw = re.sub(r'```json\s*', '', raw)
            raw = re.sub(r'```\s*$', '', raw)
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                raw = raw[start:end+1]
            raw = re.sub(r',\s*}', '}', raw)
            raw = re.sub(r',\s*]', ']', raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"JSON parse error in {stage}: {e}")
                print(f"Raw response (first 500 chars): {raw[:500]}")
                return {}
        elif r.status_code == 429:
            wait = int(r.headers.get("retry-after", r.headers.get("Retry-After", 30)))
            # Cap wait at 30s max; skip retry if too close to deadline
            if _deadline:
                max_wait = max(0, int(_deadline - _t.time()) - 20)
                wait = min(wait, max_wait, 30)
            else:
                wait = min(wait, 30)
            if attempt == 0 and wait > 0:
                await asyncio.sleep(wait)
                continue
            # Embed retry-after so callers (call_cerebras) can parse it
            raise ValueError(f"Rate limited on {stage} (retry-after={wait})")
        elif r.status_code in (401, 403):
            raise ValueError(f"Invalid key on {stage}")
        else:
            raise ValueError(f"HTTP {r.status_code} on {stage}")
    raise ValueError(f"Rate limited on {stage} (retry-after=30)")

# -- Technology sanitiser - removes non-tech words from Pipeline 1 output ------
# Words that commonly leak into the tech list from JD prose/boilerplate
_NON_TECH_WORDS = {
    # months / time
    "january","february","march","april","may","june","july","august",
    "september","october","november","december","year","years","month","months",
    # generic business / HR words
    "about","as","at","bank","benefits","bonus","business","china","client",
    "clients","competitive","covergo","dai","environment","global","group",
    "insurance","insurtech","international","join","leader","leading","market",
    "mission","our","platform","reasons","remote","salary","team","top","us",
    "we","work","working","worldwide","bupa","axa","msig","asia","americas",
    # common English filler
    "able","ability","across","also","and","any","are","based","be","been",
    "both","but","by","can","company","cross","detail","driven","each","end",
    "for","from","full","have","help","high","in","including","into","its",
    "key","large","like","make","more","multiple","new","not","of","on","one",
    "or","other","our","out","over","own","people","per","plus","product",
    "proven","provide","role","scale","set","software","stack","strong","such",
    "the","their","them","then","they","this","through","to","up","use","using",
    "very","well","will","with","within","you","your",
}

def _is_likely_tech(word: str) -> bool:
    """Return True only if the word looks like a real technology name."""
    w = word.strip()
    if not w or len(w) < 2:
        return False
    # Reject purely lowercase common words
    if w.lower() in _NON_TECH_WORDS:
        return False
    # Reject strings that are purely numeric
    if w.isdigit():
        return False
    # Reject strings with spaces that contain only generic words
    if " " in w:
        parts = w.lower().split()
        if all(p in _NON_TECH_WORDS for p in parts):
            return False
    # Reject strings > 30 chars (likely phrases, not tool names)
    if len(w) > 30:
        return False
    return True


def _sanitise_techs(techs: dict) -> dict:
    """Strip non-technology words from all three tech arrays."""
    clean = {}
    for key in ("core", "preferred", "ecosystem"):
        raw = techs.get(key, [])
        clean[key] = [t for t in raw if isinstance(t, str) and _is_likely_tech(t)]
    return clean


# ===================================================================
# SEPARATE VALIDATION PIPELINE
# Runs AFTER generation and BEFORE final output.
# Enforces: JD relevance, backend tech alignment, realism, title-match.
# This runs independently of the main generation so models can't skip it.
# ===================================================================

# Tools that are ONLY relevant in specific business contexts — never generic backend tools
_CONTEXT_SPECIFIC_TOOLS = {
    # Communication/messaging APIs - only relevant if JD explicitly mentions them
    "whatsapp": ["whatsapp", "whatsapp business", "whatsapp api"],
    "twilio":   ["twilio", "twilio sms", "twilio voice"],
    "sendgrid": ["sendgrid"],
    "mailgun":  ["mailgun"],
    "pusher":   ["pusher"],
    "intercom": ["intercom"],
    "zendesk":  ["zendesk"],
    "chatwoot": ["chatwoot"],
    "freshdesk":["freshdesk"],
    "salesforce":["salesforce", "salesforce crm"],
    "hubspot":  ["hubspot"],
    "stripe":   ["stripe"],
    "paypal":   ["paypal"],
    "twilio":   ["twilio"],
}

def _is_context_tool_relevant(tool: str, jd: str, job_title: str) -> bool:
    """
    Check if a context-specific tool (e.g. WhatsApp, Stripe) is actually
    mentioned in the JD or job title. If not, it should be stripped.
    """
    tl = tool.lower().strip()
    jd_lower = jd.lower()
    title_lower = job_title.lower()
    combined = jd_lower + " " + title_lower

    for group_key, variants in _CONTEXT_SPECIFIC_TOOLS.items():
        if any(v in tl for v in variants):
            # This tool belongs to a context-specific group — check if JD mentions any variant
            return any(v in combined for v in variants)
    return True  # Not a context-specific tool — always allowed


def _filter_irrelevant_backend_techs(cv: dict, jd: str, job_title: str) -> dict:
    """
    VALIDATION STEP 1 — Backend Technology Alignment.

    Problem: The AI sometimes injects tools like 'WhatsApp Business', 'Stripe',
    'Salesforce' etc. into backend skill rows even when the JD has nothing to do
    with those domains. This function strips them from skills and company tech tags
    unless they are explicitly present in the job description or job title.

    Rules:
    - A context-specific tool is ONLY kept if it appears in the JD or job title.
    - This applies to: skills rows, company tech tags, project tech tags.
    - Tools that are universally relevant (Git, Docker, PostgreSQL, etc.) are never stripped.
    """
    jd_lower = jd.lower()
    title_lower = job_title.lower()

    def _filter_tech_string(tech_str: str) -> str:
        if not tech_str:
            return tech_str
        sep = "|" if "|" in tech_str else ","
        parts = [t.strip() for t in tech_str.split(sep) if t.strip()]
        kept = [t for t in parts if _is_context_tool_relevant(t, jd, job_title)]
        return " | ".join(kept) if "|" in tech_str else ", ".join(kept)

    def _filter_tech_list(tech_list: list) -> list:
        return [t for t in tech_list if _is_context_tool_relevant(t, jd, job_title)]

    # 1. Filter skills rows
    new_skills = []
    for row in cv.get("skills", []):
        if ":" not in row:
            new_skills.append(row)
            continue
        colon = row.index(":")
        cat = row[:colon].strip()
        items_str = row[colon + 1:].strip()
        items = [t.strip() for t in items_str.split(",") if t.strip()]
        kept = [t for t in items if _is_context_tool_relevant(t, jd, job_title)]
        if len(kept) >= 3:
            new_skills.append(f"{cat}: {', '.join(kept)}")
        else:
            new_skills.append(row)  # Keep original if too many would be stripped
    cv["skills"] = new_skills

    # 2. Filter company tech tags
    for co in cv.get("companies", []):
        co["tech"] = _filter_tech_string(co.get("tech", ""))

    # 3. Filter project tech tags
    for proj in cv.get("projects", []):
        if isinstance(proj.get("techTags"), list):
            proj["techTags"] = _filter_tech_list(proj["techTags"])

    # 4. Filter technologies block
    tech_block = cv.get("technologies", {})
    if isinstance(tech_block, dict):
        for key in ("mustHave", "niceToHave", "additional"):
            if isinstance(tech_block.get(key), list):
                tech_block[key] = _filter_tech_list(tech_block[key])
        cv["technologies"] = tech_block

    return cv

# Tools that are universally acceptable in any CV regardless of JD
_NEUTRAL_TECHS = [
    "git", "github", "gitlab", "bitbucket",
    "docker", "linux", "bash", "powershell",
    "vs code", "visual studio", "intellij", "pycharm", "rider",
    "jira", "confluence", "slack", "notion", "trello",
    "postman", "swagger", "openapi",
    "npm", "yarn", "pip", "maven", "gradle", "nuget",
    "json", "yaml", "xml", "rest", "http",
    "agile", "scrum", "kanban",
]
def _validate_jd_relevance(cv: dict, jd: str, job_title: str, techs: dict) -> dict:
    """
    VALIDATION STEP 2 — Strict JD Relevance Enforcement.

    Problem: AI models sometimes inject technologies from their training data
    that are NOT present in the JD or its ecosystem. This validator removes
    any technology that cannot be traced back to the JD.

    Algorithm:
    1. Build the full allowed tech set from the extracted techs dict.
    2. For any skill item that is NOT in the allowed set AND is not a neutral/cross-stack tool,
       remove it.
    3. Applies to skills, company tags, project tags, and the technologies block.

    Priority rule: JD relevance > default AI assumptions.
    """
    # Build allowed set: everything in techs + neutral cross-stack tools
    allowed: set = set()
    for arr in ("core", "preferred", "ecosystem"):
        for t in techs.get(arr, []):
            allowed.add(t.lower().strip())

    # Add all neutral tools to allowed (they are always acceptable)
    for n in _NEUTRAL_TECHS:
        allowed.add(n.lower().strip())

    # If allowed set is too thin (<10 items), skip enforcement
    # (thin set means tech extraction failed; don't strip everything)
    if len(allowed) < 10:
        return cv

    def _is_jd_relevant(tech: str) -> bool:
        tl = tech.lower().strip()
        # Direct match
        if tl in allowed:
            return True
        # Prefix/substring match for compound names (e.g. "AWS Lambda" matches "aws")
        if any(tl.startswith(a) or a.startswith(tl) for a in allowed if len(a) > 3):
            return True
        # If the tech is short (2-3 chars, e.g. "Go", "C#"), be permissive
        if len(tl) <= 3:
            return True
        return False

    def _filter_items(items: list) -> list:
        return [t for t in items if not _is_real_tech(t) or _is_jd_relevant(t)]

    # Filter skills rows
    new_skills = []
    for row in cv.get("skills", []):
        if ":" not in row:
            new_skills.append(row)
            continue
        colon = row.index(":")
        cat = row[:colon].strip()
        items = [t.strip() for t in row[colon + 1:].split(",") if t.strip()]
        kept = _filter_items(items)
        if len(kept) >= 3:
            new_skills.append(f"{cat}: {', '.join(kept)}")
        else:
            new_skills.append(row)  # Don't destroy rows with too few left
    cv["skills"] = new_skills

    return cv


def _validate_realism(cv: dict, job_title: str, jd: str) -> dict:
    """
    VALIDATION STEP 3 — Realism Check.

    Detects and fixes unrealistic or AI-generated patterns:
    1. Repeated metric formats across bullets (e.g. all "X% improvement")
    2. Repeated action verbs across bullets in the same company
    3. Overly broad or repeated technology lists in projects
    4. Non-technical role detecting technical injection

    For non-technical roles (PM, SEO, Finance, Design), removes
    backend/infrastructure technologies from skills and experience
    unless they are explicitly in the JD.
    """
    title_lower = job_title.lower()
    jd_lower    = jd.lower()

    # Detect if this is a non-technical role
    _NON_TECH_ROLE_KW = (
        "seo", "digital marketing", "social media", "content manager",
        "project manager", "product manager", "scrum master",
        "financial analyst", "accountant", "business analyst",
        "ux designer", "ui designer", "graphic designer",
        "sales", "customer success", "account manager",
    )
    _is_non_tech = any(kw in title_lower for kw in _NON_TECH_ROLE_KW)

    # For non-technical roles, strip backend/infra technologies from skills
    # unless they appear in the JD
    if _is_non_tech:
        _BACKEND_INFRA_KW = (
            "docker", "kubernetes", "terraform", "ansible",
            "postgresql", "mysql", "mongodb", "redis",
            "node.js", "express", "django", "flask",
            "react", "angular", "vue", "typescript", "javascript",
            "aws ec2", "aws lambda", "azure functions", "gcp",
        )
        def _should_strip_for_nontechrole(tech: str) -> bool:
            tl = tech.lower().strip()
            return any(kw in tl for kw in _BACKEND_INFRA_KW) and kw not in jd_lower

        new_skills = []
        for row in cv.get("skills", []):
            if ":" not in row:
                new_skills.append(row)
                continue
            colon = row.index(":")
            cat = row[:colon].strip()
            items = [t.strip() for t in row[colon + 1:].split(",") if t.strip()]
            kept = []
            for t in items:
                tl = t.lower().strip()
                strip = False
                for kw in _BACKEND_INFRA_KW:
                    if kw in tl and kw not in jd_lower:
                        strip = True
                        break
                if not strip:
                    kept.append(t)
            if len(kept) >= 3:
                new_skills.append(f"{cat}: {', '.join(kept)}")
            else:
                new_skills.append(row)
        cv["skills"] = new_skills

    # Deduplicate bullet verbs within each company (realism: same verb twice = AI flag)
    for co in cv.get("companies", []):
        bullets = co.get("bullets", [])
        if not bullets:
            continue
        seen_verbs: set = set()
        _VERB_REPLACEMENTS = [
            "Developed", "Engineered", "Implemented", "Designed",
            "Built", "Deployed", "Integrated", "Optimised",
            "Delivered", "Established", "Streamlined", "Configured",
        ]
        replacement_idx = 0
        new_bullets = []
        for b in bullets:
            if not b:
                new_bullets.append(b)
                continue
            first_word = b.split()[0] if b.split() else ""
            fl = first_word.lower()
            if fl in seen_verbs:
                # Replace the first verb with an unused one
                while replacement_idx < len(_VERB_REPLACEMENTS):
                    rep = _VERB_REPLACEMENTS[replacement_idx]
                    replacement_idx += 1
                    if rep.lower() not in seen_verbs:
                        rest = b[len(first_word):] if len(b) > len(first_word) else b
                        b = rep + rest
                        seen_verbs.add(rep.lower())
                        break
            else:
                seen_verbs.add(fl)
            new_bullets.append(b)
        co["bullets"] = new_bullets

    return cv


def _validate_title_alignment(cv: dict, job_title: str, jd: str) -> dict:
    """
    VALIDATION STEP 4 — Title Alignment.

    Ensures the CV headline title:
    1. Is not identical to the input job title.
    2. Does not contain location words.
    3. Has exactly 3 technologies after the pipe.
    4. Does not use 'Architect' unless the JD explicitly requests it.
    """
    title = cv.get("title", "")
    if not title:
        return cv

    jd_lower    = jd.lower()
    title_clean = title.strip()

    # Rule 1: Remove location words from title
    _LOCATION_WORDS = (
        "dallas", "texas", "tx", "pakistan", "lahore", "islamabad",
        "karachi", "remote", "based", "located", "usa", "uk", "london",
        "new york", "california", "ca", "ny", "dubai", "uae",
    )
    for loc in _LOCATION_WORDS:
        pattern = r'(?i)\b' + re.escape(loc) + r'\b'
        title_clean = re.sub(pattern, '', title_clean).strip()
    title_clean = re.sub(r'\s+', ' ', title_clean).strip()
    title_clean = title_clean.strip('|').strip()

    # Rule 2: Remove 'Architect' unless JD explicitly mentions it
    if "architect" not in jd_lower:
        title_clean = re.sub(r'(?i)\barchitect\b', 'Engineer', title_clean)
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()

    # Rule 3: Don't let title be identical to the job title (case-insensitive)
    if title_clean.lower().split("|")[0].strip() == job_title.lower().split("|")[0].strip():
        # Add a domain qualifier to differentiate
        parts = title_clean.split("|")
        if len(parts) >= 2:
            domain_part = parts[0].strip()
            tech_part   = parts[1].strip()
            title_clean = f"Specialist {domain_part} | {tech_part}"

    # Rule 4: Normalize double spaces, leading/trailing pipe chars
    title_clean = re.sub(r'\s+', ' ', title_clean).strip().strip('|').strip()

    cv["title"] = title_clean
    return cv


def _validate_skills_category_names(cv: dict, jd: str, job_title: str) -> dict:
    """
    VALIDATION STEP 5 — Skill Category Name Sanity.

    Detects any skill category whose name is a bare generic word or a numbered
    placeholder (e.g. "Backend", "Skills", "Technical Skills 4") and asks the AI
    to rename it using the actual JD domain. Everything is derived from the JD —
    no hardcoded prefix map.
    """
    import json as _json

    skills = cv.get("skills", [])
    if not skills:
        return cv

    # Bare single-word generics or numbered placeholders that must be renamed
    _GENERIC = {
        "backend technologies", "backend", "frontend technologies", "frontend",
        "database", "databases", "cloud", "devops", "skills", "tools",
        "technologies", "other technologies", "additional skills",
        "technical skills", "core skills", "key skills", "other skills",
        "programming languages", "soft skills", "general skills",
    }

    def _is_generic(cat: str) -> bool:
        cl = cat.lower().strip()
        # Any "Technical Skills N" or bare "Technical Skills"
        if re.match(r"^technical skills\s*\d*$", cl):
            return True
        # Any short "X N" numbered pattern: "Skills 2", "Category 3", etc.
        if re.match(r"^[\w &/]+\s+\d+$", cl) and len(cl.split()) <= 4:
            return True
        return cl in _GENERIC

    # Collect rows that need renaming
    bad_indices = []
    for i, row in enumerate(skills):
        if ":" not in row:
            continue
        cat = row[:row.index(":")].strip()
        if _is_generic(cat):
            bad_indices.append(i)

    if not bad_indices:
        return cv  # all names are fine — nothing to do

    # Build one AI call to rename all bad categories at once
    bad_cats = [skills[i][:skills[i].index(":")].strip() for i in bad_indices]
    bad_items = [skills[i][skills[i].index(":")+1:].strip() for i in bad_indices]

    prompt = (
        f"Job title: {job_title}\n"
        f"Job description (first 500 chars): {jd[:500]}\n\n"
        f"The following skill category names are too generic and must be renamed to reflect "
        f"the actual technology domain of this specific JD:\n"
    )
    for cat, items in zip(bad_cats, bad_items):
        prompt += f'  - Category "{cat}" contains: {items[:120]}\n'
    prompt += (
        "\nReturn ONLY a JSON array of new category name strings, one per category above, "
        "in the same order. Each name must be a real technology domain label derived from "
        "this JD (e.g. for a DevOps JD: 'IaC & Config Management', 'CI/CD & Automation', "
        "'Container Orchestration', 'Monitoring & Observability', 'Scripting & Version Control'). "
        "No markdown, no extra text."
    )

    try:
        import urllib.request as _ur
        import concurrent.futures as _cf
        payload = _json.dumps({
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
        }).encode()
        def _sync_rename():
            req_obj = _ur.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {GROQ_API_KEY}"},
                method="POST",
            )
            with _ur.urlopen(req_obj, timeout=10) as resp:
                return _json.loads(resp.read())
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_sync_rename)
            body = future.result(timeout=12)
        raw = body["choices"][0]["message"]["content"].strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        new_names = _json.loads(raw)
        if isinstance(new_names, list) and len(new_names) == len(bad_indices):
            for idx, new_name in zip(bad_indices, new_names):
                if isinstance(new_name, str) and new_name.strip():
                    old = skills[idx]
                    colon = old.index(":")
                    skills[idx] = f"{new_name.strip()}{old[colon:]}"
    except Exception:
        # AI call failed — use deterministic item-based labelling instead of leaving broken names
        for idx, items_str in zip(bad_indices, bad_items):
            items_list = [t.strip() for t in items_str.split(",") if t.strip()]
            new_name = _infer_category_name(items_list, job_title, idx)
            if new_name:
                old = skills[idx]
                colon = old.index(":")
                skills[idx] = f"{new_name}{old[colon:]}"

    cv["skills"] = skills
    return cv



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HARD ROLE-TITLE ENFORCER — runs inside run_validation_pipeline         ║
# ║  Executes AFTER every AI model on EVERY code path.                      ║
# ║  Cannot be skipped or bypassed by any AI output.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_BANNED_ROLE_DOMAINS_SET = {
    "devops", "web", "software", "backend", "frontend", "full-stack",
    "fullstack", "it", "tech", "digital", "full stack",
}
_BAD_TITLE_ADJECTIVES = {
    "transformed", "innovative", "dynamic", "versatile", "experienced",
    "seasoned", "skilled", "expert", "creative", "enhanced", "advanced",
    "optimized", "optimised", "title", "placeholder", "related", "generic",
    "digital solutions",   # catches "Digital Solutions Engineer" etc.
}
_FUNC_WORDS  = {"engineer","developer","specialist","analyst","programmer",
                "consultant","designer","technologist","builder","integrator"}
_SENIOR_WORDS = {"senior","junior","associate","graduate","lead","principal","staff","intern","internship"}
_FUNC_LIST   = ["Engineer","Developer","Specialist","Analyst","Programmer"]


def _jt_primary_tech(job_title: str) -> str:
    """First non-seniority, non-function word from job_title."""
    if not job_title:
        return "WordPress"
    clean = re.sub(
        r"(?i)\b(senior|junior|lead|staff|principal|associate|mid.?level)\b\s*",
        "", job_title).strip()
    for w in re.split(r"[\s&|/,]+", clean):
        wl = w.lower().strip(".")
        if len(wl) > 2 and wl not in _FUNC_WORDS and wl not in _SENIOR_WORDS:
            return w.strip()
    return "WordPress"


def _role_domain_is_banned(role: str) -> bool:
    """True when every non-function, non-seniority word is a banned generic."""
    core = re.sub(
        r"(?i)\b(senior|junior|associate|lead|staff|principal)\b\s*", "", role
    ).strip().lower()
    words = [w for w in re.split(r"[\s\-/]+", core)
             if w and w not in _FUNC_WORDS and len(w) > 1]
    return bool(words) and all(w in _BANNED_ROLE_DOMAINS_SET for w in words)


def _title_has_bad_word(title: str) -> bool:
    before = title.split("|")[0].lower() if title else ""
    words  = re.split(r"[\s\-&/|,]+", before)
    return any(w in _BAD_TITLE_ADJECTIVES for w in words)


def _enforce_role_titles(cv: dict, job_title: str) -> dict:
    """
    Hard post-processor — replaces any banned/generic role domain with
    the primary named technology extracted from job_title.
    Also fixes the CV headline title if it contains invented adjectives.
    """
    if not job_title:
        return cv

    primary = _jt_primary_tech(job_title)

    # Build a list of 3 distinct tech words from the job_title for variety
    all_words = [w for w in re.split(r"[\s&|/,\-]+", job_title)
                 if len(w) > 2
                 and w.lower() not in _FUNC_WORDS
                 and w.lower() not in _SENIOR_WORDS
                 and w.lower() not in {"full","stack","theme","plugin",
                                       "development","the","and","for","with"}]
    tech_pool = list(dict.fromkeys(all_words))   # dedup, keep order

    companies = cv.get("companies", [])
    num = len(companies)
    tiers = (["Junior"] if num == 1
             else ["", "Junior"] if num == 2
             else ["Senior", "", "Junior"])

    for i, co in enumerate(companies[:3]):
        role = (co.get("role") or "").strip()
        # Always strip "Intern/Internship" from role titles
        role = re.sub(r"(?i)\b(intern(?:ship)?)\b\s*", "", role).strip()
        role = re.sub(r"\s+", " ", role).strip()
        co["role"] = role
        if not role or _role_domain_is_banned(role):
            tech = tech_pool[i % len(tech_pool)] if tech_pool else primary
            fn   = _FUNC_LIST[i % len(_FUNC_LIST)]
            tier = tiers[min(i, len(tiers)-1)]
            co["role"] = (f"{tier} {tech} {fn}" if tier else f"{tech} {fn}").strip()

    # Deduplicate: if two roles are identical after fix, rotate function word
    seen = {}
    for i, co in enumerate(companies[:3]):
        key = co.get("role","").lower()
        if key in seen:
            fn = _FUNC_LIST[(i + 2) % len(_FUNC_LIST)]
            co["role"] = re.sub(
                r"(?i)\b(" + "|".join(_FUNC_LIST) + r")\b",
                fn, co.get("role",""), count=1
            ).strip()
        seen[co.get("role","").lower()] = i

    # Fix CV headline title if it contains invented adjectives
    title = (cv.get("title") or "").strip()
    if _title_has_bad_word(title) or not title:
        pipe = ""
        if "|" in title:
            rhs = title.split("|",1)[1].strip()
            if not _title_has_bad_word(rhs):
                pipe = " | " + rhs
        # Build clean title from job_title (first 3 meaningful words)
        jt_clean = re.sub(
            r"(?i)\b(senior|junior|lead|staff|principal|associate)\b\s*",
            "", job_title).strip()
        words = jt_clean.split()[:3]
        cv["title"] = " ".join(words) + pipe

    return cv

def run_validation_pipeline(cv: dict, jd: str, job_title: str, techs: dict) -> dict:
    """
    MASTER VALIDATION PIPELINE — runs ALL validation steps in order.

    This is a separate, independent pipeline that enforces quality rules
    AFTER the AI has generated the CV. It runs before final output and
    cannot be bypassed by any AI model.

    Steps run in order (each builds on the previous):
      1. Filter irrelevant backend technologies (e.g. WhatsApp in a Java CV)
      2. Validate strict JD relevance (remove off-stack technologies)
      3. Realism check (deduplicate verbs, handle non-tech roles)
      4. Title alignment (location words, architect guard, uniqueness)
      5. Skill category name sanity

    Priority: JD relevance ALWAYS overrides AI defaults.
    """
    if not cv or not jd:
        return cv

    # Step 1: Remove context-specific tools not in the JD
    cv = _filter_irrelevant_backend_techs(cv, jd, job_title)

    # Step 2: Remove technologies not traceable to the JD
    cv = _validate_jd_relevance(cv, jd, job_title, techs)

    # Step 3: Realism check
    cv = _validate_realism(cv, job_title, jd)

    # Step 4: Title alignment
    cv = _validate_title_alignment(cv, job_title, jd)

    # Step 5: Skill category name sanity
    cv = _validate_skills_category_names(cv, jd, job_title)

    # Step 6: Hard role-title enforcer — last step, cannot be bypassed
    cv = _enforce_role_titles(cv, job_title)

    return cv


# ===================================================================
# MAIN ATOMIC GENERATION - 5 pipelines, never truncated
# ===================================================================

# ===================================================================
# DEDICATED PIPES FOR ALL JOB TYPES (Technical + Non-Technical)
# Works for: SEO, Marketing, HR, Finance, Design, Sales, PM, etc.
# ===================================================================

def build_pipeline1_tech_extraction_universal(req: CVRequest) -> tuple:
    """
    PIPE 1 — Exhaustive JD keyword extraction.
    Scans EVERY sentence, EVERY section (required, nice-to-have, responsibilities,
    title) and extracts every named technology, tool, platform, language, framework,
    or service. Nothing is skipped. Output feeds Pipe 2 for companion expansion.
    """
    jd    = req.job_description.strip()
    title = req.job_title.strip()

    system = """You are a forensic JD keyword extractor. Output ONLY valid JSON.
No markdown, no backticks. Start { end }.

MISSION: Read the job description WORD BY WORD and extract EVERY technology, tool,
language, framework, platform, library, database, cloud service, or software product
mentioned — including in "nice to have", "good to have", "preferred", "bonus",
"responsibilities", "requirements", and even in passing.

══════════════════════════════════════════════════════════════
ABSOLUTE RULES (no exceptions):
1. Extract EVERY named software product regardless of where it appears.
2. "Required", "must have", "nice to have", "preferred", "bonus", "advantageous",
   "beneficial", "desirable", "good to have", "a plus" — ALL sections are extracted.
   Do NOT skip preferred/nice-to-have sections.
3. If the same tool appears multiple times, include it once.
4. Include version numbers stripped: "React 18" → "React", ".NET 8" → ".NET 8" (keep if meaningful).
5. Do NOT include soft skills: "communication", "teamwork", "problem solving", etc.
6. Do NOT include job metadata: salaries, locations, company descriptions, benefits.
7. Include acronyms: CI/CD, ORM, REST, API, SOAP, gRPC, JWT, OAuth.
══════════════════════════════════════════════════════════════

WHAT TO EXTRACT (every category applies simultaneously):
• Programming languages: C#, Python, Java, JavaScript, TypeScript, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, R, MATLAB, VBA, PowerShell, Bash
• Frameworks & libraries: ASP.NET Core, .NET, Entity Framework, Dapper, Spring Boot, Django, FastAPI, Flask, React, Angular, Vue.js, Next.js, NestJS, Express.js, Laravel, Rails, Gin, Fiber
• Databases: SQL Server, PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, DynamoDB, Cosmos DB, Oracle DB, SQLite, Cassandra, InfluxDB, Neo4j
• Cloud platforms: Azure, AWS, GCP, Firebase, Heroku, DigitalOcean, Vercel, Netlify — AND their specific services (Azure App Service, AWS Lambda, S3, EC2, RDS, etc.)
• DevOps & CI/CD: Docker, Kubernetes, Helm, Terraform, Ansible, GitHub Actions, Azure DevOps, Jenkins, GitLab CI, CircleCI, ArgoCD
• Testing: xUnit, NUnit, MSTest, Jest, Cypress, Playwright, Selenium, Pytest, JUnit, Moq, Jasmine, Karma
• Monitoring: Prometheus, Grafana, Datadog, New Relic, Sentry, ELK Stack, Splunk, Application Insights
• Security: OAuth 2.0, JWT, OpenID Connect, OWASP, IdentityServer, Azure AD, AWS IAM
• Architecture patterns (if named): Microservices, CQRS, Event Sourcing, DDD, Clean Architecture, Hexagonal
• Version control: Git, GitHub, GitLab, Bitbucket, Azure Repos
• Project tools: Jira, Confluence, Trello, Asana, Monday.com, Slack, Teams
• SEO tools: SEMrush, Ahrefs, Moz, Google Analytics, Google Search Console, Screaming Frog
• Design tools: Figma, Adobe XD, Sketch, Canva, InVision
• Any OTHER named software not listed above

OUTPUT FORMAT — fill ALL four arrays, even if some overlap:
{
  "required_techs": [...],    // from "required", "must have", "essential" sections
  "preferred_techs": [...],   // from "nice to have", "good to have", "preferred", "bonus", "advantageous"
  "title_techs": [...],       // from the job title itself
  "all_extracted": [...],     // UNION of everything above — every single named tool found anywhere
  "job_category": "tech|seo|hr|finance|pm|design|sales|content"
}

CRITICAL COMPLETENESS CHECK before outputting:
Re-read the JD one more time. For each paragraph/bullet, confirm you captured every tool named.
If you find any tool not yet in all_extracted, add it now.
Minimum all_extracted size = number of distinct named tools in the JD. Never output fewer."""

    user = f"""Job Title: {title}

Full Job Description (extract from EVERY line):
{jd}

Extract ALL named technologies — required AND nice-to-have AND mentioned in passing.
Output JSON only:"""

    return system, user


def build_pipeline2_tech_enrichment_universal(jd: str, job_title: str, extracted_techs: dict) -> tuple:
    """
    PIPE 2 — Aggressive ecosystem companion expansion.
    Takes every tool from Pipe 1 and adds its standard production companions.
    Companions are ALWAYS closely related tools — never random tech from other stacks.
    """
    all_techs   = extracted_techs.get("all_extracted",
                  extracted_techs.get("all_technologies", []))
    required    = extracted_techs.get("required_techs", [])
    preferred   = extracted_techs.get("preferred_techs",
                  extracted_techs.get("preferred_techs", []))
    title_techs = extracted_techs.get("title_techs",
                  extracted_techs.get("job_title_techs", []))
    job_category = extracted_techs.get("job_category", "detect_from_jd")

    all_techs_str = ", ".join(all_techs) if all_techs else "tools from the JD"

    system = f"""You are a senior software architect and domain expert. Output ONLY valid JSON.
No markdown, no backticks. Start {{ end }}.

JOB CATEGORY: {job_category if job_category != "detect_from_jd" else "detect from job title and JD"}

YOUR JOB:
1. Keep ALL tools already extracted (never remove any).
2. For EACH extracted tool, add its 3–6 closest standard production companions.
   A companion must be: (a) a real named product, (b) commonly used alongside the parent tool,
   (c) from the SAME technology domain/ecosystem.

═══════════════════════════════════════════════════════════════
COMPANION EXPANSION RULES BY DOMAIN
(expand EVERY tool found — not just the ones listed here)
═══════════════════════════════════════════════════════════════

.NET / C# ECOSYSTEM companions:
  C# → ASP.NET Core, .NET 8, .NET 6, LINQ, Entity Framework Core, Dapper
  ASP.NET Core → C#, .NET 8, Entity Framework Core, Dapper, MediatR, AutoMapper, Swagger/OpenAPI, SignalR
  Entity Framework Core → Dapper, SQL Server, PostgreSQL, Migrations, T-SQL
  .NET → C#, ASP.NET Core, NuGet, MSBuild, Roslyn, dotnet CLI
  WPF / WinForms → C#, .NET, MVVM, INotifyPropertyChanged
  MediatR → CQRS, AutoMapper, FluentValidation, Command Pattern
  SignalR → WebSockets, Real-Time, Hub, .NET
  NUnit/xUnit/MSTest → Moq, FluentAssertions, AutoFixture, TestContainers, Shouldly

Angular ECOSYSTEM companions:
  Angular → TypeScript, RxJS, NgRx, Angular Material, Angular CLI, Angular Router
  TypeScript → JavaScript, TSC, tsconfig, ESLint, Prettier
  RxJS → Angular, Observables, Subjects, Operators, BehaviorSubject
  NgRx → Redux, Store, Effects, Selectors, Entity Adapter

React ECOSYSTEM companions:
  React → TypeScript, Redux Toolkit, React Query, React Router, React Hook Form
  Next.js → React, TypeScript, Vercel, tRPC, Server Components
  Redux → Redux Toolkit, Zustand, Recoil, Context API

Node.js ECOSYSTEM companions:
  Node.js → Express.js, NestJS, TypeScript, npm, Yarn, pnpm
  NestJS → TypeScript, Node.js, Fastify, Prisma, TypeORM, class-validator
  Express.js → Node.js, Middleware, REST APIs, Morgan, Helmet, cors

Python ECOSYSTEM companions:
  Python → pip, pyenv, virtualenv, Mypy, Black, Flake8, Ruff
  Django → Django REST Framework, Celery, Redis, PostgreSQL, Gunicorn
  FastAPI → Pydantic, Uvicorn, SQLAlchemy, Alembic, asyncio
  Flask → Gunicorn, SQLAlchemy, Flask-RESTful, WTForms
  Pandas → NumPy, Matplotlib, Seaborn, Jupyter, Scipy
  PyTest → unittest, coverage, hypothesis, faker, factory-boy

Java ECOSYSTEM companions:
  Java → Maven, Gradle, JDK, Lombok, Jackson, SLF4J
  Spring Boot → Spring MVC, Spring Security, Spring Data JPA, Spring Cloud, Hibernate
  Hibernate → JPA, Spring Data, PostgreSQL, MySQL, Flyway, Liquibase

PHP ECOSYSTEM companions:
  PHP → Composer, Laravel, Symfony, PHPUnit, PHP-FPM
  Laravel → Eloquent ORM, Blade, Artisan, Queue Workers, Horizon, Valet

DATABASE companions:
  SQL Server → T-SQL, SSMS, Entity Framework Core, Dapper, Always On, SSIS
  PostgreSQL → pgAdmin, PgBouncer, Flyway, Liquibase, pg_stat, TimescaleDB
  MySQL → phpMyAdmin, Percona, MariaDB, Sequelize, MySQL Workbench
  MongoDB → Mongoose, Atlas, Compass, Aggregation Pipeline, Change Streams
  Redis → Redis Cluster, Redis Sentinel, StackExchange.Redis, Lettuce, Jedis, Cache-Aside
  Elasticsearch → Kibana, Logstash, Beats, ELK Stack, NEST client

CLOUD companions:
  Azure → Azure App Service, Azure Functions, Azure DevOps, Azure Service Bus, Azure Blob Storage,
           Azure AD, Azure Key Vault, Azure Monitor, Azure SQL Database, Azure Container Registry,
           Azure Cache for Redis, Azure CDN, Azure Logic Apps
  AWS → EC2, S3, Lambda, RDS, ECS, EKS, CloudFormation, CloudWatch, SQS, SNS, API Gateway, IAM
  GCP → Cloud Run, Cloud Functions, BigQuery, Pub/Sub, GKE, Cloud Storage, Firestore

DEVOPS companions:
  Docker → Docker Compose, Kubernetes, Docker Hub, Container Registry, Dockerfile, BuildKit
  Kubernetes → Helm, kubectl, ArgoCD, Istio, Prometheus, Grafana, K3s
  Terraform → Ansible, Pulumi, CloudFormation, Helm, Terragrunt
  GitHub Actions → CI/CD, Workflows, Secrets, Environments, Actions Marketplace
  Azure DevOps → Pipelines, Boards, Repos, Artifacts, Release Gates

TESTING companions:
  xUnit → Moq, FluentAssertions, AutoFixture, TestContainers, Bogus
  Jest → React Testing Library, Enzyme, Supertest, MSW, Vitest
  Cypress → Playwright, Selenium, WebDriver, E2E Testing, Allure
  Pytest → unittest, coverage, hypothesis, mock, requests-mock

MONITORING companions:
  Prometheus → Grafana, Alertmanager, Node Exporter, PushGateway, PromQL
  Datadog → APM, Logs, Metrics, Synthetics, Watchdog
  Application Insights → Azure Monitor, Log Analytics, KQL, Kusto

SEO / MARKETING companions:
  Google Analytics → Google Tag Manager, Google Search Console, Looker Studio, BigQuery
  SEMrush → Ahrefs, Moz, Screaming Frog, Sitebulb, Keyword Planner
  WordPress → Yoast SEO, Rank Math, Elementor, WooCommerce, WP Rocket, ACF

═══════════════════════════════════════════════════════════════

RULES:
- Keep EVERY tool from the input list (never remove).
- Add companions only from the SAME ecosystem — never inject React into a .NET role.
- Aim for 35–55 total tools across core + preferred + ecosystem.
- No duplicates across arrays.
- core_techs = required tools from JD (highest priority)
- preferred_techs = nice-to-have tools from JD
- ecosystem_techs = companions (not mentioned in JD but standard for this stack)

OUTPUT:
{{
  "core_techs": [...],
  "preferred_techs": [...],
  "ecosystem_techs": [...],
  "detected_category": "category name"
}}"""

    user = f"""Job Title: {job_title}

ALL tools extracted from the JD (keep every single one, then add companions):
{all_techs_str}

Required/Must-have tools: {", ".join(required) if required else "(see all_extracted)"}
Preferred/Nice-to-have tools: {", ".join(preferred) if preferred else "(see all_extracted)"}
From job title: {", ".join(title_techs) if title_techs else "(none)"}

Add 3–6 ecosystem companions for EACH tool above. Output JSON:"""

    return system, user


def build_pipeline3_tech_validation_universal(techs: dict, job_title: str, jd: str) -> tuple:
    """
    PIPE 3 — Minimal cross-domain contamination filter.
    ONLY removes tools from a clearly different domain (e.g. React in a finance role).
    Never removes tools that belong to the correct domain.
    Default is KEEP — only remove when 100% certain it is wrong domain.
    """
    core_list      = techs.get("core_techs",      techs.get("core",      []))
    preferred_list = techs.get("preferred_techs",  techs.get("preferred", []))
    ecosystem_list = techs.get("ecosystem_techs",  techs.get("ecosystem", []))
    all_tools_str  = ", ".join(core_list + preferred_list + ecosystem_list)

    system = """You are a precision domain validator. Output ONLY valid JSON.
No markdown, no backticks. Start { end }.

YOUR ONLY JOB: Remove tools that are OBVIOUSLY from the WRONG domain.

KEEP EVERYTHING BY DEFAULT. Only remove when 100% certain the tool has no place here.

REMOVAL RULES (only these specific cases):
1. NON-TECHNICAL role (SEO, HR, Finance, Design, Sales, PM):
   Remove ONLY backend developer frameworks like: Django, Spring Boot, FastAPI, NestJS, Rails.
   KEEP: Any cloud tool, CI/CD tool, analytics tool, or general productivity tool.
   KEEP: REST, API, JSON, Excel, SQL — these appear in many non-developer roles.

2. PURE FRONTEND role (UI developer, React developer, Angular developer):
   Remove ONLY pure infrastructure/server tools with zero frontend relevance:
   e.g. Terraform, Ansible, Kubernetes (unless JD explicitly mentions them).
   KEEP: All frontend libs, testing tools, CI/CD, cloud (for deployment), databases (frontend devs query APIs).

3. TECHNICAL role (any developer/engineer/architect/DevOps/data):
   Remove NOTHING. All technical tools are valid.
   KEEP: Everything.

ABSOLUTE RULES:
- Never remove a tool mentioned ANYWHERE in the JD (required OR preferred OR nice-to-have).
- Never remove a tool just because it seems "advanced" for the seniority level.
- Never remove a tool just because it wasn't in your training data.
- When in doubt: KEEP.

OUTPUT:
{
  "validated_core": [...],
  "validated_preferred": [...],
  "validated_ecosystem": [...],
  "removed_techs": [...],
  "primary_domain": "...",
  "validation_notes": "..."
}"""

    user = f"""Job Title: {job_title}

Job Description (for domain detection):
{jd[:600]}

Tools to validate (keep almost everything — only remove clear cross-domain contamination):
{all_tools_str}

Output JSON:"""

    return system, user


# ===================================================================
# UPDATED generate_cv_atomic with universal tech pipes
# ===================================================================

async def generate_cv_atomic(req: CVRequest, client, key: str, model: str, 
                              url: str, headers: dict) -> dict:
    """Generate CV using 5 pipelines - with DEDICATED universal tech extraction pipes first."""
    import asyncio as _asyncio
    import time as _atomic_time
    
    # Hard 270-second wall clock deadline — every call_llm_atomic checks this
    _deadline = _atomic_time.time() + 270  # 270s gives ~30s buffer before asyncio.wait_for fires
    
    # Delay between calls - only for deepseek-r1 which hits token limits
    _is_r1 = "deepseek-r1" in model.lower()
    _delay = 3 if (_is_r1 and url == GROQ_URL) else 0
    
    years_exp = (req.years_exp or "").strip()
    total_years = _calc_total_years(years_exp)
    companies_list = _build_dynamic_companies(years_exp)
    edu = _build_education_year(years_exp)
    num_cos = len(companies_list)
    jd = req.job_description.strip()[:1500]
    company_name = (req.company_name or "").strip()
    company_ctx = (req.company_context or "").strip()[:400]
    
    # ===================================================================
    # STEP 1: DEDICATED TECH EXTRACTION PIPE (UNIVERSAL - ALL JOB TYPES)
    # ===================================================================
    
    # ── Pipe 1: Exhaustive JD keyword extraction ──────────────────────────────
    # Also run a Python-level regex pass on the raw JD BEFORE the LLM call.
    # This guarantees that even if the LLM misses something, the regex catches it.
    # The two sets are merged so nothing is ever lost.

    _TECH_REGEX = re.compile(
        r'\b(?:'
        # .NET / C# ecosystem
        r'C#|VB\.NET|F#|ASP\.NET(?:\s+Core)?|(?:\.NET(?:\s+\d+)?)|Entity\s+Framework(?:\s+Core)?|'
        r'Dapper|MediatR|AutoMapper|SignalR|NuGet|Roslyn|LINQ|WPF|WinForms|MAUI|Blazor|'
        r'xUnit|NUnit|MSTest|Moq|FluentAssertions|SpecFlow|BenchmarkDotNet|'
        # Java ecosystem
        r'Java(?:\s+\d+)?|Spring(?:\s+Boot)?|Hibernate|Maven|Gradle|JPA|Lombok|Jackson|JUnit(?:\s+5)?|'
        r'Mockito|Tomcat|Jetty|Kafka|RabbitMQ|ActiveMQ|'
        # Python ecosystem
        r'Python(?:\s+3\.\w+)?|Django(?:\s+REST\s+Framework)?|FastAPI|Flask|SQLAlchemy|Alembic|'
        r'Pandas|NumPy|Celery|Pytest|Pydantic|Gunicorn|Uvicorn|Scrapy|BeautifulSoup|'
        # JavaScript / TypeScript ecosystem
        r'JavaScript|TypeScript|Node\.js|Express\.js|NestJS|Next\.js|Nuxt(?:\.js)?|'
        r'React(?:\.js)?|Angular(?:\s+\d+)?|Vue(?:\.js)?|Svelte|'
        r'Redux(?:\s+Toolkit)?|NgRx|RxJS|Zustand|Vite|Webpack|Rollup|Babel|ESLint|Prettier|'
        r'Jest|Cypress|Playwright|Jasmine|Karma|Vitest|Storybook|'
        # PHP ecosystem
        r'PHP(?:\s+\d+)?|Laravel|Symfony|Composer|Eloquent|Blade|'
        # Ruby
        r'Ruby(?:\s+on\s+Rails)?|Rails|RSpec|Sidekiq|'
        # Go / Rust / others
        r'Golang|Go\b|Rust|Swift|Kotlin|Scala|Erlang|Elixir|'
        # Databases
        r'SQL\s+Server|PostgreSQL|MySQL|MariaDB|MongoDB|Redis|Elasticsearch|'
        r'DynamoDB|Cosmos\s+DB|Oracle\s+DB|SQLite|Cassandra|InfluxDB|Neo4j|'
        r'Firestore|Supabase|PlanetScale|CockroachDB|'
        # Cloud
        r'AWS|Azure|GCP|Google\s+Cloud|Firebase|Heroku|Vercel|Netlify|DigitalOcean|'
        r'EC2|S3|Lambda|RDS|ECS|EKS|CloudFront|CloudFormation|CloudWatch|SQS|SNS|'
        r'API\s+Gateway|Route\s+53|IAM|VPC|'
        r'Azure\s+App\s+Service|Azure\s+Functions|Azure\s+DevOps|Azure\s+Service\s+Bus|'
        r'Azure\s+Blob\s+Storage|Azure\s+AD|Azure\s+Key\s+Vault|Azure\s+Monitor|'
        r'Azure\s+SQL|Azure\s+Container\s+Registry|Azure\s+Cache|Azure\s+CDN|'
        r'Cloud\s+Run|Cloud\s+Functions|BigQuery|Pub/Sub|GKE|Cloud\s+Storage|'
        # DevOps
        r'Docker(?:\s+Compose)?|Kubernetes|Helm|Terraform|Ansible|Puppet|Chef|'
        r'GitHub\s+Actions|Azure\s+Pipelines|Jenkins|GitLab\s+CI|CircleCI|'
        r'ArgoCD|Flux|Istio|Linkerd|Prometheus|Grafana|Datadog|'
        r'New\s+Relic|Sentry|ELK(?:\s+Stack)?|Splunk|PagerDuty|'
        # Version control / tools
        r'Git(?:Hub|Lab|)?|Bitbucket|Azure\s+Repos|Jira|Confluence|Trello|'
        r'Slack|Teams|Zoom|Notion|Linear|ClickUp|Asana|Monday\.com|'
        # SEO / Marketing tools
        r'SEMrush|Ahrefs|Moz|Google\s+Analytics(?:\s+4)?|Google\s+Search\s+Console|'
        r'Google\s+Tag\s+Manager|Screaming\s+Frog|Sitebulb|Looker\s+Studio|'
        r'Google\s+Ads|Facebook\s+Ads|LinkedIn\s+Ads|TikTok\s+Ads|Microsoft\s+Ads|'
        r'HubSpot|Mailchimp|Klaviyo|ActiveCampaign|SendGrid|Marketo|Pardot|'
        r'WordPress|Yoast\s+SEO|Rank\s+Math|Elementor|Webflow|Shopify|WooCommerce|'
        r'Hotjar|Crazy\s+Egg|Mixpanel|Heap|Amplitude|Optimizely|'
        # Design tools
        r'Figma|Adobe\s+XD|Sketch|InVision|Zeplin|Framer|Canva|Storybook|'
        r'Photoshop|Illustrator|After\s+Effects|Premiere\s+Pro|InDesign|'
        # Finance / HR / PM tools
        r'QuickBooks|Xero|SAP|Oracle\s+Financials|NetSuite|Sage|FreshBooks|'
        r'Workday|BambooHR|Greenhouse|Lever|ADP|SAP\s+SuccessFactors|iCIMS|Taleo|'
        r'Tableau|Power\s+BI|Looker|QlikView|Domo|Excel|PowerPoint|'
        r'Salesforce|Pipedrive|Zoho\s+CRM|HubSpot\s+CRM|Outreach|SalesLoft|ZoomInfo|'
        # Architecture patterns (when explicitly named)
        r'Microservices|CQRS|Event\s+Sourcing|Clean\s+Architecture|DDD|'
        r'OAuth(?:\s+2\.0)?|JWT|OpenID\s+Connect|OWASP|SAML|LDAP|SSO|'
        r'REST(?:ful)?|GraphQL|gRPC|WebSockets|SOAP|OpenAPI|Swagger|'
        r'CI/CD|DevOps|Agile|Scrum|Kanban'
        r')\b',
        re.IGNORECASE
    )

    # Run regex extraction on raw JD + job title
    _jd_full  = req.job_description.strip()
    _jt_full  = req.job_title.strip()
    _regex_raw  = re.findall(_TECH_REGEX, _jd_full + " " + _jt_full)
    # Deduplicate preserving original casing of first occurrence
    _seen_lower = set()
    _regex_found = []
    for t in _regex_raw:
        if t.lower() not in _seen_lower:
            _seen_lower.add(t.lower())
            _regex_found.append(t)
    print(f"[PIPE1-REGEX] Extracted {len(_regex_found)} tools directly from JD: {_regex_found[:15]}")

    # LLM extraction (catches tools the regex doesn't know about)
    sys1, usr1 = build_pipeline1_tech_extraction_universal(req)
    result1 = await call_llm_atomic(client, key, model, url, sys1, usr1, "Pipe1-TechExtraction", headers, max_tokens=1000, _deadline=_deadline)

    # Merge regex + LLM results — regex results go first (highest confidence)
    _llm_all   = result1.get("all_extracted", result1.get("all_technologies", []))
    _llm_req   = result1.get("required_techs", [])
    _llm_pref  = result1.get("preferred_techs", [])
    _llm_title = result1.get("title_techs", result1.get("job_title_techs", []))

    # Build merged all_extracted: regex first, then LLM additions
    _merged_all = list(_regex_found)
    for t in _llm_all:
        if t and t.lower() not in _seen_lower and len(t) > 1:
            _seen_lower.add(t.lower())
            _merged_all.append(t)

    # Ensure required/preferred from LLM are in merged_all
    for t in (_llm_req + _llm_pref + _llm_title):
        if t and t.lower() not in _seen_lower and len(t) > 1:
            _seen_lower.add(t.lower())
            _merged_all.append(t)

    print(f"[PIPE1-MERGED] Total after regex+LLM merge: {len(_merged_all)} tools: {_merged_all[:20]}")

    extracted_techs = {
        "all_extracted":   _merged_all,
        "required_techs":  _llm_req  or [t for t in _regex_found[:8]],
        "preferred_techs": _llm_pref or [],
        "title_techs":     _llm_title or list(re.findall(_TECH_REGEX, _jt_full)),
        "job_category":    result1.get("job_category", "detect_from_jd"),
    }

    await _asyncio.sleep(_delay)

    # ── Pipe 2: Aggressive ecosystem companion expansion ────────────────────────
    sys2, usr2 = build_pipeline2_tech_enrichment_universal(jd, req.job_title, extracted_techs)
    result2 = await call_llm_atomic(client, key, model, url, sys2, usr2, "Pipe2-TechEnrichment", headers, max_tokens=1400, _deadline=_deadline)

    _p2_core  = result2.get("core_techs",      [])
    _p2_pref  = result2.get("preferred_techs", [])
    _p2_eco   = result2.get("ecosystem_techs", [])
    _detected = result2.get("detected_category", extracted_techs.get("job_category", "tech"))

    # Safety: always include ALL merged_all in core — Pipe 2 must never lose JD keywords
    _p2_seen = set(t.lower() for t in _p2_core + _p2_pref + _p2_eco)
    _missing_from_jd = [t for t in _merged_all if t.lower() not in _p2_seen]
    if _missing_from_jd:
        print(f"[PIPE2-SAFETY] Re-adding {len(_missing_from_jd)} JD tools dropped by Pipe2: {_missing_from_jd[:10]}")
        _p2_core = list(dict.fromkeys(_p2_core + _missing_from_jd))

    techs = {
        "core":      _p2_core  or extracted_techs["required_techs"]  or _merged_all[:15],
        "preferred": _p2_pref  or extracted_techs["preferred_techs"] or [],
        "ecosystem": _p2_eco,
        "detected_domain": _detected,
    }

    print(f"[PIPE2] core={len(techs['core'])} preferred={len(techs['preferred'])} ecosystem={len(techs['ecosystem'])}")
    print(f"[PIPE2] core: {techs['core'][:12]}")

    await _asyncio.sleep(_delay)

    # ── Pipe 3: Minimal cross-domain filter (keep everything when in doubt) ─────
    sys3, usr3 = build_pipeline3_tech_validation_universal(techs, req.job_title, jd)
    result3 = await call_llm_atomic(client, key, model, url, sys3, usr3, "Pipe3-TechValidation", headers, max_tokens=700, _deadline=_deadline)

    validated_core      = result3.get("validated_core",      techs["core"])
    validated_preferred = result3.get("validated_preferred",  techs["preferred"])
    validated_ecosystem = result3.get("validated_ecosystem",  techs["ecosystem"])
    detected_domain     = result3.get("primary_domain", techs.get("detected_domain", _detected))

    # Critical safety: never let Pipe 3 remove tools that were in the original JD
    removed      = result3.get("removed_techs", [])
    removed_lower = set(r.lower() for r in removed)
    # Any tool that was in the regex extraction or LLM all_extracted is JD-confirmed
    # and must NEVER be removed by validation
    jd_confirmed = set(t.lower() for t in _merged_all)
    # Only allow removal of ecosystem tools (companions), not JD-confirmed tools
    safe_removed = {r for r in removed_lower if r not in jd_confirmed}

    techs = {
        "core":      [t for t in validated_core      if t.lower() not in safe_removed],
        "preferred": [t for t in validated_preferred  if t.lower() not in safe_removed],
        "ecosystem": [t for t in validated_ecosystem  if t.lower() not in safe_removed],
        "detected_domain": detected_domain,
    }

    # Final safety: re-add any JD-confirmed tools that slipped through
    _final_seen = set(t.lower() for t in techs["core"] + techs["preferred"] + techs["ecosystem"])
    _still_missing = [t for t in _merged_all if t.lower() not in _final_seen]
    if _still_missing:
        print(f"[PIPE3-SAFETY] Re-adding {len(_still_missing)} JD tools after validation: {_still_missing[:10]}")
        techs["core"] = list(dict.fromkeys(techs["core"] + _still_missing))

    print(f"[PIPE3] Final — core={len(techs['core'])} preferred={len(techs['preferred'])} ecosystem={len(techs['ecosystem'])}")
    print(f"[PIPE3] Final core: {techs['core'][:12]}")

    # Ensure minimum counts
    if len(techs["core"]) < 3:
        techs["core"] = list(dict.fromkeys(techs["core"] + techs["preferred"] + techs["ecosystem"]))[:10]
    if len(techs["preferred"]) < 2:
        techs["preferred"] = techs["preferred"] + techs["ecosystem"][:max(0, 2-len(techs["preferred"]))]

    all_techs_flat = list(dict.fromkeys(techs["core"] + techs["preferred"] + techs["ecosystem"]))[:50]
    techs_str = ", ".join(all_techs_flat) if all_techs_flat else "skills from the job description"

    await _asyncio.sleep(_delay)
    
    # ===================================================================
    # STEP 2: TITLE + SUMMARY (uses validated techs for all job types)
    # ===================================================================
    
    # Determine appropriate function words based on domain
    domain = detected_domain.lower()
    if "seo" in domain or "marketing" in domain:
        function_options = ["Specialist", "Strategist", "Analyst", "Manager", "Consultant"]
    elif "hr" in domain or "recruitment" in domain:
        function_options = ["Specialist", "Generalist", "Partner", "Manager", "Coordinator"]
    elif "finance" in domain or "accounting" in domain:
        function_options = ["Analyst", "Accountant", "Manager", "Controller", "Specialist"]
    elif "project" in domain or "pm" in domain:
        function_options = ["Manager", "Coordinator", "Lead", "Scrum Master", "Analyst"]
    elif "design" in domain or "creative" in domain:
        function_options = ["Designer", "Specialist", "Artist", "Lead", "Manager"]
    elif "sales" in domain or "crm" in domain:
        function_options = ["Specialist", "Manager", "Representative", "Consultant", "Lead"]
    elif "content" in domain or "writing" in domain:
        function_options = ["Writer", "Specialist", "Manager", "Strategist", "Editor"]
    else:
        function_options = ["Engineer", "Developer", "Specialist", "Analyst", "Manager"]
    
    sys4 = (
        f"You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
        f"JOB DOMAIN DETECTED: {detected_domain}\n\n"
        "TITLE RULES:\n"
        "- MUST be DIFFERENT from the original job title\n"
        f"- Format: '[Seniority] [Domain Focus] [{'/'.join(function_options[:3])}] | Tool1, Tool2, Tool3'\n"
        "- Use ONLY tools from the allowed list below\n"
        "- For non-technical roles: use appropriate function words (Strategist, Analyst, Manager, Specialist)\n\n"
        "SUMMARY RULES:\n"
        f"- 4 sentences, minimum 70 words, start with '{total_years} years of experience in [domain]...'\n"
        "- Name 4-5 tools from the allowed list below\n"
        "- Include metrics and concrete outcomes appropriate for the role\n"
        "- NO AI buzzwords, NO location words, NO company names\n\n"
        f"Allowed tools (use ONLY these): {techs_str}"
    )
    
    core_sample = techs['core'][:3] if len(techs['core']) >= 3 else techs['core'] + ['Skill'] * (3 - len(techs['core']))
    
    usr4 = f"""Job Title: {req.job_title}
Experience: {total_years} years
Detected Domain: {detected_domain}

Allowed tools: {techs_str}

Output JSON:
{{"title":"{req.job_title.split('|')[0].strip()[:30]} | {core_sample[0]}, {core_sample[1]}, {core_sample[2]}",
"summary":"{total_years} years of experience in {detected_domain} with expertise in {techs_str[:80]}... (4 sentences, 70+ words, using ONLY tools from the allowed list)"}}"""
    
    result4 = await call_llm_atomic(client, key, model, url, sys4, usr4, "Pipe4-TitleSummary", headers, max_tokens=700, _deadline=_deadline)
    
    title_out = result4.get("title", "")
    summary_out = result4.get("summary", "")
    
    if not title_out:
        title_out = f"{req.job_title.split('|')[0].strip()} | {core_sample[0]}, {core_sample[1]}, {core_sample[2]}"
    if not summary_out:
        summary_out = f"{total_years} years of experience in {detected_domain} with expertise in {techs_str[:100]}."
    
    await _asyncio.sleep(_delay)
    
    # ===================================================================
    # STEP 3: SKILLS placeholder — filled by dedicated call AFTER experience
    # Running skills AFTER experience means the model sees exactly which
    # technologies are already in bullets/tech-tags → perfect alignment.
    # No extra LLM call here; the dedicated call replaces this entirely.
    # ===================================================================
    skills_out = []  # populated by extract_dedicated_skills() below
    
    # ===================================================================
    # STEP 4: EXPERIENCE
    # ===================================================================
    
    if num_cos == 1:
        sen_rules = 'Co1 (only): appropriate seniority based on experience'
        verb_guide = "Co1: Led, Managed, Developed, Implemented, Executed, Delivered."
    elif num_cos == 2:
        sen_rules = 'Co1 (current): appropriate seniority. Co2 (oldest): Junior/entry-level'
        verb_guide = "Co1: Led, Managed, Developed, Implemented.\nCo2: Assisted, Supported, Executed, Delivered."
    else:
        sen_rules = 'Co1 (current): Senior level. Co2 (mid): Standard level. Co3 (oldest): Junior/entry-level'
        verb_guide = "Co1 (Senior): Led, Architected, Managed, Spearheaded.\nCo2: Developed, Implemented, Executed, Delivered.\nCo3 (Junior): Assisted, Supported, Learned, Contributed."
    
    co_lines = "\n".join(
        f'Co{j+1}: name="{c["name"]}", dates="{c["start"]} - {c["end"]}"' 
        for j, c in enumerate(companies_list)
    )
    
    sys6 = (
        f"You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
        f"JOB DOMAIN: {detected_domain}\n"
        f"Allowed tools (use ONLY these): {techs_str}\n\n"
        "EXPERIENCE RULES:\n"
        f"Produce EXACTLY {num_cos} companies.\n"
        f"Seniority: {sen_rules}\n"
        f"Verb guide:\n{verb_guide}\n"
        "Each company: unique role title, 4 bullets (20-30 words each), 5-6 tool tags from allowed list.\n"
        "Bullets: DIFFERENT achievement per bullet, UNIQUE metric per bullet, no repeated verbs.\n"
        "Tool tags: use only allowed tools, preferably varied across companies.\n"
        f"Role title format: '[Seniority] [Domain Focus] [{'/'.join(function_options[:3])}]'"
    )
    
    usr6 = f"""Job Title: {req.job_title}
Experience: {total_years} years
Detected Domain: {detected_domain}

Companies (use exact names and dates):
{co_lines}

Allowed tools: {techs_str}

Output JSON:
{{"companies":[
  {{"company":"EXACT NAME","role":"Seniority Domain Function","dateRange":"Start - End",
    "bullets":["Achievement using tool from allowed list (20-30w)","Achievement (20-30w)","Achievement (20-30w)","Achievement (20-30w)"],
    "tech":"Tool1 | Tool2 | Tool3 | Tool4 | Tool5"}},
  ...({num_cos} total)
]}}"""
    
    result6 = await call_llm_atomic(client, key, model, url, sys6, usr6, "Pipe6-Experience", headers, max_tokens=1600, _deadline=_deadline)
    companies_out = result6.get("companies", [])
    
    # Fix company names/dates
    for i, co in enumerate(companies_out):
        if i < len(companies_list):
            co["company"] = companies_list[i]["name"]
            co["dateRange"] = f'{companies_list[i]["start"]} - {companies_list[i]["end"]}'
    
    if not companies_out:
        companies_out = []
        for i, c in enumerate(companies_list):
            companies_out.append({
                "company": c["name"],
                "role": f"{detected_domain.capitalize()} {function_options[0]}",
                "dateRange": f'{c["start"]} - {c["end"]}',
                "bullets": [
                    f"Delivered key initiatives using {techs['core'][0] if techs['core'] else 'core tools'}",
                    "Achieved measurable improvements in operational efficiency",
                    "Collaborated with cross-functional teams to drive results",
                    "Maintained high standards of quality and compliance"
                ],
                "tech": " | ".join(all_techs_flat[:5])
            })
    
    await _asyncio.sleep(_delay)
    
    # ===================================================================
    # STEP 3: SKILLS — DEDICATED SECOND LLM REQUEST
    # This runs AFTER experience is assembled so the skills section
    # sees all technologies used in experience bullets and tech tags.
    # ===================================================================
    
    print(f"\n[ATOMIC] Starting DEDICATED SKILLS extraction (separate LLM call)")
    print(f"[ATOMIC] Detected domain: {detected_domain}")
    print(f"[ATOMIC] Tech pool: core={len(techs.get('core',[]))} preferred={len(techs.get('preferred',[]))} ecosystem={len(techs.get('ecosystem',[]))}")
    print(f"[ATOMIC] Core: {techs.get('core', [])[:8]}")
    
    # Build a partial CV so the skills extractor can see companies context
    cv_for_skills = {
        "companies":    companies_out,
        "technologies": {
            "mustHave":   techs.get("core", [])[:12],
            "niceToHave": techs.get("preferred", [])[:10],
            "additional": techs.get("ecosystem", [])[:10],
        },
    }
    
    # Use OpenAI-compatible call for all atomic providers
    dedicated_skills = await extract_dedicated_skills(
        client   = client,
        key      = key,
        model    = model,
        url      = url,
        headers  = headers,
        req      = req,
        cv       = cv_for_skills,
        techs    = techs,
        max_tokens = 1800,
        provider = "openai_compat",
        _deadline = _deadline
    )
    
    print(f"[ATOMIC] Dedicated skills: {len(dedicated_skills)} categories")
    for i, s in enumerate(dedicated_skills):
        colon = s.find(":")
        cat   = s[:colon].strip() if colon > 0 else "?"
        items = [t.strip() for t in s[colon+1:].split(",")] if colon > 0 else []
        print(f"[ATOMIC]   [{i+1}] {cat}: {len(items)} items → {items[:4]}")
    
    # Use dedicated skills if we got good results; fallback to pipeline skills
    skills_out = dedicated_skills if len(dedicated_skills) >= 3 else skills_out
    
    await _asyncio.sleep(_delay)
    
    # ===================================================================
    # STEP 5: PROJECTS (or relevant achievements for non-technical roles)
    # ===================================================================
    
    used_systems = []
    for co in companies_out:
        for b in (co.get("bullets") or []):
            if len(b) > 20:
                used_systems.append(b[:60])
    used_str = "\n".join(f"  - {s}" for s in used_systems[:4]) or "  (none)"
    
    co_intel = ""
    if company_ctx and company_name:
        co_intel = f"Target Company: {company_name}\nContext: {company_ctx}\nProjects 3&4: analogous domain in adjacent sector."
    elif company_name:
        co_intel = f"Target Company: {company_name}\nProjects 3&4: use your knowledge of {company_name}'s industry for analogous projects."
    
    # Determine if this is a project-heavy role or achievement-heavy role
    is_technical = any(kw in domain.lower() for kw in ['tech', 'developer', 'engineer', 'devops', 'data'])
    
    sys7 = (
        f"You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
        f"JOB DOMAIN: {detected_domain}\n"
        f"Is Technical Role: {is_technical}\n"
        f"Allowed tools: {techs_str}\n\n"
    )
    
    if is_technical:
        sys7 += (
            "PROJECTS RULES:\n"
            "- Exactly 4 projects\n"
            "- Each project name: format 'CoinedWord - what it does [Tool1, Tool2]'\n"
            "- Each overview: 3-4 natural sentences (problem -> solution -> functionality -> impact)\n"
            "- Each bullets: 3 strings (20-30 words, different verb, unique metric)\n"
            "- Each techTags: 5-6 tools from allowed list, none repeated across projects\n"
            "BANNED: generic names like 'Web App', real company names in projects, repeated verbs/metrics\n\n"
            "RELATED TECH: exactly 5 category boxes, 5 items each, all from allowed list, zero duplicates"
        )
    else:
        sys7 += (
            "ACHIEVEMENTS & INITIATIVES RULES (for non-technical roles):\n"
            "- Exactly 4 key achievements or initiatives\n"
            "- Each name: format 'Achievement Name - brief description [Tool1, Tool2]'\n"
            "- Each overview: 3-4 natural sentences (challenge -> approach -> execution -> result)\n"
            "- Each bullets: 3 strings (20-30 words, different verb, unique metric)\n"
            "- Each techTags: 4-5 tools from allowed list\n"
            "BANNED: generic achievements, real company names, repeated verbs/metrics\n\n"
            "RELATED TOOLS: exactly 5 category boxes, 5 items each, all from allowed list, zero duplicates"
        )
    
    usr7 = f"""Job Title: {req.job_title}
Detected Domain: {detected_domain}
{co_intel}

Allowed tools: {techs_str}
Previous experience used (do NOT repeat): {used_str}

Output JSON:
{{"{'projects' if is_technical else 'achievements'}":[
  {{"name":"{'CoinedName' if is_technical else 'Achievement Name'} - description [{techs['core'][0] if techs['core'] else 'Tool'}, {techs['core'][1] if len(techs['core']) > 1 else 'Tool'}]",
    "overview":"3-4 sentence story.",
    "bullets":["verb + outcome (20-30w)","verb + unique metric (20-30w)","verb + business impact (20-30w)"],
    "techTags":["{techs['core'][0] if techs['core'] else 'Tool'}", "{techs['core'][1] if len(techs['core']) > 1 else 'Tool'}", "{techs['core'][2] if len(techs['core']) > 2 else 'Tool'}", "{techs['core'][3] if len(techs['core']) > 3 else 'Tool'}"]}}
],
"relatedTech":[{{"category":"Domain Category","items":["{techs['core'][0] if techs['core'] else 'Tool'}", "{techs['core'][1] if len(techs['core']) > 1 else 'Tool'}", "{techs['core'][2] if len(techs['core']) > 2 else 'Tool'}", "{techs['core'][3] if len(techs['core']) > 3 else 'Tool'}", "{techs['core'][4] if len(techs['core']) > 4 else 'Tool'}"]}}]}}"""
    
    result7 = await call_llm_atomic(client, key, model, url, sys7, usr7, "Pipe7-ProjectsAchievements", headers, max_tokens=1800, _deadline=_deadline)
    
    projects_out = result7.get("projects", []) or result7.get("achievements", [])
    related_out = result7.get("relatedTech", [])
    if projects_out:
        projects_out = validate_project_techs({"projects": projects_out}, techs).get("projects", projects_out)
    
    # ===================================================================
    # ASSEMBLE FINAL CV
    # ===================================================================
    
    # Generate competencies based on domain
    if "seo" in domain:
        competencies = "Keyword Research * On-Page SEO * Technical SEO * Link Building * Content Strategy * Google Analytics * Search Console * Performance Tracking * Competitor Analysis * SEO Auditing"
    elif "hr" in domain:
        competencies = "Talent Acquisition * Employee Relations * Performance Management * HRIS Administration * Benefits Administration * Compliance * Recruitment Strategy * Onboarding * Offboarding * HR Analytics"
    elif "finance" in domain:
        competencies = "Financial Analysis * Budgeting & Forecasting * Financial Reporting * Tax Compliance * Audit * Risk Management * Data Visualization * Process Improvement * Strategic Planning * Cost Optimization"
    elif "project" in domain:
        competencies = "Project Planning * Risk Management * Stakeholder Management * Agile Methodologies * Scrum * Budget Management * Resource Allocation * Timeline Management * Quality Assurance * Change Management"
    elif "design" in domain:
        competencies = "UI Design * UX Research * Wireframing * Prototyping * User Testing * Visual Design * Design Systems * Brand Identity * Interaction Design * Accessibility"
    else:
        competencies = " * ".join(techs["core"][:10]) if len(techs["core"]) >= 10 else "Technical Leadership * Problem Solving * Team Collaboration * Project Delivery * Quality Assurance * Process Improvement * Stakeholder Management * Strategic Planning * Innovation * Communication"
    
    cv = {
        "totalYears": total_years,
        "title": title_out,
        "summary": summary_out,
        "skills": skills_out,
        "competencies": competencies,
        "companies": companies_out,
        "projects": projects_out,
        "relatedTech": related_out,
        "technologies": {
            "mustHave": techs.get("core", [])[:10],
            "niceToHave": techs.get("preferred", [])[:8],
            "additional": techs.get("ecosystem", [])[:8],
        },
        "education": {
            "university": "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
            "degree": "Bachelor of Science in Computer Science (BSCS)",
            "cgpa": "3.97/4.0",
            "years": f"{edu['start']} - {edu['end']}",
            "achievement": "Gold Medalist for Academic Excellence",
        },
        "keywords": ", ".join(all_techs_flat[:15]),
        "architectures": [],
        "_techs": techs,
        "_job_title": req.job_title,
        "_domain": detected_domain,
    }
    
    print(f"[ATOMIC] Starting post-processing pipeline")
    print(f"[ATOMIC] Skills going into post-processing: {len(skills_out)} categories")
    cv_sanitised = sanitise_cv(cv)
    cv_companies = fix_companies(cv_sanitised)
    cv_skills = fix_skills(cv_companies)
    print(f"[ATOMIC] After fix_skills: {len(cv_skills.get('skills', []))} categories")
    for i, s in enumerate(cv_skills.get("skills", [])):
        colon = s.find(":")
        cat   = s[:colon].strip() if colon > 0 else "?"
        items = [t.strip() for t in s[colon+1:].split(",")] if colon > 0 else []
        print(f"[ATOMIC]   [{i+1}] {cat}: {len(items)} items")
    cv_enforced = _enforce_skill_domains(cv_skills, techs, req.job_title)
    if projects_out:
        cv_projtags = _repair_project_tech_tags(cv_enforced, techs)
    else:
        cv_projtags = cv_enforced
    cv_polished = final_polish(fix_skills_dedup(fix_projects(cv_projtags)), years_exp=years_exp)
    
    final_skills = cv_polished.get("skills", [])
    print(f"[ATOMIC] FINAL skills count: {len(final_skills)} categories")
    for i, s in enumerate(final_skills):
        colon = s.find(":")
        cat   = s[:colon].strip() if colon > 0 else "?"
        items = [t.strip() for t in s[colon+1:].split(",")] if colon > 0 else []
        missing_from_jd = []
        print(f"[ATOMIC] FINAL [{i+1}] {cat}: {len(items)} items → {items[:5]}")
    
    return run_validation_pipeline(cv_polished, req.job_description, req.job_title, techs)
# ===================================================================
# call_cerebras - robust key rotation with detailed error reporting
# ===================================================================

async def call_cerebras(req: CVRequest) -> tuple:
    raw_keys = req.cerebras_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Cerebras keys provided.")

    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid Cerebras keys found.")

    model = req.model or "llama3.1-8b"
    # Remove any bad/known-unavailable models - auto-fallback to working models
    _BAD_CEREBRAS_MODELS = {"qwen-3-235b-a22b-instruct-2507", "zai-glm-4.7", "gpt-oss-120b"}
    _FALLBACK_MODELS = ["llama3.1-8b", "llama-3.3-70b"]
    if model.lower() in _BAD_CEREBRAS_MODELS:
        model = "llama3.1-8b"  # Force override bad models
    
    sorted_keys = _prioritised_keys(valid_keys)

    last_error = "Unknown error"
    errors_by_key: list = []

    # Overall hard deadline: 260 seconds total for all keys (gives 40s buffer vs 300s client)
    import time as _cb_time
    _cb_deadline = _cb_time.time() + 260  # 260s hard wall clock limit

    # Session timeout aligned to 5-minute goal
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=180, write=15, pool=10)) as client:
        # -- PROBE: First check if the model is available on this key ----------
        # This prevents spending 3 pipeline calls on a 404 model.
        for probe_key in sorted_keys:
            pk_mk = mask(probe_key)
            try:
                probe_r = await client.post(
                    CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {probe_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 1},
                    timeout=10,
                )
                if probe_r.status_code == 404:
                    # Model not found - try fallback models
                    fallback_used = None
                    for fb_model in _FALLBACK_MODELS:
                        if fb_model == model:
                            continue
                        fb_r = await client.post(
                            CEREBRAS_URL,
                            headers={"Authorization": f"Bearer {probe_key}", "Content-Type": "application/json"},
                            json={"model": fb_model, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 1},
                            timeout=10,
                        )
                        if fb_r.status_code == 200:
                            fallback_used = fb_model
                            break
                    if fallback_used:
                        errors_by_key.append(f"Model '{model}' not found - auto-switched to '{fallback_used}'")
                        model = fallback_used
                        break
                    else:
                        errors_by_key.append(f"Model '{model}' not found and no fallback available - try llama3.1-8b")
                        last_error = "model not found"
                        continue
                elif probe_r.status_code in (401, 403):
                    errors_by_key.append(f"Key {pk_mk}: invalid key - skipping")
                    continue
                elif probe_r.status_code == 429:
                    errors_by_key.append(f"Key {pk_mk}: rate limited - skipping")
                    continue
                elif probe_r.status_code == 200:
                    # Model found and key works - keep this key
                    sorted_keys = [probe_key] + [k for k in sorted_keys if k != probe_key]
                    break
            except Exception:
                continue
        else:
            # All probes failed
            pass  # Continue anyway - may get better errors from real calls
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            try:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

                # -- NO probe for Cerebras -------------------------------------
                # The probe burns one request slot and can falsely trigger a 429
                # before the real generation starts. Auth/model errors are caught
                # below during the first pipeline call instead.

                # Key is live - run full atomic pipeline generation
                cv = await generate_cv_atomic(req, client, key, model, CEREBRAS_URL, headers)

                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                return cv, mk, i

            except ValueError as e:
                err_str = str(e)
                # -- 429: Skip retry if it would push us past the 2-min budget ---
                if "rate limited" in err_str.lower() or "429" in err_str:
                    retry_match = re.search(r"retry.after[=:\s]+(\d+)", err_str, re.I)
                    wait_s = int(retry_match.group(1)) if retry_match else 30
                    wait_s = min(wait_s, 20)   # cap at 20s to stay within 2-min budget
                    errors_by_key.append(
                        f"Key {i+1} ({mk}): rate limited (429) - "
                        f"waiting {wait_s}s then retrying"
                    )
                    last_error = "rate limited"
                    await asyncio.sleep(wait_s)
                    try:
                        cv = await generate_cv_atomic(req, client, key, model, CEREBRAS_URL, headers)
                        _key_usage[mk] = _key_usage.get(mk, 0) + 1
                        _log_generation(req.job_title, mk, i, 0, model, True)
                        errors_by_key.pop()   # remove the "waiting?" notice on success
                        return cv, mk, i
                    except Exception as retry_exc:
                        retry_str = str(retry_exc)
                        if "rate limited" in retry_str.lower() or "429" in retry_str:
                            errors_by_key.append(
                                f"Key {i+1} ({mk}): still rate limited after {wait_s}s - "
                                "daily limit likely reached; try a different key"
                            )
                            last_error = "rate limited (daily limit)"
                        else:
                            errors_by_key.append(f"Key {i+1} ({mk}): retry failed - {retry_str[:120]}")
                            last_error = retry_str[:120]
                    continue
                if "invalid key" in err_str.lower() or "401" in err_str or "403" in err_str:
                    errors_by_key.append(f"Key {i+1} ({mk}): key rejected - invalid or expired")
                    last_error = "invalid key"
                    continue
                if "not found" in err_str.lower() and ("model" in err_str.lower() or "404" in err_str):
                    errors_by_key.append(
                        f"Key {i+1} ({mk}): model \'{model}\' not found - "
                        "open Keys & Model tab and switch to llama3.1-8b"
                    )
                    last_error = f"model \'{model}\' not found"
                    break
                errors_by_key.append(f"Key {i+1} ({mk}): pipeline error - {err_str[:140]}")
                last_error = err_str[:140]
                continue
            except httpx.TimeoutException:
                errors_by_key.append(
                    f"Key {i+1} ({mk}): generation timed out (>90 s per call) - "
                    "Cerebras may be under load; try again or switch to Groq"
                )
                last_error = "generation timeout"
                continue
            except httpx.ConnectError as ce:
                errors_by_key.append(f"Key {i+1} ({mk}): lost connection - {ce}")
                last_error = "connection lost"
                break
            except Exception as e:
                errors_by_key.append(
                    f"Key {i+1} ({mk}): {type(e).__name__} - {str(e)[:120]}"
                )
                last_error = f"{type(e).__name__}: {str(e)[:120]}"
                continue

    # Build a detailed, actionable error message
    lines = ["All Cerebras keys failed:"]
    lines.extend(f"  * {err}" for err in errors_by_key)
    lines.append("")
    if "rate limited" in last_error:
        lines.append("FIX: Daily token limit reached. Add more keys (cloud.cerebras.ai - free per email) OR switch to Groq in the Keys & Model tab.")
    elif "model" in last_error and "not found" in last_error:
        lines.append(f"FIX: Model \'{model}\' is unavailable on your plan. Open Keys & Model tab -> select llama3.1-8b.")
    elif "invalid key" in last_error or "unauthorized" in last_error:
        lines.append("FIX: Key is invalid or expired. Get a fresh key at cloud.cerebras.ai (free, no credit card).")
    elif "connection" in last_error:
        lines.append("FIX: Cannot reach api.cerebras.ai. Check your internet connection or VPN. Alternatively switch to Groq.")
    elif "timeout" in last_error:
        lines.append("FIX: Cerebras is slow right now. Switch to Groq (console.groq.com - free) in Keys & Model tab.")
    else:
        lines.append("FIX: Try switching provider to Groq in the Keys & Model tab (console.groq.com - free & fast).")

    raise HTTPException(502, "\n".join(lines))



async def call_ollama(req: CVRequest) -> dict:
    model  = req.ollama_model or "llama3.2:3b"
    cfg    = OLLAMA_CONFIGS.get(model, OLLAMA_DEFAULT)

    is_small     = any(x in model for x in ("3b", "mini", "small", "1b"))
    jd_chars     = 800 if is_small else 1600
    sys_p, usr_p = build_prompt(req, jd_chars=jd_chars)
    prompt       = sys_p + "\n\n" + usr_p

    try:
        async with httpx.AsyncClient(timeout=cfg["timeout"]) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model, "prompt": prompt, "stream": False,
                "options": {
                    "temperature":    cfg["temperature"],
                    "top_p":          0.85,
                    "top_k":          40,
                    "repeat_penalty": 1.1,
                    "num_predict":    cfg["num_predict"],
                    "num_ctx":        cfg["num_ctx"]
                }
            })
    except httpx.TimeoutException:
        raise HTTPException(504,
            f"Ollama model '{model}' timed out after {cfg['timeout']}s. Try qwen2.5:7b instead.")
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot connect to Ollama on localhost:11434. Run: ollama serve")

    if r.status_code == 404:
        raise HTTPException(404, f"Model '{model}' not found. Run: ollama pull {model}")
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama error {r.status_code}: {r.text[:300]}")

    raw = r.json().get("response", "")
    if not raw.strip():
        raise HTTPException(502, f"Ollama '{model}' returned empty response. Try qwen2.5:7b.")

    _cv_raw  = sanitise_cv(extract_json(raw))
    _cv_raw  = fix_companies(_cv_raw)
    _cv_raw  = fix_skills(_cv_raw)
    _techs_o = {"core": _cv_raw.get("technologies",{}).get("mustHave",[]),
                "preferred": _cv_raw.get("technologies",{}).get("niceToHave",[]),
                "ecosystem": _cv_raw.get("technologies",{}).get("additional",[])}
    _cv_raw  = _enforce_skill_domains(_cv_raw, _techs_o, req.job_title)
    _cv_raw  = _repair_project_tech_tags(_cv_raw, _techs_o)
    _cv_polished = final_polish(fix_skills_dedup(fix_projects(_cv_raw)), years_exp=(req.years_exp or ''))
    return run_validation_pipeline(_cv_polished, req.job_description, req.job_title, _techs_o)


# -- Main endpoint -------------------------------------------------------------

# -- DeepSeek caller (OpenAI-compatible) ---------------------------------------
async def call_deepseek(req: CVRequest) -> tuple:
    raw_keys = req.deepseek_keys or []
    if not raw_keys:
        raise HTTPException(400, "No DeepSeek API keys provided. Get one at platform.deepseek.com")

    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid DeepSeek keys found.")

    model        = req.model or "deepseek-chat"
    sys_p, usr_p = build_prompt(req, jd_chars=1600)

    # deepseek-reasoner uses a special reasoning_effort param; strip <think> from output
    is_reasoner  = "reasoner" in model

    sorted_keys  = _prioritised_keys(valid_keys)
    last_error   = ""
    exhausted    = []

    async with httpx.AsyncClient(timeout=120) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            try:
                payload = {
                    "model":       model,
                    "messages":    [
                        {"role": "system", "content": sys_p},
                        {"role": "user",   "content": usr_p},
                    ],
                    "temperature": 0 if is_reasoner else 0.3,
                    "max_tokens":  4096,
                }
                r = await client.post(
                    DEEPSEEK_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload
                )

                if r.status_code == 200:
                    _key_usage[mk] = _key_usage.get(mk, 0) + 1
                    resp_json = r.json()
                    raw_text  = resp_json["choices"][0]["message"]["content"]
                    usage     = resp_json.get("usage", {})
                    _log_generation(
                        job_title     = req.job_title,
                        key_masked    = mk,
                        key_index     = i,
                        prompt_tokens = usage.get("prompt_tokens", est_tokens(sys_p + usr_p)),
                        model         = model,
                        success       = True,
                    )
                    _cv_raw  = sanitise_cv(extract_json(raw_text))
                    _cv_raw  = fix_companies(_cv_raw)
                    _cv_raw  = fix_skills(_cv_raw)
                    _techs_d = {"core": _cv_raw.get("technologies",{}).get("mustHave",[]),
                                "preferred": _cv_raw.get("technologies",{}).get("niceToHave",[]),
                                "ecosystem": _cv_raw.get("technologies",{}).get("additional",[])}
                    _cv_raw  = _enforce_skill_domains(_cv_raw, _techs_d, req.job_title)
                    _cv_raw  = _repair_project_tech_tags(_cv_raw, _techs_d)
                    _cv_polished = final_polish(fix_skills_dedup(fix_projects(_cv_raw)), years_exp=(req.years_exp or ''))
                    cv = run_validation_pipeline(_cv_polished, req.job_description, req.job_title, _techs_d)
                    return cv, mk, i

                elif r.status_code == 429:
                    exhausted.append(mk)
                    last_error = f"{mk} rate limited"
                    continue

                elif r.status_code in (401, 403):
                    exhausted.append(mk)
                    last_error = f"{mk} invalid key"
                    continue

                elif r.status_code == 402:
                    exhausted.append(mk)
                    last_error = (
                        f"Insufficient balance on key {mk}. "
                        "Top up your DeepSeek account at platform.deepseek.com/top_up"
                    )
                    continue

                else:
                    try:   body = r.json()
                    except Exception: body = {}
                    err = body.get("error", {})
                    msg = err.get("message", f"HTTP {r.status_code}") if isinstance(err, dict) else str(err)
                    if "insufficient" in str(msg).lower() or "balance" in str(msg).lower():
                        last_error = (
                            f"Insufficient balance on key {mk}. "
                            "Top up your DeepSeek account at platform.deepseek.com/top_up"
                        )
                    else:
                        last_error = str(msg)
                    exhausted.append(mk)
                    continue

            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(422, f"JSON parse error: {e}")
            except httpx.TimeoutException:
                last_error = f"{mk} timed out"
                exhausted.append(mk)
                continue
            except Exception as e:
                last_error = str(e)
                exhausted.append(mk)
                continue

    n_limited = sum(1 for e in exhausted if "rate" in e)
    n_balance = sum(1 for e in exhausted if "balance" in e.lower() or "top up" in e.lower())
    if n_balance:
        raise HTTPException(402, f"Insufficient DeepSeek balance. Top up at platform.deepseek.com/top_up")
    if n_limited == len(sorted_keys):
        raise HTTPException(429, f"All DeepSeek key(s) rate limited. Add more at platform.deepseek.com.")
    raise HTTPException(502, f"All DeepSeek keys failed. Last error: {last_error}")


# -- OpenAI caller -------------------------------------------------------------
async def call_openai(req: CVRequest) -> tuple:
    raw_keys = req.openai_keys or []
    if not raw_keys:
        raise HTTPException(400, "No OpenAI API keys provided. Get one at platform.openai.com")

    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid OpenAI keys found.")

    model        = req.model or "gpt-4o-mini"
    sys_p, usr_p = build_prompt(req, jd_chars=1600)

    # o-series reasoning models don't support system role or temperature
    is_o_model   = model.startswith("o")

    sorted_keys  = _prioritised_keys(valid_keys)
    last_error   = ""
    exhausted    = []

    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            try:
                if is_o_model:
                    messages = [{"role": "user", "content": sys_p + "\n\n" + usr_p}]
                    payload  = {"model": model, "messages": messages, "max_completion_tokens": 4096}
                else:
                    messages = [
                        {"role": "system", "content": sys_p},
                        {"role": "user",   "content": usr_p},
                    ]
                    payload  = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 4096}

                r = await client.post(
                    OPENAI_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload
                )

                if r.status_code == 200:
                    _key_usage[mk] = _key_usage.get(mk, 0) + 1
                    resp_json = r.json()
                    raw_text  = resp_json["choices"][0]["message"]["content"]
                    usage     = resp_json.get("usage", {})
                    _log_generation(
                        job_title     = req.job_title,
                        key_masked    = mk,
                        key_index     = i,
                        prompt_tokens = usage.get("prompt_tokens", est_tokens(sys_p + usr_p)),
                        model         = model,
                        success       = True,
                    )
                    _cv_raw  = sanitise_cv(extract_json(raw_text))
                    _cv_raw  = fix_companies(_cv_raw)
                    _cv_raw  = fix_skills(_cv_raw)
                    _techs_oa = {"core": _cv_raw.get("technologies",{}).get("mustHave",[]),
                                 "preferred": _cv_raw.get("technologies",{}).get("niceToHave",[]),
                                 "ecosystem": _cv_raw.get("technologies",{}).get("additional",[])}
                    _cv_raw  = _enforce_skill_domains(_cv_raw, _techs_oa, req.job_title)
                    _cv_raw  = _repair_project_tech_tags(_cv_raw, _techs_oa)
                    _cv_polished = final_polish(fix_skills_dedup(fix_projects(_cv_raw)), years_exp=(req.years_exp or ''))
                    cv = run_validation_pipeline(_cv_polished, req.job_description, req.job_title, _techs_oa)
                    return cv, mk, i

                elif r.status_code == 429:
                    exhausted.append(mk)
                    last_error = f"{mk} rate limited / quota exceeded"
                    continue

                elif r.status_code == 401:
                    exhausted.append(mk)
                    last_error = f"{mk} invalid API key"
                    continue

                elif r.status_code == 403:
                    exhausted.append(mk)
                    last_error = f"{mk} access denied - check your OpenAI plan"
                    continue

                elif r.status_code == 400:
                    try:   body = r.json()
                    except Exception: body = {}
                    err_msg = body.get("error", {}).get("message", f"HTTP 400")
                    if isinstance(err_msg, dict): err_msg = str(err_msg)
                    last_error = err_msg
                    exhausted.append(mk)
                    continue

                else:
                    try:   body = r.json()
                    except Exception: body = {}
                    last_error = body.get("error", {}).get("message", f"HTTP {r.status_code}")
                    if isinstance(last_error, dict): last_error = str(last_error)
                    exhausted.append(mk)
                    continue

            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(422, f"JSON parse error: {e}")
            except httpx.TimeoutException:
                last_error = f"{mk} timed out - model may be slow"
                exhausted.append(mk)
                continue
            except Exception as e:
                last_error = str(e)
                exhausted.append(mk)
                continue

    raise HTTPException(502, f"All OpenAI keys failed. Last error: {last_error}")




# -- Gemini caller (Google AI Studio - Free) -----------------------------------
async def call_gemini(req: CVRequest) -> tuple:
    """
    Calls Gemini using a 3-call atomic pipeline.
    KEY ROTATION: each of the 3 calls uses a DIFFERENT key from the pool,
    so 4 keys = 3 calls spread across 3 keys = no single key gets rate-limited.
    """
    import time as _time

    raw_keys = req.gemini_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Gemini API keys provided. Get one free at aistudio.google.com")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid Gemini keys found.")

    model = req.model or "gemini-2.0-flash"

    # -- Helper: single Gemini call with one Retry-After retry -----------------
    # Resolve actual model name - 2.5 has varied preview IDs; try and fall back
    _resolved_model = model
    _MODEL_FALLBACKS = {
        "gemini-2.5-flash-preview-05-20": ["gemini-2.5-flash-preview-05-20", "gemini-2.5-flash", "gemini-2.0-flash"],
        "gemini-2.5-flash":               ["gemini-2.5-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.0-flash"],
    }

    async def _gcall(client, key: str, system: str, user: str, max_tokens: int, stage: str) -> dict:
        nonlocal _resolved_model
        # Use higher max_tokens for gemini-2.5-flash (needs more output)
        actual_max_tokens = max_tokens
        if "2.5" in _resolved_model:
            actual_max_tokens = max(max_tokens, 3200)
        
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": actual_max_tokens},
        }
        # Try model variants if we get 404
        model_candidates = _MODEL_FALLBACKS.get(_resolved_model, [_resolved_model])
        for attempt in range(2):
            url = f"{GEMINI_URL}/{_resolved_model}:generateContent?key={key}"
            r = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)
            if r.status_code == 200:
                try:
                    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    raw = re.sub(r'```json\s*', '', raw)
                    raw = re.sub(r'```\s*$', '', raw)
                    s, e = raw.find('{'), raw.rfind('}')
                    if s != -1 and e != -1: raw = raw[s:e+1]
                    raw = re.sub(r',\s*}', '}', raw)
                    raw = re.sub(r',\s*]', ']', raw)
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        # -- Robust JSON repair: handle unterminated strings --
                        # The substring approach: extract only what can be parsed
                        i = len(raw)
                        while i > 0:
                            try:
                                # Try progressively shorter JSON until it parses
                                candidate = raw[:i]
                                # Close any open brackets
                                opens = candidate.count('{') - candidate.count('}')
                                aopens = candidate.count('[') - candidate.count(']')
                                fixed = candidate.rstrip(', \t\n')
                                # Remove unterminated string at end (e.g. "summary\":\"4 years of...)
                                fixed = re.sub(r',\s*"[^"]*$', '', fixed)
                                fixed = re.sub(r':\s*"[^"]*$', '', fixed)
                                fixed = re.sub(r'"\s*$', '', fixed)
                                fixed += ']' * max(0, aopens) + '}' * max(0, opens)
                                return json.loads(fixed)
                            except (json.JSONDecodeError, ValueError):
                                i -= 1
                                continue
                        # Last resort: return empty dict
                        print(f"Gemini {stage} - ALL JSON repair attempts failed for raw: {raw[:200]}")
                        return {}
                except Exception as ex:
                    raise ValueError(f"Parse error on {stage}: {ex}")
            elif r.status_code == 404:
                # Try next model variant
                tried_idx = model_candidates.index(_resolved_model) if _resolved_model in model_candidates else -1
                if tried_idx < len(model_candidates) - 1:
                    _resolved_model = model_candidates[tried_idx + 1]
                    print(f"Gemini 404 on {_resolved_model} - trying fallback")
                    continue
                raise ValueError(f"HTTP 404 on {stage} - model not available")
            elif r.status_code == 429:
                if attempt == 0:
                    wait = int(r.headers.get("Retry-After", r.headers.get("retry-after", 12)))
                    wait = min(wait, 20)
                    print(f"Gemini {stage} 429 - wait {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise ValueError(f"429 rate-limited on {stage} (both attempts)")
            elif r.status_code in (401, 403):
                try: msg = r.json()["error"]["message"]
                except Exception: msg = r.text[:150]
                raise ValueError(f"Invalid key ({r.status_code}) on {stage}: {msg}")
            else:
                raise ValueError(f"HTTP {r.status_code} on {stage}")
        raise ValueError(f"Rate-limited on {stage}")

    # -- Try each key as the "primary" key; rotate remaining keys for calls 2+3 -
    sorted_keys = _prioritised_keys(valid_keys)
    last_error  = ""
    exhausted   = []

    # Hard deadline: 110 seconds for Gemini (3 calls x ~25s each + overhead)
    import time as _time
    _deadline = _time.time() + 110  # 110s hard wall clock limit

    async with httpx.AsyncClient(timeout=115) as client:

        # If ALL keys are rate-limited, wait for the soonest one to become free
        # (up to 20s max) rather than failing immediately
        async def _maybe_wait_for_free_key():
            now = _time.time()
            if now >= _deadline:
                return
            limited = [_key_rate_limited_until.get(mask(k), 0) for k in valid_keys]
            soonest = min(limited)
            if soonest > now:
                wait = min(soonest - now + 1, 20, _deadline - now)
                if wait > 0:
                    print(f"All Gemini keys rate-limited - waiting {wait:.0f}s for cooldown")
                    await asyncio.sleep(wait)

        await _maybe_wait_for_free_key()

        for i, primary_key in enumerate(sorted_keys):
            if _time.time() >= _deadline:
                last_error = "2-minute timeout reached"
                break
            mk = mask(primary_key)
            try:
                years_exp      = (req.years_exp or "").strip()
                total_years    = _calc_total_years(years_exp)
                companies_list = _build_dynamic_companies(years_exp)
                edu            = _build_education_year(years_exp)
                num_cos        = len(companies_list)
                jd             = req.job_description.strip()[:1200]
                company_name   = (req.company_name or "").strip()
                company_ctx    = (req.company_context or "").strip()[:400]

                # Build a rotation pool: primary key first, then the rest round-robin
                # This means Call1->key[i], Call2->key[i+1], Call3->key[i+2]
                pool = sorted_keys[i:] + sorted_keys[:i]   # rotate starting at primary

                def _key_for_call(call_n: int) -> str:
                    # Skip exhausted keys when picking for call 2/3
                    for offset in range(len(pool)):
                        k = pool[(call_n + offset) % len(pool)]
                        if mask(k) not in exhausted:
                            return k
                    return primary_key  # fallback

                if num_cos == 1:
                    sen_rules  = "Co1 (only): Junior prefix"
                    verb_guide = "Co1: Implemented, Built, Developed, Deployed."
                elif num_cos == 2:
                    sen_rules  = "Co1 (current): no prefix. Co2 (oldest): Junior prefix"
                    verb_guide = "Co1: Developed, Engineered, Integrated.\nCo2: Implemented, Configured, Deployed."
                else:
                    sen_rules  = "Co1 (current): Senior prefix. Co2 (mid): no prefix. Co3 (oldest): Junior prefix"
                    verb_guide = "Co1: Architected, Engineered, Led.\nCo2: Optimised, Refactored, Scaled.\nCo3: Implemented, Built, Deployed."

                # -- CALL 1 via primary key -------------------------------------
                sys1 = (
                    "You are an expert CV writer and tech extractor. Output ONLY valid JSON. No markdown, no backticks.\n\n"
                    "Do ALL of these tasks from the job description below:\n"
                    "1. TECH: Extract every named technology into core/preferred/ecosystem arrays (aim 30+ total items).\n"
                    "2. TITLE: A transformed role title. Format: \'Transformed Title | Tech1, Tech2, Tech3\'\n"
                    f"3. SUMMARY: 4 sentences, 70+ words, start with \'{total_years} years of experience...\'\n"
                    "4. COMPETENCIES: exactly 10 domain-specific 2-4 word phrases separated by \' * \'"
                )
                usr1 = f"""Job Title: {req.job_title}
{"Target Company: " + company_name if company_name else ""}
Experience: {total_years} years

Job Description:
{jd}

Output JSON:
{{"core":["tech1"],"preferred":["tech1"],"ecosystem":["companion1"],
"title":"Transformed Title | Tech1, Tech2, Tech3",
"summary":"{total_years} years of experience in [domain]... (4 sentences, 70+ words)",
"competencies":"Phrase1 * Phrase2 * Phrase3 * Phrase4 * Phrase5 * Phrase6 * Phrase7 * Phrase8 * Phrase9 * Phrase10"}}"""

                result1 = await _gcall(client, primary_key, sys1, usr1, 1200, "Call1")

                techs = _sanitise_techs({
                    "core":      result1.get("core", []),
                    "preferred": result1.get("preferred", []),
                    "ecosystem": result1.get("ecosystem", []),
                })
                if not techs.get("core") or len(techs["core"]) < 3:
                    found  = re.findall(r'\b([A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z]+)?(?:js|JS|TS|ts)?)\b', req.job_description)
                    common = {"The","And","For","With","From","Have","Will","Are","Can","You","Your","Our","This","That"}
                    found  = [f for f in found if len(f) > 2 and f not in common][:15]
                    techs  = {"core": list(dict.fromkeys(found)), "preferred": [], "ecosystem": []}

                all_techs_flat   = list(dict.fromkeys(techs["core"] + techs["preferred"] + techs["ecosystem"]))[:40]
                techs_str        = ", ".join(all_techs_flat) if all_techs_flat else "technologies from the JD"
                title_out        = result1.get("title", "")
                summary_out      = result1.get("summary", "")
                competencies_out = result1.get("competencies", "")
                if not title_out:
                    core      = techs.get("core", ["Technology"])
                    title_out = f"{req.job_title.split('|')[0].strip()} | {', '.join((core + ['Technology']*3)[:3])}"
                if not summary_out:
                    summary_out = f"{total_years} years of experience in software development with expertise in {techs_str[:100]}."
                if not competencies_out:
                    competencies_out = "Digital Marketing * SEO Strategy * Content Development * Analytics & Reporting * Campaign Management * E-Commerce Growth * Brand Positioning * Social Media * Conversion Optimisation * Performance Marketing"

                # -- CALL 2 via next available key -----------------------------
                key2 = _key_for_call(1)
                co_lines = "\n".join(
                    f'Co{j+1}: name="{c["name"]}", dates="{c["start"]} - {c["end"]}"'
                    for j, c in enumerate(companies_list)
                )
                sys2 = (
                    "You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
                    f"Allowed technologies (use ONLY these): {techs_str}\n\n"
                    f"TASK — EXPERIENCE: exactly {num_cos} companies.\n"
                    f"Seniority: {sen_rules}\n"
                    f"Verb guide:\n{verb_guide}\n"
                    "Each company: unique role title, 4 bullets (20-30 words each), 6 tech tags from allowed list.\n"
                    "BANNED: tools outside the allowed list, <6 tech tags, repeated verbs across bullets."
                )
                usr2 = f"""Job Title: {req.job_title}
Experience: {total_years} years
Companies (use exact names and dates):
{co_lines}
Job Description context:
{jd[:600]}
Output JSON:
{{"companies":[{{"company":"EXACT NAME","role":"Seniority Domain Function","dateRange":"Start - End","bullets":["20-30w bullet","20-30w bullet","20-30w bullet","20-30w bullet"],"tech":"Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6"}}]}}"""

                result2 = await _gcall(client, key2, sys2, usr2, 2400, "Call2")
                _key_usage[mask(key2)] = _key_usage.get(mask(key2), 0) + 1

                companies_out = result2.get("companies", [])
                for j, co in enumerate(companies_out):
                    if j < len(companies_list):
                        co["company"]   = companies_list[j]["name"]
                        co["dateRange"] = f'{companies_list[j]["start"]} - {companies_list[j]["end"]}'
                if not companies_out:
                    companies_out = [
                        {"company": c["name"], "role": "Software Developer",
                         "dateRange": f'{c["start"]} - {c["end"]}',
                         "bullets": ["Developed scalable solutions using " + (techs.get("core", ["the primary stack"])[0] if techs.get("core") else "the primary stack"),
                                     "Implemented features improving efficiency by 30%",
                                     "Collaborated with cross-functional teams on key deliverables",
                                     "Maintained high code quality and system reliability"],
                         "tech": " | ".join(all_techs_flat[:6])}
                        for c in companies_list
                    ]

                # -- CALL 3 via next available key -----------------------------
                key3 = _key_for_call(2)
                used_systems = []
                for co in companies_out:
                    for b in (co.get("bullets") or []):
                        if len(b) > 20: used_systems.append(b[:60])
                used_str = "\n".join(f"  - {s}" for s in used_systems[:4]) or "  (none)"
                co_intel = ""
                if company_ctx and company_name:
                    co_intel = f"Target Company: {company_name}\nContext: {company_ctx}"
                elif company_name:
                    co_intel = f"Target Company: {company_name}"

                sys3 = (
                    "You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
                    f"Allowed technologies: {techs_str}\n\n"
                    "TASK A - PROJECTS: exactly 4 projects. Each:\n"
                    "  name: \'CoinedWord - what it does [Tech1, Tech2]\'\n"
                    "  overview: 3-4 natural sentences (problem -> solution -> functionality -> impact).\n"
                    "  bullets: 3 strings (20-30 words, different verb, unique metric)\n"
                    "  techTags: 5-6 from allowed list, none repeated across projects\n\n"
                    "TASK B - RELATED TECH: exactly 5 category boxes, 5 items each, all from allowed list."
                )
                usr3 = f"""Job Title: {req.job_title}
{co_intel}
Allowed technologies: {techs_str}
Systems already in experience (do NOT repeat):
{used_str}
Output JSON:
{{"projects":[{{"name":"CoinedName - what it does [Tech1, Tech2]","overview":"3-4 sentence story.","bullets":["verb + metric (20-30w)","verb + metric (20-30w)","verb + metric (20-30w)"],"techTags":["Tech1","Tech2","Tech3","Tech4","Tech5"]}}],"relatedTech":[{{"category":"Domain","items":["t1","t2","t3","t4","t5"]}}]}}"""

                result3 = await _gcall(client, key3, sys3, usr3, 2800, "Call3")
                _key_usage[mask(key3)] = _key_usage.get(mask(key3), 0) + 1

                projects_out = result3.get("projects", [])
                related_out  = result3.get("relatedTech", [])
                projects_out = validate_project_techs({"projects": projects_out}, techs).get("projects", [])

                # -- Assemble partial CV (without skills yet) -------------------
                cv_partial = {
                    "totalYears":   total_years,
                    "title":        title_out,
                    "summary":      summary_out,
                    "skills":       [],  # will be filled by dedicated skills call below
                    "competencies": competencies_out,
                    "companies":    companies_out,
                    "projects":     projects_out,
                    "relatedTech":  related_out,
                    "technologies": {
                        "mustHave":   techs.get("core", [])[:12],
                        "niceToHave": techs.get("preferred", [])[:10],
                        "additional": techs.get("ecosystem", [])[:10],
                    },
                    "education": {
                        "university":  "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
                        "degree":      "Bachelor of Science in Computer Science (BSCS)",
                        "cgpa":        "3.97/4.0",
                        "years":       f"{edu['start']} - {edu['end']}",
                        "achievement": "Gold Medalist for Academic Excellence",
                    },
                    "keywords":      ", ".join(all_techs_flat[:15]),
                    "architectures": [],
                    "_techs":        techs,
                    "_job_title":    req.job_title,
                }

                # -- CALL 4 (DEDICATED SKILLS) via next available key ----------
                # This is the second dedicated request for Technical Skills.
                # It sees the full tech list + companies already generated,
                # guaranteeing the skills section is never dropped or corrupted.
                print(f"[GEMINI] Starting DEDICATED SKILLS extraction (Call4)")
                key4 = _key_for_call(3)
                
                # Build dedicated Gemini URL with key4
                skills_gemini_url = f"{GEMINI_URL}/{model}:generateContent?key={key4}"
                dedicated_skills = await extract_dedicated_skills(
                    client       = client,
                    key          = key4,
                    model        = model,
                    url          = skills_gemini_url,
                    headers      = {},  # Gemini uses key in URL
                    req          = req,
                    cv           = cv_partial,
                    techs        = techs,
                    max_tokens   = 2000,
                    provider     = "gemini"
                )
                _key_usage[mask(key4)] = _key_usage.get(mask(key4), 0) + 1
                
                print(f"[GEMINI] Dedicated skills returned {len(dedicated_skills)} categories")
                for i2, s in enumerate(dedicated_skills):
                    colon = s.find(":")
                    cat   = s[:colon].strip() if colon > 0 else "?"
                    items = [t.strip() for t in s[colon+1:].split(",")] if colon > 0 else []
                    print(f"[GEMINI]   [{i2+1}] {cat}: {len(items)} items")
                
                cv_partial["skills"] = dedicated_skills

                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _key_rate_limited_until.pop(mk, None)
                _log_generation(
                    job_title     = req.job_title,
                    key_masked    = mk,
                    key_index     = i,
                    prompt_tokens = est_tokens(jd),
                    model         = model,
                    success       = True,
                )

                print(f"[GEMINI] Starting post-processing pipeline")
                cv_s = sanitise_cv(cv_partial)
                cv_s = fix_companies(cv_s)
                cv_s = fix_skills(cv_s)
                print(f"[GEMINI] After fix_skills: {len(cv_s.get('skills', []))} categories")
                _tg  = {"core":      cv_s.get("technologies", {}).get("mustHave",   []),
                        "preferred": cv_s.get("technologies", {}).get("niceToHave", []),
                        "ecosystem": cv_s.get("technologies", {}).get("additional", [])}
                cv_s = _enforce_skill_domains(cv_s, _tg, req.job_title)
                cv_s = _repair_project_tech_tags(cv_s, _tg)
                _cv_polished = final_polish(fix_skills_dedup(fix_projects(cv_s)), years_exp=(req.years_exp or ""))
                print(f"[GEMINI] Final skills: {len(_cv_polished.get('skills', []))} categories")
                for i2, s in enumerate(_cv_polished.get("skills", [])):
                    colon = s.find(":")
                    cat   = s[:colon].strip() if colon > 0 else "?"
                    items = [t.strip() for t in s[colon+1:].split(",")] if colon > 0 else []
                    print(f"[GEMINI]   [{i2+1}] {cat}: {len(items)} items → {items[:4]}")
                return run_validation_pipeline(_cv_polished, req.job_description, req.job_title, _tg), mk, i

            except ValueError as e:
                err_str = str(e)
                if "429" in err_str or "rate-limited" in err_str.lower() or "rate limited" in err_str.lower():
                    wait = 12
                    _key_rate_limited_until[mk] = _time.time() + wait + 2
                    last_error = f"{mk} rate limited - {err_str[:100]}"
                elif "401" in err_str or "403" in err_str or "Invalid key" in err_str:
                    last_error = f"{mk} invalid key - {err_str[:100]}"
                else:
                    last_error = f"{mk}: {err_str[:140]}"
                exhausted.append(mk)
                continue
            except httpx.TimeoutException:
                last_error = f"{mk} timed out"
                exhausted.append(mk)
                continue
            except Exception as e:
                last_error = str(e)[:200]
                exhausted.append(mk)
                continue

    if "2-minute timeout" in last_error:
        raise HTTPException(504, "Gemini timed out after 3.5 minutes. Try again - keys may need a moment to reset.")
    n_limited = sum(1 for e in exhausted if "rate" in e.lower())
    n_keys    = len(sorted_keys)
    if n_limited >= n_keys:
        cooldowns = [_key_rate_limited_until.get(mask(k), 0) for k in valid_keys]
        soonest   = min(cooldowns) if cooldowns else 0
        wait_sec  = max(0, int(soonest - _time.time()))
        wait_msg  = f" Retry in ~{wait_sec}s." if wait_sec > 0 else " Try again in ~60s."
        raise HTTPException(
            429,
            f"All {n_keys} Gemini key(s) are rate-limited (15 req/min free tier).{wait_msg}\n"
            f"Add more keys at aistudio.google.com"
        )
    raise HTTPException(502, f"All Gemini keys failed. Last error: {last_error}")


@app.post("/generate-cv")
async def generate_cv(req: CVRequest):
    _reset_infer_category_name()  # clear used-label cache for each new CV

    async def _run():
        if req.provider == "cerebras":
            cv_data, key_used, key_idx = await call_cerebras(req)
            return {"cv": cv_data, "provider": "cerebras", "model": req.model,
                    "key_used": key_used, "key_index": key_idx}
        elif req.provider == "groq":
            cv_data, key_used, key_idx = await call_groq(req)
            return {"cv": cv_data, "provider": "groq", "model": req.model,
                    "key_used": key_used, "key_index": key_idx}
        elif req.provider == "deepseek":
            cv_data, key_used, key_idx = await call_deepseek(req)
            return {"cv": cv_data, "provider": "deepseek", "model": req.model,
                    "key_used": key_used, "key_index": key_idx}
        elif req.provider == "openai":
            cv_data, key_used, key_idx = await call_openai(req)
            return {"cv": cv_data, "provider": "openai", "model": req.model,
                    "key_used": key_used, "key_index": key_idx}
        elif req.provider == "gemini":
            cv_data, key_used, key_idx = await call_gemini(req)
            return {"cv": cv_data, "provider": "gemini", "model": req.model,
                    "key_used": key_used, "key_index": key_idx}
        else:
            cv_data = await call_ollama(req)
            return {"cv": cv_data, "provider": "ollama", "model": req.ollama_model}

    try:
        # Hard 300-second server-side wall-clock limit — client gets a clean error
        # if any provider hangs, instead of waiting forever.
        return await asyncio.wait_for(_run(), timeout=300)

    except asyncio.TimeoutError:
        raise HTTPException(
            504,
            "CV generation timed out (> 5 minutes). "
            "The AI provider is overloaded or your key is rate-limited. "
            "Try again, add more keys, or switch to a different provider."
        )
    except HTTPException:
        raise
    except httpx.ConnectError as e:
        if "11434" in str(e):
            raise HTTPException(503, "Cannot connect to Ollama. Make sure it is running.")
        raise HTTPException(503, f"Connection error: {e}")
    except Exception as e:
        raise HTTPException(500, str(e))


# ================================================================================
#  /generate-pdf  - takes CV JSON, returns a true text-based PDF (ATS-readable)
# ================================================================================

STATIC_CANDIDATE = {
    "name":        "MUHAMMAD JUNAID",
    "address":     "G-8 Markaz, Islamabad, Pakistan",
    "open_to":     "Open To Remote Opportunities",
    "email":       "muhammad.junaid.software@gmail.com",
    "phone":       "+92 308 2550767",
    "github":      "github.com/muhammad-junaid-code",
    "portfolio":   "muhammad-junaid-portfolio-view.netlify.app",
    "linkedin":    "linkedin.com/in/muhammad-junaid-6986b5305",
    "university":  "QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY",
    "degree":      "Bachelor of Science in Computer Science (BSCS)",
    "gpa":         "CGPA: 3.97 / 4.0",
    "grad_year":   "2017 - 2021",
    "achievement": "Gold Medalist for Academic Excellence",
    "companies":   ["MULTYLOGICS SOLUTIONS", "ENCS NETWORKS", "NOW TECHNOLOGIES (NOW.NET.PK)"],
    "dates":       ["May 2024 - Present", "May 2022 - May 2024", "May 2020 - May 2022"],
}

# Full clickable URLs for PDF links
_CONTACT_LINKS = {
    "email":     "mailto:muhammad.junaid.software@gmail.com",
    "phone":     "tel:+923082550767",
    "github":    "https://github.com/muhammad-junaid-code",
    "portfolio": "https://muhammad-junaid-portfolio-view.netlify.app",
    "linkedin":  "https://linkedin.com/in/muhammad-junaid-6986b5305",
}


def _safe(val, fallback=""):
    if val is None:
        return fallback
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val).strip()


def _strip_brackets(name: str) -> str:
    """Remove [Tech, Tech] suffix from project names."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip()


# In the PDF generation section (build_cv_pdf function),
# replace the style definitions with these compact versions:

def _detect_job_domain(cv: dict) -> str:
    """
    Detect the job domain from CV title, summary, and skills.
    Returns one of: 'seo', 'data_science', 'mobile', 'devops', 'security',
                    'design', 'project_management', 'finance', 'software_dev'
    Defaults to 'software_dev' if domain cannot be determined.

    IMPORTANT: software_dev signals take priority - a .NET or Angular developer CV
    that happens to mention Terraform in a misplaced skills row must NOT be classified
    as 'devops'. Only classify as 'devops' if the title/role explicitly calls it out.
    """
    title = str(cv.get("title", "")).lower()
    summary = str(cv.get("summary", "")).lower()
    skills_text = " ".join(str(s) for s in (cv.get("skills") or [])).lower()
    signals = f"{title} {summary} {skills_text}"

    # -- Hard software_dev signals in TITLE - these always win ----------------
    _SOFTDEV_TITLE_KW = (
        ".net", "c#", "asp.net", "angular", "react", "vue", "node.js",
        "python developer", "django", "flask", "fastapi", "java developer",
        "spring boot", "php developer", "laravel", "ruby on rails",
        "full stack", "fullstack", "frontend developer", "backend developer",
        "software engineer", "software developer",
    )
    if any(kw in title for kw in _SOFTDEV_TITLE_KW):
        return "software_dev"

    if any(w in signals for w in [
        "seo", "search engine optim", "keyword research", "semrush", "ahrefs",
        "google analytics", "google search console", "moz", "backlink",
        "organic traffic", "link building", "on-page", "serp", "content marketing",
        "digital marketing", "ppc", "google ads", "social media marketing",
        "google tag manager", "keyword planner", "screaming frog",
    ]):
        return "seo"

    if any(w in signals for w in [
        "machine learning", "deep learning", "tensorflow", "pytorch", "pandas",
        "numpy", "scikit", "jupyter", "data science", "bigquery", "spark",
        "airflow", "mlops", "nlp", "computer vision", "etl", "dbt",
    ]):
        return "data_science"

    if any(w in signals for w in [
        "react native", "flutter", "swift", "swiftui", "kotlin", "android",
        "ios", "mobile app", "xcode", "android studio", "fastlane", "testflight",
    ]):
        return "mobile"

    # devops only if TITLE explicitly says devops/sre/platform - not just because
    # skills rows mention Docker or Terraform (those get correctly bucketed anyway)
    if any(w in title for w in [
        "devops", "site reliability", "sre", "infrastructure engineer",
        "platform engineer", "cloud engineer", "devsecops",
    ]):
        return "devops"

    if any(w in signals for w in [
        "cybersecurity", "penetration test", "ethical hack", "soc analyst",
        "information security", "owasp", "firewalls", "ids", "ips",
        "vulnerability", "incident response", "siem", "splunk",
    ]):
        return "security"

    if any(w in signals for w in [
        "ux", "ui design", "figma", "sketch", "adobe xd", "product design",
        "user experience", "wireframe", "prototyping", "illustrator",
    ]):
        return "design"

    if any(w in signals for w in [
        "project manager", "scrum master", "product manager", "product owner",
        "agile coach", "program manager", "delivery manager", "jira", "confluence",
        "roadmap", "stakeholder",
    ]) and "developer" not in signals and "engineer" not in signals:
        return "project_management"

    if any(w in signals for w in [
        "financial analyst", "accountant", "cfa", "quickbooks", "sap finance",
        "bloomberg", "excel financial", "financial model", "ifrs", "gaap",
    ]):
        return "finance"

    return "software_dev"


def _get_domain_fallback_pools(domain: str, techs: list = None, job_title: str = "", jd_text: str = "") -> list:
    """
    Returns 5 fallback tech pools — one per skill category slot.
    When techs + job_title are provided the AI derives the pools dynamically.
    The compact static map is only used when the AI call fails.
    """
    import json as _json

    if techs and job_title:
        tech_str = ", ".join(t for t in techs if _is_real_tech(t))
        jd_hint  = f" JD context: {jd_text[:300]}." if jd_text else ""
        prompt = (
            f"Job title: {job_title}.{jd_hint}\n"
            f"JD technologies: {tech_str}\n\n"
            "Return EXACTLY a JSON array of 5 arrays. Each inner array lists real named "
            "technologies for one skill domain of this specific job. "
            "Order by this job's actual domains (e.g. DevOps: IaC, CI/CD, Container, "
            "Monitoring, Cloud). Use only the JD technologies above plus their direct "
            "ecosystem companions. No markdown, no extra text. "
            "Example: [[\"Terraform\",\"Ansible\"],[\"Jenkins\",\"ArgoCD\"]]"
        )
        try:
            import urllib.request as _ur
            payload = _json.dumps({
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500,
            }).encode()
            req = _ur.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {GROQ_API_KEY}"},
                method="POST",
            )
            with _ur.urlopen(req, timeout=12) as resp:
                body = _json.loads(resp.read())
            raw = body["choices"][0]["message"]["content"].strip()
            raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
            pools = _json.loads(raw)
            if isinstance(pools, list):
                valid = [p for p in pools
                         if isinstance(p, list) and len(p) >= 3
                         and all(isinstance(x, str) for x in p)]
                if len(valid) >= 3:
                    while len(valid) < 5:
                        valid.append(valid[-1])
                    return valid[:5]
        except Exception:
            pass

    # Compact static fallback — only reached when AI call fails
    _STATIC = {
        "devops": [
            ["Terraform","Ansible","Pulumi","CloudFormation","Chef","Puppet","Packer","Vault","Consul","SaltStack"],
            ["Jenkins","GitHub Actions","GitLab CI","CircleCI","ArgoCD","Azure Pipelines","TeamCity","Spinnaker","Tekton"],
            ["Docker","Kubernetes","Helm","Istio","Envoy","containerd","Podman","Rancher","OpenShift","Docker Compose"],
            ["Prometheus","Grafana","Datadog","ELK Stack","Splunk","Jaeger","PagerDuty","New Relic","Dynatrace","CloudWatch","Loki"],
            ["AWS","Azure","GCP","DigitalOcean","Cloudflare","Nginx","HAProxy","Bash","Python","Git","Linux"],
        ],
        "software_dev": [
            ["React","Angular","Vue.js","Next.js","TypeScript","Tailwind CSS","SCSS","Webpack","Vite","Redux"],
            ["Node.js","Express.js","Django","FastAPI","Spring Boot","GraphQL","RabbitMQ","JWT","OAuth 2.0","gRPC"],
            ["PostgreSQL","MySQL","MongoDB","Redis","SQL Server","Elasticsearch","DynamoDB","Cosmos DB","Prisma","Flyway"],
            ["Docker","Kubernetes","AWS EC2","AWS S3","Azure App Service","GitHub Actions","Terraform","Nginx","Helm"],
            ["xUnit","Pytest","Selenium","Playwright","Cypress","Postman","SonarQube","OWASP ZAP","Snyk","Sentry"],
        ],
        "data_science": [
            ["Python","R","SQL","Scala","PySpark","Bash","Jupyter Notebook"],
            ["TensorFlow","PyTorch","Scikit-learn","Keras","XGBoost","Hugging Face Transformers","Pandas","NumPy"],
            ["Apache Spark","Apache Kafka","Apache Airflow","dbt","BigQuery","Snowflake","Databricks"],
            ["Vertex AI","AWS SageMaker","Azure ML","MLflow","Kubeflow","Docker","Kubernetes","DVC"],
            ["Tableau","Power BI","Looker","Matplotlib","Seaborn","Plotly","Grafana","Streamlit"],
        ],
        "security": [
            ["Metasploit","Burp Suite","Nmap","Wireshark","Nessus","OWASP ZAP","sqlmap"],
            ["Palo Alto","Fortinet FortiGate","Cisco ASA","pfSense","Snort","CrowdStrike","SentinelOne"],
            ["Splunk","IBM QRadar","Microsoft Sentinel","LogRhythm","Elastic SIEM","ArcSight"],
            ["ISO 27001","NIST CSF","SOC 2","GDPR","PCI DSS","HIPAA","CIS Benchmarks","MITRE ATT&CK"],
            ["Python","Bash","PowerShell","Go","Yara","Ansible","Docker","Kali Linux"],
        ],
        "mobile": [
            ["React Native","Flutter","Ionic","Expo","Xamarin","Kotlin Multiplatform"],
            ["Swift","SwiftUI","UIKit","Xcode","Core Data","Combine","CocoaPods"],
            ["Kotlin","Java","Jetpack Compose","Android Studio","Gradle","Room Database","Retrofit"],
            ["Firebase","AWS Amplify","Supabase","Node.js","GraphQL"],
            ["Fastlane","Bitrise","GitHub Actions","Detox","Appium","XCTest","Espresso","Crashlytics"],
        ],
    }
    return _STATIC.get(domain, _STATIC["software_dev"])


def _sanitize_skills(raw_skills: list, cv: dict = None) -> list:
    """
    Fully dynamic skills sanitiser — preserves the AI's category names and groupings exactly.

    What this function does:
      1. Parse every "Category: item1, item2, ..." string from AI output.
      2. Strip items that are provably NOT real technology names (verbs, generic nouns, HR words).
      3. Global dedup — if an item appears in multiple categories, keep it only in the first.
      4. Drop categories that end up with fewer than 3 real items after stripping.
      5. If fewer than 5 valid categories remain, backfill using domain-appropriate fallback pools
         BUT only with tools consistent with the detected tech stack.
      6. Cap at exactly 5 categories.

    What this function does NOT do:
      - It never renames AI category names (e.g. "Infrastructure as Code" stays as-is).
      - It never moves items between categories.
      - It never reassigns items to hardcoded bucket positions.
      - It never adds non-JD technologies from training data.

    cv (optional): the CV dict — used to detect job domain for fallback pool selection only.
    """
    # -- Step 1: Parse AI skill categories, strip only provably fake items ----
    BANNED_TERMS = {
        "web development", "web", "hands", "good", "ability", "strong", "dev",
        "remote", "setup", "mindset", "detail", "attention", "focus", "solid",
        "working", "independent", "effectively", "efficiently",
        "restful", "rest", "web apis", "web api",
        "relational databases", "relational database",
        "clean architecture", "solid principles",
        "agile", "scrum", "kanban", "tdd", "bdd", "ddd",
        "architecture", "infrastructure", "environment", "system",
        "server", "network", "platform", "application", "solution",
        "module", "component", "interface", "integration", "requirements",
        "design", "development", "backend", "frontend", "full stack",
        ".net", "net",
        "claude code", "cursor", "copilot", "github copilot", "anthropic",
        "chatgpt", "openai",
    }

    # -- Step 2: Parse every AI category, filter fake items, global dedup ------
    parsed: list = []  # [(cat_name, [real_items])]
    seen_global: set = set()

    for entry in raw_skills:
        if not isinstance(entry, str):
            continue
        colon = entry.find(":")
        if colon <= 0:
            continue
        cat_name = entry[:colon].strip()
        items_raw = entry[colon + 1:].strip()
        items = [i.strip() for i in re.split(r"[,|;?·•]", items_raw) if i.strip()]

        real_items = []
        for it in items:
            key = it.lower().strip()
            if key in BANNED_TERMS:
                continue
            if not _is_real_tech(it):
                continue
            if key in seen_global:
                continue
            seen_global.add(key)
            real_items.append(it)

        if real_items:
            parsed.append((cat_name, real_items))

    if not parsed:
        return raw_skills  # nothing to fix

    # -- Step 3: Drop categories with fewer than 3 real items, keep first 5 ---
    valid = [(cat, items) for cat, items in parsed if len(items) >= 3]

    # -- Step 4: If AI gave us >=5 good categories, output them AS-IS ----------
    # This is the primary path — trust the AI for DevOps, SEO, HR, etc.
    if len(valid) >= 5:
        result = []
        for cat_name, items in valid[:5]:
            result.append(f"{cat_name}: {', '.join(items)}")
        return result

    # -- Step 5: Fewer than 5 valid categories — need fallback -----------------
    # Detect domain for fallback pool selection
    domain     = _detect_job_domain(cv) if cv else "software_dev"
    job_title  = (cv or {}).get("title", "") or (cv or {}).get("_job_title", "")
    jd_text    = (cv or {}).get("_jd_text", "")
    all_jd_techs = list(seen_global)  # all real tools seen across parsed categories

    # AI-powered fallback pools — passes JD context so AI picks the right tools & order
    fallback_pools = _get_domain_fallback_pools(domain, all_jd_techs, job_title, jd_text)

    # Output valid AI categories first; fill gaps with AI-derived pool rows
    seen_fb: set = set(seen_global)
    result = [f"{cat}: {', '.join(items)}" for cat, items in valid]

    needed = 5 - len(valid)
    if needed > 0:
        import json as _json

        # Build pool items for each missing slot
        pool_groups = []
        for i in range(needed):
            pool = fallback_pools[i] if i < len(fallback_pools) else []
            items = [t for t in pool if t.lower() not in seen_fb and _is_real_tech(t)][:10]
            pool_groups.append(items)
            for t in items:
                seen_fb.add(t.lower())

        # Ask AI to name each group according to the JD
        if job_title and any(pool_groups):
            groups_str = _json.dumps(pool_groups)
            prompt = (
                f"Job title: {job_title}.\n"
                f"Name each of these technology groups with a domain heading that matches "
                f"this job. Return ONLY a JSON array of strings, one name per group.\n"
                f"Groups: {groups_str}"
            )
            try:
                import urllib.request as _ur
                payload = _json.dumps({
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1, "max_tokens": 200,
                }).encode()
                req_obj = _ur.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {GROQ_API_KEY}"},
                    method="POST",
                )
                with _ur.urlopen(req_obj, timeout=10) as resp:
                    body = _json.loads(resp.read())
                raw = body["choices"][0]["message"]["content"].strip()
                raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
                names = _json.loads(raw)
                if isinstance(names, list):
                    for name, items in zip(names, pool_groups):
                        if items and isinstance(name, str) and name.strip():
                            result.append(f"{name.strip()}: {', '.join(items)}")
                    return result if result else raw_skills
            except Exception:
                pass

        # If AI naming fails, derive a domain label from the items themselves
        for i, items in enumerate(pool_groups):
            if items:
                label = _infer_category_name(items, job_title, len(valid) + i)
                result.append(f"{label}: {', '.join(items)}")

    return result if result else raw_skills


def build_cv_pdf(cv: dict, profile_data: dict = None) -> bytes:
    """Build a 100% text-based PDF from CV JSON + dynamic profileData. Returns raw PDF bytes."""

    from reportlab.platypus import KeepTogether

    # ── Resolve profile fields — prefer dynamic profileData, fall back to STATIC ──
    _pd        = profile_data or {}
    p_name     = (_pd.get("name")  or "").strip() or STATIC_CANDIDATE["name"]
    p_links    = _pd.get("links")  or []   # [{icon, label, value}]
    p_work     = _pd.get("work")   or []   # [{company, role, from, to, bullets}]
    p_edu      = _pd.get("edu")    or []   # [{institution, degree, from, to, note}]

    buf = io.BytesIO()

    PAGE_W, _ = A4
    ML = 13 * mm
    MR = 13 * mm
    MT = 11 * mm
    MB = 11 * mm

    # Build with a very tall page first to measure actual content height
    PAGE_H_SINGLE = 841.89 * 2.2
    doc = SimpleDocTemplate(
        buf,
        pagesize=(PAGE_W, PAGE_H_SINGLE),
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
        title=f"{p_name} CV - {_safe(cv.get('title',''))}",
        author=p_name,
        subject=_safe(cv.get("title", "")),
        keywords=_safe(cv.get("keywords", "")),
    )

    TW = PAGE_W - ML - MR

    # -- STYLES WITH INCREASED FONT SIZE AND SPACING -----------------------------
    def ps(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=10, leading=14,
                        spaceAfter=0, spaceBefore=0, textColor=colors.HexColor("#111111"))
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    S = {
        "name":        ps("name",   fontName="Helvetica-Bold", fontSize=18, leading=24,
                           textColor=colors.HexColor("#111111"), spaceAfter=3,
                           alignment=TA_CENTER),
        "role":        ps("role",   fontName="Helvetica", fontSize=8, leading=12,
                           textColor=colors.HexColor("#444444"), spaceAfter=1,
                           alignment=TA_CENTER),
        "contact":     ps("contact",fontName="Helvetica", fontSize=8, leading=11,
                           textColor=colors.HexColor("#0057A8"), spaceAfter=1,
                           alignment=TA_CENTER),
        "contact_plain": ps("cp",   fontName="Helvetica", fontSize=8, leading=11,
                           textColor=colors.HexColor("#555555"), spaceAfter=1,
                           alignment=TA_CENTER),
        "sec_title":   ps("sec",    fontName="Helvetica-Bold", fontSize=11, leading=14,
                           textColor=colors.HexColor("#222222"), spaceBefore=4, spaceAfter=2),
        "company":     ps("co",     fontName="Helvetica-Bold", fontSize=11, leading=14,
                           textColor=colors.HexColor("#111111"), spaceAfter=2),
        "role_title":  ps("rt",     fontName="Helvetica-Oblique", fontSize=10, leading=13,
                           textColor=colors.HexColor("#555555"), spaceAfter=2),
        "bullet":      ps("bul",    fontName="Helvetica", fontSize=9.5, leading=13,
                           leftIndent=12, firstLineIndent=0, textColor=colors.HexColor("#222222"),
                           spaceAfter=2),
        "tech_line":   ps("tech",   fontName="Helvetica", fontSize=8.5, leading=11,
                           leftIndent=12, textColor=colors.HexColor("#666666"), spaceAfter=3),
        "skill_items": ps("sitm",   fontName="Helvetica", fontSize=9, leading=12,
                           textColor=colors.HexColor("#333333"), spaceAfter=1),
        "proj_type":   ps("pt",     fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                           textColor=colors.HexColor("#ffffff"),
                           backColor=colors.HexColor("#1a5fa8"),
                           spaceBefore=4, spaceAfter=2, leftIndent=0, borderPadding=(1,4,1,4)),
        "proj_name":   ps("pn",     fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                           textColor=colors.HexColor("#111111"), spaceAfter=1),
        "proj_body":   ps("pb",     fontName="Helvetica", fontSize=9.5, leading=13,
                           textColor=colors.HexColor("#333333"), spaceAfter=1),
        "proj_bullet": ps("pbul",   fontName="Helvetica", fontSize=9.5, leading=12.5,
                           leftIndent=12, textColor=colors.HexColor("#333333"), spaceAfter=2),
        "proj_stack":  ps("pst",    fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                           textColor=colors.HexColor("#555555"), spaceAfter=2),
        "competency":  ps("comp",   fontName="Helvetica", fontSize=9.5, leading=13,
                           textColor=colors.HexColor("#333333"), spaceAfter=1),
        "edu_uni":     ps("uni",    fontName="Helvetica-Bold", fontSize=11, leading=14,
                           textColor=colors.HexColor("#111111"), spaceAfter=1),
        "edu_deg":     ps("deg",    fontName="Helvetica", fontSize=10, leading=13,
                           textColor=colors.HexColor("#444444"), spaceAfter=2),
        "edu_medal":   ps("med",    fontName="Helvetica-Bold", fontSize=10, leading=13,
                           textColor=colors.HexColor("#166534"), spaceAfter=1),
    }

    def HR():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"),
                          spaceAfter=3, spaceBefore=1)

    def BOLD_HR():
        return HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#111111"),
                          spaceAfter=2, spaceBefore=1)

    def _link(text: str, href: str) -> str:
        return f'<a href="{href}" color="#0057A8">{text}</a>'

    story = []

            # -- HEADER ------------------------------------------------------------------
    total_years = _safe(cv.get("totalYears", ""))
    cv_title    = _safe(cv.get("title", ""))

    # Name — dynamic from profile
    story.append(Paragraph(p_name.upper(), S["name"]))
    if cv_title:
        # Cap tech list to max 3 technologies
        title_parts = cv_title.split("|")
        role_part   = title_parts[0].strip()
        if len(title_parts) > 1:
            techs     = [t.strip() for t in title_parts[1].split(",") if t.strip()]
            tech_part = ", ".join(techs[:3])
            cv_title  = role_part + " | " + tech_part
        else:
            cv_title  = role_part
        if total_years:
            try:
                years_num     = float(total_years.replace("+", "").strip())
                years_display = " | " + str(int(years_num)) + "+ Years Experience" \
                    if years_num == int(years_num) else " | " + total_years + " Years Experience"
            except Exception:
                years_display = " | " + total_years + " Years Experience"
        else:
            years_display = ""
        story.append(Paragraph(cv_title.upper() + years_display, S["role"]))

    story.append(BOLD_HR())

    # ── Contact strip — dynamic from profile links, fallback to STATIC ───────
    SEP = '  <font color="#aaaaaa">|</font>  '

    def _make_link(val):
        v = (val or "").strip()
        if not v:
            return v
        if v.startswith("http://") or v.startswith("https://"):
            return _link(v, v)
        if re.match(r"^[\w.+-]+@[\w-]+\.[a-z]{2,}$", v, re.IGNORECASE):
            return _link(v, "mailto:" + v)
        if re.match(r"^[+\d\s\-()]{7,}$", v):
            return _link(v, "tel:" + re.sub(r"[\s\-()]", "", v))
        if re.search(r"\.(com|io|net|org|app|pk|co|dev|ai)(/|$)", v, re.IGNORECASE):
            url = v if re.match(r"^https?://", v, re.IGNORECASE) else "https://" + v
            return _link(v, url)
        return v

    if p_links:
        contact_items = [_make_link(l.get("value", "")) for l in p_links]
        contact_items = [c for c in contact_items if c]
        # Single line if 4 or fewer items; otherwise two centered lines
        if len(contact_items) <= 4:
            story.append(Paragraph(SEP.join(contact_items), S["contact"]))
        else:
            mid = (len(contact_items) + 1) // 2
            story.append(Paragraph(SEP.join(contact_items[:mid]),  S["contact"]))
            story.append(Paragraph(SEP.join(contact_items[mid:]), S["contact"]))
    else:
        sc = STATIC_CANDIDATE
        line1 = [sc["address"],
                 _link(sc["email"], _CONTACT_LINKS["email"]),
                 _link(sc["phone"], _CONTACT_LINKS["phone"])]
        line2 = [_link(sc["github"],    _CONTACT_LINKS["github"]),
                 _link(sc["portfolio"], _CONTACT_LINKS["portfolio"]),
                 _link(sc["linkedin"],  _CONTACT_LINKS["linkedin"])]
        story.append(Paragraph(SEP.join(line1), S["contact"]))
        story.append(Paragraph(SEP.join(line2), S["contact"]))

    story.append(HR())
    story.append(Spacer(1, 2 * mm))

    # -- SUMMARY --
    summary_text = _safe(cv.get("summary", ""))
    if total_years and summary_text:
        summary_text = re.sub(
            r'(?:over\s+|more\s+than\s+|approximately\s+)?\b\d+\+?\s+years?\b',
            f"{total_years} years",
            summary_text, count=1
        )
    if summary_text:
        story.append(Paragraph("PROFESSIONAL SUMMARY", S["sec_title"]))
        story.append(Paragraph(summary_text, S["bullet"]))
        story.append(Spacer(1, 3 * mm))

    # -- EXPERIENCE --
    # Profile p_work entries are authoritative for company/role/dates.
    # AI cv["companies"] array provides JD-tailored bullets and tech tags.
    # Date auto-fill: if profile dates missing, split years_exp equally across companies.
    # 4+ profile companies: show first 3 in full; collapse rest into "Other Experience".

    def _date_range_str(frm, to):
        frm = (frm or "").strip()
        to  = (to  or "").strip()
        if frm and to:
            return f"{frm} \u2013 {to}"
        if frm:
            return f"{frm} \u2013 Present"
        return to if to else ""

    def _render_exp_entry(company_name, role, date_range, bullets, tech_val):
        row = [[Paragraph(company_name.upper(), S["company"]),
                Paragraph(date_range, ps("dr", fontName="Helvetica", fontSize=10, leading=12,
                                          textColor=colors.HexColor("#666666"),
                                          alignment=TA_RIGHT))]]
        t = Table(row, colWidths=[TW * 0.65, TW * 0.35])
        t.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        if role:
            story.append(Paragraph(_safe(role), S["role_title"]))
        for b in (bullets or []):
            story.append(Paragraph(f"\u2022  {_safe(b)}", S["bullet"]))
        if tech_val:
            story.append(Paragraph(f"<b>Technologies:</b> {tech_val}", S["tech_line"]))
        story.append(Spacer(1, 4 * mm))

    ai_companies = cv.get("companies") or []

    if p_work:
        shown_work = p_work[:3]
        _n_shown   = len(shown_work)

        # Auto-date fallback: split total_years equally across shown companies
        try:
            _yrs_float = float(total_years.replace("+", "").strip())
        except Exception:
            _yrs_float = 0.0
        _auto_dates = _build_dynamic_companies(total_years, num_companies=_n_shown) \
                      if _yrs_float > 0 else []

        story.append(Paragraph("WORK EXPERIENCE", S["sec_title"]))

        for i, w in enumerate(shown_work):
            ai = ai_companies[i] if i < len(ai_companies) else {}
            company_name = _safe(w.get("company")) or _safe(ai.get("company")) or ""
            role         = _safe(w.get("role"))    or _safe(ai.get("role"))    or ""
            # Strip AI internal labels like "Co1", "Co2" that leak into role text
            import re as _re
            role = _re.sub(r'\bCo\d+\b', '', role).strip()
            role = _re.sub(r'\s{2,}', ' ', role).strip()

            p_from = _safe(w.get("from", ""))
            p_to   = _safe(w.get("to",   ""))
            if p_from or p_to:
                date_range = _date_range_str(p_from, p_to)
            elif i < len(_auto_dates):
                date_range = _date_range_str(_auto_dates[i]["start"], _auto_dates[i]["end"])
            else:
                date_range = ""

            raw_bullets = ai.get("bullets") or []
            if not raw_bullets and w.get("bullets"):
                raw_bullets = [b.strip() for b in w["bullets"].split("\n") if b.strip()]
            tech_val = _safe(ai.get("tech") or ai.get("technologies") or "")
            _render_exp_entry(company_name, role, date_range, raw_bullets, tech_val)

        # Companies 4+: collapse into a single "Other Experience" block
        if len(p_work) > 3:
            extra_work = p_work[3:]
            _e_froms = [_safe(w.get("from")) for w in extra_work if w.get("from")]
            _e_tos   = [_safe(w.get("to"))   for w in extra_work if w.get("to")]
            _span_s  = _e_froms[-1] if _e_froms else ""
            _span_e  = _e_tos[0]   if _e_tos   else "Present"
            other_date = _date_range_str(_span_s, _span_e)
            other_names = ", ".join(
                _safe(w.get("company") or w.get("role") or "")
                for w in extra_work if (w.get("company") or w.get("role"))
            )
            other_role = f"Roles at: {other_names}" if other_names else "Various Roles"
            extra_ai_bullets = []
            for j in range(3, len(ai_companies)):
                extra_ai_bullets.extend(ai_companies[j].get("bullets") or [])
            _render_exp_entry("Other Experience", other_role, other_date,
                              extra_ai_bullets[:4], "")

    else:
        # No profile work: pure AI companies + STATIC date fallback
        if ai_companies:
            story.append(Paragraph("WORK EXPERIENCE", S["sec_title"]))
            for i, c in enumerate(ai_companies):
                dr  = _safe(c.get("dateRange", ""))
                if not dr and i < len(STATIC_CANDIDATE["dates"]):
                    dr = STATIC_CANDIDATE["dates"][i]
                cn = _safe(c.get("company", ""))
                if not cn and i < len(STATIC_CANDIDATE["companies"]):
                    cn = STATIC_CANDIDATE["companies"][i]
                _render_exp_entry(cn, _safe(c.get("role", "")), dr,
                                  c.get("bullets") or [],
                                  _safe(c.get("tech") or c.get("technologies") or ""))

    # -- TECHNICAL SKILLS --
    skills_raw = cv.get("skills") or []
    skills = _sanitize_skills(skills_raw, cv)   # enforce 5 buckets, dedup, re-sort
    if skills:
        story.append(Paragraph("TECHNICAL SKILLS", S["sec_title"]))

        cat_style = ps("sk_cat",
            fontName="Helvetica-Bold", fontSize=9, leading=13,
            textColor=colors.HexColor("#111111"),
        )
        itm_style = ps("sk_itm",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=colors.HexColor("#333333"),
        )

        LABEL_W = 46 * mm
        ITEMS_W = TW - LABEL_W

        rows = []
        for entry in skills:
            colon = entry.find(":")
            if colon > 0:
                cat   = entry[:colon].strip()
                items_str = entry[colon + 1:].strip()
            else:
                cat   = ""
                items_str = entry.strip()

            item_list  = [i.strip() for i in re.split(r"[,|;?·•]", items_str) if i.strip()]
            items_text = "  ·  ".join(item_list)

            rows.append([
                Paragraph(cat, cat_style),
                Paragraph(items_text, itm_style),
            ])

        t = Table(rows, colWidths=[LABEL_W, ITEMS_W], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW",     (0, 0), (-1, -2),
             0.3, colors.HexColor("#e0e0e0")),
        ]))
        story.append(t)
        story.append(Spacer(1, 3 * mm))

    # -- PROJECTS --
    projects = cv.get("projects") or []
    if projects:
        story.append(Paragraph("KEY PROJECTS", S["sec_title"]))
        for p in projects:
            raw_name = _safe(p.get("name", ""))
            # Normalize separator: "PREFIX - Description" → "PREFIX: Description"
            # Only fix when there's no colon already and a clear UPPERCASE prefix before " - "
            if ":" not in raw_name and " - " in raw_name:
                _parts = raw_name.split(" - ", 1)
                if len(_parts) == 2 and len(_parts[0].split()) <= 5:
                    raw_name = _parts[0].strip() + ": " + _parts[1].strip()
            clean_name = _strip_brackets(raw_name)

            # Split "PREFIX: Full Project Name" into prefix label + project name
            # Supports format: "ETL: Real-Time Data Pipeline" or "ERP: Internal HR System"
            _prefix_label = ""
            _proj_display = clean_name
            if ":" in clean_name:
                _colon_idx = clean_name.index(":")
                _candidate_prefix = clean_name[:_colon_idx].strip()
                _candidate_name   = clean_name[_colon_idx + 1:].strip()
                # Only treat as prefix if it's short (≤5 words) and the rest is substantial
                if _candidate_prefix and len(_candidate_prefix.split()) <= 5 and len(_candidate_name) > 8:
                    _prefix_label = _candidate_prefix
                    _proj_display = _candidate_name

            # Fallback: try systemType field if name has no colon prefix
            if not _prefix_label:
                _prefix_label = _safe(p.get("systemType", ""))

            if _prefix_label:
                combined = f'{_prefix_label.upper()} : {_proj_display}'
                combined_style = ps("pch", fontName="Helvetica-Bold", fontSize=10.5,
                                    leading=16, spaceAfter=2, spaceBefore=4,
                                    textColor=colors.HexColor("#111111"))
                story.append(Paragraph(combined, combined_style))
            else:
                story.append(Paragraph(_proj_display, S["proj_name"]))

            overview = _safe(p.get("overview") or p.get("desc") or "")
            if overview:
                story.append(Paragraph(overview, S["proj_body"]))

            for b in (p.get("bullets") or []):
                story.append(Paragraph(f"\u2022  {_safe(b)}", S["proj_bullet"]))

            # Extract tech tags
                        # Extract tech tags - ensure ALL tags are shown
            tech_tags = []
            if p.get("techTags"):
                tt = p["techTags"]
                tech_tags = tt if isinstance(tt, list) else [tt]
            elif p.get("tech"):
                tt = p["tech"]
                tech_tags = tt if isinstance(tt, list) else [tt]
            
            # Also check name for tags as fallback
            if not tech_tags:
                tech_match = re.search(r"\[([^\]]+)\]", raw_name)
                if tech_match:
                    tech_tags = [t.strip() for t in tech_match.group(1).split(",") if t.strip()]

            # Display ALL tech tags - always show Stack line
            if tech_tags:
                story.append(Paragraph(f"<b>Stack:</b> {', '.join(tech_tags[:7])}", S["proj_stack"]))

            story.append(Spacer(1, 4 * mm))  # Space between projects

    # -- KEY COMPETENCIES --
    competencies = _safe(cv.get("competencies", ""))
    if competencies:
        story.append(Paragraph("KEY COMPETENCIES", S["sec_title"]))
        competencies_display = competencies.replace(" * ", ", ").replace("* ", ", ").replace(" *", ", ")
        story.append(Paragraph(competencies_display, S["competency"]))
        story.append(Spacer(1, 2 * mm))

    # -- EDUCATION --
    # Priority: profile p_edu entries (dates static if provided, else auto-calculated)
    #           → AI cv.education → STATIC_CANDIDATE
    story.append(Paragraph("EDUCATION", S["sec_title"]))

    def _render_edu_entry(institution, degree_str, grad_yr, gpa_str, achievement_str, note_str):
        edu_row = [[Paragraph(institution.upper(), S["edu_uni"]),
                    Paragraph(grad_yr, ps("gy", fontName="Helvetica", fontSize=10, leading=12,
                                          textColor=colors.HexColor("#666666"), alignment=TA_RIGHT))]]
        t = Table(edu_row, colWidths=[TW * 0.72, TW * 0.28])
        t.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        deg_line = degree_str
        if gpa_str:
            deg_line += f"  |  {gpa_str}"
        if note_str and note_str not in deg_line:
            deg_line += f"  |  {note_str}"
        story.append(Paragraph(deg_line, S["edu_deg"]))
        # Green achievement line — shown for any medal/honour/award text
        if achievement_str:
            story.append(Paragraph(f"\U0001f3c5 {achievement_str}", S["edu_medal"]))
        story.append(Spacer(1, 3 * mm))

    _HONOUR_KEYWORDS = ("gold", "medal", "honour", "honor", "distinction",
                        "award", "excellence", "merit", "top", "prize",
                        "first", "dean", "valedictorian", "summa", "magna")

    if p_edu:
        # Auto-calculate dates from total_years for entries missing both from+to
        _auto_edu = _build_education_year(total_years, profile_edu=p_edu)

        for e in p_edu:
            institution = _safe(e.get("institution")) or STATIC_CANDIDATE["university"]
            degree_str  = _safe(e.get("degree"))      or STATIC_CANDIDATE["degree"]
            note_str    = _safe(e.get("note", ""))

            p_e_from = _safe(e.get("from", ""))
            p_e_to   = _safe(e.get("to",   ""))
            if p_e_from and p_e_to:
                # Both provided: use as-is (static)
                grad_yr = f"{p_e_from} \u2013 {p_e_to}"
            elif p_e_from and not p_e_to:
                # Only start: add 4 years
                try:
                    grad_yr = f"{p_e_from} \u2013 {int(p_e_from[:4]) + 4}"
                except Exception:
                    grad_yr = p_e_from
            elif p_e_to and not p_e_from:
                # Only end: subtract 4 years
                try:
                    grad_yr = f"{int(p_e_to[:4]) - 4} \u2013 {p_e_to}"
                except Exception:
                    grad_yr = p_e_to
            else:
                # Neither: auto-calculate from years_exp
                grad_yr = f"{_auto_edu['start']} \u2013 {_auto_edu['end']}"

            # Extract CGPA from note (e.g. "Gold Medal — CGPA 3.97/4.0")
            gpa_str = ""
            _gpa_match = re.search(r"CGPA\s*:?\s*([\d.]+\s*/\s*[\d.]+)", note_str, re.IGNORECASE)
            if _gpa_match:
                gpa_str  = f"CGPA: {_gpa_match.group(1)}"
                note_str = re.sub(r"[—\-\u2013]?\s*CGPA\s*:?\s*[\d.]+\s*/\s*[\d.]+", "",
                                  note_str, flags=re.IGNORECASE).strip().strip("— \t")

            # Detect achievement keywords → show as green medal line
            achievement_str = ""
            if any(kw in note_str.lower() for kw in _HONOUR_KEYWORDS):
                # Expand short/abbreviated inputs to a full readable phrase
                _note_lower = note_str.strip().lower()
                if _note_lower in ("gold medal", "gold medalist", "gold", "gold medallist"):
                    achievement_str = "Gold Medalist for Academic Excellence"
                elif _note_lower in ("silver medal", "silver medalist", "silver medallist"):
                    achievement_str = "Silver Medalist for Academic Excellence"
                else:
                    achievement_str = note_str
                note_str = ""   # don't also print in degree line

            _render_edu_entry(institution, degree_str, grad_yr, gpa_str, achievement_str, note_str)

    else:
        # Fallback: AI education + STATIC_CANDIDATE, with auto-calculated dates
        edu_data    = cv.get("education", {})
        _auto_edu   = _build_education_year(total_years)
        _auto_range = f"{_auto_edu['start']} \u2013 {_auto_edu['end']}"
        grad_year   = (_safe(edu_data.get("years")) or _auto_range) if isinstance(edu_data, dict) else _auto_range
        university  = (_safe(edu_data.get("university")) or STATIC_CANDIDATE["university"]) \
                      if isinstance(edu_data, dict) else STATIC_CANDIDATE["university"]
        degree      = (_safe(edu_data.get("degree")) or STATIC_CANDIDATE["degree"]) \
                      if isinstance(edu_data, dict) else STATIC_CANDIDATE["degree"]
        gpa         = (_safe(edu_data.get("cgpa")) or STATIC_CANDIDATE["gpa"]) \
                      if isinstance(edu_data, dict) else STATIC_CANDIDATE["gpa"]
        achievement = (_safe(edu_data.get("achievement")) or STATIC_CANDIDATE["achievement"]) \
                      if isinstance(edu_data, dict) else STATIC_CANDIDATE["achievement"]
        if gpa and not gpa.upper().startswith("CGPA"):
            gpa = f"CGPA: {gpa}"
        _render_edu_entry(university, degree, grad_year, gpa, achievement, "")

    # ── Single pass: build on tall canvas, then crop MediaBox to content ──
    doc.build(story)

    last_y  = doc.frame._y
    tight_h = (PAGE_H_SINGLE - MT) - last_y + MT + MB + 4 * mm
    tight_h = max(tight_h, 100 * mm)

    # ReportLab draws content at the TOP of the tall page.
    # To crop, we set the MediaBox to window into the top tight_h points:
    #   lower_left  = (0, PAGE_H_SINGLE - tight_h)   ← start just above content
    #   upper_right = (PAGE_W, PAGE_H_SINGLE)         ← top of page
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
    page.mediabox.lower_left  = (0, crop_bottom)
    page.mediabox.upper_right = (PAGE_W, PAGE_H_SINGLE)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()

class PDFRequest(BaseModel):
    cv:          dict
    filename:    str             = "CV.pdf"
    profileData: Optional[dict] = None


@app.post("/generate-pdf")
async def generate_pdf(req: PDFRequest):
    """Accept CV JSON + profileData, return a true text-based PDF file."""
    try:
        pdf_bytes = build_cv_pdf(req.cv, profile_data=req.profileData)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{req.filename}"'}
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")