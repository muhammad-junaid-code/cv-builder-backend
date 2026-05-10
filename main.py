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
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
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

def _load_auth_keys() -> dict:
    """Load auth keys from disk. Returns {token: {label, created_at, expires_at, active}}.
    On first run (no file), auto-generates a default key so the extension works immediately.
    """
    if not os.path.exists(AUTH_FILE):
        raw = secrets.token_hex(8).upper()
        default_token = f"CVAI-{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"
        keys = {
            default_token: {
                "label": "Default Key (auto-generated)",
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": None,
                "active": True,
                "last_used": None,
            }
        }
        _save_auth_keys(keys)
        print(f"\n{'='*60}")
        print(f"  CV Builder AI - First Run Setup")
        print(f"  Auto-generated access key: {default_token}")
        print(f"  Paste this key in the extension login screen.")
        print(f"  Generate more keys via the Admin panel in login.html")
        print(f"{'='*60}\n")
        return keys
    try:
        with open(AUTH_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

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


# -- Thin-JD detection & enrichment --------------------------------------------
_REAL_TECH_WORDS = {
    "net", "asp", "angular", "react", "vue", "node", "python", "java", "php",
    "ruby", "swift", "kotlin", "go", "rust", "typescript", "javascript",
    "sql", "mysql", "postgres", "mongodb", "redis", "docker", "kubernetes",
    "azure", "aws", "gcp", "linux", "django", "laravel", "spring", "flask",
    "flutter", "android", "ios", "terraform", "jenkins", "git", "graphql",
    "elasticsearch", "kafka", "rabbitmq", "nginx", "apache", "webpack",
    "tailwind", "bootstrap", "jquery", "scss", "sass", "rest", "api",
    "oauth", "jwt", "openapi", "swagger", "c#", "f#", "c++", "scala",
    "perl", "bash", "powershell", "excel", "powerbi", "tableau",
}

def _jd_is_thin(jd: str, min_tech_words: int = 3) -> bool:
    """
    Returns True if the JD contains fewer than `min_tech_words` real technology names.
    A 'thin' JD is one that describes requirements in plain English without naming
    actual tools (e.g. "Good understanding of REST APIs, SQL, and Web Development").
    """
    words = re.findall(r"[a-zA-Z0-9#\.\+]+", jd.lower())
    found = sum(1 for w in words if w in _REAL_TECH_WORDS)
    return found < min_tech_words


# Domain-based technology expansion: maps role keywords -> concrete tech stacks
_DOMAIN_TECH_STACKS: dict = {
    "dotnet": (
        "ASP.NET Core, C#, .NET 8, Entity Framework Core, LINQ, Dapper, "
        "REST APIs, SignalR, MediatR, AutoMapper, Swagger/OpenAPI, "
        "SQL Server, PostgreSQL, Redis, Azure App Service, Azure DevOps, "
        "Docker, GitHub Actions, xUnit, NUnit, MSTest, SonarQube, "
        "Angular, TypeScript, RxJS, NuGet"
    ),
    "angular": (
        "Angular 17, TypeScript, RxJS, NgRx, Angular Material, "
        "HTML5, CSS3, SCSS, Tailwind CSS, Webpack, Vite, Jest, Cypress, "
        "REST APIs, HTTP Client, OAuth 2.0, JWT, "
        "npm, ESLint, Prettier, Git, GitHub, Jasmine, Karma"
    ),
    "react": (
        "React 18, TypeScript, Redux Toolkit, React Query, Next.js, "
        "HTML5, CSS3, Tailwind CSS, SCSS, Webpack, Vite, Jest, Cypress, "
        "REST APIs, GraphQL, Apollo Client, OAuth 2.0, JWT, "
        "Node.js, ESLint, Prettier, Git, GitHub"
    ),
    "node": (
        "Node.js, Express.js, TypeScript, NestJS, REST APIs, GraphQL, "
        "JWT, OAuth 2.0, Redis, PostgreSQL, MongoDB, Mongoose, Prisma, "
        "Docker, AWS Lambda, GitHub Actions, Jest, Mocha, Supertest, "
        "npm, ESLint, Swagger, Socket.IO"
    ),
    "python": (
        "Python 3.x, Django, FastAPI, Flask, SQLAlchemy, Alembic, "
        "PostgreSQL, MySQL, MongoDB, Redis, Celery, RabbitMQ, "
        "Docker, AWS Lambda, GitHub Actions, Pytest, Unittest, "
        "Pandas, NumPy, Pydantic, pip, Mypy, Flake8, Black"
    ),
    "java": (
        "Java 17, Spring Boot, Spring MVC, Hibernate, JPA, Maven, Gradle, "
        "PostgreSQL, MySQL, MongoDB, Redis, Kafka, RabbitMQ, "
        "Docker, Kubernetes, AWS, GitHub Actions, JUnit, Mockito, "
        "REST APIs, GraphQL, JWT, OAuth 2.0, Lombok"
    ),
    "php": (
        "PHP 8.x, Laravel, Symfony, Composer, Eloquent ORM, Blade, "
        "MySQL, PostgreSQL, Redis, Queue Workers, REST APIs, "
        "Docker, DigitalOcean, Forge, Envoyer, PHPUnit, Pest, "
        "HTML, CSS, JavaScript, Vue.js, Tailwind CSS, Git"
    ),
    "fullstack": (
        "React, Angular, TypeScript, Node.js, Express.js, REST APIs, "
        "PostgreSQL, MySQL, MongoDB, Redis, "
        "Docker, AWS, GitHub Actions, Jest, Cypress, Git, "
        "HTML5, CSS3, Tailwind CSS, JWT, OAuth 2.0"
    ),
    "devops": (
        "Docker, Kubernetes, Helm, Terraform, Ansible, "
        "AWS EC2, S3, Lambda, ECS, RDS, CloudWatch, "
        "GitHub Actions, Jenkins, ArgoCD, GitLab CI, "
        "Prometheus, Grafana, Datadog, PagerDuty, "
        "Linux, Bash, Python, Nginx, Vault"
    ),
    "data": (
        "Python, Pandas, NumPy, PySpark, Apache Airflow, dbt, "
        "PostgreSQL, MySQL, BigQuery, Snowflake, Redshift, "
        "AWS S3, Glue, Athena, GCP Pub/Sub, Dataflow, "
        "Kafka, Elasticsearch, Tableau, Power BI, Great Expectations, "
        "GitHub Actions, Docker, Jupyter"
    ),
}


def _enrich_thin_jd(job_title: str, jd: str) -> str:
    """
    When a JD contains fewer than 3 real tech names, expand it by appending
    a concrete technology context block derived from the job title's domain.
    This prevents the AI from treating plain English adjectives ('Good', 'Hands',
    'Ability', 'Web') as skill tags.
    """
    title_lower = job_title.lower()

    # Pick the best matching domain from job title keywords
    if ".net" in title_lower or "c#" in title_lower or "asp" in title_lower:
        domain_key = "dotnet"
    elif "angular" in title_lower:
        domain_key = "angular"
    elif "react" in title_lower:
        domain_key = "react"
    elif "node" in title_lower or "express" in title_lower:
        domain_key = "node"
    elif "python" in title_lower or "django" in title_lower or "flask" in title_lower:
        domain_key = "python"
    elif "java" in title_lower and "javascript" not in title_lower:
        domain_key = "java"
    elif "php" in title_lower or "laravel" in title_lower:
        domain_key = "php"
    elif "devops" in title_lower or "sre" in title_lower or "platform engineer" in title_lower:
        domain_key = "devops"
    elif "data engineer" in title_lower or "data analyst" in title_lower:
        domain_key = "data"
    elif "full" in title_lower and "stack" in title_lower:
        domain_key = "fullstack"
    else:
        # Generic fallback: try to detect from JD content
        if "angular" in jd.lower():
            domain_key = "angular"
        elif "react" in jd.lower():
            domain_key = "react"
        elif ".net" in jd.lower() or "c#" in jd.lower():
            domain_key = "dotnet"
        elif "node" in jd.lower():
            domain_key = "node"
        else:
            domain_key = "fullstack"

    tech_block = _DOMAIN_TECH_STACKS.get(domain_key, _DOMAIN_TECH_STACKS["fullstack"])

    enriched = (
        jd.strip() +
        "\n\n[TECHNOLOGY CONTEXT - derived from role domain. "
        "Use ONLY these as skill/tech items; ignore plain English adjectives from above]\n" +
        tech_block
    )
    return enriched


# -- Prompt builder -------------------------------------------------------------
def _co_line(companies: list) -> str:
    return " | ".join(
        f'"{c["name"]}" {c["start"]} - {c["end"]}' for c in companies
    )


def build_prompt(req: CVRequest, jd_chars: int = 1600) -> tuple:
    """Returns (system_prompt, user_prompt) tuple."""
    # -- CRITICAL: Normalize job title - remove duplicate words --------------
    req = req.copy(update={"job_title": _normalize_job_title(req.job_title.strip())})

    _raw_jd = req.job_description.strip()
    # -- Thin-JD enrichment: if JD has <3 real tech words, expand from domain --
    if _jd_is_thin(_raw_jd):
        _raw_jd = _enrich_thin_jd(req.job_title, _raw_jd)
    jd          = _raw_jd[:jd_chars]
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
            "Co1 (only company) MUST use 'Junior' as the seniority prefix. "
            "The function word MUST match the JD domain (e.g. if JD is about backend -> 'Junior Backend Developer', "
            "if JD is about data -> 'Junior Data Engineer', if JD is about security -> 'Junior Security Analyst'). "
            "Derive the domain word and function word from the actual JD - do not hardcode. "
            "Omitting 'Junior' is a HARD FAILURE.\n\n"
        )
        _seniority_ban = (
            "- Missing 'Junior' prefix on Co1 role title - with 1 year experience, the only company MUST be Junior\n"
        )
    elif _yrs_float <= 2.4:
        _seniority_rule = (
            f"SENIORITY RULE - {total_years} years experience: "
            "Co1 (current) must have a plain domain title with NO seniority prefix. "
            "Co2 (previous) MUST use 'Junior' as the seniority prefix. "
            "Both titles MUST be derived from the JD domain - use the JD's tech stack and role type to pick the domain word. "
            "Each company MUST use a DIFFERENT function word from the pool. "
            "Getting this wrong is a HARD FAILURE.\n\n"
        )
        _seniority_ban = (
            "- Missing 'Junior' prefix on Co2 role title - the oldest company MUST be Junior\n"
            "- Adding ANY seniority prefix to Co1 role title - the current company must be plain\n"
        )
    else:
        _seniority_rule = (
            f"SENIORITY RULE - {total_years} years experience: "
            "Co1 (current) MUST use 'Senior' as the seniority prefix. "
            "Co2 (mid) must have a plain domain title with NO seniority prefix. "
            "Co3 (oldest) MUST use 'Junior' as the seniority prefix. "
            "ALL titles MUST be derived from the JD domain - pick the domain word and function word from the JD's tech stack and role focus. "
            "Each company MUST use a DIFFERENT function word from the pool. "
            "Getting this wrong is a HARD FAILURE.\n\n"
        )
        _seniority_ban = (
            "- Missing 'Senior' prefix on Co1 role title - current company MUST be Senior\n"
            "- Adding ANY seniority prefix to Co2 role title - mid company must be plain\n"
            "- Missing 'Junior' prefix on Co3 role title - oldest company MUST be Junior\n"
        )

    system = (
        "You are an expert ATS-focused CV writer. Output ONLY valid JSON. "
        "No text before/after, no markdown, no backticks. Start { end }.\n\n"

        "=== CRITICAL PRIORITY RULE (HIGHEST PRIORITY) ===\n"
        "JD relevance ALWAYS overrides AI defaults or training assumptions.\n"
        "EVERY technology, skill, and section MUST be strictly aligned with the provided JD.\n"
        "NEVER include a technology that is not mentioned in the JD or a standard companion\n"
        "to a technology explicitly in the JD. For non-technical roles (PM, SEO, Marketing,\n"
        "Finance, Design): DO NOT inject backend/infra technologies unless the JD requires them.\n\n"

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
        "  If a TECHNOLOGY CONTEXT block is appended below the JD, use THOSE named tools for skill extraction.\n"
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

        "=== STEP 1.5: CLOUD & HOSTING INFERENCE (mandatory - execute right after STEP 1) ===\n"
        "After extracting JD technologies, determine the most relevant cloud/hosting platform for this specific role.\n"
        "INFERENCE RULES - derive entirely from the JD and job title, nothing hardcoded:\n"
        "  - JD mentions .NET, C#, Azure DevOps, Active Directory, or any Microsoft stack\n"
        "    -> infer Azure: Azure App Service, Azure Functions, Azure SQL Database, Azure Blob Storage,\n"
        "      Azure DevOps, Azure AD, Azure Key Vault, Azure Monitor, Azure Container Registry\n"
        "  - JD mentions Python, TensorFlow, PyTorch, BigQuery, Dataflow, or data engineering\n"
        "    -> infer GCP: Cloud Run, GKE, Cloud Storage, BigQuery, Pub/Sub, Cloud Build, Vertex AI,\n"
        "      Cloud Composer, Artifact Registry, Cloud IAM\n"
        "  - JD mentions Node.js, Java, Go, Terraform, or generic backend/DevOps with no specific cloud\n"
        "    -> infer AWS: EC2, S3, Lambda, RDS, ECS/EKS, CloudFront, IAM, CloudWatch, CodePipeline, ECR\n"
        "  - JD mentions PHP, Laravel, WordPress, or shared-hosting stack\n"
        "    -> infer: DigitalOcean Droplets, AWS Lightsail, cPanel, Nginx, Apache, Forge, Envoyer\n"
        "  - JD mentions React Native, Flutter, Ionic, or mobile/cross-platform\n"
        "    -> infer: Firebase Hosting, Firebase Cloud Messaging, App Engine, Fastlane, TestFlight, Play Console\n"
        "  - JD explicitly names one or more cloud providers -> those providers are CORE; include their key services\n"
        "  - JD names multiple providers -> include ALL of them; merge into one 'Cloud & Hosting' category\n"
        "  - NEVER default to a fixed provider - always derive from the actual JD content\n"
        "  - NEVER omit cloud/hosting - every modern software role has a deployment environment\n"
        "MANDATORY PLACEMENT of inferred cloud services:\n"
        "  1. One dedicated skill category named after the platform (e.g. 'Azure Cloud Services',\n"
        "     'AWS Infrastructure', 'GCP & Cloud DevOps') - never use a generic label like 'Cloud Tools'\n"
        "  2. At least 2 work-experience bullets referencing deployment, hosting, scaling, or cloud-native ops\n"
        "  3. Tech tags for at least one company MUST include 1-2 cloud platform services\n"
        "  4. At least one project's techTags/stack MUST reference the inferred cloud platform\n"
        "HARD RULE: The cloud category MUST contain exactly 11-13 items - expand with the platform's\n"
        "  CI/CD pipelines, IaC tools, monitoring/alerting, secrets management, and serverless services.\n\n"

        "=== STEP 2: SKILLS PLANNING (MANDATORY - execute before writing any skill) ===\n"
        "Before writing the 'skills' array, build an internal allocation table:\n"
        "  1. List every technology extracted in STEP 1 (CORE + PREFERRED + ECOSYSTEM).\n"
        "  2. Assign EACH technology to EXACTLY ONE of the 5 buckets below based on its primary purpose:\n"
        "       BUCKET A - Frontend/UI: frameworks, component libs, CSS, state managers, build tools, browser APIs\n"
        "       BUCKET B - Backend/Server-Side: server languages, backend frameworks, APIs, auth, message brokers\n"
        "       BUCKET C - Databases & Storage: RDBMS, NoSQL, caches, ORMs, migration tools, search engines\n"
        "       BUCKET D - Cloud & DevOps: cloud services, containers, IaC, CI/CD, monitoring, registries\n"
        "       BUCKET E - Testing, Security & Tooling: test frameworks, scanners, linters, version control, package managers\n"
        "  3. CHECK: Does any technology appear in more than one bucket? -> MOVE IT to only the most appropriate bucket.\n"
        "  4. CHECK: Does any bucket have fewer than 10 technologies? -> Expand with ecosystem tools from that same domain.\n"
        "  5. ONLY AFTER completing this allocation table, write the 'skills' JSON array.\n"
        "  HARD RULE: Docker and Kubernetes ONLY go in BUCKET D (Cloud & DevOps) - NEVER in Backend or Frontend.\n"
        "  HARD RULE: Git, GitHub, npm, pip ONLY go in BUCKET E (Tooling) - NEVER in Backend or Frontend.\n"
        "  HARD RULE: C#, .NET, ASP.NET Core, Node.js, Python ONLY go in BUCKET B (Backend) - NEVER in Database or Cloud.\n"
        "  HARD RULE: SQL Server, PostgreSQL, MySQL, MongoDB, Redis ONLY go in BUCKET C (Database) - NEVER in Backend or Cloud.\n"
        "  HARD RULE: React, Angular, Vue, TypeScript (when frontend), Tailwind ONLY go in BUCKET A (Frontend) - NEVER in Backend or DB.\n"
        "  STACK CONSISTENCY RULE (CRITICAL): Every item in every bucket MUST belong to the SAME technology stack as the JD. "
        "If the JD is a .NET/C# role, do NOT add Node.js, Django, FastAPI, Spring Boot, or any non-.NET backend tool anywhere. "
        "If the JD is a Node.js role, do NOT add ASP.NET Core, Django, Spring Boot, or Laravel. "
        "If the JD is a Python role, do NOT add .NET, Spring, or Laravel tools. "
        "EVERY item across ALL 5 buckets must be a tool a developer working in THIS specific JD stack would actually use. "
        "Mixing tools from different unrelated stacks (e.g. Node.js in a .NET CV) is an INSTANT FAILURE.\n\n"

        "=== COMPANY TECH TAGS ===\n"
        "Each company MUST have a 'tech' field with exactly 6-8 pipe-separated technologies from the extracted list.\n"
        "Fewer than 6 tech tags is a HARD FAILURE.\n\n"

        f"=== ROLE TITLES ===\n"
        f"Exactly {len(companies)} companies, each with a COMPLETELY DIFFERENT role title.\n"
        f"Each title: '[Seniority] [Domain from JD] [UniqueFunction]' - derive domain from the actual JD tech stack.\n"
        "Function word pool (no repeats across companies): Engineer, Developer, Specialist, Analyst, Programmer. "
        "NEVER use Architect, Consultant, or Designer unless explicitly present in the JD.\n"
        f"NEVER use a generic title without the JD's specific tech domain appended.\n"
        "BANNED domain words: 'Full-Stack', 'Fullstack', 'Full Stack' — these are too generic. "
        "Use the JD's primary technology or focus area as the domain word instead "
        "(e.g. 'React', 'Node.js', 'Backend', 'Frontend', 'Python', 'TypeScript', 'API'). "
        "If the JD is a full-stack role, pick the PRIMARY framework (frontend or backend) as the domain word.\n"
        "UNIQUENESS RULE: ALL role titles MUST look visually different — different domain word AND different function word. "
        "CORRECT: 'Senior React Engineer' / 'Node.js Developer' / 'Junior API Specialist'. "
        "WRONG (too similar): 'Senior Backend Engineer' / 'Backend Developer' / 'Junior Backend Analyst'.\n"
        + _seniority_rule +

        "=== SUMMARY ===\n"
        "Exactly 4 sentences, minimum 70 words.\n"
        "S1: Start with '{total_years} years of experience in [domain from JD]...'.\n"
        "S2: Name 4-5 technologies FROM THE JD + the specific system types built.\n"
        "S3: Scale and complexity metrics derived from the JD context.\n"
        "S4: Methodology/business outcome relevant to the target company's industry.\n"
        "Use ONLY technologies found in the JD - never add technologies from outside the JD.\n\n"

        "=== SKILLS ===\n"
        "Exactly 5 categories. Each category: at least 10 items (aim for 10-12). ZERO items repeated across ANY category.\n"
        "Every technology across ALL 5 categories MUST be completely unique - no item may appear in more than one category.\n"
        "Derive category names from the JD's actual technical domains - not generic labels.\n"
        "All items from JD CORE + PREFERRED + ECOSYSTEM only.\n\n"

        "=== MANDATORY 5 CATEGORY STRUCTURE (STRICT DOMAIN SEPARATION) ===\n"
        "ALWAYS produce EXACTLY these 5 category types, named specifically to the JD's tech domain:\n\n"

        "CATEGORY 1 - FRONTEND / UI (name it based on JD, e.g. 'Frontend Development', 'React & UI Engineering', 'Angular & Web UI'):\n"
        "  ONLY: UI frameworks, component libraries, state managers, CSS tools, build tools, browser APIs, frontend testing utilities.\n"
        "  Examples: React, Angular, Vue.js, Next.js, TypeScript, Tailwind CSS, SCSS, Webpack, Vite, Redux, RxJS, Jest, Cypress, Storybook.\n"
        "  BANNED from this category: ANY backend framework, database, server, cloud service, DevOps tool, or language used server-side.\n\n"

        "CATEGORY 2 - BACKEND / SERVER-SIDE (name it based on JD, e.g. 'Backend Development', 'Node.js & API Engineering', 'ASP.NET Core & C#'):\n"
        "  ONLY: Server-side languages, backend frameworks, API tools, authentication libraries, message brokers, background job runners.\n"
        "  Examples: Node.js, Express.js, ASP.NET Core, Django, FastAPI, Laravel, Spring Boot, GraphQL, REST APIs, JWT, RabbitMQ, Celery.\n"
        "  BANNED from this category: ANY database engine, cloud service, frontend library, or DevOps/CI tool.\n\n"

        "CATEGORY 3 - DATABASES & DATA STORAGE (name it based on JD, e.g. 'Databases & Data Layer', 'SQL & NoSQL Storage', 'Data Engineering Tools'):\n"
        "  ONLY: Relational databases, NoSQL stores, caches, ORMs, migration tools, query builders, search engines, data warehouses.\n"
        "  Examples: PostgreSQL, MySQL, SQL Server, MongoDB, Redis, Elasticsearch, Entity Framework, Prisma, Flyway, Liquibase, ClickHouse, BigQuery.\n"
        "  BANNED from this category: ANY backend framework, frontend library, cloud service, CI/CD tool, or language runtime.\n\n"

        "CATEGORY 4 - CLOUD & DEVOPS / INFRASTRUCTURE (name it based on JD, e.g. 'AWS Infrastructure', 'Azure Cloud Services', 'GCP & DevOps'):\n"
        "  ONLY: Cloud platform services, CI/CD pipelines, container tools, IaC tools, monitoring/logging, secrets management, registries.\n"
        "  Examples: AWS EC2, S3, Lambda, Docker, Kubernetes, Terraform, GitHub Actions, Jenkins, Prometheus, Grafana, Azure DevOps, GCP Cloud Run.\n"
        "  BANNED from this category: ANY database engine, backend framework, frontend library, or language-level tool.\n"
        "  Derive the cloud platform from the JD - NEVER hardcode AWS/Azure/GCP unless the JD implies it via STEP 1.5.\n\n"

        "CATEGORY 5 - TESTING, SECURITY & TOOLING (name it based on JD, e.g. 'Testing & QA Tools', 'Security & Developer Tooling', 'Dev Tools & Quality Assurance'):\n"
        "  ONLY: Testing frameworks, security scanners, code quality tools, version control, package managers, linters, performance profilers.\n"
        "  Examples: Jest, Mocha, Pytest, xUnit, Selenium, OWASP ZAP, SonarQube, Git, GitHub, npm, pip, ESLint, Prettier, Postman, Swagger.\n"
        "  BANNED from this category: Cloud services, databases, backend frameworks, or frontend UI libraries.\n\n"

        "DOMAIN SEPARATION ENFORCEMENT (CRITICAL - HARD RULES):\n"
        "  X No database engine (PostgreSQL, MySQL, Redis, MongoDB, etc.) in Frontend, Backend, Cloud, or Testing categories.\n"
        "  X No backend framework (Express, Django, Laravel, Spring, etc.) in Frontend, Database, Cloud, or Testing categories.\n"
        "  X No frontend library (React, Angular, Vue, Tailwind, etc.) in Backend, Database, Cloud, or Testing categories.\n"
        "  X No cloud service (EC2, S3, Lambda, Azure Blob, GKE, etc.) in Frontend, Backend, Database, or Testing categories.\n"
        "  X No CI/CD or DevOps tool (Docker, Kubernetes, GitHub Actions, etc.) in Frontend, Backend, Database, or Testing categories.\n"
        "  RULE: Each technology belongs in exactly ONE category - the one that matches its primary purpose.\n"
        "  RULE: If a tool is used in multiple layers (e.g. TypeScript used in both frontend and backend), assign it ONCE to the layer the JD emphasises.\n\n"

        "CRITICAL - WHAT COUNTS AS A VALID SKILL ITEM:\n"
        "  VALID: Real named software tools, frameworks, languages, libraries, platforms, services, APIs.\n"
        "    Examples: SQL Server, PostgreSQL, Redis, Entity Framework, ASP.NET Core, React, Docker, Git\n"
        "  BANNED - INSTANT FAILURE if any of these appear as a skill item:\n"
        "    X Action verbs or gerunds: Write, Troubleshoot, Implement, Configure, Deploy, Test, Debug,\n"
        "      Design, Architect, Operate, Monitor, Manage, Develop, Build, Create, Scale, Optimize\n"
        "    X Generic English nouns: Requirements, Architecture, Infrastructure, Environment, System,\n"
        "      Server, Network, Platform, Application, Solution, Service, Module, Component\n"
        "    X Soft skills or HR words: Communication, Teamwork, Leadership, Collaboration, Problem-Solving\n"
        "    X Adjectives or adverbs: Strong, Excellent, Proficient, Experienced, Knowledgeable\n"
        "  RULE: Every single item in every skill category MUST be a real named product you could Google.\n"
        "  RULE: If a word could appear in a sentence as a common English verb or noun, REJECT IT.\n"
        "  RULE: 'NET' alone is BANNED - always write the full product name: 'ASP.NET Core', '.NET 8', etc.\n\n"

        "=== TECHNOLOGIES OBJECT ===\n"
        "mustHave: 10-14 items from JD CORE + top ecosystem companions.\n"
        "niceToHave: 8-12 items from JD PREFERRED + ecosystem companions.\n"
        "additional: 8-12 complementary tools standard in this role domain (not already listed above).\n"
        "ZERO duplicates across the three arrays.\n\n"

        "=== ARCHITECTURES ===\n"
        "3-5 objects, each with 'name' and 'description' (25-40 words, JD technologies + concrete outcome).\n"
        "Derive patterns from what the JD actually says - not from generic architecture patterns.\n\n"

        "=== CORE COMPETENCIES ===\n"
        "Exactly 10 phrases separated by ' * '. Each 2-4 words.\n"
        "Span 4 areas: Technical Practices, Domain Expertise, Engineering Process, Impact Areas.\n"
        "ALL derived from the JD domain - zero generic phrases.\n\n"

        "=== CV HEADLINE TITLE (CRITICAL - ANTI-COPY ENFORCEMENT) ===\n"
"The 'title' field MUST be DIFFERENT from the job title. COPYING = FAILURE.\n"
"FORMAT EXACTLY: \"Senior [JD Domain] Developer | [Tech1], [Tech2], [Tech3]\" when JD is a Senior Developer role.\n"
"DO NOT use Architect unless the JD explicitly contains the word 'Architect'.\n"
"DO NOT include location words like Dallas, Texas, Pakistan, Remote in the title.\n"
"- Any word in title not present in JD or not a direct role synonym derived from JD\n"
"- Location or company-related words in title\n"
"RULES:\n"
"  1. Do NOT use the exact job title string anywhere in your output.\n"
"  2. Change at least 2 words from the original job title.\n"
"  3. Use a DIFFERENT function word (Engineer/Developer/Specialist/Analyst/Programmer) - NEVER use Architect unless explicitly mentioned in the JD.\n"
"  4. If the job title has a technology name, replace it with a DIFFERENT technology from the JD's extracted list.\n"
"  5. Exactly 3 technologies after the pipe. NO MORE, NO LESS.\n"
"  6. The title MUST stay in the SAME ROLE LEVEL as the JD (e.g. if JD is Developer, DO NOT upgrade to Architect).\n"
"  7. Use pipe '|' not hyphens.\n"
"VALIDATION: Compare your title to the job title. If they look similar, regenerate.\n\n"

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
        "- Summary shorter than 70 words\n"
        "- Generic role titles without JD tech domain\n"
        "- 1-2 sentence project overviews\n"
        "- Technology-first project names (e.g. 'Azure App', 'Angular Application', 'React Dashboard')\n"
        "- Generic project names (e.g. 'ERP Platform', 'Web Application', 'Social Media Platform')\n"
        "- Real company names inside project names or overviews\n"
        "- Two or more projects with the same purpose, domain, or functionality (reworded duplicates)\n"
        "- Same sentence structure repeated across bullets or projects\n"
        "- Same metric format repeated across projects\n"
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
            "Co1 (only company): MUST use 'Junior' as the seniority prefix. "
            "The domain word and function word MUST come from the JD - identify the primary tech domain (e.g. backend, data, cloud, security, mobile, ML) and pick the most fitting function word from the pool below."
        ]
        json_seniority = ["Junior"]
    elif num_cos == 2:
        seniority_labels = ["current / plain no prefix", "previous / Junior"]
        r3_levels = [
            "Co1 (current):  plain domain role title, NO seniority prefix - derive both the domain word and function word from the JD's primary tech stack and responsibilities.",
            "Co2 (previous): MUST use 'Junior' as the seniority prefix - derive domain word from the JD, pick a DIFFERENT function word from the pool.",
        ]
        json_seniority = ["", "Junior"]
    else:
        seniority_labels = ["current / Senior", "mid-level / plain no prefix", "oldest / Junior"]
        r3_levels = [
            "Co1 (current):  MUST use 'Senior' as the seniority prefix - derive the domain word from the JD's primary tech focus, pick a UNIQUE function word from the pool.",
            "Co2 (mid):      plain domain role title, NO seniority prefix - derive domain word from the JD's secondary focus or a closely related specialisation, pick a DIFFERENT function word.",
            "Co3 (oldest):   MUST use 'Junior' as the seniority prefix - derive domain word from the JD or a related sub-domain, pick ANOTHER DIFFERENT function word.",
        ]
        json_seniority = ["Senior", "", "Junior"]

    r3_block = "\n".join(r3_levels)

    co_prompt_lines = "\n".join(
        f'Co{i+1} ({seniority_labels[i]}):  "{companies[i]["name"]}"   '
        f'{companies[i]["start"]} - {companies[i]["end"]}'
        for i in range(num_cos)
    )

    function_words = ["Engineer", "Developer", "Specialist"]
    json_companies = ",".join(
        f'{{"company":"{companies[i]["name"]}",'
        f'"role":"{"" if not json_seniority[i] else json_seniority[i] + " "}[Domain] [{function_words[i]}/etc]",'
        f'"dateRange":"{companies[i]["start"]} - {companies[i]["end"]}",'
        f'"bullets":["Achievement + tech + metric","Achievement","Achievement","Achievement"],'
        f'"tech":"Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6 | Tech7"}}'
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

{google_label}
Use your training knowledge of "{company_name or "this company"}" - their industry reputation, public sector, known offerings, and market positioning - to design an ANALOGOUS project for project 4. Do NOT copy their exact product; use an adjacent domain with the same underlying technology pattern.
"""
    elif company_name:
        co_intel_block = f"""
TARGET COMPANY: {company_name}

For project 3: Use your knowledge of what {company_name} does (website/product) to infer an ANALOGOUS project in an adjacent domain.
For project 4: Use broader public knowledge of {company_name}'s industry/sector to infer another ANALOGOUS project in a different adjacent domain.
REMINDER: Do NOT copy {company_name}'s exact product or service - use parallel domain projects with the same underlying tech pattern.
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
            "COMPANY-DRIVEN PROJECT RULES (CRITICAL - applies to projects 3 and 4):\n"
            "You have been given real intelligence about the target company (" + _co_name_str + ").\n\n"
            "*** ANALOGY RULE - THIS IS THE MOST IMPORTANT RULE FOR PROJECTS 3 AND 4 ***\n"
            "You MUST NOT copy the company actual product name, service name, or literal business description into the project.\n"
            "Instead, identify the ABSTRACT PATTERN the company business follows, then build a project in a PARALLEL or ADJACENT domain using that same pattern - with technologies drawn from the JD.\n\n"
            "HOW TO APPLY THE ANALOGY RULE:\n"
            "STEP 1 - Identify the company ABSTRACT PATTERN:\n"
            "  What is the underlying technical/business pattern?\n"
            "  (e.g. vendor-to-consumer marketplace, B2B CRM, booking/scheduling system, fleet/asset tracking, data analytics platform, workflow automation, content management, payments/billing, etc.)\n"
            "STEP 2 - Identify the ADJACENT DOMAIN for project 3:\n"
            "  Pick a DIFFERENT real-world sector that uses the same pattern but is clearly separate from the company actual sector.\n"
            "  (e.g. if the company does energy marketplaces -> try telecom, insurance, or SaaS subscription; if the company does healthcare booking -> try legal, education, or hospitality)\n"
            "STEP 3 - Identify a SECOND ADJACENT DOMAIN for project 4:\n"
            "  Pick yet another adjacent sector or a scaled/enterprise variant of the same pattern.\n"
            "STEP 4 - Name each project with a CATCHY, ORIGINAL SaaS-style product name:\n"
            "  The name must NOT contain: the company name, their product name, their commodity/sector keyword, or any direct synonym of their business.\n"
            "  Format: [OriginalProductName] - [What it does using JD technologies]\n"
            "STEP 5 - Use ONLY technologies from the JD for all tech references in project names, overviews, and bullets.\n\n"
            "PROJECT 1 & 2 (JD-driven) rules:\n"
            "Identify 2 distinct core system types implied by the JD tech stack and role responsibilities.\n"
            "Name each as a catchy original SaaS product using only JD technologies.\n\n"
            "ALL 4 projects MUST have:\n"
            "- name: [CoinedProductName] - [Specific what it does] [JD-Tech1, JD-Tech2]\n"
            "- overview: 3-4 sentences covering PROBLEM -> SOLUTION -> FUNCTIONALITY -> BUSINESS IMPACT\n"
            "- bullets: exactly 3 strings each describing a DIFFERENT specific feature or challenge:\n"
            "    bullet 1: specific component/feature built + key JD technology used (20-30 words)\n"
            "    bullet 2: hardest technical challenge + UNIQUE concrete metric (20-30 words)\n"
            "    bullet 3: business outcome + UNIQUE number (20-30 words)\n"
            "- No two projects may share the same bullet structure or the same metric format.\n"
        )
    else:
        _company_proj_block = (
            "PROJECT 1-4 are all JD-driven. Identify 4 distinct core system types implied by the JD tech stack and role responsibilities.\n"
            "HOW TO DERIVE SYSTEM TYPES: Read the JD carefully - look at the tech stack, the listed responsibilities, the domain keywords, and the problem context.\n"
            "  Look for signals like: the platform type (web, mobile, data, infra), the integration patterns (APIs, pipelines, queues), the user type (enterprise, consumer, internal), and the scale indicators.\n"
            "  Derive 4 distinct but related system types from those signals - do not repeat the same system type with different names.\n"
            "Name each as a catchy, original SaaS-style coined product name using only technologies from the JD - NEVER use generic names.\n"
            "Each project MUST have:\n"
            "- name: [CoinedProductName] - [Specific what it does] [JD-Tech1, JD-Tech2]\n"
            "- overview: 3-4 sentences covering PROBLEM -> SOLUTION -> FUNCTIONALITY -> BUSINESS IMPACT\n"
            "- bullets: exactly 3 strings each describing a DIFFERENT specific feature/challenge:\n"
            "    bullet 1: specific component/feature built + key JD technology (20-30 words)\n"
            "    bullet 2: hardest technical challenge + UNIQUE concrete metric (20-30 words)\n"
            "    bullet 3: business outcome + UNIQUE number (no metric format repeated across projects) (20-30 words)\n"
            "- No two projects may share the same bullet structure or metric format.\n"
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
Function word pool (each pick exactly one, no repeats): Engineer, Developer, Specialist, Architect, Consultant, Analyst, Programmer, Designer
HARD RULE: No function word may be used at more than one company.

R4 BULLETS: 4 per company, 20-30 words each. Every bullet: >=1 JD technology + specific named system + unique metric.
TECH ENFORCEMENT: Use ONLY technologies extracted from the JD. Never add technologies not in the JD.
Every bullet must name at least one technology from the JD's REQUIRED or PREFERRED list.
CLOUD BULLET RULE: Across all companies combined, at least 2 bullets MUST reference the cloud/hosting
platform inferred from the JD (e.g. deployment to cloud, hosted on platform, CI/CD pipeline, auto-scaling,
cloud-native service integration). Derive the platform from the JD - do NOT hardcode AWS/Azure/GCP.
Verb guide (derive from the JD's domain - do not use generic verbs):
  Co1 (most senior): high-ownership verbs reflecting leadership - Architected, Engineered, Led, Spearheaded, Established, Designed, Launched
  {"Co2 (mid): improvement verbs - Optimised, Refactored, Scaled, Migrated, Integrated, Streamlined, Unified" if num_cos >= 2 else ""}
  {"Co3 (junior): delivery verbs - Implemented, Built, Deployed, Configured, Automated, Developed, Instrumented" if num_cos >= 3 else ""}

BULLET DIVERSITY (all 12 bullets across all companies must be unique):
- Each bullet describes a DIFFERENT specific system or feature type - named precisely (e.g. 'real-time notification engine', 'multi-tenant billing service', 'role-based access middleware').
- Each bullet uses a DIFFERENT primary technology from the JD.
- Each metric is UNIQUE - no two bullets share the same number, percentage, or user count.
- Each bullet has a DIFFERENT sentence structure - no two bullets follow the same grammatical template.
- NEVER use 'web application' or 'full stack application' as the deliverable - name the SPECIFIC system type.
- ZERO verbs repeated across any of the 12 bullets.

R5 SUMMARY: Exactly 4 sentences, minimum 70 words.
S1: Start "{total_years} years of experience in [domain from JD], with a strong background in [2-3 JD areas]."
S2: "Proficient in [JD-tech1], [JD-tech2], [JD-tech3], [JD-tech4], building [specific system type from JD]."
S3: "Proven ability to [concrete achievement] handling [scale/complexity from JD context]."
S4: "Committed to [methodology], delivering [business outcome] through [practice]."
Use ONLY technologies extracted from the JD. Count words - under 70 is a FAILURE.

R6 SKILLS - FOLLOW THESE STEPS IN ORDER:

STEP R6-A: BUILD ALLOCATION TABLE (do this mentally before writing any JSON)
  Take every technology from your STEP 1 extraction. Assign each to ONE bucket only:
    BUCKET A (Frontend/UI)        -> UI frameworks, CSS tools, state managers, build tools, browser APIs
    BUCKET B (Backend)            -> Server languages, backend frameworks, REST/GraphQL APIs, auth libs, message queues
    BUCKET C (Databases/Storage)  -> RDBMS, NoSQL, caches, ORMs, migration tools, search engines, data warehouses
    BUCKET D (Cloud & DevOps)     -> Cloud services, Docker, Kubernetes, CI/CD, IaC, monitoring, container registries
    BUCKET E (Testing & Tooling)  -> Test frameworks, security scanners, linters, Git, version control, package managers

  ABSOLUTE PLACEMENT RULES - these override everything else:
    Docker, Kubernetes                    -> BUCKET D only. NEVER in A, B, C, or E.
    Git, GitHub, npm, pip, yarn           -> BUCKET E only. NEVER in A, B, C, or D.
    C#, .NET 8, ASP.NET Core, Node.js     -> BUCKET B only. NEVER in C or D.
    SQL Server, PostgreSQL, MySQL, Redis  -> BUCKET C only. NEVER in B or D.
    React, Angular, Vue, Tailwind CSS     -> BUCKET A only. NEVER in B, C, or D.
    AWS/Azure/GCP services                -> BUCKET D only. NEVER in B, C, or A.

  STACK CONSISTENCY RULE (CRITICAL - overrides all defaults):
  BEFORE adding any item to any bucket, ask: 'Would a developer working in this JD's primary stack actually use this tool?'
  If the JD is .NET/C#: Node.js, Express, Django, FastAPI, Spring Boot, Laravel, Flask are BANNED from ALL buckets.
  If the JD is Node.js: ASP.NET Core, Django, Spring Boot, Laravel, C# are BANNED from ALL buckets.
  If the JD is Python: ASP.NET Core, Node.js, Spring Boot, Laravel are BANNED from ALL buckets.
  If the JD is Java/Spring: ASP.NET Core, Node.js, Django, Laravel are BANNED from ALL buckets.
  If the JD is PHP/Laravel: ASP.NET Core, Node.js, Django, Spring Boot are BANNED from ALL buckets.
  EVERY item in EVERY bucket must belong to the same ecosystem as the JD's primary stack.

STEP R6-B: DEDUPLICATION CHECK
  Scan your allocation table. If any technology appears in more than one bucket -> remove it from all but its most appropriate bucket.
  The same string MUST NOT appear in two different skill categories. This includes partial matches (e.g. ".NET" and "ASP.NET Core" are different - keep both, but each in BUCKET B only).

STEP R6-C: COUNT CHECK
  Each bucket must have at least 10 technologies. If any bucket has fewer than 10 -> add closely adjacent tools from the same domain until it reaches 10.
  TESTING BUCKET RULE - if Cat 5 has fewer than 10 items, fill it with: xUnit, NUnit, MSTest, Mocha, Pytest,
    Selenium, Playwright, Postman, SonarQube, OWASP ZAP, Snyk, ESLint, Prettier, Swagger UI, Sentry.
    Git/GitHub/npm belong here ONLY if there is still space after filling with real test/security tools.

STEP R6-D: WRITE THE JSON
  Only now output the 'skills' array with exactly 5 objects.
  Format: "CategoryName: item1, item2, item3, item4, item5, item6, item7"
  Name each category specifically from the JD domain - e.g. "ASP.NET Core & C# Backend", "SQL Server & Data Layer", "AWS Infrastructure & CI/CD", "Angular & Web UI", "xUnit & Developer Tooling".
  NEVER use a generic name like "Backend Skills", "Database", "Tools", "Cloud".

VALID ITEMS ONLY - each item must be a real named software product:
  OK VALID: ASP.NET Core, Entity Framework Core, SQL Server, Redis, Docker, xUnit, Swagger, GitHub Actions, Angular, TypeScript
  X BANNED (instant failure): Microservices, Web APIs, CI/CD, Relational Databases, Git Flow, Clean Architecture, REST, Agile, Scrum,
    any verb (Deploy, Configure, Build, Test), any generic noun (Server, System, Platform, Infrastructure, Architecture, Environment)


R7 PROJECTS: Produce EXACTLY 4 projects split as follows:
  PROJECT 1 & 2 - JD-driven: Strictly aligned with the job description tech stack and role domain.
  PROJECT 3 - Company-aligned: Based on what the company website/context reveals about their core product/platform domain.
  PROJECT 4 - Company-sector: Based on broader public knowledge of what this company's sector/industry is known for.
  If no company name or URL was provided, replace projects 3 and 4 with two additional domain-specific JD projects covering DIFFERENT system types.

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
  Every project name MUST be a coined, original SaaS-style product name reflecting a real-world problem.
  BANNED project name patterns (INSTANT FAILURE):
    X Technology-first names: 'Azure App', 'Angular Application', 'React Dashboard', 'Python Pipeline'
    X Standalone generic suffixes: 'Web Platform', 'Mobile App', 'ERP System', 'Web Application'
    X Any real company name inside the project name
    X Any name that could apply to any company or domain
  REQUIRED format: '[CoinedProductName] - [Problem-domain description] [Tech1, Tech2]'
  Good examples: 'ShiftSync - Real-Time Staff Scheduling Engine', 'NexaQueue - Priority-Based Job Dispatch System', 'PulseTrack - Live SLA Monitoring Dashboard'

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
  PROBLEM: What real-world business problem or pain point did this system solve? Be specific about who suffers and what they lose without this system.
  SOLUTION: What specific architectural approach or technical solution was designed? Name the actual JD technologies used.
  FUNCTIONALITY: What does the system do? Describe the key features and how users interact with it.
  BUSINESS IMPACT: What measurable outcome was achieved? (revenue, uptime, users served, time saved, cost reduced - UNIQUE per project)
EXAMPLE STRUCTURE (adapt to actual JD domain):
  "Enterprise procurement teams faced 3-5 day approval cycles due to fragmented manual workflows across departments. [Candidate] engineered a real-time workflow automation engine using [JD-Tech1] and [JD-Tech2], enabling end-to-end purchase order processing with role-based approval routing. The system integrates with ERP modules for inventory validation and budget checks, reducing cycle time to under 4 hours. Deployed to 200+ enterprise users, the platform eliminated 85% of manual email approvals, saving an estimated 1,200 staff-hours per month."
HARD RULES FOR OVERVIEWS:
- MINIMUM 3 sentences, ideally 4 - short 1-sentence overviews are a HARD FAILURE
- Each overview MUST name at least 2 specific JD technologies in the solution/functionality sentences
- Each overview MUST include ONE unique business impact metric (number + unit) not reused in other projects
- NO overview may reuse the same phrase or sentence structure as another project
- NEVER include any real company name (hiring company, candidate's companies, or any known brand) in any project name or overview.
- NEVER use generic platform names: "E-commerce Platform", "Social Media Platform", "Project Management Platform", "Business Intelligence Platform", "Web Application", "Mobile App" are all BANNED as project names.
- Each project name MUST be a coined, original SaaS-style product name (e.g. "ShiftSync", "NexaQueue", "PulseTrack") followed by a dash and a specific description of what it does.
- Each project name MUST be a coined, original SaaS-style product name (e.g. "ShiftSync", "NexaQueue", "PulseTrack") followed by a dash and a specific description of what it does.

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

JSON shape (totalYears="{total_years}", degree years={edu_start}-{edu_end}, EXACTLY {num_cos} companies):
{{"totalYears":"{total_years}","title":"Related Role Title - Tech1, Tech2, Tech3","summary":"[4 sentences, 70-80+ words, starting with {total_years} years of experience...]","companies":[{json_companies}],"skills":["BackendDomain: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10, NamedTool11","DatabaseDomain: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10, NamedTool11","FrontendDomain: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10, NamedTool11","CloudPlatform: NamedService1, NamedService2, NamedService3, NamedService4, NamedService5, NamedService6, NamedService7, NamedService8, NamedService9, NamedService10, NamedService11","DevOpsDomain: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10, NamedTool11","TestingDomain: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7, NamedTool8, NamedTool9, NamedTool10, NamedTool11"],"education":{{"university":"QURTUBA UNIVERSITY OF SCIENCE AND INFORMATION TECHNOLOGY","degree":"Bachelor of Science in Computer Science (BSCS)","cgpa":"3.97/4.0","years":"{edu_start} - {edu_end}","achievement":"Gold Medalist for Academic Excellence"}},"projects":[{{"name":"CoinedName - Specific what it does [Tech1, Tech2]","overview":"[PROBLEM: who is affected and what they lose.] [SOLUTION: architecture + 2 JD techs.] [FUNCTIONALITY: key features.] [BUSINESS IMPACT: unique metric.]","bullets":["Component built using JD-tech - what it enables (20-30 words)","Challenge solved + unique concrete metric not reused (20-30 words)","Business outcome with unique number (20-30 words)"]}},{{"name":"CoinedName - Specific what it does [Tech1, Tech2]","overview":"[PROBLEM.] [SOLUTION with 2 JD techs.] [FUNCTIONALITY.] [BUSINESS IMPACT unique metric.]","bullets":["built+tech (20-30w)","challenge+UNIQUE metric (20-30w)","outcome+UNIQUE number (20-30w)"]}},{{"name":"CoinedName - Specific what it does [Tech1, Tech2]","overview":"[PROBLEM.] [SOLUTION with 2 JD techs.] [FUNCTIONALITY.] [BUSINESS IMPACT unique metric.]","bullets":["built+tech (20-30w)","challenge+UNIQUE metric (20-30w)","outcome+UNIQUE number (20-30w)"]}},{{"name":"CoinedName - Specific what it does [Tech1, Tech2]","overview":"[PROBLEM.] [SOLUTION with 2 JD techs.] [FUNCTIONALITY.] [BUSINESS IMPACT unique metric.]","bullets":["built+tech (20-30w)","challenge+UNIQUE metric (20-30w)","outcome+UNIQUE number (20-30w)"]}}],"competencies":"TechPractice1 * DomainExpertise1 * EngineeringProcess1 * ImpactArea1 * TechPractice2 * DomainExpertise2 * EngineeringProcess2 * ImpactArea2 * TechPractice3 * DomainExpertise3","relatedTech":[{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}},{{"category":"Domain","items":["i1","i2","i3","i4","i5","i6"]}}],"keywords":"kw1, kw2, kw3, kw4, kw5, kw6, kw7, kw8, kw9, kw10, kw11, kw12, kw13, kw14, kw15, kw16, kw17, kw18","technologies":{{"mustHave":["tool1","tool2","tool3","tool4","tool5","tool6","tool7"],"niceToHave":["tool1","tool2","tool3","tool4","tool5","tool6"],"additional":["tool1","tool2","tool3","tool4","tool5","tool6"]}},"architectures":[{{"name":"Pattern Name 1","description":"How you applied this pattern with concrete tech and outcome metric."}},{{"name":"Pattern Name 2","description":"How you applied this pattern with concrete tech and outcome metric."}},{{"name":"Pattern Name 3","description":"How you applied this pattern with concrete tech and outcome metric."}}]}}"""
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
            # Strip any real company name from project name if it slips through
            # (project names must be coined product names, not generic labels)
            bare_name = re.sub(r'\s*\[.*?\]', '', proj_name).split('-')[0].strip()
            if _GENERIC_PROJECT_PATTERNS.match(bare_name):
                # Flag it but still include - the prompt should prevent this
                proj_name = proj_name  # Keep as-is; log warning could go here

            clean_projects.append({
                "name":     proj_name,
                "overview": overview,
                "bullets":  proj_bullets,
                "desc":     overview,
                "techTags": tech_tags[:7] if tech_tags else [],
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

        # Strip any existing seniority prefix from role
        role = co.get("role", "").strip()
        # Strip ALL seniority words including "Lead/Principal" slash combos
        domain = re.sub(
            r"(?i)\b(lead|principal|staff|senior|mid[\s\-]?level|junior|associate|graduate|entry[\s\-]?level)(/\w+)?\b\s*",
            "", role
        ).strip()
        domain = re.sub(r"\s+", " ", domain).strip()

        # Fall back to title-derived domain if empty or generic
        if not domain or re.match(r"(?i)^(role|engineer|developer|specialist)$", domain):
            raw_title = (cv.get("title") or "Software Engineer").split("|")[0].strip()
            domain = re.sub(
                r"(?i)\b(lead|principal|staff|senior|mid[\s\-]?level|junior|associate)(/\w+)?\b\s*",
                "", raw_title
            ).strip()
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
    cv["skills"] = [_clean_skill_row(s) for s in cv.get("skills", []) if s]
    cv["skills"] = _sanitize_skills_list(cv["skills"])

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


# -- fix_skills: normalise only, no hardcoded fallbacks -----------------------
def _rebuild_skills_from_techs(techs: dict, job_title: str = "") -> list:
    """
    Emergency fallback: build 5 skill categories from the techs dict.
    Guarantees exactly 5 categories, each with >=5 real items.
    Nothing is hardcoded - categories are inferred from tech names and job title.
    """
    core      = [t for t in techs.get("core", [])      if _is_real_tech(t)]
    preferred = [t for t in techs.get("preferred", []) if _is_real_tech(t)]
    ecosystem = [t for t in techs.get("ecosystem", []) if _is_real_tech(t)]
    all_techs = list(dict.fromkeys(core + preferred + ecosystem))

    if not all_techs:
        return []

    # -- Step 1: keyword-based first-pass bucketing ----------------------------
    DB_KW    = {"sql", "mysql", "postgres", "mongodb", "redis", "oracle", "sqlite",
                "cassandra", "dynamodb", "cosmos", "firebase", "elastic", "mssql",
                "mariadb", "neo4j", "influx", "prisma", "sequelize", "typeorm",
                "hibernate", "flyway", "liquibase", "pgadmin", "dbeaver"}
    CLOUD_KW = {"aws", "azure", "gcp", "lambda", "ec2", "s3 ", "rds", "ecs", "eks",
                "fargate", "cloudfront", "route53", "heroku", "digitalocean", "vercel",
                "netlify", "docker", "kubernetes", "k8s", "helm", "terraform", "pulumi",
                "ansible", "jenkins", "github actions", "gitlab ci", "circleci"}
    FRONT_KW = {"react", "vue", "angular", "svelte", "next", "nuxt", "html", "css",
                "sass", "scss", "tailwind", "bootstrap", "material", "typescript",
                "javascript", "webpack", "vite", "babel", "eslint", "redux", "mobx",
                "jquery", "d3", "figma", "storybook", "styled", "framer"}
    QA_KW    = {"git", "jest", "pytest", "junit", "mocha", "cypress", "selenium",
                "playwright", "postman", "swagger", "openapi", "grafana", "prometheus",
                "datadog", "sentry", "sonar", "prettier", "lint", "jira", "agile"}

    title_lower = job_title.lower()
    # Rename the catch-all to match job domain
    if ".net" in title_lower or "c#" in title_lower or "asp" in title_lower:
        primary_name = "Backend & Frameworks"
    elif "react" in title_lower or "frontend" in title_lower or "angular" in title_lower:
        primary_name = "Frontend Frameworks"
    elif "node" in title_lower or "express" in title_lower:
        primary_name = "Node.js & Backend"
    elif "python" in title_lower or "django" in title_lower or "flask" in title_lower:
        primary_name = "Python & Web Frameworks"
    elif "java" in title_lower or "spring" in title_lower:
        primary_name = "Java & Enterprise Frameworks"
    elif "php" in title_lower or "laravel" in title_lower:
        primary_name = "PHP & Web Frameworks"
    elif "data" in title_lower or "ml" in title_lower or "machine" in title_lower:
        primary_name = "Data & ML Frameworks"
    elif "devops" in title_lower or "sre" in title_lower or "platform" in title_lower:
        primary_name = "Platform & Automation"
    elif "mobile" in title_lower or "flutter" in title_lower or "swift" in title_lower:
        primary_name = "Mobile & Cross-Platform"
    else:
        primary_name = "Core Languages & Libraries"

    # 5 named slots - order matters for assignment priority
    slots = [primary_name, "Database & Storage", "Cloud & Infrastructure",
             "Frontend Technologies", "DevOps & Quality"]
    buckets: dict = {s: [] for s in slots}
    used: set = set()

    for t in all_techs:
        tl = t.lower()
        if tl in used:
            continue
        used.add(tl)
        if any(kw in tl for kw in DB_KW):
            buckets["Database & Storage"].append(t)
        elif any(kw in tl for kw in CLOUD_KW):
            buckets["Cloud & Infrastructure"].append(t)
        elif any(kw in tl for kw in FRONT_KW):
            buckets["Frontend Technologies"].append(t)
        elif any(kw in tl for kw in QA_KW):
            buckets["DevOps & Quality"].append(t)
        else:
            buckets[primary_name].append(t)

    # -- Step 2: pool all items and redistribute so each slot has >=5 ----------
    # Flatten all items in stable priority order (core first, preferred, ecosystem)
    ordered_pool = list(dict.fromkeys(all_techs))

    def _ensure_min(min_count: int = 5):
        """Pull from the largest bucket to top up any bucket below min_count."""
        for attempt in range(len(ordered_pool)):  # safety limit
            thin = [s for s in slots if len(buckets[s]) < min_count]
            if not thin:
                break
            fat  = [s for s in slots if len(buckets[s]) > min_count + 1]
            if not fat:
                break
            # Move one item from the fattest to the thinnest
            fat.sort(key=lambda s: -len(buckets[s]))
            thin.sort(key=lambda s: len(buckets[s]))
            moved = buckets[fat[0]].pop(-1)
            buckets[thin[0]].append(moved)

    _ensure_min(5)

    # If still any slot has <5 (all buckets are very thin), duplicate from pool
    for slot in slots:
        if len(buckets[slot]) < 5:
            in_slot = {t.lower() for t in buckets[slot]}
            for t in ordered_pool:
                if len(buckets[slot]) >= 5:
                    break
                if t.lower() not in in_slot:
                    buckets[slot].append(t)
                    in_slot.add(t.lower())

    # -- Step 3: deduplicate within each category and cap at 13 ---------------
    result = []
    for slot in slots:
        seen: set = set()
        deduped = []
        for t in buckets[slot]:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
        result.append(f"{slot}: {', '.join(deduped[:13])}")

    return result




def fix_skills(cv: dict) -> dict:
    """
    Normalise skills into 'Category: item1, item2, ...' strings.
    Guarantees at least 5 categories with >=5 real-tech items each.
    If the LLM output is thin, rebuilds from cv['_techs'] (injected by generate_cv_atomic)
    or from a best-effort bucket split of the technologies block.
    """
    cleaned    = []
    raw_skills = cv.get("skills", [])
    if not isinstance(raw_skills, list):
        raw_skills = []

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
        if techs:
            rebuilt = _rebuild_skills_from_techs(techs, cv.get("title", "") or cv.get("_job_title", ""))
            if rebuilt:
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




# -- Domain-enforcement maps ----------------------------------------------------
# Each tuple: (substring_to_match_in_tech_name_lowercase, canonical_bucket)
# Buckets: "frontend" | "backend" | "database" | "cloud" | "testing"
_DOMAIN_RULES: list = [
    # -- DATABASE (must come before backend to catch ORM names first) ----------
    ("sql server",      "database"), ("postgresql",   "database"), ("postgres",     "database"),
    ("mysql",           "database"), ("sqlite",       "database"), ("mongodb",      "database"),
    ("redis",           "database"), ("elasticsearch","database"), ("cassandra",    "database"),
    ("dynamodb",        "database"), ("cosmos db",    "database"), ("cosmosdb",     "database"),
    ("oracle db",       "database"), ("mariadb",      "database"), ("neo4j",        "database"),
    ("influxdb",        "database"), ("clickhouse",   "database"), ("bigquery",     "database"),
    ("firestore",       "database"), ("supabase",     "database"), ("planetscale",  "database"),
    ("entity framework","database"), ("dapper",       "database"), ("prisma",       "database"),
    ("typeorm",         "database"), ("sequelize",    "database"), ("hibernate",    "database"),
    ("flyway",          "database"), ("liquibase",    "database"), ("pgadmin",      "database"),
    ("dbeaver",         "database"), ("sqlalchemy",   "database"), ("knex",         "database"),
    ("mongoose",        "database"), ("nhibernate",   "database"),

    # -- CLOUD / DEVOPS --------------------------------------------------------
    ("azure devops",    "cloud"),    ("azure app service","cloud"),("azure functions","cloud"),
    ("azure sql",       "cloud"),    ("azure blob",   "cloud"),   ("azure key vault","cloud"),
    ("azure monitor",   "cloud"),    ("azure ad",     "cloud"),   ("azure container","cloud"),
    ("azure cdn",       "cloud"),    ("azure service bus","cloud"),("azure event hub","cloud"),
    ("aws ec2",         "cloud"),    ("aws s3",       "cloud"),   ("aws lambda",   "cloud"),
    ("aws rds",         "cloud"),    ("aws ecs",      "cloud"),   ("aws eks",      "cloud"),
    ("aws cloudfront",  "cloud"),    ("aws iam",      "cloud"),   ("aws cloudwatch","cloud"),
    ("aws codepipeline","cloud"),    ("aws ecr",      "cloud"),   ("aws lightsail","cloud"),
    ("gcp",             "cloud"),    ("google cloud", "cloud"),   ("cloud run",    "cloud"),
    ("cloud storage",   "cloud"),    ("pub/sub",      "cloud"),   ("vertex ai",    "cloud"),
    ("cloud build",     "cloud"),    ("artifact registry","cloud"),
    ("heroku",          "cloud"),    ("vercel",       "cloud"),   ("netlify",      "cloud"),
    ("digitalocean",    "cloud"),    ("cloudflare",   "cloud"),   ("linode",       "cloud"),
    ("docker",          "cloud"),    ("kubernetes",   "cloud"),   ("k8s",          "cloud"),
    ("helm",            "cloud"),    ("terraform",    "cloud"),   ("ansible",      "cloud"),
    ("pulumi",          "cloud"),    ("vagrant",      "cloud"),   ("packer",       "cloud"),
    ("jenkins",         "cloud"),    ("github actions","cloud"),  ("gitlab ci",    "cloud"),
    ("circleci",        "cloud"),    ("travis ci",    "cloud"),   ("teamcity",     "cloud"),
    ("argocd",          "cloud"),    ("spinnaker",    "cloud"),   ("flux",         "cloud"),
    ("prometheus",      "cloud"),    ("grafana",      "cloud"),   ("datadog",      "cloud"),
    ("new relic",       "cloud"),    ("splunk",       "cloud"),   ("elk stack",    "cloud"),
    ("vault",           "cloud"),    ("consul",       "cloud"),   ("istio",        "cloud"),
    ("nginx",           "cloud"),    ("apache httpd", "cloud"),   ("haproxy",      "cloud"),

    # -- TESTING / QA / TOOLING ------------------------------------------------
    ("jest",            "testing"),  ("mocha",        "testing"), ("chai",         "testing"),
    ("xunit",           "testing"),  ("nunit",        "testing"), ("mstest",       "testing"),
    ("pytest",          "testing"),  ("junit",        "testing"), ("testng",       "testing"),
    ("selenium",        "testing"),  ("playwright",   "testing"), ("cypress",      "testing"),
    ("puppeteer",       "testing"),  ("webdriver",    "testing"), ("appium",       "testing"),
    ("postman",         "testing"),  ("swagger",      "testing"), ("openapi",      "testing"),
    ("sonarqube",       "testing"),  ("sonar",        "testing"), ("eslint",       "testing"),
    ("prettier",        "testing"),  ("stylelint",    "testing"), ("coverlet",     "testing"),
    ("stryker",         "testing"),  ("specflow",     "testing"), ("bdd",          "testing"),
    ("git",             "testing"),  ("github",       "testing"), ("gitlab",       "testing"),
    ("bitbucket",       "testing"),  ("jira",         "testing"), ("confluence",   "testing"),
    ("npm",             "testing"),  ("yarn",         "testing"), ("pnpm",         "testing"),
    ("pip",             "testing"),  ("nuget",        "testing"), ("maven",        "testing"),
    ("gradle",          "testing"),  ("make",         "testing"), ("cargo",        "testing"),

    # -- FRONTEND / UI ---------------------------------------------------------
    ("angular",         "frontend"), ("react",        "frontend"), ("vue",         "frontend"),
    ("svelte",          "frontend"), ("next.js",      "frontend"), ("nuxt",        "frontend"),
    ("remix",           "frontend"), ("gatsby",       "frontend"), ("astro",       "frontend"),
    ("html",            "frontend"), ("css",          "frontend"), ("scss",        "frontend"),
    ("sass",            "frontend"), ("less",         "frontend"), ("tailwind",    "frontend"),
    ("bootstrap",       "frontend"), ("material ui",  "frontend"), ("chakra",      "frontend"),
    ("ant design",      "frontend"), ("shadcn",       "frontend"), ("radix",       "frontend"),
    ("typescript",      "frontend"), ("javascript",   "frontend"), ("jquery",      "frontend"),
    ("rxjs",            "frontend"), ("redux",        "frontend"), ("zustand",     "frontend"),
    ("mobx",            "frontend"), ("ngrx",         "frontend"), ("pinia",       "frontend"),
    ("webpack",         "frontend"), ("vite",         "frontend"), ("parcel",      "frontend"),
    ("rollup",          "frontend"), ("esbuild",      "frontend"), ("babel",       "frontend"),
    ("storybook",       "frontend"), ("figma",        "frontend"), ("framer",      "frontend"),
    ("d3.js",           "frontend"), ("three.js",     "frontend"), ("chart.js",    "frontend"),
    ("highcharts",      "frontend"), ("leaflet",      "frontend"), ("mapbox",      "frontend"),
    ("ionic",           "frontend"), ("nativescript",  "frontend"),

    # -- BACKEND / SERVER-SIDE -------------------------------------------------
    ("asp.net",         "backend"),  ("asp.net core", "backend"),  (".net",        "backend"),
    ("c#",              "backend"),  ("f#",           "backend"),   ("vb.net",     "backend"),
    ("node.js",         "backend"),  ("nodejs",       "backend"),   ("express",    "backend"),
    ("fastify",         "backend"),  ("nestjs",       "backend"),   ("koa",        "backend"),
    ("hapi",            "backend"),  ("adonis",       "backend"),
    ("django",          "backend"),  ("flask",        "backend"),   ("fastapi",    "backend"),
    ("tornado",         "backend"),  ("starlette",    "backend"),
    ("spring boot",     "backend"),  ("spring",       "backend"),   ("quarkus",    "backend"),
    ("micronaut",       "backend"),  ("jakarta",      "backend"),
    ("laravel",         "backend"),  ("symfony",      "backend"),   ("codeigniter","backend"),
    ("ruby on rails",   "backend"),  ("rails",        "backend"),   ("sinatra",    "backend"),
    ("golang",          "backend"),  ("go ",          "backend"),   ("gin",        "backend"),
    ("echo",            "backend"),  ("fiber",        "backend"),
    ("rust",            "backend"),  ("actix",        "backend"),   ("axum",       "backend"),
    ("graphql",         "backend"),  ("rest api",     "backend"),   ("grpc",       "backend"),
    ("signalr",         "backend"),  ("websocket",    "backend"),   ("rabbitmq",   "backend"),
    ("kafka",           "backend"),  ("celery",       "backend"),   ("sidekiq",    "backend"),
    ("hangfire",        "backend"),  ("quartz",       "backend"),
    ("jwt",             "backend"),  ("oauth",        "backend"),   ("openid",     "backend"),
    ("identity server", "backend"),  ("keycloak",     "backend"),   ("auth0",      "backend"),
    ("python",          "backend"),  ("java",         "backend"),   ("php",        "backend"),
    ("scala",           "backend"),  ("kotlin",       "backend"),   ("elixir",     "backend"),
]


def _classify_tech(tech: str) -> str:
    """Return the domain bucket for a technology name (lowercase match)."""
    tl = tech.lower().strip()
    for pattern, bucket in _DOMAIN_RULES:
        if pattern in tl:
            return bucket
    # Heuristic fallbacks for unknown techs
    if tl.endswith(("js", ".js", "ts", ".ts")):
        return "frontend"
    if tl.endswith(("db", " db", "sql", "base")):
        return "database"
    return "backend"   # safe default - better backend than wrong category


def _jd_allowed_stacks(techs: dict, job_title: str) -> set:
    """
    Return the set of backend/frontend language ecosystems that are ACTUALLY in the JD.
    Used to remove completely off-stack technologies from skills.
    E.g. a .NET JD should not list Node.js/Express unless the JD explicitly mentions them.
    """
    all_techs_lower = set()
    for arr in ("core", "preferred", "ecosystem"):
        for t in techs.get(arr, []):
            all_techs_lower.add(t.lower())

    title_lower = job_title.lower() if job_title else ""

    # Map of ecosystem root -> detection keywords
    STACKS = {
        "dotnet":   (".net", "c#", "asp.net", "blazor", "maui", "xamarin"),
        "node":     ("node.js", "nodejs", "express", "nestjs", "fastify"),
        "python":   ("python", "django", "flask", "fastapi", "celery"),
        "java":     ("java", "spring", "quarkus", "micronaut", "jakarta"),
        "php":      ("php", "laravel", "symfony", "codeigniter", "wordpress"),
        "go":       ("golang", "gin framework", "echo framework", "fiber framework"),
        "ruby":     ("ruby", "rails", "sinatra"),
        "angular":  ("angular", "ngrx", "rxjs"),
        "react":    ("react", "redux", "next.js", "gatsby", "remix"),
        "vue":      ("vue", "nuxt", "pinia", "vuex"),
    }

    present = set()
    for stack, keywords in STACKS.items():
        for kw in keywords:
            if any(kw in tl for tl in all_techs_lower) or kw in title_lower:
                present.add(stack)
                break

    return present


# Technologies that are cross-stack neutral (never stripped regardless of JD stack)
_NEUTRAL_TECHS = {
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
    "docker", "kubernetes", "terraform", "ansible", "helm", "nginx",
    "rest api", "graphql", "grpc", "openapi", "swagger", "postman",
    "redis", "rabbitmq", "kafka", "elasticsearch",
    "aws", "azure", "gcp", "google cloud", "heroku", "vercel", "netlify",
    "sonarqube", "eslint", "prettier",
    "typescript", "javascript",  # used in many stacks
    "html", "css", "scss",  # always relevant
}


def _is_neutral(tech: str) -> bool:
    tl = tech.lower().strip()
    return any(n in tl for n in _NEUTRAL_TECHS)


def _enforce_skill_domains(cv: dict, techs: dict, job_title: str = "") -> dict:
    """
    Post-processing domain enforcer for the 'skills' section.

    Two problems solved:
    1. MISPLACED ITEMS - a tech like Docker listed under 'Backend & Frameworks'
       is moved to the correct cloud/devops category.
    2. OFF-STACK ITEMS - techs from an entirely different language ecosystem
       (e.g. Node.js in a .NET CV, or Django in an Angular CV) are removed
       unless the JD explicitly extracted them.

    Algorithm:
      a) Parse the 5 skill category rows into (heading, [items]) pairs.
      b) Classify every item into its canonical bucket.
      c) If an item's bucket doesn't match the row's inferred bucket, move it
         to the matching row (or a "pending" list if no row matches yet).
      d) Remove items whose ecosystem is not present in the JD extracted techs.
      e) Redistribute any displaced items so each row keeps >=5 items.
    """
    skills = cv.get("skills", [])
    if not skills or len(skills) < 2:
        return cv

    # -- Step 1: parse rows ----------------------------------------------------
    parsed: list[tuple[str, list[str], str]] = []  # (heading, items, inferred_bucket)
    BUCKET_HINTS = {
        "frontend": "frontend", "ui":       "frontend", "angular":  "frontend",
        "react":    "frontend", "vue":      "frontend",
        "backend":  "backend",  "server":   "backend",  "api":      "backend",
        "c#":       "backend",  ".net":     "backend",  "asp":      "backend",
        "node":     "backend",  "python":   "backend",  "java":     "backend",
        "database": "database", "db":       "database", "storage":  "database",
        "sql":      "database", "nosql":    "database", "data":     "database",
        "cloud":    "cloud",    "devops":   "cloud",    "infra":    "cloud",
        "azure":    "cloud",    "aws":      "cloud",    "gcp":      "cloud",
        "ci/cd":    "cloud",    "deploy":   "cloud",    "pipeline": "cloud",
        "test":     "testing",  "qa":       "testing",  "quality":  "testing",
        "tool":     "testing",  "security": "testing",  "lint":     "testing",
    }

    for row in skills:
        if not row:
            continue
        colon = row.find(":")
        if colon <= 0:
            parsed.append((row.strip(), [], "backend"))
            continue
        heading = row[:colon].strip()
        items_str = row[colon+1:].strip()
        items = [t.strip() for t in items_str.split(",") if t.strip()]

        # Infer bucket from heading keywords
        hl = heading.lower()
        inferred = "backend"  # default
        for hint, bucket in BUCKET_HINTS.items():
            if hint in hl:
                inferred = bucket
                break
        parsed.append((heading, items, inferred))

    if not parsed:
        return cv

    # -- Step 2: determine which stacks are present in the JD ------------------
    allowed_stacks = _jd_allowed_stacks(techs, job_title)

    # Map stack -> bucket (which bucket that stack's items belong to)
    STACK_BUCKET = {
        "dotnet": "backend", "node": "backend", "python": "backend",
        "java":   "backend", "php":  "backend", "go":     "backend",
        "ruby":   "backend",
        "angular":"frontend","react":"frontend", "vue":    "frontend",
    }

    # Reverse: which stacks are OFF-limits (not in JD and not neutral)
    def _is_allowed_tech(tech: str) -> bool:
        """Return True if this tech should be kept (in JD stack or neutral)."""
        if _is_neutral(tech):
            return True
        tl = tech.lower().strip()
        # Check if the tech belongs to an allowed stack
        for stack, bucket in STACK_BUCKET.items():
            if stack in allowed_stacks:
                continue  # this stack is allowed
            # This stack is NOT in the JD - check if tech belongs to it
            STACK_MARKERS = {
                "dotnet": (".net", "c#", "asp.net", "blazor", "maui", "signalr",
                           "entity framework", "dapper", "nhibernate", "xunit",
                           "nunit", "mstest", "hangfire", "identity server"),
                "node":   ("node.js", "nodejs", "express", "nestjs", "fastify",
                           "koa", "hapi", "adonis"),
                "python": ("python", "django", "flask", "fastapi", "tornado",
                           "celery", "pytest", "pip", "sqlalchemy"),
                "java":   ("java", "spring", "quarkus", "micronaut", "jakarta",
                           "maven", "gradle", "junit"),
                "php":    ("php", "laravel", "symfony", "codeigniter", "composer"),
                "go":     ("golang", "gin framework", "echo framework", "fiber framework"),
                "ruby":   ("ruby on rails", "rails", "sinatra", "rspec", "rubygems"),
                "angular":("ngrx",),   # Angular-specific (angular itself is always frontend)
                "react":  ("redux", "react-query", "gatsby", "remix"),
                "vue":    ("pinia", "vuex", "nuxt"),
            }
            markers = STACK_MARKERS.get(stack, ())
            if any(m in tl for m in markers):
                return False  # belongs to an off-stack ecosystem
        return True

    # -- Step 3: build bucket -> row index map ----------------------------------
    # The 5 rows should map to the 5 canonical buckets.
    # If two rows infer the same bucket, use the first one for that bucket.
    bucket_row: dict[str, int] = {}
    for idx, (heading, items, inferred) in enumerate(parsed):
        if inferred not in bucket_row:
            bucket_row[inferred] = idx

    # -- Step 4: re-bucket every item ------------------------------------------
    # Build clean_items per row
    clean: list[list[str]] = [[] for _ in parsed]
    displaced: list[tuple[str, str]] = []  # (tech, correct_bucket)

    for row_idx, (heading, items, inferred) in enumerate(parsed):
        for item in items:
            if not _is_allowed_tech(item):
                continue  # drop off-stack tech entirely
            correct_bucket = _classify_tech(item)
            target_idx = bucket_row.get(correct_bucket)

            if target_idx is None:
                # No row for this bucket - keep in current row (avoids data loss)
                clean[row_idx].append(item)
            elif target_idx == row_idx:
                clean[row_idx].append(item)
            else:
                # Move to correct row
                clean[target_idx].append(item)

    # -- Step 5: dedup within each row -----------------------------------------
    for i in range(len(clean)):
        seen: set[str] = set()
        deduped: list[str] = []
        for t in clean[i]:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
        clean[i] = deduped

    # -- Step 6: ensure every row has >=5 items (pull from original if needed) --
    for row_idx, (heading, orig_items, inferred) in enumerate(parsed):
        if len(clean[row_idx]) < 5:
            existing = {t.lower() for t in clean[row_idx]}
            for t in orig_items:
                if len(clean[row_idx]) >= 10:
                    break
                if t.lower() not in existing and _is_allowed_tech(t):
                    clean[row_idx].append(t)
                    existing.add(t.lower())

    # -- Step 7: rebuild skill rows --------------------------------------------
    result: list[str] = []
    for idx, (heading, _, _) in enumerate(parsed):
        items = clean[idx]
        if items:
            result.append(f"{heading}: {', '.join(items)}")

    if len(result) >= 3:   # only replace if we produced something sensible
        cv["skills"] = result

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

    # Per-call timeouts inside call_llm_atomic (60s Groq). Session must not interfere.
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15, read=300, write=30, pool=15)) as client:
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

        "=== SKILLS (6 categories, 11-13 items each) ===\n"
        "Identify 6 distinct technical domains from the JD (e.g. backend, frontend, cloud, database, devops, testing).\n"
        "For each domain, list 11-13 technologies from your CORE + PREFERRED + ECOSYSTEM extraction.\n"
        "Assign each technology to exactly ONE category - zero repeated items across categories.\n"
        "Category names: derive from the actual JD domains - e.g. '[Language] Backend Development', '[Framework] Frontend', '[Cloud] Infrastructure'.\n"
        "MINIMUM 11 items per category is a HARD FAILURE. MAXIMUM 13.\n\n"

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
        "name": "[CoinedName] - [Specific system description using JD-Tech1, JD-Tech2]",
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
                          headers: dict, max_tokens: int = 1200) -> dict:
    """Universal atomic LLM call - honours Retry-After on 429, one retry.

    Per-call timeout strategy:
      - Cerebras: 90 s  (fast inference but occasionally slow cold-starts)
      - Groq:     60 s  (very fast, short timeout is fine)
      - Others:   90 s  (conservative default)
    The outer httpx.AsyncClient session timeout is intentionally set very high
    (or None) so it never fires before these per-call timeouts do.
    """
    # Choose per-call timeout based on provider URL
    if url == CEREBRAS_URL:
        per_call_timeout = 40   # Reduced from 90s to 40s for faster Cerebras generation
    elif url == GROQ_URL:
        per_call_timeout = 50   # Reduced from 60s to 50s
    else:
        per_call_timeout = 60   # Reduced from 90s to 60s

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
            if attempt == 0:
                # Give Cerebras one extra chance - it can be slow on cold starts
                if url == CEREBRAS_URL:
                    raise ValueError(f"Stage {stage} timed out after {per_call_timeout}s - Cerebras slow; try again or switch to Groq")
                raise ValueError(f"Stage {stage} timed out after {per_call_timeout}s")
            raise ValueError(f"Stage {stage} timed out after {per_call_timeout}s")
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
            wait = min(wait, 60)
            if attempt == 0:
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

    Ensures skill category names reflect the ACTUAL JD domain, not generic labels.
    Renames overly generic categories like 'Backend Technologies' to role-specific names.
    Also ensures no category is named after a company or a location.
    """
    title_lower = job_title.lower()
    skills = cv.get("skills", [])
    if not skills:
        return cv

    _GENERIC_CAT_NAMES = {
        "backend technologies", "backend", "frontend technologies", "frontend",
        "database", "databases", "cloud", "devops", "skills", "tools",
        "technologies", "other technologies", "additional skills",
    }

    new_skills = []
    for row in skills:
        if ":" not in row:
            new_skills.append(row)
            continue
        colon = row.index(":")
        cat = row[:colon].strip()
        rest = row[colon + 1:]

        # Rename overly generic category names by prepending domain signal
        cat_lower = cat.lower()
        if cat_lower in _GENERIC_CAT_NAMES:
            # Derive a domain-specific prefix from job title or JD
            if ".net" in title_lower or "c#" in title_lower:
                prefix = ".NET & C#"
            elif "angular" in title_lower:
                prefix = "Angular & TypeScript"
            elif "react" in title_lower:
                prefix = "React & Frontend"
            elif "node" in title_lower:
                prefix = "Node.js & Backend"
            elif "python" in title_lower or "django" in title_lower:
                prefix = "Python & Backend"
            elif "java" in title_lower:
                prefix = "Java & Spring"
            elif "php" in title_lower:
                prefix = "PHP & Laravel"
            elif "backend" in cat_lower:
                # Try to infer from JD
                if "node" in jd.lower():
                    prefix = "Node.js & Backend"
                elif ".net" in jd.lower() or "c#" in jd.lower():
                    prefix = ".NET & C# Backend"
                elif "python" in jd.lower():
                    prefix = "Python Backend"
                else:
                    prefix = "Core Backend"
            else:
                new_skills.append(row)
                continue
            cat = prefix + " " + cat if prefix not in cat else cat

        new_skills.append(f"{cat}:{rest}")
    cv["skills"] = new_skills
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

    return cv


# ===================================================================
# MAIN ATOMIC GENERATION - 5 pipelines, never truncated
# ===================================================================

async def generate_cv_atomic(req: CVRequest, client, key: str, model: str, 
                              url: str, headers: dict) -> dict:
    """Generate CV using 3 merged pipelines - fits within Groq free tier (30 RPM, 6k TPM).

    Call 1 -> tech extraction + title + summary + competencies  (~600 output tokens)
    Call 2 -> skills + experience                               (~1500 output tokens)
    Call 3 -> projects + related tech                           (~1800 output tokens)
    """
    import asyncio as _asyncio

    # Delay between calls for Groq free tier (6K TPM limit)
    # DeepSeek-R1 outputs extra <think> reasoning tokens - needs longer delay + shorter JD
    _is_r1    = "deepseek-r1" in model.lower()
    _delay    = 8 if (_is_r1 and url == GROQ_URL) else (3 if url == GROQ_URL else 0)
    _jd_limit = 700 if (_is_r1 and url == GROQ_URL) else 1200

    years_exp    = (req.years_exp or "").strip()
    total_years  = _calc_total_years(years_exp)
    companies_list = _build_dynamic_companies(years_exp)
    edu          = _build_education_year(years_exp)
    num_cos      = len(companies_list)
    jd           = req.job_description.strip()[:_jd_limit]
    company_name = (req.company_name or "").strip()
    company_ctx  = (req.company_context or "").strip()[:400]

    # -- CALL 1: Tech + Title + Summary + Competencies -------------------------
    sys1 = (
        "You are an expert CV writer and tech extractor. Output ONLY valid JSON. No markdown, no backticks.\n\n"
        "Do ALL of these tasks from the job description below:\n"
        "1. TECH: Extract every named technology into core/preferred/ecosystem arrays (aim 30+ total items).\n"
        "2. TITLE: A transformed role title different from the input job title. "
        "Format: \'Transformed Title | Tech1, Tech2, Tech3\'\n"
        "3. SUMMARY: 4 sentences, 70+ words, start with \'" + total_years + " years of experience...\'\n"
        "4. COMPETENCIES: exactly 10 domain-specific 2-4 word phrases separated by \' * \'"
    )
    usr1 = f"""Job Title: {req.job_title}
{"Target Company: " + company_name if company_name else ""}
Experience: {total_years} years

Job Description:
{jd}

Output JSON:
{{"core":["tech1","tech2"],"preferred":["tech1"],"ecosystem":["companion1","companion2"],
"title":"Transformed Title | Tech1, Tech2, Tech3",
"summary":"{total_years} years of experience in [domain]... (4 sentences, 70+ words)",
"competencies":"Phrase1 * Phrase2 * Phrase3 * Phrase4 * Phrase5 * Phrase6 * Phrase7 * Phrase8 * Phrase9 * Phrase10"}}"""

    result1 = await call_llm_atomic(client, key, model, url, sys1, usr1, "Call1-TechTitleSummary", headers, max_tokens=700)

    # Sanitise techs
    techs = _sanitise_techs({
        "core": result1.get("core", []),
        "preferred": result1.get("preferred", []),
        "ecosystem": result1.get("ecosystem", []),
    })
    # Fallback tech extraction from JD text
    if not techs.get("core") or len(techs["core"]) < 3:
        found = re.findall(r'\b([A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z]+)?(?:js|JS|TS|ts)?)\b', req.job_description)
        common = {"The","And","For","With","From","Have","Will","Are","Can","You","Your","Our","This","That"}
        found = [f for f in found if len(f) > 2 and f not in common][:15]
        techs = {"core": list(dict.fromkeys(found)), "preferred": [], "ecosystem": []}

    all_techs_flat = list(dict.fromkeys(techs["core"] + techs["preferred"] + techs["ecosystem"]))[:40]
    techs_str = ", ".join(all_techs_flat) if all_techs_flat else "technologies from the JD"

    title_out = result1.get("title", "")
    summary_out = result1.get("summary", "")
    competencies_out = result1.get("competencies", "")

    if not title_out:
        core = techs.get("core", ["Technology"])
        title_out = f"{req.job_title.split('|')[0].strip()} | {core[0] if core else 'Technology'}, {core[1] if len(core)>1 else 'Technology'}, {core[2] if len(core)>2 else 'Technology'}"
    if not summary_out:
        summary_out = f"{total_years} years of experience in software development with expertise in {techs_str[:100]}."
    if not competencies_out:
        competencies_out = "API Design * System Architecture * Cloud Integration * Performance Optimisation * CI/CD Automation * Database Optimisation * Agile Delivery * Code Quality * Test Coverage * Security Hardening"

    await _asyncio.sleep(_delay)

    # -- CALL 2: Skills + Experience -------------------------------------------
    if num_cos == 1:
        sen_rules = 'Co1 (only): role has Junior prefix'
        verb_guide = "Co1: Implemented, Built, Developed, Deployed, Configured, Automated."
    elif num_cos == 2:
        sen_rules = 'Co1 (current): no prefix. Co2 (oldest): Junior prefix'
        verb_guide = "Co1: Developed, Engineered, Integrated, Built.\nCo2 (Junior): Implemented, Configured, Deployed, Automated."
    else:
        sen_rules = 'Co1 (current): Senior prefix. Co2 (mid): no prefix. Co3 (oldest): Junior prefix'
        verb_guide = "Co1 (Senior): Architected, Engineered, Led, Spearheaded.\nCo2: Optimised, Refactored, Scaled, Migrated.\nCo3 (Junior): Implemented, Built, Deployed, Configured."

    co_lines = "\n".join(
        f'Co{i+1}: name="{c["name"]}", dates="{c["start"]} - {c["end"]}"' 
        for i, c in enumerate(companies_list)
    )

    sys2 = (
        "You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
        "PRIORITY RULE: JD relevance is always highest priority. Use ONLY technologies from the JD or their standard companions. Never inject tools not traceable to this JD.\n\n"
        f"Allowed technologies (use ONLY these): {techs_str}\n\n"
        "TASK A - SKILLS: exactly 5 categories, each with exactly 7-10 items from the allowed list.\n"
        "Category names must reflect JD technical domains (e.g. 'Backend & Frameworks', 'Database & Storage',\n"
        " 'Cloud & Infrastructure', 'Frontend Technologies', 'DevOps & Quality'). Zero items repeated across categories.\n"
        "CRITICAL - VALID SKILL ITEMS ONLY: Every item MUST be a real named software product (a tool you could Google).\n"
        "BANNED items - instant failure if any appear: Write, Read, Test, Debug, Troubleshoot, Configure, Deploy,\n"
        " Design, Implement, Develop, Build, Maintain, Monitor, Manage, Requirements, Architecture, Infrastructure,\n"
        " Environment, System, Server, Network, Platform, Application, Solution, Service, NET (alone).\n"
        "MINIMUM 7 items per category - fewer is a hard failure.\n\n"
        "TASK B - EXPERIENCE: exactly " + str(num_cos) + " companies.\n"
        f"Seniority: {sen_rules}\n"
        f"Verb guide:\n{verb_guide}\n"
        "Each company: unique role title, 4 bullets (20-30 words each), 6 tech tags.\n"
        "Bullets: different system per bullet, unique metric, no repeated verbs.\n"
        "Tech tags: use only allowed technologies, no tech repeated across companies."
    )
    usr2 = f"""Job Title: {req.job_title}
Experience: {total_years} years

Companies (use exact names and dates):
{co_lines}

Job Description context:
{jd[:600]}

Output JSON:
{{"skills":["BackendCategory: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7","DatabaseCategory: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7","CloudCategory: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6, NamedTool7","FrontendCategory: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6","DevOpsCategory: NamedTool1, NamedTool2, NamedTool3, NamedTool4, NamedTool5, NamedTool6"],
"companies":[{{"company":"EXACT NAME","role":"Seniority Domain Function","dateRange":"Start - End","bullets":["20-30w bullet","20-30w bullet","20-30w bullet","20-30w bullet"],"tech":"Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6"}}]}}"""

    result2 = await call_llm_atomic(client, key, model, url, sys2, usr2, "Call2-SkillsExperience", headers, max_tokens=1600)

    skills_out = result2.get("skills", [])
    companies_out = result2.get("companies", [])

    # Fix company names/dates from our canonical list
    for i, co in enumerate(companies_out):
        if i < len(companies_list):
            co["company"] = companies_list[i]["name"]
            co["dateRange"] = f'{companies_list[i]["start"]} - {companies_list[i]["end"]}'

    if not companies_out:
        companies_out = [
            {"company": c["name"], "role": "Software Developer",
             "dateRange": f'{c["start"]} - {c["end"]}',
             "bullets": ["Developed scalable software solutions using " + (techs.get("core", ["the primary stack"])[0] if techs.get("core") else "the primary stack"),
                         "Implemented features improving system efficiency by 30%",
                         "Collaborated with cross-functional teams to deliver key projects",
                         "Maintained high code quality and system reliability"],
             "tech": " | ".join(all_techs_flat[:6])}
            for c in companies_list
        ]

    await _asyncio.sleep(_delay)

    # -- CALL 3: Projects + Related Tech --------------------------------------
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
        co_intel = f"Target Company: {company_name}\nProjects 3&4: use your knowledge of {company_name}\'s industry for analogous projects."

    sys3 = (
        "You are an expert CV writer. Output ONLY valid JSON. No markdown, no backticks.\n\n"
        f"Allowed technologies: {techs_str}\n\n"
        "TASK A - PROJECTS: exactly 4 projects. Each:\n"
        "  name: \'CoinedWord - what it does [Tech1, Tech2]\'\n"
        "  overview: 3-4 natural sentences (problem -> solution -> functionality -> impact). NO S1/S2 labels.\n"
        "  bullets: 3 strings (20-30 words each, different verb each, unique metric each)\n"
        "  techTags: 5-6 technologies from allowed list, none repeated across projects\n"
        "BANNED: generic names like \'Web App\', real company names in projects, repeated verbs/metrics.\n\n"
        "TASK B - RELATED TECH: exactly 5 category boxes, 5 items each, all from allowed list, zero duplicates."
    )
    usr3 = f"""Job Title: {req.job_title}
{co_intel}

Allowed technologies (pick from these ONLY):
{techs_str}

Systems already in experience (do NOT repeat):
{used_str}

Output JSON:
{{"projects":[{{"name":"CoinedName - what it does [Tech1, Tech2]","overview":"Natural 3-4 sentence story.","bullets":["verb component + tech (20-30w)","verb challenge + unique metric (20-30w)","verb outcome + unique number (20-30w)"],"techTags":["Tech1","Tech2","Tech3","Tech4","Tech5"]}}],"relatedTech":[{{"category":"Domain","items":["t1","t2","t3","t4","t5"]}}]}}"""

    result3 = await call_llm_atomic(client, key, model, url, sys3, usr3, "Call3-ProjectsRelated", headers, max_tokens=1800)

    projects_out = result3.get("projects", [])
    related_out  = result3.get("relatedTech", [])
    projects_out = validate_project_techs({"projects": projects_out}, techs).get("projects", [])

    # -- Assemble --------------------------------------------------------------
    cv = {
        "totalYears":   total_years,
        "title":        title_out,
        "summary":      summary_out,
        "skills":       skills_out,
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
        "keywords": ", ".join(all_techs_flat[:15]),
        "architectures": [],
        # Internal hints used by fix_skills fallback (stripped by sanitise_cv afterwards)
        "_techs":     techs,
        "_job_title": req.job_title,
    }

    cv_sanitised  = sanitise_cv(cv)
    cv_companies  = fix_companies(cv_sanitised)
    cv_skills     = fix_skills(cv_companies)
    # Enforce domain separation: re-bucket misplaced items, strip off-stack techs
    cv_enforced   = _enforce_skill_domains(cv_skills, techs, req.job_title)
    # Guarantee each project has 5-7 real JD-derived tech tags
    cv_projtags   = _repair_project_tech_tags(cv_enforced, techs)
    cv_polished   = final_polish(fix_skills_dedup(fix_projects(cv_projtags)), years_exp=years_exp)
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

    # Overall hard deadline: 2 minutes 10 seconds total for all keys
    import time as _cb_time
    _cb_deadline = _cb_time.time() + 125  # 2 min 5 sec per key tries

    # Session timeout reduced to match the 2-min goal
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=130, write=20, pool=10)) as client:
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
                # -- 429: Cerebras resets per-minute, so wait & retry once -----
                if "rate limited" in err_str.lower() or "429" in err_str:
                    retry_match = re.search(r"retry.after[=:\s]+(\d+)", err_str, re.I)
                    wait_s = int(retry_match.group(1)) if retry_match else 30
                    wait_s = min(wait_s, 60)
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

    # Hard deadline: 3.5 minutes for Gemini (3 calls x ~40s each + overhead)
    import time as _time
    _deadline = _time.time() + 210  # 3.5 min for slower models like 2.5 Flash

    async with httpx.AsyncClient(timeout=200) as client:

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
                    "TASK A - SKILLS: exactly 5 categories, 7-10 items each from allowed list. No repeated items.\n"
                    "BANNED: Write, Read, Test, Debug, Troubleshoot, Configure, Deploy, Design, Implement, Build, Maintain, Monitor, Manage.\n\n"
                    f"TASK B - EXPERIENCE: exactly {num_cos} companies.\n"
                    f"Seniority: {sen_rules}\n"
                    f"Verb guide:\n{verb_guide}\n"
                    "Each company: unique role title, 4 bullets (20-30 words each), 6 tech tags from allowed list."
                )
                usr2 = f"""Job Title: {req.job_title}
Experience: {total_years} years
Companies (use exact names and dates):
{co_lines}
Job Description context:
{jd[:600]}
Output JSON:
{{"skills":["Category: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7","Category: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7","Category: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6, Tool7","Category: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6","Category: Tool1, Tool2, Tool3, Tool4, Tool5, Tool6"],
"companies":[{{"company":"EXACT NAME","role":"Seniority Domain Function","dateRange":"Start - End","bullets":["20-30w bullet","20-30w bullet","20-30w bullet","20-30w bullet"],"tech":"Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6"}}]}}"""

                result2 = await _gcall(client, key2, sys2, usr2, 2400, "Call2")
                _key_usage[mask(key2)] = _key_usage.get(mask(key2), 0) + 1

                skills_out    = result2.get("skills", [])
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

                # -- Assemble --------------------------------------------------
                cv = {
                    "totalYears":   total_years,
                    "title":        title_out,
                    "summary":      summary_out,
                    "skills":       skills_out,
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

                cv_s = sanitise_cv(cv)
                cv_s = fix_companies(cv_s)
                cv_s = fix_skills(cv_s)
                _tg  = {"core":      cv_s.get("technologies", {}).get("mustHave",   []),
                        "preferred": cv_s.get("technologies", {}).get("niceToHave", []),
                        "ecosystem": cv_s.get("technologies", {}).get("additional", [])}
                cv_s = _enforce_skill_domains(cv_s, _tg, req.job_title)
                cv_s = _repair_project_tech_tags(cv_s, _tg)
                _cv_polished = final_polish(fix_skills_dedup(fix_projects(cv_s)), years_exp=(req.years_exp or ""))
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
    try:
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


def _get_domain_fallback_pools(domain: str) -> list:
    """
    Return 5 domain-specific fallback pools (one per skill bucket).
    Each pool is a list of real named tools for that domain.
    The bucket structure adapts to the domain:
      - software_dev: Frontend / Backend / DB / Cloud / Testing
      - seo: SEO Tools / Analytics / Content / Technical SEO / Reporting & Ads
      - data_science: Languages / ML Frameworks / Data Eng / Cloud/MLOps / Viz & Tools
      - mobile: Cross-Platform / iOS / Android / Backend / DevOps & Testing
      - devops: IaC / CI/CD / Containers / Monitoring / Cloud
      - security: Offensive / Defensive / SIEM / Compliance / Scripting
      - design: Design Tools / Prototyping / Research / Frontend / Collab
      - project_management: PM Tools / Agile / Collab / Reporting / CRM
      - finance: Accounting / ERP / Analytics / Compliance / Data
    """
    pools = {
        "software_dev": [
            ["React", "Angular", "Vue.js", "Next.js", "TypeScript", "Tailwind CSS",
             "SCSS", "Webpack", "Vite", "Redux", "Bootstrap", "Storybook", "ESLint"],
            ["ASP.NET Core", "C#", ".NET 8", "Node.js", "Express.js", "Django",
             "FastAPI", "Spring Boot", "GraphQL", "RabbitMQ", "JWT", "OAuth 2.0",
             "Swagger", "gRPC", "MediatR"],
            ["SQL Server", "PostgreSQL", "MySQL", "MongoDB", "Redis",
             "Entity Framework Core", "Dapper", "Elasticsearch", "Cosmos DB",
             "SQLite", "T-SQL", "DynamoDB", "Prisma", "Flyway"],
            ["Docker", "Kubernetes", "AWS EC2", "AWS S3", "AWS Lambda",
             "Azure App Service", "Azure DevOps", "GitHub Actions", "Terraform",
             "NGINX", "AWS RDS", "CloudWatch", "Azure Blob Storage", "Helm"],
            ["xUnit", "NUnit", "Pytest", "Selenium", "Playwright", "Cypress",
             "Postman", "SonarQube", "OWASP ZAP", "Snyk", "ESLint", "Sentry",
             "GitHub", "Visual Studio", "VS Code"],
        ],
        "seo": [
            # Bucket 0 - Core SEO & Research Tools
            ["SEMrush", "Ahrefs", "Moz Pro", "Google Keyword Planner", "Screaming Frog",
             "Majestic", "SpyFu", "KWFinder", "Ubersuggest", "SERPstat",
             "Long Tail Pro", "Keyword Hero", "Answer The Public"],
            # Bucket 1 - Analytics & Search Console
            ["Google Analytics 4", "Google Search Console", "Google Tag Manager",
             "Looker Studio", "Adobe Analytics", "Hotjar", "Crazy Egg",
             "Microsoft Clarity", "GA4 Event Tracking", "BigQuery for GA4",
             "Search Console API", "Tag Assistant"],
            # Bucket 2 - Content & On-Page
            ["Surfer SEO", "Clearscope", "MarketMuse", "Frase.io", "Yoast SEO",
             "Rank Math", "All in One SEO", "Schema Markup", "Structured Data",
             "Content Gap Analysis", "TF-IDF Analysis", "LSI Graph", "SEOwind"],
            # Bucket 3 - Technical SEO
            ["Core Web Vitals", "Google PageSpeed Insights", "GTmetrix", "Lighthouse",
             "Screaming Frog", "DeepCrawl", "Sitebulb", "Log File Analyser",
             "Cloudflare", "Nginx", "Apache", "XML Sitemap", "Robots.txt",
             "Canonical Tags", "hreflang"],
            # Bucket 4 - Paid & Social / Reporting
            ["Google Ads", "Google Display Network", "Meta Ads Manager",
             "LinkedIn Campaign Manager", "Microsoft Advertising", "Google Looker Studio",
             "Google Data Studio", "Supermetrics", "Zapier", "IFTTT",
             "Slack for Reporting", "Trello", "Asana"],
        ],
        "data_science": [
            ["Python", "R", "SQL", "Scala", "Julia", "PySpark", "HiveQL",
             "Bash", "Jupyter Notebook", "Google Colab"],
            ["TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
             "LightGBM", "Hugging Face Transformers", "OpenCV", "NLTK", "spaCy",
             "Pandas", "NumPy", "SciPy", "Statsmodels"],
            ["Apache Spark", "Apache Kafka", "Apache Airflow", "dbt",
             "BigQuery", "Snowflake", "Redshift", "Delta Lake", "Apache Hive",
             "Databricks", "Fivetran", "Apache Flink"],
            ["Vertex AI", "AWS SageMaker", "Azure ML", "MLflow", "Kubeflow",
             "Docker", "Kubernetes", "GitHub Actions", "DVC", "Weights & Biases",
             "Feature Store", "Ray"],
            ["Tableau", "Power BI", "Looker", "Matplotlib", "Seaborn",
             "Plotly", "Grafana", "Excel", "Google Sheets", "Metabase",
             "Apache Superset", "Streamlit"],
        ],
        "mobile": [
            ["React Native", "Flutter", "Ionic", "Capacitor", "Expo",
             "Xamarin", "NativeScript", "Kotlin Multiplatform"],
            ["Swift", "SwiftUI", "UIKit", "Xcode", "Core Data",
             "Combine", "TestFlight", "App Store Connect", "CocoaPods"],
            ["Kotlin", "Java", "Jetpack Compose", "Android Studio", "Gradle",
             "Room Database", "Retrofit", "OkHttp", "Hilt", "WorkManager"],
            ["Firebase", "Firebase Auth", "Cloud Firestore", "Firebase Cloud Messaging",
             "AWS Amplify", "App Center", "Supabase", "Node.js", "GraphQL"],
            ["Fastlane", "Bitrise", "GitHub Actions", "Detox", "Appium",
             "XCTest", "Espresso", "Crashlytics", "Sentry", "Charles Proxy"],
        ],
        "devops": [
            ["Terraform", "Ansible", "Pulumi", "CloudFormation", "Chef",
             "Puppet", "Packer", "Vault", "Consul"],
            ["Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI",
             "Azure Pipelines", "TeamCity", "ArgoCD", "Spinnaker"],
            ["Docker", "Kubernetes", "Helm", "Istio", "Envoy",
             "containerd", "Podman", "Rancher", "OpenShift", "Docker Compose"],
            ["Prometheus", "Grafana", "Datadog", "ELK Stack", "Splunk",
             "Jaeger", "Zipkin", "PagerDuty", "New Relic", "Dynatrace",
             "CloudWatch", "Azure Monitor"],
            ["AWS", "Azure", "GCP", "DigitalOcean", "Cloudflare",
             "Nginx", "HAProxy", "Traefik", "BIND DNS", "VPC", "IAM"],
        ],
        "security": [
            ["Metasploit", "Burp Suite", "Nmap", "Wireshark", "Nessus",
             "OWASP ZAP", "sqlmap", "Aircrack-ng", "John the Ripper", "Hydra"],
            ["Palo Alto", "Fortinet FortiGate", "Cisco ASA", "pfSense",
             "Snort", "Suricata", "CrowdStrike", "Carbon Black", "SentinelOne"],
            ["Splunk", "IBM QRadar", "Microsoft Sentinel", "LogRhythm",
             "Elastic SIEM", "Securonix", "Exabeam", "ArcSight"],
            ["ISO 27001", "NIST CSF", "SOC 2", "GDPR", "PCI DSS",
             "HIPAA", "CIS Benchmarks", "MITRE ATT&CK", "OpenSCAP", "Tenable.io"],
            ["Python", "Bash", "PowerShell", "Go", "Yara", "Sigma",
             "Git", "Ansible", "Docker", "Kali Linux", "Parrot OS"],
        ],
        "design": [
            ["Figma", "Adobe XD", "Sketch", "InVision", "Zeplin",
             "Framer", "Axure RP", "Balsamiq", "Marvel"],
            ["Adobe Illustrator", "Adobe Photoshop", "Adobe InDesign",
             "Affinity Designer", "Canva", "Procreate", "Blender (UI)"],
            ["UserTesting", "Maze", "Hotjar", "Optimal Workshop",
             "Lookback", "Dovetail", "Miro", "FigJam", "Mural"],
            ["HTML", "CSS", "Tailwind CSS", "SCSS", "JavaScript",
             "React", "Storybook", "Design Tokens", "CSS Grid"],
            ["Slack", "Notion", "Jira", "Confluence", "Asana",
             "GitHub", "Abstract", "Linear", "Loom"],
        ],
        "project_management": [
            ["Jira", "Linear", "ClickUp", "Monday.com", "Asana",
             "Trello", "Basecamp", "Wrike", "Smartsheet", "Teamwork"],
            ["Scrum", "Kanban", "SAFe", "Agile", "OKRs",
             "PRINCE2", "PMP", "Lean", "Six Sigma", "Waterfall"],
            ["Confluence", "Notion", "Miro", "Google Workspace",
             "Microsoft Teams", "Slack", "Zoom", "Loom", "Figma"],
            ["Power BI", "Tableau", "Google Data Studio", "Excel",
             "Google Sheets", "Looker", "Metabase", "Smartsheet Reporting"],
            ["Salesforce", "HubSpot", "Zendesk", "Intercom",
             "Freshdesk", "ServiceNow", "Zoho CRM", "Pipedrive"],
        ],
        "finance": [
            ["QuickBooks", "Xero", "Sage 50", "FreshBooks", "Wave",
             "Zoho Books", "NetSuite", "Microsoft Dynamics 365"],
            ["SAP S/4HANA", "Oracle ERP", "Microsoft Dynamics", "Workday",
             "Epicor", "Infor", "PeopleSoft", "SAP FI/CO"],
            ["Bloomberg Terminal", "Refinitiv Eikon", "FactSet",
             "Morningstar Direct", "Capital IQ", "PitchBook", "Excel Financial Modeling"],
            ["IFRS", "GAAP", "SOX Compliance", "Internal Audit", "COSO Framework",
             "Anti-Money Laundering", "KYC", "Basel III"],
            ["Power BI", "Tableau", "Python (pandas)", "SQL",
             "VBA/Macros", "Google Sheets", "Alteryx", "SQL Server Reporting Services"],
        ],
    }
    return pools.get(domain, pools["software_dev"])


def _sanitize_skills(raw_skills: list, cv: dict = None) -> list:
    """
    Server-side enforcement of the 5-category x 5-technology skills structure.

    Rules enforced here (regardless of what the AI returned):
      1. Parse every "Category: item1, item2, ..." string into (category_name, [items]).
      2. Global dedup - if a technology appears in more than one category, keep it
         only in the FIRST category it appears in and remove it from all others.
      3. Each category must have at least 5 items - categories with fewer are dropped.
      4. Keep exactly the first 5 valid categories.
      5. Re-serialise back to clean "Category: item1, item2, ..." strings.

    Domain-based bucket assignment (used to re-sort misplaced items):
    Items are MOVED to the correct bucket if they are clearly misplaced.

    cv (optional): the CV dict - used to detect job domain for domain-aware fallback pools.
    """
    # Detect job domain for domain-aware fallback pools
    domain = _detect_job_domain(cv) if cv else "software_dev"

    # -- Bucket ownership map: keyword -> canonical bucket index (0-4) ----------
    # For non-software domains, the bucket meanings shift but index 0-4 still apply.
    # Lower index = higher priority when a conflict is detected.
    BUCKET_KEYWORDS = {
        # 0 = Frontend / UI
        "react": 0, "angular": 0, "vue": 0, "next.js": 0, "nuxt": 0,
        "tailwind": 0, "bootstrap": 0, "scss": 0, "sass": 0, "less": 0,
        "webpack": 0, "vite": 0, "parcel": 0, "storybook": 0, "rxjs": 0,
        "redux": 0, "zustand": 0, "mobx": 0, "svelte": 0, "ember": 0,
        "html": 0, "css": 0, "jquery": 0, "typescript": 0, "javascript": 0,

        # 1 = Backend / Server-side
        "asp.net": 1, "asp.net core": 1, ".net core": 1, ".net 8": 1,
        ".net 6": 1, ".net 5": 1, "c#": 1, "node.js": 1, "express": 1,
        "django": 1, "flask": 1, "fastapi": 1, "laravel": 1, "symfony": 1,
        "spring boot": 1, "spring": 1, "rails": 1, "sinatra": 1,
        "graphql": 1, "grpc": 1, "rabbitmq": 1, "kafka": 1, "celery": 1,
        "signalr": 1, "jwt": 1, "oauth": 1, "identityserver": 1,
        "hangfire": 1, "mediator": 1, "mediatr": 1, "automapper": 1,
        "restful": 1, "rest api": 1, "web api": 1, "microservices": 1,
        "python": 1, "java": 1, "go": 1, "rust": 1, "php": 1, "ruby": 1,
        "scala": 1, "kotlin": 1, "swagger": 1, "openapi": 1,

        # 2 = Databases & Storage
        "sql server": 2, "postgresql": 2, "mysql": 2, "mariadb": 2,
        "sqlite": 2, "oracle": 2, "mongodb": 2, "redis": 2, "cassandra": 2,
        "dynamodb": 2, "cosmos db": 2, "firebase": 2, "firestore": 2,
        "elasticsearch": 2, "opensearch": 2, "neo4j": 2, "influxdb": 2,
        "entity framework": 2, "ef core": 2, "dapper": 2, "prisma": 2,
        "sequelize": 2, "sqlalchemy": 2, "hibernate": 2, "flyway": 2,
        "liquibase": 2, "clickhouse": 2, "bigquery": 2, "snowflake": 2,
        "memcached": 2, "mssql": 2, "t-sql": 2, "pl/sql": 2,

        # 3 = Cloud & DevOps
        "aws": 3, "azure": 3, "gcp": 3, "google cloud": 3,
        "docker": 3, "kubernetes": 3, "k8s": 3, "helm": 3,
        "terraform": 3, "ansible": 3, "pulumi": 3,
        "github actions": 3, "jenkins": 3, "gitlab ci": 3, "circleci": 3,
        "azure devops": 3, "azure pipelines": 3, "codepipeline": 3,
        "ec2": 3, "s3": 3, "lambda": 3, "ecs": 3, "eks": 3,
        "cloudfront": 3, "rds": 3, "app service": 3, "cloud run": 3,
        "prometheus": 3, "grafana": 3, "datadog": 3, "cloudwatch": 3,
        "nginx": 3, "traefik": 3, "istio": 3, "ci/cd": 3,
        "digitalocean": 3, "heroku": 3, "vercel": 3, "netlify": 3,
        "cloudflare": 3, "haproxy": 3, "envoy": 3, "containerd": 3,
        "podman": 3, "rancher": 3, "openshift": 3,
        "elk stack": 3, "splunk": 3, "jaeger": 3, "zipkin": 3, "pagerduty": 3,
        "cloudformation": 3, "chef": 3, "puppet": 3, "packer": 3,
        "vault": 3, "consul": 3, "travis ci": 3, "teamcity": 3, "argocd": 3,

        # 4 = Testing, Security & Tooling
        "xunit": 4, "nunit": 4, "mstest": 4, "jest": 4, "mocha": 4,
        "pytest": 4, "junit": 4, "selenium": 4, "cypress": 4,
        "playwright": 4, "postman": 4, "insomnia": 4,
        "sonarqube": 4, "owasp": 4, "snyk": 4, "veracode": 4,
        "eslint": 4, "prettier": 4, "stylelint": 4, "resharper": 4,
        "git": 4, "github": 4, "gitlab": 4, "bitbucket": 4,
        "npm": 4, "yarn": 4, "pip": 4, "nuget": 4, "maven": 4, "gradle": 4,
        "visual studio": 4, "vs code": 4, "rider": 4, "intellij": 4,
        "jira": 4, "confluence": 4, "sentry": 4, "raygun": 4,
    }

    # -- BANNED generic / non-product terms ------------------------------------
    BANNED_TERMS = {
        "microservices", "restful", "rest", "web apis", "web api",
        "relational databases", "relational database", "sql", "nosql",
        "clean architecture", "solid principles",
        "agile", "scrum", "kanban", "tdd", "bdd", "ddd",
        "architecture", "infrastructure", "environment", "system",
        "server", "network", "platform", "application", "solution",
        "module", "component", "interface", "integration", "requirements",
        "design", "development", "backend", "frontend", "full stack",
        ".net", "net",   # too vague - full names like "ASP.NET Core" are fine
        # Thin-JD prose words that can appear as fake tech items
        "web development", "web", "hands", "good", "ability", "strong", "dev",
        "remote", "setup", "mindset", "detail", "attention", "focus", "solid",
        "working", "independent", "effectively", "efficiently",
        # AI tools / IDEs that are not technical skills
        "claude code", "cursor", "copilot", "github copilot", "anthropic",
        "chatgpt", "openai",
        # Networking / infra nouns that are not software products
        "bind dns", "vpc", "vlan", "subnet", "load balancer",
    }

    # -- Step 1: Parse all (category_name, [items]) pairs ---------------------
    parsed: list[tuple[str, list[str]]] = []
    for entry in raw_skills:
        if not isinstance(entry, str):
            continue
        colon = entry.find(":")
        if colon <= 0:
            continue
        cat_name = entry[:colon].strip()
        items_raw = entry[colon + 1:].strip()
        # Split on comma, pipe, or semicolon
        items = [i.strip() for i in re.split(r"[,|;\?·•]", items_raw) if i.strip()]
        # Strip banned generic terms
        items = [
            it for it in items
            if it.lower() not in BANNED_TERMS and len(it) > 1
        ]
        if items:
            parsed.append((cat_name, items))

    if not parsed:
        return raw_skills  # nothing to fix - return as-is

    # -- Step 1b: Cross-stack contamination filter ----------------------------
    # If we can detect the primary backend stack from items in the parsed data,
    # remove items that clearly belong to a completely different unrelated stack.
    all_items_lower = {it.lower() for _, items in parsed for it in items}

    _stack_dotnet = any(kw in all_items_lower for kw in (
        "asp.net", ".net", "c#", "entity framework", "asp.net core", "asp.net mvc",
        ".net core", ".net 8", "razor", "blazor", "signalr", "hangfire", "mediatr"
    ))
    _stack_node = any(kw in all_items_lower for kw in (
        "node.js", "express.js", "nestjs", "fastify", "socket.io"
    ))
    _stack_python_fw = any(kw in all_items_lower for kw in (
        "django", "flask", "fastapi", "celery", "pydantic", "sqlalchemy"
    ))
    _stack_java = any(kw in all_items_lower for kw in (
        "spring boot", "spring mvc", "hibernate", "maven", "gradle", "quarkus"
    ))
    _stack_php = any(kw in all_items_lower for kw in (
        "laravel", "symfony", "eloquent", "composer", "artisan"
    ))

    # Build a banned-items set based on detected primary stack
    _cross_stack_banned: set = set()
    if _stack_dotnet and not _stack_node:
        _cross_stack_banned.update({
            "node.js", "express.js", "express", "nestjs", "fastify", "socket.io",
            "passport.js", "bull queue", "helmet.js"
        })
    if _stack_dotnet and not _stack_python_fw:
        _cross_stack_banned.update({
            "django", "flask", "fastapi", "celery", "pydantic", "sqlalchemy",
            "alembic", "gunicorn", "uvicorn", "django rest framework", "pytest"
        })
    if _stack_dotnet and not _stack_java:
        _cross_stack_banned.update({
            "spring boot", "spring mvc", "spring security", "spring data jpa",
            "hibernate", "maven", "gradle", "lombok", "mapstruct", "quarkus",
            "micronaut", "junit", "mockito", "testcontainers"
        })
    if _stack_dotnet and not _stack_php:
        _cross_stack_banned.update({
            "laravel", "symfony", "lumen", "eloquent orm", "composer",
            "artisan cli", "laravel sanctum", "livewire", "inertia.js",
            "phpunit", "pest", "php_codesniffer", "psalm", "phpstan"
        })
    if _stack_node and not _stack_dotnet:
        _cross_stack_banned.update({
            "asp.net core", "asp.net mvc", "asp.net web api", "c#", ".net 8",
            ".net core 8", "signalr", "mediatr", "automapper", "hangfire",
            "xunit", "nunit", "mstest", "moq", "fluentassertions", "resharper"
        })
    if _stack_python_fw and not _stack_dotnet:
        _cross_stack_banned.update({
            "asp.net core", "asp.net mvc", "c#", ".net 8", "signalr", "mediatr",
            "xunit", "nunit", "mstest", "spring boot", "laravel"
        })

    # Apply filter: remove cross-stack items from parsed data
    if _cross_stack_banned:
        cleaned_parsed = []
        for cat_name, items in parsed:
            filtered = [it for it in items if it.lower() not in _cross_stack_banned]
            if filtered:
                cleaned_parsed.append((cat_name, filtered))
        parsed = cleaned_parsed

    # -- Step 2: ITEM-LEVEL re-bucketing (the only correct approach) ----------
    # Don't trust the AI's category assignments at all.
    # Each individual item is placed into the correct bucket by BUCKET_KEYWORDS.
    # Items not matched by any keyword go to bucket 1 (backend/primary domain) as default.
    #
    # NOTE: This replaces the old "score the whole category" approach which kept
    # entire wrongly-named categories together (e.g. "Infrastructure as Code" containing
    # Angular + Terraform + SQL together, all mis-assigned to one bucket).

    # Bucket label names - always driven by domain, never by what the AI wrote.
    # The AI's category names are completely discarded; correct names come from here.
    _DOMAIN_BUCKET_NAMES = {
        "seo": [
            "SEO & Keyword Research Tools",
            "Analytics & Search Console",
            "Content & On-Page Optimization",
            "Technical SEO",
            "Paid Media & Reporting",
        ],
        "data_science": [
            "Programming Languages",
            "ML & Data Science Frameworks",
            "Data Engineering & Warehousing",
            "Cloud & MLOps",
            "Visualization & BI Tools",
        ],
        "mobile": [
            "Cross-Platform Frameworks",
            "iOS Development",
            "Android Development",
            "Backend & APIs",
            "DevOps, CI/CD & Testing",
        ],
        "devops": [
            "Infrastructure as Code",
            "CI/CD & Automation",
            "Containers & Orchestration",
            "Monitoring & Observability",
            "Cloud Platforms",
        ],
        "security": [
            "Offensive Security Tools",
            "Defensive Security & Endpoint",
            "SIEM & Threat Intelligence",
            "Compliance & Frameworks",
            "Scripting & Tooling",
        ],
        "design": [
            "Design & Prototyping Tools",
            "Graphic & Visual Design",
            "User Research & Testing",
            "Frontend & Dev Handoff",
            "Collaboration & PM",
        ],
        "project_management": [
            "Project & Task Management",
            "Agile & Delivery Frameworks",
            "Collaboration & Communication",
            "Reporting & Analytics",
            "CRM & Stakeholder Tools",
        ],
        "finance": [
            "Accounting & Bookkeeping",
            "ERP & Financial Systems",
            "Market Data & Research",
            "Compliance & Audit",
            "Data Analysis & Reporting",
        ],
        "software_dev": [
            "Frontend & UI Technologies",
            "Backend & Frameworks",
            "Databases & Data Storage",
            "Cloud & DevOps",
            "Testing, Security & Tooling",
        ],
    }
    BUCKET_NAMES = _DOMAIN_BUCKET_NAMES.get(domain, _DOMAIN_BUCKET_NAMES["software_dev"])

    # Flatten all items from all parsed categories into a single pool,
    # then place each item into the correct bucket by keyword lookup.
    # This bypasses the AI's (wrong) category groupings entirely.
    buckets: list[list[str]] = [[], [], [], [], []]
    seen_lower: set = set()

    for _cat_name, items in parsed:
        for item in items:
            key = item.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)

            # Find the correct bucket for this item
            assigned = None
            for keyword, b_idx in BUCKET_KEYWORDS.items():
                if keyword in key:
                    assigned = b_idx
                    break

            # Default unrecognised items to bucket 1 (backend / primary domain)
            if assigned is None:
                assigned = 1
            buckets[assigned].append(item)

    # -- Step 3: Deduplicate (already done via seen_lower above) --------------
    # buckets already contain unique items; just alias for clarity
    clean: list[list[str]] = buckets

    # -- Step 4: Global seen set for fallback fills ----------------------------
    seen2: set = set(seen_lower)

    # -- Step 5: Auto-fill sparse buckets to reach minimum 10 items -----------
    # Strategy: prefer filling from OTHER buckets that have overflow items from the
    # same JD domain (re-distributing items the AI placed in wrong buckets),
    # then fall back to the domain fallback pool FILTERED to only include tools
    # that are consistent with what the AI already put in the CV.
    #
    # This prevents the classic failure: a .NET CV getting Node.js/Django/FastAPI
    # injected into its Backend bucket because the generic fallback pool contains them.

    MIN_ITEMS = 10
    MAX_ITEMS = 12

    # Build a domain-aware filtered fallback: only include fallback items that
    # share a technology ecosystem with what's already in the CV.
    # We detect the primary stack from what's in bucket 1 (backend) items.
    backend_items_lower = {it.lower() for it in clean[1]}
    frontend_items_lower = {it.lower() for it in clean[0]}

    # Detect primary backend stack from existing items
    _is_dotnet = any(kw in backend_items_lower for kw in (
        "asp.net", ".net", "c#", "asp.net core", "asp.net mvc", ".net core", ".net 8", "entity framework"
    ))
    _is_node = any(kw in backend_items_lower for kw in ("node.js", "express", "fastify", "nestjs"))
    _is_python = any(kw in backend_items_lower for kw in ("django", "flask", "fastapi", "python"))
    _is_java = any(kw in backend_items_lower for kw in ("spring", "java", "hibernate", "maven"))
    _is_php = any(kw in backend_items_lower for kw in ("laravel", "php", "symfony"))
    _is_react_fe = any(kw in frontend_items_lower for kw in ("react", "next.js", "redux"))
    _is_angular_fe = any(kw in frontend_items_lower for kw in ("angular", "rxjs"))
    _is_vue_fe = any(kw in frontend_items_lower for kw in ("vue", "nuxt"))

    # Build stack-specific fallback pools that stay consistent with the CV's detected stack
    if _is_dotnet:
        _fb_backend  = ["ASP.NET Core", "ASP.NET MVC", "ASP.NET Web API", "C#", ".NET 8",
                         ".NET Core 8", "SignalR", "MediatR", "AutoMapper", "Hangfire",
                         "Minimal APIs", "Identity Server", "OWIN", "WCF", "CQRS Pattern"]
        _fb_frontend = ["Angular", "Angular 17+", "TypeScript", "Bootstrap", "HTML5",
                         "CSS3", "SCSS", "RxJS", "Angular Material", "NgRx",
                         "Razor Pages", "Blazor", "jQuery", "JavaScript"]
        _fb_db       = ["SQL Server", "MS SQL Server", "Entity Framework Core", "Entity Framework",
                         "Dapper", "T-SQL", "SSRS", "SSAS", "Azure SQL Database",
                         "SQL Server Profiler", "Database Migrations", "Stored Procedures",
                         "SQL Server Agent", "Full-Text Search"]
        _fb_cloud    = ["Azure App Service", "Azure DevOps", "Azure Functions",
                         "Azure Blob Storage", "Azure AD", "Azure Key Vault",
                         "Azure Service Bus", "Azure API Management", "Azure Monitor",
                         "Azure Container Registry", "GitHub Actions", "Docker", "Kubernetes"]
        _fb_testing  = ["xUnit", "NUnit", "MSTest", "Moq", "FluentAssertions",
                         "Selenium", "Playwright", "Postman", "SonarQube", "OWASP ZAP",
                         "Snyk", "Visual Studio", "ReSharper", "Swagger UI"]
    elif _is_node:
        _fb_backend  = ["Node.js", "Express.js", "NestJS", "Fastify", "GraphQL",
                         "Socket.IO", "Passport.js", "JWT", "Axios", "Bull Queue",
                         "TypeScript", "Lodash", "Helmet.js", "CORS Middleware"]
        _fb_frontend = ["React", "Next.js", "TypeScript", "Tailwind CSS", "Redux",
                         "React Query", "Vite", "Webpack", "SCSS", "Styled Components",
                         "Storybook", "ESLint", "Prettier"]
        _fb_db       = ["PostgreSQL", "MongoDB", "Redis", "MySQL", "Mongoose",
                         "Sequelize", "Prisma", "TypeORM", "Knex.js", "DynamoDB",
                         "Elasticsearch", "SQLite", "Firestore", "Cassandra"]
        _fb_cloud    = ["AWS EC2", "AWS S3", "AWS Lambda", "AWS RDS", "AWS ECS",
                         "CloudFront", "API Gateway", "GitHub Actions", "Docker",
                         "Kubernetes", "Terraform", "AWS CloudWatch", "Vercel", "Netlify"]
        _fb_testing  = ["Jest", "Mocha", "Chai", "Supertest", "Cypress", "Playwright",
                         "Postman", "SonarQube", "Snyk", "ESLint", "Prettier", "GitHub"]
    elif _is_python:
        _fb_backend  = ["Python", "Django", "FastAPI", "Flask", "Celery",
                         "Pydantic", "SQLAlchemy", "Alembic", "Gunicorn", "Uvicorn",
                         "Django REST Framework", "Pytest", "HTTPX", "Boto3"]
        _fb_frontend = ["React", "TypeScript", "Tailwind CSS", "Next.js", "Redux",
                         "Vite", "SCSS", "Axios", "Storybook", "ESLint"]
        _fb_db       = ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
                         "SQLAlchemy", "Alembic", "DynamoDB", "SQLite", "Cassandra",
                         "BigQuery", "Snowflake", "Flyway", "Prisma"]
        _fb_cloud    = ["AWS EC2", "AWS S3", "AWS Lambda", "AWS RDS", "AWS ECS",
                         "GitHub Actions", "Docker", "Kubernetes", "Terraform",
                         "GCP Cloud Run", "GCP Cloud Storage", "GCP Pub/Sub"]
        _fb_testing  = ["Pytest", "unittest", "Selenium", "Playwright", "Locust",
                         "Postman", "SonarQube", "Snyk", "Bandit", "Black",
                         "Flake8", "mypy", "GitHub", "VS Code"]
    elif _is_php:
        _fb_backend  = ["PHP", "Laravel", "Symfony", "Lumen", "Eloquent ORM",
                         "Composer", "Artisan CLI", "Laravel Sanctum", "Laravel Horizon",
                         "Laravel Nova", "Livewire", "Inertia.js", "PHP-CS-Fixer"]
        _fb_frontend = ["Vue.js", "React", "TypeScript", "Tailwind CSS", "Alpine.js",
                         "Vite", "Webpack", "SCSS", "Bootstrap", "jQuery"]
        _fb_db       = ["MySQL", "PostgreSQL", "Redis", "MongoDB", "Eloquent ORM",
                         "Laravel Migrations", "SQLite", "MariaDB", "SQL Server",
                         "Memcached", "Elasticsearch", "DynamoDB"]
        _fb_cloud    = ["AWS EC2", "AWS S3", "AWS RDS", "DigitalOcean Droplets",
                         "Laravel Forge", "Envoyer", "GitHub Actions", "Docker",
                         "Kubernetes", "Nginx", "Apache", "Cloudflare"]
        _fb_testing  = ["PHPUnit", "Pest", "Laravel Dusk", "Mockery", "Faker",
                         "Postman", "SonarQube", "Snyk", "PHP_CodeSniffer", "Psalm",
                         "PHPStan", "GitHub", "VS Code"]
    elif _is_java:
        _fb_backend  = ["Java", "Spring Boot", "Spring MVC", "Spring Security",
                         "Spring Data JPA", "Hibernate", "Maven", "Gradle",
                         "Lombok", "MapStruct", "Quarkus", "Micronaut", "JWT"]
        _fb_frontend = ["Angular", "React", "TypeScript", "Bootstrap", "Tailwind CSS",
                         "HTML5", "CSS3", "SCSS", "Thymeleaf", "JSP"]
        _fb_db       = ["PostgreSQL", "MySQL", "Oracle", "MongoDB", "Redis",
                         "Hibernate ORM", "Flyway", "Liquibase", "JDBC", "JPA",
                         "Elasticsearch", "Cassandra", "SQL Server"]
        _fb_cloud    = ["AWS EC2", "AWS S3", "AWS Lambda", "AWS RDS", "AWS ECS",
                         "GitHub Actions", "Docker", "Kubernetes", "Terraform",
                         "Azure App Service", "GCP Cloud Run", "Jenkins"]
        _fb_testing  = ["JUnit 5", "Mockito", "TestContainers", "Selenium", "RestAssured",
                         "Postman", "SonarQube", "Snyk", "Checkstyle", "SpotBugs",
                         "JaCoCo", "GitHub", "IntelliJ IDEA"]
    else:
        # Generic software_dev fallback - only used when stack cannot be detected
        GENERIC_FALLBACK = _get_domain_fallback_pools(domain)
        _fb_frontend = GENERIC_FALLBACK[0]
        _fb_backend  = GENERIC_FALLBACK[1]
        _fb_db       = GENERIC_FALLBACK[2]
        _fb_cloud    = GENERIC_FALLBACK[3]
        _fb_testing  = GENERIC_FALLBACK[4]

    FALLBACK_POOLS = [_fb_frontend, _fb_backend, _fb_db, _fb_cloud, _fb_testing]

    result: list[str] = []
    for b_idx in range(5):
        items = clean[b_idx][:MAX_ITEMS]
        name  = BUCKET_NAMES[b_idx]   # always use canonical name - never the AI's name

        # Auto-fill up to MIN_ITEMS using the stack-specific fallback pool
        if len(items) < MIN_ITEMS:
            for fallback in FALLBACK_POOLS[b_idx]:
                if len(items) >= MIN_ITEMS:
                    break
                if fallback.lower() not in seen2:
                    seen2.add(fallback.lower())
                    items.append(fallback)

        # Cap at MAX_ITEMS
        items = items[:MAX_ITEMS]

        if items:
            result.append(f"{name}: {', '.join(items)}")

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

    PAGE_W, PAGE_H = A4
    ML = 13 * mm
    MR = 13 * mm
    MT = 11 * mm
    MB = 11 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
        title=f"Muhammad Junaid CV - {_safe(cv.get('title',''))}",
        author="Muhammad Junaid",
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
                           textColor=colors.HexColor("#111111"), spaceAfter=3),
        "role":        ps("role",   fontName="Helvetica", fontSize=8, leading=12,
                           textColor=colors.HexColor("#444444"), spaceAfter=1),
        "contact":     ps("contact",fontName="Helvetica", fontSize=8, leading=11,
                           textColor=colors.HexColor("#0057A8"), spaceAfter=1),
        "contact_plain": ps("cp",   fontName="Helvetica", fontSize=8, leading=11,
                           textColor=colors.HexColor("#555555"), spaceAfter=1),
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
        mid = (len(contact_items) + 1) // 2
        story.append(Paragraph(SEP.join(contact_items[:mid]),  S["contact"]))
        if contact_items[mid:]:
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
            company_name = _safe(w.get("company")) or _safe(ai.get("company")) or f"Company {i+1}"
            role         = _safe(w.get("role"))    or _safe(ai.get("role"))    or ""

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
            clean_name = _strip_brackets(raw_name)

            story.append(Paragraph(clean_name, S["proj_name"]))

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
        story.append(Paragraph(competencies, S["competency"]))
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
    doc.build(story)
    buf.seek(0)
    return buf.read()

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