"""
CV Builder AI - FastAPI Backend v12
Port: 8001  |  Start: uvicorn main:app --host 0.0.0.0 --port 8001

Providers: Groq, Cerebras, Gemini, Qwen (DashScope), Ollama
100% Dynamic - No hardcoded technologies, no static fallbacks, no predefined categories
Everything derived from Job Description + Job Title only
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
import httpx, json, re, math, io, asyncio, secrets, string, logging
from datetime import date, datetime, timedelta

# ── Structured logger — outputs to uvicorn/server console ────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("cvai")

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
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models"
OLLAMA_URL   = "http://localhost:11434"
# Qwen / Alibaba DashScope — OpenAI-compatible international endpoint (Singapore region)
# Sign up: https://bailian.console.aliyun.com → API Keys → Create API Key (sk-...)
# Free tier: 1M input + 1M output tokens for 90 days on new accounts
QWEN_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

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
    """Always returns a clean number string WITHOUT trailing + (e.g. '5', '3', '6').
    The + sign is added only at display time to avoid double-+ bugs."""
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
    if y >= 5: return "5"
    elif y >= 4: return "4"
    elif y >= 3: return "3"
    elif y >= 2: return "2"
    else: return "1"

def _build_dynamic_companies(years_exp: str, num_companies: int = 0,
                              profile_work: list = None) -> list:
    """
    Build a list of company timeline dicts: [{"name":…, "start":…, "end":…}, …]

    DATE CALCULATION — fully dynamic, never hardcoded:
      1. Convert years_exp to total_months  (e.g. "2" → 24 months).
      2. Anchor to date.today().
      3. Walk backwards from today, assigning each company an equal share of months.
      4. Result: company[0] ends "Present", company[-1] starts exactly
         total_months before today.

      Example (today = May 2026, years_exp = "2"):
        total_months = 24, num_cos = 2, each = 12
        company[0]: May 2025 → Present   (12 months)
        company[1]: May 2024 → May 2025  (12 months)
        Career start = May 2024 = today − 24 months  ✓

    Priority / what profile_work contributes:
      • profile_work entries            → company NAMES only. Dates in profile entries
        are always ignored so that years_exp is the single source of truth for spans.
        This guarantees "2 years selected → exactly 24 months shown", regardless of
        what dates the user previously stored in their profile.
      • years_exp (UI input)            → total duration + company count + date ranges.
      • Neither provided                → compute duration from CANDIDATE_COMPANIES spans.

    Company names: profile_work names → generic "Company N" (never hardcoded).
    """
    today = date.today()

    def fmt(d: date) -> str:
        return f"{_month_name(d.month)} {d.year}"

    # ── Step 1: collect company names from profile (dates are ALWAYS recalculated) ──
    # Profile work entries supply company names only.
    # Their from/to dates are intentionally ignored here — dates are always
    # derived from years_exp so the selected experience duration is always honoured.
    # Example: user selects "2 years" but profile has "September 2025 → Present"
    # (only 8 months) — without this rule the wrong span would appear on the CV.
    if profile_work:
        profile_names = [
            (w.get("company") or "").strip() or f"Company {i + 1}"
            for i, w in enumerate(profile_work)
        ]
    else:
        profile_names = None

    # ── Step 2: resolve numeric years from UI input ───────────────────────────
    n: float | None = None
    if years_exp:
        raw = years_exp.strip().replace("+", "").strip()
        try:
            n = float(raw)
        except ValueError:
            n = None

    if n is None:
        # Fallback: compute from CANDIDATE_COMPANIES actual date spans
        total_months_fb = 0
        for co in CANDIDATE_COMPANIES:
            try:
                s = _parse_month_year(co["start"])
                e = _parse_month_year(co["end"])
                total_months_fb += _months_between(s, e)
            except Exception:
                pass
        n = total_months_fb / 12 if total_months_fb else 3.0

    # Total career duration in whole months — e.g. "2" → 24, "1.5" → 18
    # Add 2 extra months so the experience always looks slightly more seasoned
    # (e.g. "2 years" → 26 months, career starts ~2 months earlier than round figure).
    # This is applied uniformly for every input so behaviour is fully dynamic.
    total_months = int(round(n * 12)) + 2

    # ── Step 3: determine company count — always driven by years_exp ────────────
    # num_companies (explicit override) takes highest priority.
    # Profile entry count is NOT used — it would let a 1-entry profile force
    # a single company even when the user selects "5 years".
    if num_companies > 0:
        num_cos = num_companies
    elif n <= 1.4:
        num_cos = 1
    elif n <= 2.4:
        num_cos = 2
    else:
        num_cos = 3

    # Guard: at least 1 company, and never more months than we have
    num_cos = max(1, num_cos)

    # ── Step 4: distribute months with natural, increasing progression ──────────
    # Instead of equal splits we use percentage weights that grow from oldest to
    # newest company, mirroring real career patterns (early jobs shorter, later
    # roles progressively longer).
    #
    # Weights: index 0 = most recent company (listed first on CV, longest span),
    #          index k-1 = oldest/earliest company (shortest span).
    # The final (oldest) company always absorbs the true remainder so the total
    # is always mathematically exact.
    #
    # Preset weight tables (most-recent → oldest):
    #   1 company : [1.00]
    #   2 companies: [0.55, 0.45]
    #   3 companies: [0.40, 0.35, 0.25]
    #   4 companies: [0.35, 0.30, 0.22, 0.13]
    #   5+ : geometric decay r=0.80, normalised

    _weight_presets = {
        1: [1.00],
        2: [0.55, 0.45],
        3: [0.40, 0.35, 0.25],
        4: [0.35, 0.30, 0.22, 0.13],
    }
    if num_cos in _weight_presets:
        _weights = _weight_presets[num_cos]
    else:
        # Geometric decay: most recent largest, older progressively smaller
        _r = 0.80
        _raw = [_r ** i for i in range(num_cos)]
        _tw  = sum(_raw)
        _weights = [w / _tw for w in _raw]

    # Convert weights to integer months (floor), then fix rounding in last slot
    spans = [max(1, int(total_months * w)) for w in _weights]
    spans[-1] = max(1, total_months - sum(spans[:-1]))   # exact remainder

    # ── Step 5: walk backwards from today assigning date ranges ──────────────
    result = []
    cursor = today   # marks the *end* boundary for the current company
    for i in range(num_cos):
        span     = spans[i]
        co_start = _subtract_months(cursor, span)
        co_end   = "Present" if i == 0 else fmt(cursor)

        # Name: UI-supplied → generic placeholder (never a hardcoded company name)
        if profile_names and i < len(profile_names):
            name = profile_names[i]
        else:
            name = f"Company {i + 1}"

        result.append({"name": name, "start": fmt(co_start), "end": co_end})
        cursor = co_start   # next company ends where this one started

    return result

def _normalise_edu_entry(e: dict) -> dict:
    """
    Normalise a profile edu entry so server-side code always uses consistent keys.

    The UI stores education notes in the 'note' field (e.g. "Gold Medal, CGPA 3.97/4.0").
    The server previously read 'achievement' and 'cgpa' which don't exist in the UI profile.

    This function:
      • Maps 'note' → 'achievement' (if 'achievement' not already set)
      • Extracts a CGPA pattern from the note and stores it under 'cgpa' (if not set)
      • Leaves all other fields untouched
    """
    if not e:
        return e
    out = dict(e)  # shallow copy — never mutate the original

    # Pull raw note text
    note = (out.get("note") or "").strip()

    # achievement: prefer explicit field, fall back to note
    if not out.get("achievement") and note:
        out["achievement"] = note

    # cgpa: prefer explicit field, then try to extract from note
    if not out.get("cgpa") and note:
        cgpa_match = re.search(r'(?:cgpa|gpa)[:\s]*(\d+\.\d+(?:/\d+(?:\.\d+)?)?)', note, re.IGNORECASE)
        if cgpa_match:
            out["cgpa"] = cgpa_match.group(1).strip()

    return out


def _infer_degree_duration(degree_str: str) -> int:
    """
    Infer typical program duration in years from the degree name.
    Fully dynamic — no hardcoded institution or country assumptions.
    Scans for standard academic level keywords and returns the most
    commonly accepted duration for that level worldwide.

    Returns an integer number of years (default 4 if unrecognised).
    """
    d = (degree_str or "").lower()

    # ── Doctoral / research doctorates (5–6 yrs, use 4 for CV purposes) ──────
    if any(k in d for k in ("phd", "ph.d", "d.phil", "doctor of philosophy",
                             "dphil", "dsc", "d.sc", "doctor of science")):
        return 4

    # ── Professional doctorates & long integrated masters (3–4 yrs post-grad) ─
    if any(k in d for k in ("md", "m.d", "doctor of medicine", "mbbs", "bds",
                             "llb", "l.l.b", "juris doctor", "j.d")):
        return 5

    # ── Masters / postgraduate taught (1–2 yrs) ──────────────────────────────
    if any(k in d for k in ("m.phil", "mphil", "m phil",
                             "master", "msc", "m.sc", "m.s.", " ms ",
                             "mba", "m.b.a", "med", "m.ed",
                             "meng", "m.eng", "mtech", "m.tech",
                             "ma ", "m.a.", "mfa", "m.f.a",
                             "postgrad", "post-grad", "pgd", "pg dip")):
        return 2

    # ── Bachelor / undergraduate (3–4 yrs; default 4) ────────────────────────
    if any(k in d for k in ("bachelor", "bsc", "b.sc", "b.s.",
                             "beng", "b.eng", "btech", "b.tech",
                             "ba ", "b.a.", "bba", "b.b.a",
                             "bcom", "b.com", "bca", "b.ca",
                             "bcs", "b.cs", "bscs", "b.s.c.s",
                             "bsit", "b.s.i.t", "bsee", "bsce",
                             "undergraduate", "honours", "hons")):
        return 4

    # ── Associate / diploma / certificate (1–2 yrs) ──────────────────────────
    if any(k in d for k in ("associate", "diploma", "dip.", "certificate",
                             "hnd", "hnc", "technician", "vocational")):
        return 2

    # ── Default: 4-year program if unrecognised ───────────────────────────────
    return 4


def _build_education_year(years_exp: str, profile_edu: list = None) -> dict:
    """
    Returns {"start": "YYYY", "end": "YYYY"} for the FIRST education entry.

    Priority (per entry):
      1. Both dates explicit in profile_edu[0]  → use exactly as provided.
      2. Only one date provided                  → infer the other using
         _infer_degree_duration() on the degree name — fully dynamic.
      3. years_exp provided (UI input)           → graduation = current_year - years_exp + 1
                                                   duration   = inferred from degree name
      4. No info at all                          → graduation = current_year - 3,
                                                   duration   = 4 years.

    No hardcoded durations — duration is always derived from the degree string.
    """
    today = date.today()

    # ── Priority 1 & 2: profile data ─────────────────────────────────────────
    if profile_edu:
        e      = profile_edu[0]
        degree = (e.get("degree") or "").strip()
        dur    = _infer_degree_duration(degree)
        p_from = str(e.get("from", "") or "").strip()
        p_to   = str(e.get("to",   "") or "").strip()
        if p_from and p_to:
            return {"start": p_from, "end": p_to}
        if p_to and not p_from:
            try:
                end_yr = int(p_to[:4])
                return {"start": str(end_yr - dur), "end": p_to}
            except (ValueError, TypeError):
                pass
        if p_from and not p_to:
            try:
                start_yr = int(p_from[:4])
                return {"start": p_from, "end": str(start_yr + dur)}
            except (ValueError, TypeError):
                pass

    # ── Priority 3: years_exp from UI ────────────────────────────────────────
    if years_exp:
        try:
            n      = int(float(years_exp.strip().replace("+", "")))
            degree = (profile_edu[0].get("degree") or "") if profile_edu else ""
            dur    = _infer_degree_duration(degree)
            grad_year  = today.year - max(n, 0) + 1
            start_year = grad_year - dur
            return {"start": str(start_year), "end": str(grad_year)}
        except (ValueError, TypeError):
            pass

    # ── Priority 4: no info ───────────────────────────────────────────────────
    degree     = (profile_edu[0].get("degree") or "") if profile_edu else ""
    dur        = _infer_degree_duration(degree)
    grad_year  = today.year - 3 + 1
    start_year = grad_year - dur
    return {"start": str(start_year), "end": str(grad_year)}

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
    gemini_keys: Optional[List[str]] = []
    qwen_keys: Optional[List[str]] = []
    ollama_model: Optional[str] = "qwen2.5:7b"
    profile: str = ""
    profile_data: Optional[dict] = None
    static_data: Optional[dict] = None
    company_name: Optional[str] = ""
    company_context: Optional[str] = ""
    ui_template: Optional[str] = "ui1"   # "ui1" | "ui2" | "ui3"

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
    
    # Normalise to a plain integer-like string ("5", "3", etc.) — NO trailing +
    # This is the single source of truth; we add + only in display strings below.
    raw_years = years_exp.replace("+", "").strip()
    try:
        float(raw_years)      # validate it is numeric
    except ValueError:
        raw_years = _calc_total_years("")   # fallback: compute from CANDIDATE_COMPANIES
    
    # Human-readable display — always exactly "N+" (no double +)
    years_display = raw_years + "+"

    # ── Profile data extraction — must happen BEFORE companies/edu are computed ──
    _p_data   = req.profile_data or {}
    _raw_edu  = _p_data.get("edu") or []
    _p_edu_l  = [_normalise_edu_entry(e) for e in _raw_edu] if _raw_edu else []
    _p_edu0   = _p_edu_l[0] if _p_edu_l else {}
    _p_work_l = _p_data.get("work") or []

    companies = _build_dynamic_companies(years_exp, profile_work=_p_work_l or None)
    num_cos = len(companies)
    edu = _build_education_year(years_exp, profile_edu=_p_edu_l or None)
    
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

    # ── Build education block for prompt — one entry per qualification ──────────
    # Uses the same UI-first, auto-sequencing logic as the post-AI merge so the
    # AI sees the exact years that will appear in the final CV.
    import json as _json_mod

    def _resolve_edu_yr_prompt(pe: dict, anchor_start_yr: int = None) -> str:
        _ef  = str(pe.get("from") or "").strip()
        _et  = str(pe.get("to")   or "").strip()
        _deg = (pe.get("degree")  or "").strip()
        _dur = _infer_degree_duration(_deg)
        if _ef and _et:
            return f"{_ef} - {_et}"
        if _et and not _ef:
            try:
                return f"{int(_et[:4]) - _dur} - {_et}"
            except (ValueError, TypeError):
                pass
        if _ef and not _et:
            try:
                return f"{_ef} - {int(_ef[:4]) + _dur}"
            except (ValueError, TypeError):
                pass
        if anchor_start_yr is not None:
            return f"{anchor_start_yr - _dur} - {anchor_start_yr}"
        return f"{edu['start']} - {edu['end']}"

    _edu_entries_for_prompt = []
    _prev_start_p = None
    for _pe in _p_edu_l:
        _e_yr = _resolve_edu_yr_prompt(_pe, anchor_start_yr=_prev_start_p)
        try:
            _prev_start_p = int(_e_yr.split("-")[0].strip()[:4])
        except (ValueError, TypeError, IndexError):
            pass
        _edu_entries_for_prompt.append({
            "university": (_pe.get("institution") or "").strip(),
            "degree":     (_pe.get("degree")      or "").strip(),
            "cgpa":       (_pe.get("cgpa")        or "").strip(),
            "years":      _e_yr,
            "achievement":(_pe.get("achievement") or "").strip(),
        })
    if not _edu_entries_for_prompt:
        _edu_entries_for_prompt = [{
            "university": "", "degree": "", "cgpa": "",
            "years": f"{edu['start']} - {edu['end']}", "achievement": ""
        }]
    _edu_json_block = _json_mod.dumps(_edu_entries_for_prompt, indent=4)
    # Keep backward-compat alias for single-entry references still in scope
    _p_edu0 = _p_edu_l[0] if _p_edu_l else {}

    # Build work history context from profile work entries if provided
    _work_ctx_lines = []
    for _w in _p_work_l:
        _wc = (_w.get("company") or "").strip()
        _wr = (_w.get("role")    or "").strip()
        _wf = (_w.get("from")    or "").strip()
        _wt = (_w.get("to")      or "").strip()
        if _wc:
            _wline = f'  • {_wc}'
            if _wr: _wline += f' — {_wr}'
            if _wf or _wt: _wline += f' ({_wf}–{_wt})'
            _work_ctx_lines.append(_wline)
    _profile_work_ctx = (
        "PROFILE WORK ENTRIES (use these company names exactly):\n" + "\n".join(_work_ctx_lines)
        if _work_ctx_lines else ""
    )
    
    # ── Seniority ladder — computed from numeric years, passed into the prompt ──
    # Rule: derive a realistic progression so the AI has concrete guidance
    # without any hardcoded role names.
    try:
        yrs_float = float(raw_years)
    except (ValueError, TypeError):
        yrs_float = 3.0

    if yrs_float <= 1.0:
        seniority_guidance = (
            f"Total experience: {years_display}\n"
            f"There is exactly 1 company entry — no more, no less.\n"
            f"Position 1 (only role): use a JUNIOR-level seniority prefix "
            f"in the role title (e.g. 'Junior [domain] [function]')."
        )
    elif yrs_float <= 2.0:
        if num_cos == 1:
            seniority_guidance = (
                f"Total experience: {years_display}\n"
                f"There is exactly 1 company entry.\n"
                f"Position 1 (only / most recent role): use a JUNIOR-level seniority prefix."
            )
        else:
            seniority_guidance = (
                f"Total experience: {years_display}\n"
                f"There are exactly 2 company entries — no more, no less.\n"
                f"Position 1 (most recent role, listed first in JSON): NO seniority prefix — "
                f"just the domain + function title derived from the JD.\n"
                f"Position 2 (oldest/earliest role, listed second in JSON): use a JUNIOR-level seniority prefix."
            )
    else:
        # > 2 years: realistic ascending progression
        lines = []
        for idx in range(num_cos):
            pos_num   = idx + 1
            # Oldest first (highest index = most recent for the candidate)
            career_pos = num_cos - idx   # 1 = oldest job
            if career_pos == 1:
                lines.append(f"Position {pos_num} (earliest role): JUNIOR-level prefix.")
            elif career_pos == num_cos and num_cos >= 3:
                lines.append(
                    f"Position {pos_num} (most recent role): use a SENIOR or LEAD-level prefix "
                    f"only if {years_display} years justifies it and the JD responsibilities "
                    f"align with senior-level ownership. Otherwise omit any prefix."
                )
            else:
                lines.append(
                    f"Position {pos_num} (mid-career role): NO seniority prefix — "
                    f"just the domain + function title derived from the JD."
                )
        seniority_guidance = (
            f"Total experience: {years_display}\n"
            + "\n".join(lines)
            + "\nSeniority labels must feel realistic, not inflated. "
            + "Infer them from the JD responsibilities and the experience level above."
        )

    # Build candidate contact block for prompt
    _p_name_str  = (_p_data.get("name") or "").strip()
    _p_links_raw = _p_data.get("links") or []
    _p_contact_str = ", ".join(
        l.get("value", "") for l in _p_links_raw[:4] if l.get("value", "")
    )

    system_prompt = f"""You are a world-class CV writer. Your sole task is to generate a single, complete, \
ATS-optimised, humanised CV as a JSON object. Every word must come from the job description below — \
zero hardcoded content, zero static templates, zero predefined examples.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Job Title Provided : {job_title}
Work Positions     : {num_cos}
Education Period   : {edu['start']} – {edu['end']}
{f"Candidate Name     : {_p_name_str}" if _p_name_str else ""}
{f"Contact            : {_p_contact_str}" if _p_contact_str else ""}

WORK HISTORY (use these exact names and date ranges, do not invent others):
{co_lines}
{_profile_work_ctx}

{company_context_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENIORITY PROGRESSION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{seniority_guidance}

These rules are mandatory. Role titles must feel natural, realistic, and human.
Never use generic labels like "Developer 1 / Developer 2". Infer the full role
title (seniority + domain + function) from the JD itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NON-NEGOTIABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[R1] EXPERIENCE FORMAT — CRITICAL, NO EXCEPTIONS
  • Experience value = "{years_display}"  (exactly one + sign, already included)
  • Title last segment  → exactly "{years_display}"  (never "5++" or "5+ +")
  • Summary first words → "{years_display} years of experience in …"
  • Nowhere else in the CV should a bare year-count appear as a standalone token.

[R2] TITLE FORMAT
  • Infer a closely related but distinctly worded role title from the JD.
    Do NOT copy the provided job title verbatim.
    Example logic: "Python Developer" → "Backend Software Engineer"
                   "Data Analyst"     → "Business Intelligence & Analytics Specialist"
  • Identify the 3 single most important technologies/skills in the JD.
  • Assemble: "Inferred Role | Tech1, Tech2, Tech3 | {years_display}"
  • The last pipe-segment MUST be exactly "{years_display}" — nothing appended after it.

[R3] NO COMPANY NAMES IN FREE TEXT
  • Company names from the work history must NEVER appear in: summary, project names,
    project overviews, project bullets, achievements, or competencies.
  • They appear ONLY in the "company" field of each companies[] entry.

[R4] 100 % DERIVED FROM JD
  • Every technology, skill, keyword, competency, bullet, and project idea must be
    traceable to a word or concept in the job description.
  • Nothing hardcoded. Nothing generic. Every JD produces a completely unique CV.

[R5] HUMANISED WRITING
  • Write as a seasoned professional, not a template engine.
  • Vary sentence structure. Use active voice. Avoid repetition across bullets.
  • Technologies should weave in naturally — not feel like a keyword-stuffing exercise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION-BY-SECTION INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① title
   Inferred role (not a copy of the job title) | 3 key technologies from JD | {years_display}
   The third segment is exactly "{years_display}". Nothing more.

② summary  [70–100 words, 4–5 sentences]
   • Open with: "{years_display} years of experience in [specific JD domain]…"
   • Concise, impactful, ATS-rich. Mention 4–6 key technologies naturally.
   • No technology repeated. Reads like a polished professional wrote it.
   • Do NOT write a long paragraph — keep it punchy and scannable.

③ competencies
   Exactly 10 domain-specific skill phrases from the JD, separated by " * ".
   Phrases must name real capabilities, not generic filler.

④ keywords
   18–20 ATS keywords from the JD, comma-separated. Cover tools, methods, and domain terms.

⑤ technologies
   mustHave   : explicitly required tools/stacks in the JD (10–14 items)
   niceToHave : preferred / bonus technologies in the JD (8–12 items)
   additional : logically adjacent ecosystem tools implied by the JD (8–10 items)

⑥ skills  [TECHNICAL SKILLS section — 5 entries only]
   Format each entry as: "Short Role-Specific Category: tech1, tech2, … tech12"
   • Category labels must be short, specific to THIS role, and useful as subheadings.
   • Use small subheading style — no nested structures, no extra sections.
   • 10–12 technologies per category, all from the JD.
   • No duplicates across categories.

⑦ companies  [one entry per company listed above, in order]
   role    : Apply the seniority progression rules above. Infer the full title
             (seniority prefix + domain + function) from the JD. No hardcoded labels.
   bullets : 4 achievement bullets per company, each 20–30 words.
             Each bullet must be unique — different technology, different metric, different context.
             No copy-pasting between companies. Bullets must sound like lived experience.
   tech    : 6–8 JD technologies used at that company, pipe-separated.
             Do NOT repeat the same tech set across all companies.

⑧ projects  [EXACTLY 4 — split as described]
   PROJECT SPLIT RULE (mandatory):
     • Projects 1 & 2: Grounded in the target company's business domain, products,
       services, ecosystem, or industry context. If a target company was provided,
       align these with what that company actually does. If no company was provided,
       derive the domain from the JD. These should feel like real work done for a
       real organisation in that space.
     • Projects 3 & 4: Directly address the technical requirements, tools, and
       deliverables described in the job description. These demonstrate hands-on
       expertise in the exact skills the role demands.
   For every project:
     name     : Descriptive, specific name. Never "Project 1/2/3/4". Never a company name.
     overview : 3–4 sentences. Story arc: problem → approach → technologies → outcome.
                Reads like a real project summary a professional would write.
     bullets  : 3 achievement bullets, each with a concrete metric or outcome.
     techTags : 5–7 technologies from the JD relevant to that project.

⑨ relatedTech  [5 category objects, 5 items each — all from JD]

⑩ education
   Copy ALL education entries EXACTLY as pre-filled in the JSON array below.
   The array may contain one or more entries — preserve every entry unchanged.
   Do NOT invent, merge, or remove any entry. Do NOT change any field values.
   If a field is empty in the template, leave it empty — do not fill it in.
   Only the "years" field of the first entry was calculated dynamically; use it as-is.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON OUTPUT — no markdown, no code fences, no explanation text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "title": "Inferred Role | Tech1, Tech2, Tech3 | {years_display}",
  "summary": "{years_display} years of experience in [JD domain]… (4–5 sentences, 70–100 words)",
  "competencies": "Phrase1 * Phrase2 * Phrase3 * Phrase4 * Phrase5 * Phrase6 * Phrase7 * Phrase8 * Phrase9 * Phrase10",
  "keywords": "kw1, kw2, kw3, kw4, kw5, kw6, kw7, kw8, kw9, kw10, kw11, kw12, kw13, kw14, kw15, kw16, kw17, kw18",
  "technologies": {{
    "mustHave":   ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"],
    "niceToHave": ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10"],
    "additional": ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10"]
  }},
  "skills": [
    "Short Category Label A: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10",
    "Short Category Label B: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10",
    "Short Category Label C: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10",
    "Short Category Label D: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10",
    "Short Category Label E: tech1, tech2, tech3, tech4, tech5, tech6, tech7, tech8, tech9, tech10"
  ],
  "companies": [
    {{
      "company": "",
      "role": "Seniority-Informed Role Title Derived from JD",
      "dateRange": "",
      "bullets": [
        "Concrete achievement — JD technology + measurable result (20–30 words).",
        "Different achievement — different JD technology + quantified impact (20–30 words).",
        "Process or delivery improvement — JD technology + outcome (20–30 words).",
        "Business or stakeholder impact — JD context + result (20–30 words)."
      ],
      "tech": "Tech1 | Tech2 | Tech3 | Tech4 | Tech5 | Tech6 | Tech7 | Tech8"
    }}
  ],
  "projects": [
    {{
      "name": "Business-Domain Project Name (company/industry context)",
      "overview": "3–4 sentences: business problem → solution → technologies used → measurable impact.",
      "bullets": [
        "Specific outcome with JD technology and metric (20–30 words).",
        "Technical challenge overcome with quantified result (20–30 words).",
        "Business benefit delivered (20–30 words)."
      ],
      "techTags": ["Tech1","Tech2","Tech3","Tech4","Tech5","Tech6"]
    }},
    {{
      "name": "Second Business-Domain Project Name",
      "overview": "3–4 sentences for a different business/industry angle.",
      "bullets": ["Bullet1","Bullet2","Bullet3"],
      "techTags": ["Tech1","Tech2","Tech3","Tech4","Tech5","Tech6"]
    }},
    {{
      "name": "JD-Technical Requirements Project Name",
      "overview": "3–4 sentences aligned with the specific technical skills in the JD.",
      "bullets": ["Bullet1","Bullet2","Bullet3"],
      "techTags": ["Tech1","Tech2","Tech3","Tech4","Tech5","Tech6"]
    }},
    {{
      "name": "Second JD-Technical Requirements Project Name",
      "overview": "3–4 sentences for a different technical capability from the JD.",
      "bullets": ["Bullet1","Bullet2","Bullet3"],
      "techTags": ["Tech1","Tech2","Tech3","Tech4","Tech5","Tech6"]
    }}
  ],
  "education": {_edu_json_block},
  "relatedTech": [
    {{"category": "JD Category 1", "items": ["t1","t2","t3","t4","t5"]}},
    {{"category": "JD Category 2", "items": ["t1","t2","t3","t4","t5"]}},
    {{"category": "JD Category 3", "items": ["t1","t2","t3","t4","t5"]}},
    {{"category": "JD Category 4", "items": ["t1","t2","t3","t4","t5"]}},
    {{"category": "JD Category 5", "items": ["t1","t2","t3","t4","t5"]}}
  ]
}}

PRE-SUBMIT CHECKLIST — verify every item before writing a single character of output:
✓ title last segment is exactly "{years_display}" — not "5++" not "5+ +" not "5+ years"
✓ summary opens with exactly "{years_display} years of experience in …"
✓ summary is 70–100 words (4–5 sentences) — concise, not bloated
✓ company role titles follow the seniority progression rules above
✓ projects 1–2 are grounded in the company/industry domain
✓ projects 3–4 target the JD's specific technical requirements
✓ zero company names appear anywhere except the "company" JSON key
✓ every technology, skill, and keyword came from the job description
✓ output is raw JSON only — no markdown fences, no explanatory text
"""

    user_prompt = f"""JOB DESCRIPTION:
{jd}

Generate the complete CV JSON now.

Key reminders:
- Title must end with exactly "{years_display}" (one + sign, no more).
- Summary must open with "{years_display} years of experience in …" and be 70–100 words.
- Company roles must follow the seniority progression: {seniority_guidance.split(chr(10))[0]}
- Projects 1–2: company/industry domain. Projects 3–4: JD technical requirements.
- No company names in any free-text field.
- Every word derived from the job description above.
"""

    return system_prompt, user_prompt

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def extract_json(raw: str) -> dict:
    """Parse JSON from an LLM response, repairing truncated output if needed."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object in model response")
    raw = raw[start:]

    # Try direct parse first (the happy path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ── Attempt to repair truncated / incomplete JSON ─────────────────────────
    # 1. Remove trailing comma artefacts
    j = re.sub(r",\s*([}\]])", r"\1", raw)
    # 2. Control-character cleanup
    j = re.sub(r'[\x00-\x1f\x7f]', ' ', j)

    # 3. If the JSON was cut off (no closing brace), close all open structures
    #    by counting unmatched { and [ and appending the right closers.
    try:
        return json.loads(j)
    except json.JSONDecodeError:
        pass

    # Count unclosed brackets (ignoring those inside strings)
    opens = []
    in_str = False
    escape = False
    for ch in j:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            opens.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if opens and opens[-1] == ch:
                opens.pop()

    # Strip any trailing incomplete string / value that might confuse the parser
    j_repaired = j.rstrip().rstrip(",").rstrip()
    # Close all unclosed structures in reverse order
    j_repaired += "".join(reversed(opens))

    try:
        return json.loads(j_repaired)
    except json.JSONDecodeError:
        # Last resort: extract up to the last complete top-level value
        end = j_repaired.rfind("}")
        if end != -1:
            candidate = j_repaired[:end + 1]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)
        raise ValueError("Could not parse or repair JSON from model response")

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

    # education: normalise to a list of dicts regardless of what came in
    raw_edu = cv.get("education")
    if isinstance(raw_edu, dict):
        # Legacy single-dict — wrap in a list
        cv["education"] = [raw_edu]
    elif isinstance(raw_edu, list):
        cv["education"] = [e for e in raw_edu if isinstance(e, dict)]
    else:
        cv["education"] = []

    return cv

def final_polish(cv: dict, years_exp: str = "") -> dict:
    """Final polishing — deduplicates tech tags and ensures correct experience display.
    
    Core rule: experience is stored internally WITHOUT the trailing + sign.
    The display value (e.g. '5+') is only assembled when writing to cv fields.
    This prevents any possible '5++' bug.
    """

    # ── Resolve clean numeric years (no + sign) ───────────────────────────────
    raw = (years_exp or "").strip().replace("+", "")
    try:
        float(raw)   # validate numeric
    except (ValueError, TypeError):
        raw = _calc_total_years("")   # returns plain digits e.g. "5"

    # The one authoritative display string — always "N+" with exactly one +
    years_display = raw + "+"
    cv["totalYears"] = years_display

    # ── Fix title ─────────────────────────────────────────────────────────────
    title = cv.get("title", "")
    if title and raw:
        if "|" in title:
            parts = [p.strip() for p in title.split("|")]

            # Pattern 1 — whole trailing segment is ONLY an experience token
            # e.g. "2+", "5", "3+ years"  (no letters except optional "years")
            _exp_only = re.compile(
                r'^\s*\d+(\.\d+)?\+?\s*(years?)?\s*$',
                re.IGNORECASE
            )
            while len(parts) > 1 and _exp_only.match(parts[-1]):
                parts.pop()

            # Pattern 2 — experience token is embedded at the END of the last
            # segment (AI sometimes writes "LLM Architecture 2+" or "Tech 2+ ").
            # Strip it from that last segment so we can append cleanly.
            _exp_suffix = re.compile(
                r'\s+\d+(\.\d+)?\+?\s*(years?)?\s*$',
                re.IGNORECASE
            )
            if parts:
                parts[-1] = _exp_suffix.sub("", parts[-1]).strip()

            # Append the authoritative display value exactly once
            parts.append(years_display)
            cv["title"] = " | ".join(parts)
        else:
            # No pipes — strip any trailing experience from the bare title first
            _exp_suffix_bare = re.compile(
                r'\s+\d+(\.\d+)?\+?\s*(years?)?\s*$',
                re.IGNORECASE
            )
            title = _exp_suffix_bare.sub("", title).strip()
            cv["title"] = f"{title} | {years_display}"

    # ── Fix summary ───────────────────────────────────────────────────────────
    summary = cv.get("summary", "")
    if summary and raw:
        # Replace any existing "N years", "N+ years", "N++ years" pattern at start
        summary = re.sub(
            r'^\d+\+*\s+years?\s+of',
            f"{years_display} years of",
            summary.strip(),
            flags=re.IGNORECASE
        )
        # Also handle "With N+... years" or "Over N+... years" openers
        summary = re.sub(
            r'\b(with|over)\s+\d+\+*\s+years?\s+of',
            f"\\1 {years_display} years of",
            summary,
            count=1,
            flags=re.IGNORECASE
        )
        # Catch any remaining stray double ++ in the summary
        summary = summary.replace("++", "+")
        cv["summary"] = summary

    # ── Sanitise any stray ++ that slipped through elsewhere ─────────────────
    for field in ("title", "summary", "competencies", "keywords"):
        val = cv.get(field, "")
        if isinstance(val, str):
            cv[field] = val.replace("++", "+")

    # ── Deduplicate tech tags across companies ────────────────────────────────
    companies = cv.get("companies", [])
    used_techs: set = set()

    for co in companies:
        tech_str = co.get("tech", "")
        if tech_str:
            techs = [t.strip() for t in tech_str.split("|") if t.strip()]
            unique_techs = []
            for t in techs:
                if t.lower() not in used_techs:
                    unique_techs.append(t)
                    used_techs.add(t.lower())
            # Pad back up to 4 if we removed too many
            if len(unique_techs) < 4:
                for t in techs:
                    if len(unique_techs) >= 6:
                        break
                    if t.lower() not in used_techs:
                        unique_techs.append(t)
                        used_techs.add(t.lower())
            if unique_techs:
                co["tech"] = " | ".join(unique_techs[:8])

    # ── Deduplicate project tech tags ─────────────────────────────────────────
    for proj in cv.get("projects", []):
        tags = proj.get("techTags", [])
        if isinstance(tags, list):
            seen: set = set()
            proj["techTags"] = [
                t for t in tags
                if t and t.lower() not in seen and not seen.add(t.lower())
            ][:7]

    return cv

def fix_companies(cv: dict, companies_list: list = None, years_exp: str = "") -> dict:
    """
    Enforce correct company names/dates and clean up AI-generated role strings.

    Priority for names and dates:
      • companies_list (runtime-calculated from years_exp) — always preferred.
      • CANDIDATE_COMPANIES static fallback — only if companies_list is absent.

    Seniority logic (dynamic path only — when companies_list is provided):
      1 year  → 1 company:  Junior <role>
      2 years → 2 companies: index 0 = bare role, index 1 = Junior <role>
      3+ yrs  → 3 companies: index 0 = Senior, index 1 = bare, index 2 = Junior
    When years_exp is missing or zero, seniority is left as the AI generated it.
    When profile work entries supply hardcoded dates, years_exp still drives the
    label, preserving consistent behaviour across both paths.
    """
    companies = cv.get("companies", [])

    # ── Hard-enforce company count from companies_list ───────────────────────
    # The AI sometimes generates more companies than instructed (e.g. 3 when told 2).
    # Truncate to the exact count calculated from years_exp so the UI value is
    # always respected.  Profile-hardcoded paths also use companies_list, so the
    # count remains consistent on both paths.
    if companies_list:
        companies = companies[:len(companies_list)]
        cv["companies"] = companies

    # ── Resolve numeric years ───────────────────────────────────────────────────────
    apply_seniority = bool(companies_list)
    _yraw = (years_exp or "").strip().replace("+", "")
    if not _yraw:
        _yraw = str(cv.get("totalYears", "")).replace("+", "").strip()
    try:
        _n_years = float(_yraw)
    except (ValueError, TypeError):
        _n_years = 0.0

    _seniority_re = re.compile(
        r"^(Senior|Sr\.?|Lead|Principal|Junior|Jr\.?|Associate)\s+",
        re.IGNORECASE
    )

    def _seniority_prefix(idx: int, n_cos: int) -> str:
        """Prefix for company at idx (0 = most recent / top of CV)."""
        if n_cos == 1:
            return "Junior"
        if n_cos == 2:
            return "" if idx == 0 else "Junior"
        # 3+ companies
        if idx == 0:
            return "Senior"
        if idx == n_cos - 1:
            return "Junior"
        return ""

    for i, co in enumerate(companies):
        # ── Name & dateRange: runtime list first, static fallback second ──────
        if companies_list and i < len(companies_list):
            calc = companies_list[i]
            co["company"]   = calc["name"]
            co["dateRange"] = f"{calc['start']} - {calc['end']}"
        elif i < len(CANDIDATE_COMPANIES):
            real = CANDIDATE_COMPANIES[i]
            co["company"]   = real["name"]
            co["dateRange"] = f"{real['start']} - {real['end']}"
        elif not co.get("company"):
            co["company"] = f"Company {i + 1}"

        # ── Clean AI artefacts from role string (e.g. "Co1 ", "Co2 ") ────
        role = co.get("role", "")
        role = re.sub(r'\bCo\d+\s*', '', role).strip()

        # ── Apply experience-based seniority (dynamic path only) ─────────────
        if apply_seniority and _n_years > 0:
            n_cos  = len(companies)
            prefix = _seniority_prefix(i, n_cos)
            bare_role = _seniority_re.sub("", role).strip()
            role = f"{prefix} {bare_role}".strip() if prefix else bare_role

        co["role"] = role

    return cv

# ==============================================================================
# UNIVERSAL LLM CALLER
# ==============================================================================

class _RateLimitError(Exception):
    """Raised when a key is rate-limited so callers can skip to the next key."""
    pass

async def call_llm_atomic(client, key: str, model: str, url: str,
                          system: str, user: str, stage: str,
                          headers: dict, max_tokens: int = 4000,
                          _deadline: float = 0.0) -> dict:
    import time as _t

    # Groq can be slow on large prompts; use a generous per-call timeout.
    # The outer httpx.AsyncClient timeout is the hard ceiling — this is per-attempt.
    per_call_timeout = 120
    mk = mask(key)
    provider_tag = url.split("/")[2].split(".")[0]   # e.g. "api" → use model instead
    tag = f"[{model}|{mk}|{stage}]"

    if _deadline and _t.time() >= _deadline:
        _log.warning("%s Skipped — deadline exceeded before attempt", tag)
        raise ValueError(f"Stage {stage} skipped — deadline exceeded")

    last_error = None
    prompt_chars = len(system) + len(user)
    _log.info("%s Starting LLM call — prompt ~%d chars, max_tokens=%d, timeout=%ds",
              tag, prompt_chars, max_tokens, per_call_timeout)

    # Up to 3 attempts: handles transient 429s and single timeout blips
    for attempt in range(3):
        attempt_num = attempt + 1
        _log.info("%s Attempt %d/3 …", tag, attempt_num)
        t_start = _t.time()

        try:
            r = await client.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user}
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
                timeout=per_call_timeout,
            )

        except httpx.TimeoutException as exc:
            elapsed = round(_t.time() - t_start, 1)
            last_error = f"TimeoutException on attempt {attempt_num} after {elapsed}s"
            _log.warning("%s TIMEOUT — %s — elapsed %.1fs  (exc: %s)",
                         tag, last_error, elapsed, type(exc).__name__)
            if attempt < 2:
                wait = 4 + attempt * 4   # 4s, 8s
                _log.info("%s Waiting %ds before retry …", tag, wait)
                await asyncio.sleep(wait)
                continue
            _log.error("%s All 3 timeout attempts exhausted for key %s", tag, mk)
            raise ValueError(
                f"Stage {stage} timed out after {per_call_timeout}s "
                f"(3 attempts). Try a different key or provider."
            )

        except httpx.ReadTimeout as exc:
            elapsed = round(_t.time() - t_start, 1)
            last_error = f"ReadTimeout on attempt {attempt_num} after {elapsed}s"
            _log.warning("%s READ-TIMEOUT — %s", tag, last_error)
            if attempt < 2:
                wait = 4 + attempt * 4
                _log.info("%s Waiting %ds before retry …", tag, wait)
                await asyncio.sleep(wait)
                continue
            _log.error("%s All 3 read-timeout attempts exhausted for key %s", tag, mk)
            raise ValueError(
                f"Stage {stage} read-timeout after {per_call_timeout}s "
                f"(3 attempts). Model took too long to respond."
            )

        except Exception as e:
            elapsed = round(_t.time() - t_start, 1)
            _log.error("%s NETWORK ERROR after %.1fs — %s: %s", tag, elapsed, type(e).__name__, e)
            raise ValueError(f"Stage {stage} network error: {str(e)}")

        # ── Handle HTTP response ──────────────────────────────────────────────
        elapsed = round(_t.time() - t_start, 1)
        _log.info("%s HTTP %d received in %.1fs", tag, r.status_code, elapsed)

        if r.status_code == 200:
            raw = r.json()["choices"][0]["message"]["content"]
            _log.info("%s SUCCESS — response %d chars", tag, len(raw))
            return extract_json(raw)

        elif r.status_code == 429:
            retry_after = int(r.headers.get("retry-after", 0))
            _log.warning("%s RATE-LIMITED (429) — retry-after=%ds — attempt %d/3",
                         tag, retry_after, attempt_num)
            if attempt < 2 and retry_after > 0:
                wait = min(retry_after, 25)
                _log.info("%s Sleeping %ds (retry-after) …", tag, wait)
                await asyncio.sleep(wait)
                continue
            elif attempt < 2:
                wait = 2 ** attempt * 3
                _log.info("%s No retry-after header — exponential backoff %ds …", tag, wait)
                await asyncio.sleep(wait)
                continue
            # Retries exhausted — mark key in cooldown and signal caller
            import time as _t2
            cooldown = max(retry_after, 60)
            _key_rate_limited_until[mk] = _t2.time() + cooldown
            _log.error("%s Key %s marked rate-limited for %ds", tag, mk, cooldown)
            raise _RateLimitError(f"Key rate-limited on {stage}")

        elif r.status_code in (401, 403):
            _log.error("%s INVALID/EXPIRED KEY — HTTP %d for key %s", tag, r.status_code, mk)
            raise ValueError(f"Invalid/expired key on {stage} (HTTP {r.status_code})")

        else:
            last_error = f"HTTP {r.status_code} on {stage}"
            _log.warning("%s Unexpected status %d on attempt %d — %s",
                         tag, r.status_code, attempt_num, r.text[:200])
            if attempt < 2:
                await asyncio.sleep(3)
                continue
            raise ValueError(last_error)

    _log.error("%s Failed after 3 attempts. Last error: %s", tag, last_error)
    raise ValueError(f"Stage {stage} failed after 3 attempts. Last error: {last_error}")

# ==============================================================================
# PROVIDER CALLERS
# ==============================================================================
async def generate_cv_dynamic(req: CVRequest, client, key: str, model: str,
                               url: str, headers: dict,
                               max_output_tokens: int = None) -> dict:
    """Generate CV using single dynamic prompt - everything from JD"""
    import time as _t

    _deadline = _t.time() + 270
    years_exp = (req.years_exp or "").strip()
    years_exp_clean = years_exp.replace("+", "").strip()

    mk = mask(key)
    provider_host = url.split("/")[2]
    _log.info("[GenCV|%s] Starting — model=%s key=%s years_exp=%r job=%r",
              provider_host, model, mk, years_exp_clean, req.job_title[:50])

    # Pull profile education if available (priority data — never override with calc)
    _raw_profile_edu = (req.profile_data or {}).get("edu", []) if req.profile_data else []
    # Normalise: maps UI 'note' field → 'achievement' and extracts cgpa from note text
    profile_edu = [_normalise_edu_entry(e) for e in _raw_profile_edu] if _raw_profile_edu else []

    total_years    = _calc_total_years(years_exp_clean)
    _profile_work  = (req.profile_data or {}).get("work", []) if req.profile_data else []
    companies_list = _build_dynamic_companies(years_exp_clean, profile_work=_profile_work or None)
    edu            = _build_education_year(years_exp_clean, profile_edu=profile_edu or None)

    _log.info("[GenCV|%s] Computed — total_years=%s companies=%d edu=%s–%s",
              provider_host, total_years, len(companies_list), edu["start"], edu["end"])
    
    system_prompt, user_prompt = build_dynamic_prompt(req)
    _log.info("[GenCV|%s] Prompt built — sys=%d chars, usr=%d chars",
              provider_host, len(system_prompt), len(user_prompt))

    result = await call_llm_atomic(client, key, model, url, system_prompt, user_prompt,
                                    "FullCV", headers, max_tokens=max_output_tokens or 8000, _deadline=_deadline)

    if not result:
        _log.error("[GenCV|%s] AI returned empty/unparseable response", provider_host)
        raise ValueError("AI returned empty response")

    _log.info("[GenCV|%s] AI response parsed — companies=%d projects=%d",
              provider_host,
              len(result.get("companies", [])),
              len(result.get("projects",  [])))

    if "companies" not in result:
        result["companies"] = []

    # Hard-truncate AI output to exact count immediately — the AI frequently
    # ignores the "N companies" instruction and returns more entries.
    # This is the primary guard; fix_companies() is a second pass.
    result["companies"] = result["companies"][:len(companies_list)]

    for i, co in enumerate(result.get("companies", [])):
        if i < len(companies_list):
            co["company"]   = companies_list[i]["name"]
            co["dateRange"] = f"{companies_list[i]['start']} - {companies_list[i]['end']}"

    if "projects" not in result:
        result["projects"] = []

    # ── Build education list — all entries, dates always UI-first ───────────────
    # Priority per entry:
    #   1. Both from/to in UI  → use exactly as provided.
    #   2. One date missing     → infer from degree duration via _infer_degree_duration().
    #   3. Both dates missing   → place immediately before the entry above it started,
    #      using that entry's start year as the anchor (chronological auto-sequencing).
    _p_edu_list = profile_edu or []

    def _resolve_edu_years(pe: dict, anchor_start_yr: int = None) -> str:
        """
        Return a "YYYY - YYYY" string for one education entry.
        anchor_start_yr: the start year of the PREVIOUS (higher on CV) entry,
        used as the upper bound when this entry has no dates at all.
        """
        _ef  = str(pe.get("from") or "").strip()
        _et  = str(pe.get("to")   or "").strip()
        _deg = (pe.get("degree")  or "").strip()
        _dur = _infer_degree_duration(_deg)

        # Case 1: both dates provided — use as-is
        if _ef and _et:
            return f"{_ef} - {_et}"

        # Case 2: only end year provided — infer start from duration
        if _et and not _ef:
            try:
                return f"{int(_et[:4]) - _dur} - {_et}"
            except (ValueError, TypeError):
                pass

        # Case 3: only start year provided — infer end from duration
        if _ef and not _et:
            try:
                return f"{_ef} - {int(_ef[:4]) + _dur}"
            except (ValueError, TypeError):
                pass

        # Case 4: no dates at all — place this degree before the anchor entry
        if anchor_start_yr is not None:
            end_yr   = anchor_start_yr      # ends where the previous entry started
            start_yr = end_yr - _dur
            return f"{start_yr} - {end_yr}"

        # Case 5: no anchor either — use calculated years from the first entry
        return f"{edu['start']} - {edu['end']}"

    if _p_edu_list:
        _merged_edu = []
        _prev_start_yr = None   # start year of the entry just added (for sequencing)
        for _pe in _p_edu_list:
            _yr_str = _resolve_edu_years(_pe, anchor_start_yr=_prev_start_yr)
            # Extract the start year of THIS entry to anchor the next one
            try:
                _prev_start_yr = int(_yr_str.split("-")[0].strip()[:4])
            except (ValueError, TypeError, IndexError):
                pass
            _merged_edu.append({
                "university":  (_pe.get("institution") or "").strip(),
                "degree":      (_pe.get("degree")       or "").strip(),
                "cgpa":        (_pe.get("cgpa")         or "").strip(),
                "years":       _yr_str,
                "achievement": (_pe.get("achievement")  or "").strip(),
            })
    else:
        # No profile education — fall back to whatever the AI returned
        _ai_edu_raw = result.get("education") or {}
        if isinstance(_ai_edu_raw, list):
            _merged_edu = _ai_edu_raw
        else:
            _merged_edu = [{
                "university":  (_ai_edu_raw.get("university") or "").strip(),
                "degree":      (_ai_edu_raw.get("degree")     or "").strip(),
                "cgpa":        (_ai_edu_raw.get("cgpa")       or "").strip(),
                "years":       f"{edu['start']} - {edu['end']}",
                "achievement": "",
            }]
    result["education"] = _merged_edu

    cv = {
        "totalYears":  total_years,
        "title":       result.get("title",       req.job_title),
        "summary":     result.get("summary",     ""),
        "skills":      result.get("skills",      []),
        "competencies":result.get("competencies",""),
        "keywords":    result.get("keywords",    ""),
        "technologies":result.get("technologies",{"mustHave": [], "niceToHave": [], "additional": []}),
        "companies":   result.get("companies",   []),
        "projects":    result.get("projects",    []),
        "relatedTech": result.get("relatedTech", []),
        "education":   _merged_edu,   # always use merged (profile-priority) education
    }

    cv_sanitised = sanitise_cv(cv)
    cv_companies = fix_companies(cv_sanitised, companies_list=companies_list, years_exp=years_exp_clean)
    cv_polished  = final_polish(cv_companies, years_exp=years_exp_clean)

    _log.info("[GenCV|%s] CV post-processing complete — title=%r totalYears=%r",
              provider_host,
              cv_polished.get("title", "")[:60],
              cv_polished.get("totalYears", ""))
    return cv_polished

async def call_cerebras(req: CVRequest) -> tuple:
    import time as _t
    raw_keys = req.cerebras_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Cerebras keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid Cerebras keys found.")

    model = req.model or "llama3.1-8b"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    rate_limited_count = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=180, write=15, pool=10)) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

            # Skip keys still in their rate-limit cooldown window (same pattern as Groq)
            cooldown_until = _key_rate_limited_until.get(mk, 0)
            if cooldown_until > _t.time():
                remaining = int(cooldown_until - _t.time())
                rate_limited_count += 1
                msg = f"Key {i+1} ({mk}): still rate-limited ({remaining}s remaining)"
                errors_by_key.append(msg)
                _log.warning("[Cerebras] %s — skipping", msg)
                continue

            # Quick probe to skip obviously bad/rate-limited keys
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
                    rate_limited_count += 1
                    retry_after = int(probe.headers.get("retry-after", 60))
                    _key_rate_limited_until[mk] = _t.time() + min(retry_after, 120)
                    errors_by_key.append(f"Key {i+1} ({mk}): rate limited (retry-after {retry_after}s)")
                    # Small gap before trying next key
                    if i < len(sorted_keys) - 1:
                        await asyncio.sleep(2)
                    continue
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): probe failed - {str(e)[:50]}")
                continue

            try:
                cv = await generate_cv_dynamic(req, client, key, model, CEREBRAS_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                return cv, mk, i
            except _RateLimitError as e:
                rate_limited_count += 1
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)}")
                if i < len(sorted_keys) - 1:
                    await asyncio.sleep(3)
                continue
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue

    if rate_limited_count == len(sorted_keys):
        raise HTTPException(429, f"All Cerebras keys are rate-limited. Wait a moment and try again. Details: {'; '.join(errors_by_key[:3])}")
    raise HTTPException(502, f"All Cerebras keys failed: {'; '.join(errors_by_key[:3])}")

async def call_groq(req: CVRequest) -> tuple:
    import time as _t
    raw_keys = req.groq_keys or []
    if not raw_keys:
        _log.error("[Groq] No Groq keys provided in request")
        raise HTTPException(400, "No Groq keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip().startswith("gsk_")]
    if not valid_keys:
        _log.error("[Groq] %d keys provided but none start with 'gsk_' — all invalid format",
                   len(raw_keys))
        raise HTTPException(400, "No valid Groq keys (must start with gsk_).")

    model = req.model or "llama-3.1-8b-instant"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    rate_limited_count = 0

    _log.info("[Groq] Starting generation — model=%s, keys=%d, job_title=%r",
              model, len(sorted_keys), req.job_title[:60])

    # read=240 gives each attempt up to 4 min; call_llm_atomic uses 120s per try
    # with its own retry loop, so the outer client must not cut it short
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15, read=240, write=20, pool=10)
    ) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

            # Skip keys still in their rate-limit cooldown window
            cooldown_until = _key_rate_limited_until.get(mk, 0)
            if cooldown_until > _t.time():
                remaining = int(cooldown_until - _t.time())
                rate_limited_count += 1
                msg = f"Key {i+1} ({mk}): still rate-limited ({remaining}s remaining)"
                errors_by_key.append(msg)
                _log.warning("[Groq] %s — skipping", msg)
                continue

            _log.info("[Groq] Trying key %d/%d (%s) with model %s",
                      i + 1, len(sorted_keys), mk, model)
            try:
                cv = await generate_cv_dynamic(req, client, key, model, GROQ_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                _log.info("[Groq] SUCCESS — key %d (%s)", i + 1, mk)
                return cv, mk, i

            except _RateLimitError as e:
                rate_limited_count += 1
                msg = f"Key {i+1} ({mk}): {str(e)}"
                errors_by_key.append(msg)
                _log.warning("[Groq] RATE-LIMIT on key %d (%s): %s", i + 1, mk, e)
                if i < len(sorted_keys) - 1:
                    _log.info("[Groq] Sleeping 3s before next key …")
                    await asyncio.sleep(3)
                continue

            except Exception as e:
                msg = f"Key {i+1} ({mk}): {str(e)[:120]}"
                errors_by_key.append(msg)
                _log.error("[Groq] FAILED key %d (%s): %s", i + 1, mk, str(e)[:200])
                continue

    _log.error("[Groq] All %d key(s) failed. rate_limited=%d. Errors: %s",
               len(sorted_keys), rate_limited_count, " | ".join(errors_by_key))

    if rate_limited_count > 0 and rate_limited_count == len(sorted_keys):
        raise HTTPException(
            429,
            f"All Groq keys are currently rate-limited. "
            f"Wait ~60 seconds and try again, or switch to Cerebras/Gemini in Settings. "
            f"Details: {'; '.join(errors_by_key[:3])}"
        )
    # Check if all failures were timeouts
    timeout_count = sum(1 for e in errors_by_key if "timed out" in e.lower() or "timeout" in e.lower())
    if timeout_count == len(sorted_keys):
        raise HTTPException(
            504,
            f"Groq timed out on all keys — the model took too long to respond. "
            f"Try: (1) switch to a faster model in Settings, "
            f"(2) use Cerebras or Gemini instead, or "
            f"(3) shorten the job description. "
            f"Details: {'; '.join(errors_by_key[:2])}"
        )
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
    rate_limited_count = 0

    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)

            try:
                sys_p, usr_p = build_dynamic_prompt(req)
                url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
                payload = {
                    "systemInstruction": {"parts": [{"text": sys_p}]},
                    "contents": [{"role": "user", "parts": [{"text": usr_p}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8000},
                }
                r = await client.post(url, headers={"Content-Type": "application/json"}, json=payload)

                if r.status_code == 429:
                    rate_limited_count += 1
                    retry_after = int(r.headers.get("retry-after", 60))
                    import time as _t2
                    _key_rate_limited_until[mk] = _t2.time() + min(retry_after, 120)
                    errors_by_key.append(f"Key {i+1} ({mk}): rate limited (retry-after {retry_after}s)")
                    if i < len(sorted_keys) - 1:
                        await asyncio.sleep(3)
                    continue

                if r.status_code == 200:
                    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    result = extract_json(raw)

                    years_exp = (req.years_exp or "").strip()
                    years_exp_clean = years_exp.replace("+", "").strip()
                    total_years = _calc_total_years(years_exp_clean)
                    _g_profile_work = (req.profile_data or {}).get("work", []) if req.profile_data else []
                    companies_list = _build_dynamic_companies(years_exp_clean, profile_work=_g_profile_work or None)
                    _raw_g_edu = (req.profile_data or {}).get("edu", []) if req.profile_data else []
                    profile_edu = [_normalise_edu_entry(e) for e in _raw_g_edu] if _raw_g_edu else []
                    edu = _build_education_year(years_exp_clean, profile_edu=profile_edu or None)

                    # Hard-truncate Gemini AI output to exact company count
                    result["companies"] = result.get("companies", [])[:len(companies_list)]

                    for j, co in enumerate(result.get("companies", [])):
                        if j < len(companies_list):
                            co["company"] = companies_list[j]["name"]
                            co["dateRange"] = f"{companies_list[j]['start']} - {companies_list[j]['end']}"

                    # Education: merge AI result with profile data; profile always wins
                    # ── Gemini edu merge — same UI-first, auto-sequencing logic ──
                    def _resolve_edu_years_g(pe: dict, anchor_start_yr: int = None) -> str:
                        _ef  = str(pe.get("from") or "").strip()
                        _et  = str(pe.get("to")   or "").strip()
                        _deg = (pe.get("degree")  or "").strip()
                        _dur = _infer_degree_duration(_deg)
                        if _ef and _et:
                            return f"{_ef} - {_et}"
                        if _et and not _ef:
                            try:
                                return f"{int(_et[:4]) - _dur} - {_et}"
                            except (ValueError, TypeError):
                                pass
                        if _ef and not _et:
                            try:
                                return f"{_ef} - {int(_ef[:4]) + _dur}"
                            except (ValueError, TypeError):
                                pass
                        if anchor_start_yr is not None:
                            return f"{anchor_start_yr - _dur} - {anchor_start_yr}"
                        return f"{edu['start']} - {edu['end']}"

                    if profile_edu:
                        _merged_edu_g = []
                        _prev_s_g = None
                        for _pe_g in profile_edu:
                            _yr_g = _resolve_edu_years_g(_pe_g, anchor_start_yr=_prev_s_g)
                            try:
                                _prev_s_g = int(_yr_g.split("-")[0].strip()[:4])
                            except (ValueError, TypeError, IndexError):
                                pass
                            _merged_edu_g.append({
                                "university":  (_pe_g.get("institution") or "").strip(),
                                "degree":      (_pe_g.get("degree")       or "").strip(),
                                "cgpa":        (_pe_g.get("cgpa")         or "").strip(),
                                "years":       _yr_g,
                                "achievement": (_pe_g.get("achievement")  or "").strip(),
                            })
                    else:
                        _ai_edu_g = result.get("education") or {}
                        if isinstance(_ai_edu_g, list):
                            _merged_edu_g = _ai_edu_g
                        else:
                            _merged_edu_g = [{
                                "university":  (_ai_edu_g.get("university") or "").strip(),
                                "degree":      (_ai_edu_g.get("degree")     or "").strip(),
                                "cgpa":        (_ai_edu_g.get("cgpa")       or "").strip(),
                                "years":       f"{edu['start']} - {edu['end']}",
                                "achievement": "",
                            }]

                    cv = {
                        "totalYears":   total_years,
                        "title":        result.get("title",        req.job_title),
                        "summary":      result.get("summary",      ""),
                        "skills":       result.get("skills",       []),
                        "competencies": result.get("competencies", ""),
                        "keywords":     result.get("keywords",     ""),
                        "technologies": result.get("technologies", {}),
                        "companies":    result.get("companies",    []),
                        "projects":     result.get("projects",     []),
                        "relatedTech":  result.get("relatedTech",  []),
                        "education":    _merged_edu_g,
                    }

                    cv_sanitised = sanitise_cv(cv)
                    cv_companies = fix_companies(cv_sanitised, companies_list=companies_list, years_exp=years_exp_clean)
                    cv_polished = final_polish(cv_companies, years_exp=years_exp_clean)

                    _key_usage[mk] = _key_usage.get(mk, 0) + 1
                    return cv_polished, mk, i
                else:
                    errors_by_key.append(f"Key {i+1} ({mk}): HTTP {r.status_code}")
                    continue
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:100]}")
                continue

    if rate_limited_count == len(sorted_keys):
        raise HTTPException(429, f"All Gemini keys are rate-limited. Try again shortly. Details: {'; '.join(errors_by_key[:3])}")
    raise HTTPException(502, f"All Gemini keys failed: {'; '.join(errors_by_key[:3])}")

async def call_qwen(req: CVRequest) -> tuple:
    """
    Qwen / Alibaba DashScope — OpenAI-compatible international endpoint.
    Free tier: 1M input + 1M output tokens for 90 days on new accounts.
    API key:   https://bailian.console.aliyun.com → API Keys → Create API Key
    Key format: sk-XXXXXXXXXXXXXXXXXX
    """
    import time as _t
    raw_keys = req.qwen_keys or []
    if not raw_keys:
        raise HTTPException(400, "No Qwen / DashScope keys provided.")
    valid_keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not valid_keys:
        raise HTTPException(400, "No valid Qwen keys found.")

    model = req.model or "qwen-plus"
    sorted_keys = _prioritised_keys(valid_keys)
    errors_by_key = []
    rate_limited_count = 0

    _log.info("[Qwen] model=%s keys=%d job=%r", model, len(sorted_keys), req.job_title[:60])

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=90, write=10, pool=5)
    ) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

            cooldown_until = _key_rate_limited_until.get(mk, 0)
            if cooldown_until > _t.time():
                remaining = int(cooldown_until - _t.time())
                rate_limited_count += 1
                errors_by_key.append(f"Key {i+1} ({mk}): rate-limited ({remaining}s left)")
                continue

            try:
                cv = await generate_cv_dynamic(req, client, key, model, QWEN_URL, headers,
                                               max_output_tokens=8000)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
                _log_generation(req.job_title, mk, i, 0, model, True)
                _log.info("[Qwen] SUCCESS key %d (%s)", i + 1, mk)
                return cv, mk, i
            except _RateLimitError as e:
                rate_limited_count += 1
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)}")
                if i < len(sorted_keys) - 1:
                    await asyncio.sleep(3)
                continue
            except Exception as e:
                errors_by_key.append(f"Key {i+1} ({mk}): {str(e)[:120]}")
                _log.error("[Qwen] FAILED key %d (%s): %s", i + 1, mk, str(e)[:200])
                continue

    if rate_limited_count == len(sorted_keys):
        raise HTTPException(429, f"All Qwen keys are rate-limited. Wait ~60s then retry. Details: {'; '.join(errors_by_key[:3])}")
    raise HTTPException(502, f"All Qwen keys failed: {'; '.join(errors_by_key[:3])}")

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
        elif req.provider == "qwen":
            cv_data, key_used, key_idx = await call_qwen(req)
        else:
            raise HTTPException(400, f"Unsupported provider: {req.provider}")
        
        return {
            "cv": cv_data,
            "provider": req.provider,
            "model": req.model,
            "key_used": key_used,
            "key_index": key_idx,
            "ui_template": req.ui_template or "ui1",
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
    ui_template: Optional[str] = "ui1"   # "ui1" | "ui2" | "ui3"


# ==============================================================================
# SHARED CONTACT LINK HELPER — used by UI1, UI2, and UI3
# Detects phone numbers (any value with more than 4 digit characters) and
# returns a tel: URI. Also handles email (mailto:) and URLs (https://).
# ==============================================================================
import re as _contact_re

def _contact_href(val: str) -> str:
    """Return a clickable URI for a contact value, or '' if not linkable.

    Rules (applied in order):
      1. Already a full URI (https://, http://, mailto:, tel:) → use as-is.
      2. Contains @ → email → mailto:
      3. Contains more than 4 digit characters → phone → tel:
         Accepts any formatting: +92 318 4885878, (021) 1234-5678, etc.
      4. No spaces, contains a dot, looks like a domain → https://
      5. Anything else (location, freeform text) → '' (render as plain text).
    """
    v = (val or "").strip()
    if not v:
        return ""
    # Case 1: already a full URI
    if v.startswith(("https://", "http://", "mailto:", "tel:")):
        return v
    # Case 2: email
    if "@" in v:
        return f"mailto:{v}"
    # Case 3: phone — strip everything except digits, check count > 4
    _digits = _contact_re.sub(r"[^\d]", "", v)
    if len(_digits) > 4:
        # Keep +, digits, spaces, dashes, parens for the tel: value
        _tel_val = _contact_re.sub(r"\s+", "", v)
        return f"tel:{_tel_val}"
    # Case 4: bare URL
    if " " not in v and "." in v and _contact_re.search(r"[a-zA-Z0-9]\.[a-zA-Z]{2,}", v):
        return f"https://{v}"
    return ""


# ==============================================================================
# UI PDF BUILDERS — imported from UI/ package
# To add a new template: create UI/UI4.py with a build_cv_pdf_ui4 function,
# then register it in _PDF_BUILDERS below.
# ==============================================================================

# Inject shared helpers into the UI._shared shim BEFORE importing the builders,
# so each builder module can do `from UI._shared import ...` at module level.
import UI._shared as _ui_shared
_ui_shared._normalise_edu_entry = _normalise_edu_entry
_ui_shared._infer_degree_duration = _infer_degree_duration
_ui_shared._contact_href = _contact_href

from UI.UI1 import build_cv_pdf
from UI.UI2 import build_cv_pdf_ui2
from UI.UI3 import build_cv_pdf_ui3
from UI.UI4 import build_cv_pdf_ui4
from UI.UI5 import build_cv_pdf_ui5
from UI.UI6 import build_cv_pdf_ui6

# ==============================================================================
# PDF BUILDER REGISTRY — add new templates here only
# ==============================================================================
_PDF_BUILDERS = {
    "ui1": build_cv_pdf,       # Classic Executive (original)
    "ui2": build_cv_pdf_ui2,   # Modern Sidebar (teal two-column)
    "ui3": build_cv_pdf_ui3,   # Contemporary Card (slate-blue / gold)
    "ui4": build_cv_pdf_ui4,   # Executive Dark (charcoal sidebar / gold)
    "ui5": build_cv_pdf_ui5,   # Clean Minimal (white / emerald)
    "ui6": build_cv_pdf_ui6,   # Bold Split (indigo header / coral)
}


@app.post("/generate-pdf")
async def generate_pdf(req: PDFRequest):
    try:
        builder = _PDF_BUILDERS.get(req.ui_template or "ui1", build_cv_pdf)
        pdf_bytes = builder(req.cv, profile_data=req.profileData)
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

@app.post("/check-qwen-keys")
async def check_qwen_keys(body: dict):
    """Verify Qwen / DashScope API keys. Uses qwen-turbo with max_tokens=1 (cheapest check)."""
    keys  = body.get("keys", [])
    model = body.get("model", "qwen-turbo")
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        for key in keys:
            key = (key or "").strip()
            if not key:
                results.append({"key": "***", "status": "empty"})
                continue
            try:
                r = await client.post(
                    QWEN_URL,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                )
                if r.status_code == 200:
                    status = "ok"
                elif r.status_code == 429:
                    status = "rate_limited"
                elif r.status_code in (401, 403):
                    status = "invalid"
                else:
                    status = f"http_{r.status_code}"
                results.append({"key": mask(key), "status": status})
            except Exception:
                results.append({"key": mask(key), "status": "error"})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)