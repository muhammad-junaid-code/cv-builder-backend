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

    Priority:
      • profile_work entries (from UI)  → use their names, and dates when provided.
        If a profile work entry has both 'from' and 'to', use those dates verbatim.
        If dates are missing, calculate them from years_exp distribution.
      • years_exp (UI input)            → calculate timelines from today backwards.
      • Neither provided                → compute from CANDIDATE_COMPANIES spans.

    Company names: profile_work names → CANDIDATE_COMPANIES fallback → "Company N".
    """
    today = date.today()

    def fmt(d: date) -> str:
        return f"{_month_name(d.month)} {d.year}"

    # ── If profile_work is provided with complete date info, use it directly ──
    if profile_work:
        result = []
        for i, w in enumerate(profile_work):
            name    = (w.get("company") or "").strip() or \
                      (CANDIDATE_COMPANIES[i]["name"] if i < len(CANDIDATE_COMPANIES) else f"Company {i+1}")
            p_from  = (w.get("from") or "").strip()
            p_to    = (w.get("to")   or "").strip()
            # Use UI-provided dates when both ends are given
            if p_from and p_to:
                result.append({"name": name, "start": p_from, "end": p_to})
                continue
            # Only one end provided — store what we have; calc will fill the gap later
            result.append({"name": name, "start": p_from, "end": p_to,
                           "_partial": True})

        # If all entries have complete dates, return as-is
        if all(not r.get("_partial") for r in result):
            for r in result:
                r.pop("_partial", None)
            return result

        # Some entries lack dates — fall through to calculation for those entries
        # but preserve names from profile_work
        profile_names = [r["name"] for r in result]
    else:
        profile_names = None

    # ── Resolve numeric years ─────────────────────────────────────────────────
    if years_exp:
        raw = years_exp.strip().replace("+", "")
        try:
            n = float(raw)
        except ValueError:
            n = None
    else:
        n = None

    if n is None:
        # Compute from CANDIDATE_COMPANIES actual date spans
        total_months = 0
        for co in CANDIDATE_COMPANIES:
            try:
                s = _parse_month_year(co["start"])
                e = _parse_month_year(co["end"])
                total_months += _months_between(s, e)
            except Exception:
                pass
        n = total_months / 12 if total_months else 3.0

    total_months = int(round(n * 12))

    # ── Determine number of companies ─────────────────────────────────────────
    if num_companies > 0:
        num_cos = num_companies
    elif profile_names:
        num_cos = len(profile_names)
    elif n <= 1.4:
        num_cos = 1
    elif n <= 2.4:
        num_cos = 2
    else:
        num_cos = 3

    # ── Distribute months across companies ───────────────────────────────────
    each      = total_months // num_cos if num_cos > 0 else total_months
    remainder = total_months - each * num_cos

    result = []
    cursor = today
    for i in range(num_cos):
        span     = each + (remainder if i == 0 else 0)
        co_start = _subtract_months(cursor, span)
        co_end   = "Present" if i == 0 else fmt(cursor)
        # Name: profile_work name → CANDIDATE_COMPANIES name → generic
        if profile_names and i < len(profile_names):
            name = profile_names[i]
        elif i < len(CANDIDATE_COMPANIES):
            name = CANDIDATE_COMPANIES[i]["name"]
        else:
            name = f"Company {i + 1}"
        result.append({"name": name, "start": fmt(co_start), "end": co_end})
        cursor   = co_start

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


def _build_education_year(years_exp: str, profile_edu: list = None) -> dict:
    """
    Returns {"start": "YYYY", "end": "YYYY"} for the education block.

    Priority:
      1. Both dates explicit in profile_edu[0]  → use as-is.
      2. Only one date in profile_edu[0]         → infer the other (4-year degree).
      3. years_exp provided (UI input)           → graduation = current_year - years_exp + 1
                                                   start      = graduation - 4
      4. No info at all                          → graduation = current_year - 3 (assume ~3 yrs exp)
                                                   start      = graduation - 4

    This ensures no hardcoded fallback dates ever appear in the output.
    """
    today = date.today()

    # ── Priority 1 & 2: profile data ─────────────────────────────────────────
    if profile_edu:
        e = profile_edu[0]
        p_from = str(e.get("from", "") or "").strip()
        p_to   = str(e.get("to",   "") or "").strip()
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

    # ── Priority 3: years_exp from UI ────────────────────────────────────────
    if years_exp:
        try:
            n = int(float(years_exp.strip().replace("+", "")))
            # Graduation year = current year - experience + 1
            # (person graduated ~1 year before finishing their first year of work)
            grad_year  = today.year - max(n, 0) + 1
            start_year = grad_year - 4
            return {"start": str(start_year), "end": str(grad_year)}
        except (ValueError, TypeError):
            pass

    # ── Priority 4: no info — assume ~3 years experience ─────────────────────
    grad_year  = today.year - 3 + 1
    start_year = grad_year - 4
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

    # ── Build education block for prompt from profile data (already extracted above) ──
    # University — from profile if provided, else placeholder for AI
    _prompt_uni    = (_p_edu0.get("institution") or "").strip()
    _prompt_degree = (_p_edu0.get("degree")      or "").strip()
    _prompt_cgpa   = (_p_edu0.get("cgpa")        or "").strip()
    _prompt_ach    = (_p_edu0.get("achievement")  or "").strip()
    _prompt_edu_yr = f"{edu['start']} - {edu['end']}"

    # Only include education fields in the prompt that actually have values
    _edu_json_lines = []
    _edu_json_lines.append(f'    "university": "{_prompt_uni or ""}",')
    _edu_json_lines.append(f'    "degree": "{_prompt_degree or ""}",')
    _edu_json_lines.append(f'    "cgpa": "{_prompt_cgpa}",')
    _edu_json_lines.append(f'    "years": "{_prompt_edu_yr}",')
    _edu_json_lines.append(f'    "achievement": "{_prompt_ach}"')
    _edu_json_block = "\n".join(_edu_json_lines)

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
            f"All {num_cos} position(s) should use a JUNIOR-level seniority prefix "
            f"in the role title (e.g. 'Junior [domain] [function]')."
        )
    elif yrs_float <= 2.0:
        if num_cos == 1:
            seniority_guidance = (
                f"Total experience: {years_display}\n"
                f"Position 1 (most recent): use a JUNIOR-level seniority prefix."
            )
        else:
            seniority_guidance = (
                f"Total experience: {years_display}\n"
                f"Position 1 (most recent / earliest in career): use a JUNIOR-level seniority prefix.\n"
                f"Position 2 (later / current): NO seniority prefix — just the domain + function title."
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
   Copy the education values EXACTLY as pre-filled in the JSON template below.
   Do NOT invent or change university, degree, cgpa, years, or achievement.
   If a field is empty in the template, leave it empty — do not fill it in.
   Only the "years" field was calculated dynamically; use it as-is.

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
  "education": {{
{_edu_json_block}
  }},
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

def fix_companies(cv: dict, companies_list: list = None) -> dict:
    """
    Enforce correct company names/dates and clean up AI-generated role strings.

    Priority for names and dates:
      • companies_list (runtime-calculated from years_exp) — always preferred.
      • CANDIDATE_COMPANIES static fallback — only if companies_list is absent.

    This ensures timeline consistency: if the user said 2 years, the dates
    shown on the CV are the ones calculated from that 2-year span, not
    the static dates that might span 5 years.
    """
    companies = cv.get("companies", [])

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

        # ── Clean AI artefacts from role string (e.g. "Co1 ", "Co2 ") ────────
        role = co.get("role", "")
        role = re.sub(r'\bCo\d+\s*', '', role).strip()
        co["role"] = role

    return cv

# ==============================================================================
# CV COMPLETENESS CHECK + CONTINUATION
# ==============================================================================
_CV_REQUIRED_SECTIONS = ("companies", "projects", "skills", "competencies", "relatedTech")

def _is_cv_complete(result: dict) -> bool:
    """
    Return True only when all major sections are present and non-empty.
    Provider-agnostic: used after every AI call regardless of where it came from.
    """
    for key in _CV_REQUIRED_SECTIONS:
        val = result.get(key)
        if not val:                        # missing or empty list/string
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
    return True

def _missing_sections(result: dict) -> list:
    """Return list of section names that are absent or empty in result."""
    missing = []
    for key in _CV_REQUIRED_SECTIONS:
        val = result.get(key)
        if not val or (isinstance(val, list) and len(val) == 0):
            missing.append(key)
    return missing

async def _complete_cv_via_continuation(
    partial: dict, missing: list,
    client, key: str, model: str, url: str, headers: dict,
    req, companies_list: list, edu: dict
) -> dict:
    """
    Make a second LLM call to supply only the missing CV sections.
    Merges the continuation response into the partial result.
    Fully provider-agnostic — works for any OpenAI-compatible endpoint.
    """
    import json as _json

    already_have = {k: v for k, v in partial.items() if k not in missing}
    partial_json_str = _json.dumps(already_have, ensure_ascii=False)[:3000]

    co_lines = "\n".join(
        'Company {}: "{}" (Dates: {} - {})'.format(i + 1, c["name"], c["start"], c["end"])
        for i, c in enumerate(companies_list)
    )

    system = (
        "You are a world-class CV writer continuing an incomplete CV JSON generation. "
        "The previous response was cut off before it could finish all sections. "
        "Output ONLY raw JSON — no markdown fences, no explanation. "
        "The JSON must contain ONLY the missing sections listed below as top-level keys. "
        "Do NOT repeat sections that are already complete."
    )

    user = "\n".join([
        "JOB TITLE: " + req.job_title,
        "JOB DESCRIPTION (excerpt):",
        req.job_description[:1500],
        "",
        "WORK HISTORY:",
        co_lines,
        "",
        "ALREADY GENERATED (do NOT repeat these):",
        partial_json_str,
        "",
        "MISSING SECTIONS TO GENERATE NOW: " + ", ".join(missing),
        "",
        "Generate a JSON object containing ONLY the missing sections. "
        "Follow the same format and quality as the already-generated sections above. "
        "Output raw JSON only — no markdown, no preamble, no explanation.",
    ])

    _log.info("[ContinuationCall] Requesting missing sections: %s", missing)
    try:
        continuation = await call_llm_atomic(
            client, key, model, url, system, user,
            "Continuation", headers, max_tokens=6000
        )
        for k in missing:
            if k in continuation and continuation[k]:
                partial[k] = continuation[k]
        _log.info("[ContinuationCall] Merged — still missing: %s", _missing_sections(partial))
    except Exception as e:
        _log.warning("[ContinuationCall] Failed: %s — returning partial CV", e)

    return partial


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

        elif r.status_code == 400:
            # HTTP 400 often means max_tokens exceeds the model per-request cap.
            # Retry once with a halved token budget before giving up.
            err_body = r.text[:300]
            _log.warning("%s HTTP 400 on attempt %d — %s", tag, attempt_num, err_body)
            if attempt == 0 and max_tokens > 4000:
                reduced = max_tokens // 2
                _log.info("%s Retrying with reduced max_tokens=%d (was %d)", tag, reduced, max_tokens)
                max_tokens = reduced
                await asyncio.sleep(2)
                continue
            raise ValueError(f"HTTP 400 on {stage}: {err_body}")

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
                               url: str, headers: dict) -> dict:
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
                                    "FullCV", headers, max_tokens=8000, _deadline=_deadline)

    if not result:
        _log.error("[GenCV|%s] AI returned empty/unparseable response", provider_host)
        raise ValueError("AI returned empty response")

    _log.info("[GenCV|%s] AI response parsed — companies=%d projects=%d",
              provider_host,
              len(result.get("companies", [])),
              len(result.get("projects",  [])))

    # ── Completeness check: if the model truncated, request the missing sections ──
    missing = _missing_sections(result)
    if missing:
        _log.warning("[GenCV|%s] Incomplete CV — missing: %s — requesting continuation",
                     provider_host, missing)
        result = await _complete_cv_via_continuation(
            result, missing, client, key, model, url, headers,
            req, companies_list, edu
        )

    if "companies" not in result:
        result["companies"] = []

    for i, co in enumerate(result.get("companies", [])):
        if i < len(companies_list):
            co["company"]   = companies_list[i]["name"]
            co["dateRange"] = f"{companies_list[i]['start']} - {companies_list[i]['end']}"

    if "projects" not in result:
        result["projects"] = []

    # Build education block: profile data always wins over AI-generated values
    _p_edu_list = profile_edu or []
    _p_edu0     = _p_edu_list[0] if _p_edu_list else {}

    # Start with whatever the AI returned
    _ai_edu = result.get("education") or {}

    # Profile fields override AI fields when provided
    _merged_edu = {
        "university":  (_p_edu0.get("institution") or _ai_edu.get("university") or "").strip(),
        "degree":      (_p_edu0.get("degree")      or _ai_edu.get("degree")     or "").strip(),
        "cgpa":        (_p_edu0.get("cgpa")        or _ai_edu.get("cgpa")       or "").strip(),
        "years":       f"{edu['start']} - {edu['end']}",   # always calculated
        # Achievement ONLY from profile — never AI-invented
        "achievement": (_p_edu0.get("achievement") or "").strip(),
    }
    # Override the AI's education with the merged version
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
    cv_companies = fix_companies(cv_sanitised, companies_list=companies_list)
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

                    # ── Completeness check (provider-agnostic) ────────────────
                    _g_missing = _missing_sections(result)
                    if _g_missing:
                        _log.warning("[Gemini|%s] Incomplete CV — missing: %s — requesting continuation",
                                     mk, _g_missing)
                        import json as _jmod
                        _already = {k: v for k, v in result.items() if k not in _g_missing}
                        _partial_str = _jmod.dumps(_already, ensure_ascii=False)[:3000]
                        _co_lines_g = "\n".join(
                            'Company {}: "{}" (Dates: {} - {})'.format(_ci+1, _cc["name"], _cc["start"], _cc["end"])
                            for _ci, _cc in enumerate(companies_list)
                        )
                        _g_system = (
                            "You are a world-class CV writer continuing an incomplete CV JSON generation. "
                            "The previous response was cut off. "
                            "Output ONLY raw JSON with the missing sections as top-level keys."
                        )
                        _g_user = (
                            "JOB TITLE: {}\n"
                            "JOB DESCRIPTION (excerpt):\n{}\n\n"
                            "WORK HISTORY:\n{}\n\n"
                            "ALREADY GENERATED (do NOT repeat):\n{}\n\n"
                            "MISSING SECTIONS: {}\n"
                            "Generate ONLY the missing sections as a JSON object. Raw JSON only."
                        ).format(
                            req.job_title,
                            req.job_description[:1500],
                            _co_lines_g,
                            _partial_str,
                            ", ".join(_g_missing)
                        )
                        try:
                            _g_cont_url = "{}{}:generateContent?key={}".format(GEMINI_URL, model, key)
                            _g_cont_payload = {
                                "systemInstruction": {"parts": [{"text": _g_system}]},
                                "contents": [{"role": "user", "parts": [{"text": _g_user}]}],
                                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 6000},
                            }
                            _g_cont_r = await client.post(
                                _g_cont_url,
                                headers={"Content-Type": "application/json"},
                                json=_g_cont_payload,
                                timeout=120,
                            )
                            if _g_cont_r.status_code == 200:
                                _g_cont_raw = _g_cont_r.json()["candidates"][0]["content"]["parts"][0]["text"]
                                _g_cont_parsed = extract_json(_g_cont_raw)
                                for _gmk in _g_missing:
                                    if _gmk in _g_cont_parsed and _g_cont_parsed[_gmk]:
                                        result[_gmk] = _g_cont_parsed[_gmk]
                                _log.info("[Gemini|%s] Continuation merged — still missing: %s",
                                          mk, _missing_sections(result))
                            else:
                                _log.warning("[Gemini|%s] Continuation HTTP %d", mk, _g_cont_r.status_code)
                        except Exception as _ge:
                            _log.warning("[Gemini|%s] Continuation failed: %s", mk, _ge)

                    for j, co in enumerate(result.get("companies", [])):
                        if j < len(companies_list):
                            co["company"] = companies_list[j]["name"]
                            co["dateRange"] = f"{companies_list[j]['start']} - {companies_list[j]['end']}"

                    # Education: merge AI result with profile data; profile always wins
                    _p_edu0_g  = profile_edu[0] if profile_edu else {}
                    _ai_edu_g  = result.get("education") or {}
                    _merged_edu_g = {
                        "university":  (_p_edu0_g.get("institution") or _ai_edu_g.get("university") or "").strip(),
                        "degree":      (_p_edu0_g.get("degree")      or _ai_edu_g.get("degree")     or "").strip(),
                        "cgpa":        (_p_edu0_g.get("cgpa")        or _ai_edu_g.get("cgpa")       or "").strip(),
                        "years":       f"{edu['start']} - {edu['end']}",
                        "achievement": (_p_edu0_g.get("achievement") or "").strip(),  # never AI-invented
                    }

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
                    cv_companies = fix_companies(cv_sanitised, companies_list=companies_list)
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
    rate_limited_count = 0

    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                cv = await generate_cv_dynamic(req, client, key, model, DEEPSEEK_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
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
        raise HTTPException(429, f"All DeepSeek keys are rate-limited. Try again shortly. Details: {'; '.join(errors_by_key[:3])}")
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
    rate_limited_count = 0

    async with httpx.AsyncClient(timeout=180) as client:
        for i, key in enumerate(sorted_keys):
            mk = mask(key)
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                cv = await generate_cv_dynamic(req, client, key, model, OPENAI_URL, headers)
                _key_usage[mk] = _key_usage.get(mk, 0) + 1
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
        raise HTTPException(429, f"All OpenAI keys are rate-limited. Try again shortly. Details: {'; '.join(errors_by_key[:3])}")
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
    # Normalise edu entries: maps UI 'note' → 'achievement' and extracts cgpa from note
    p_edu = [_normalise_edu_entry(e) for e in (_pd.get("edu") or [])]
    
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
        "sec_title_center": ps("sec_c", fontName="Helvetica-Bold", fontSize=11, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#222222"), spaceBefore=10, spaceAfter=6),
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
        # ── Build a clickable token for every link ────────────────────────────
        # ReportLab supports <a href="...">text</a> inside Paragraph XML markup.
        # We auto-prefix bare addresses so they become valid URIs.
        def _make_href(val: str) -> str:
            """Return a valid URI for any link value the UI can produce."""
            v = val.strip()
            if not v:
                return ""
            # Already a full URI
            if v.startswith(("http://", "https://", "mailto:", "tel:")):
                return v
            # Email address
            if "@" in v and "." in v and "/" not in v:
                return f"mailto:{v}"
            # Phone number (digits, spaces, +, dashes, parentheses only)
            import re as _re
            if _re.fullmatch(r"[\d\s\+\-\(\)]+", v):
                return f"tel:{v.replace(' ', '')}"
            # Everything else → assume a URL; add https:// if missing
            return f"https://{v}"

        def _link_xml(label: str, val: str, color: str = "#0057A8") -> str:
            """Return ReportLab XML markup for a single clickable link token.
            Location is rendered in blue but NOT as a hyperlink.
            All other links are rendered as PDF hyperlinks (open in the system browser)."""
            safe_val = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Location: blue-coloured text, no hyperlink behaviour
            if label.strip().lower() == "location":
                return f'<font color="{color}">{safe_val}</font>'
            href = _make_href(val)
            if href:
                return f'<a href="{href}" color="{color}">{safe_val}</a>'
            return safe_val

        # Collect all non-empty link tokens
        contact_tokens = []
        for lnk in p_links:
            val = (lnk.get("value") or "").strip()
            if val:
                contact_tokens.append(_link_xml(lnk.get("label", ""), val))

        if contact_tokens:
            # Single centered paragraph — ReportLab wraps naturally at page width.
            # Center alignment (already set on S["contact"]) keeps every wrapped
            # line centered, producing the balanced layout shown in the mockup.
            SEP = ' <font color="#aaaaaa">|</font> '
            story.append(Paragraph(SEP.join(contact_tokens), S["contact"]))
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
        story.append(Paragraph("WORK EXPERIENCE", S["sec_title_center"]))
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
            colon = s.find(":")
            if colon > 0:
                category = s[:colon].strip()
                items    = s[colon + 1:].strip()
                # Bold the subheading, regular weight for items — ReportLab inline markup
                skill_html = f"<b>{category}:</b> {items}"
            else:
                skill_html = s
            story.append(Paragraph(skill_html, S["skill_items"]))
            story.append(Spacer(1, 2 * mm))
    
    # Projects
    projects = cv.get("projects", [])
    if projects:
        story.append(Paragraph("KEY PROJECTS", S["sec_title_center"]))
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
    
    # Education — only render when actual data is available
    # Priority: AI-generated edu (which itself was seeded from profile) → profile raw data
    edu = cv.get("education", {})
    uni         = (edu.get("university") or "").strip()
    degree      = (edu.get("degree")     or "").strip()
    years       = (edu.get("years")      or "").strip()
    achievement = (edu.get("achievement") or "").strip()
    cgpa        = (edu.get("cgpa")        or "").strip()

    # Profile education takes precedence for university/degree if AI left them blank
    if not uni and p_edu:
        uni = (p_edu[0].get("institution") or "").strip()
    if not degree and p_edu:
        degree = (p_edu[0].get("degree") or "").strip()
    # Achievement: only from profile — never show AI-invented medals
    if not achievement and p_edu:
        achievement = (p_edu[0].get("achievement") or "").strip()
    # CGPA: only from profile
    if not cgpa and p_edu:
        cgpa = (p_edu[0].get("cgpa") or "").strip()

    # Only render the Education section if there is something to show
    _has_edu = any([uni, degree, years, cgpa])
    if _has_edu:
        story.append(Paragraph("EDUCATION", S["sec_title_center"]))

        # ── University name (left) with years aligned to the right ───────────
        if uni:
            edu_date_style = ps("edu_dr",
                fontName="Helvetica", fontSize=10,
                alignment=TA_RIGHT,
                textColor=colors.HexColor("#666666")
            )
            uni_para   = Paragraph(uni.upper(), S["edu_uni"])
            years_para = Paragraph(years, edu_date_style) if years else Paragraph("", edu_date_style)
            edu_header_row = [[uni_para, years_para]]
            edu_header_tbl = Table(edu_header_row, colWidths=[TW * 0.65, TW * 0.35])
            edu_header_tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(edu_header_tbl)
        elif years:
            # No university name but years exist — show years on its own
            story.append(Paragraph(years, S["edu_deg"]))

        # ── Degree + CGPA line ────────────────────────────────────────────────
        deg_parts = [x for x in [degree] if x]
        if cgpa:
            deg_parts.append(f"CGPA: {cgpa}")
        deg_text = " | ".join(deg_parts)
        if deg_text:
            story.append(Paragraph(deg_text, S["edu_deg"]))

        # ── Achievement — only if explicitly in profile data ──────────────────
        if achievement:
            if "gold" in achievement.lower():
                story.append(Paragraph(f"🏅 {achievement}", S["edu_medal"]))
            else:
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