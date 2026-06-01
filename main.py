from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import httpx, json, asyncio, re, hashlib

GROQ_API_KEYS = [
    "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",
]
# llama-3.3-70b-versatile: best quality on Groq, still very fast (~200 tok/s)
GROQ_MODEL      = "llama-3.3-70b-versatile"
# llama-3.1-8b-instant: used only for fast utility calls (fix, translate, extract)
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
current_key_index = 0
current_cerebras_key_index = 0

DEEPGRAM_API_KEY = "cab42f46126a2def0c2a9c9086ba24b39f089117"
DEEPGRAM_URL     = "https://api.deepgram.com/v1/listen"
# nova-3 has significantly better accuracy for accented/Pakistani English speech.
# It uses 'keyterm' param (not 'keywords') for boosting.
DEEPGRAM_MODEL   = "nova-3"   # best accuracy for accented English; uses 'keyterm' param

CEREBRAS_URL     = "https://api.cerebras.ai/v1/chat/completions"

GEMINI_BASE_URL  = "https://generativelanguage.googleapis.com"
# Gemini model → API name mapping (kept dynamic — just the base name used in URLs)
# The popup sends "gemini-2.5-flash" or "gemini-2.5-flash-lite"; we pass straight through.

# Models that rejected the enable_thinking flag with a 400 — learned at runtime.
# Persists for the server lifetime so the same 400 round-trip never happens twice.
_CEREBRAS_NO_THINKING: set = set()

# Cross-session context extraction cache.
# Key: sha1(job_title + job_desc + cv[:200]) → extracted dict.
# Avoids the ~0.5s Groq round-trip on every reconnect for the same job.
_SESSION_CTX_CACHE: dict = {}
_SESSION_CTX_CACHE_MAX = 20  # cap memory use

app = FastAPI()

# ── Cross-reconnect session store ─────────────────────────────────────────────
# Chrome's offscreen document WebSocket can be closed by the browser after ~30s
# of apparent inactivity (no new audio arriving at the SERVER — chunks are still
# buffered client-side in pendingChunks). When this happens the client reconnects
# and sends a CONTEXT message containing the same sessionId as before.
# We keep audio_chunks alive in this dict so the reconnect picks up where it left
# off instead of starting with an empty buffer.
#
# Layout: { sessionId: { "chunks": [...], "history": [...], ...session fields } }
# Sessions are evicted after 10 minutes of inactivity to prevent memory growth.
import time as _time
_SESSION_STORE: dict = {}
_SESSION_TTL_S  = 600   # 10 minutes

def _evict_stale_sessions():
    now = _time.monotonic()
    stale = [k for k, v in _SESSION_STORE.items() if now - v.get("_ts", 0) > _SESSION_TTL_S]
    for k in stale:
        del _SESSION_STORE[k]


# Whisper model selection strategy:
#   tiny.en  — English-only, fastest (~2-3s). Used for lang_mode == "english"
#              AND as pass-1 in "auto" mode (language detection pass).
#   small    — Best multilingual accuracy on CPU. Used for lang_mode == "urdu"
#              AND as pass-2 in "auto" mode when Urdu/Roman-Urdu is detected.
#              "small" beats "base" significantly for Urdu — base mishears common
#              Roman Urdu words and hallucinates on accented Pakistani English.
#              small is ~3x more parameters and much more accurate at ~5-7s CPU cost.
#
# Auto mode — two-pass strategy:
#   Pass 1 (always):  tiny.en, forced "en" → detect_language() check (~2-3s)
#   Pass 2 (if Urdu): small,   forced "ur" → accurate Roman Urdu transcript (~5-7s total)
#   English-only speech = no extra cost. Urdu speech = pass-2 only when needed.
# ── Model thread config ───────────────────────────────────────────────────────
# cpu_threads: how many CPU threads each model's encoder/decoder may use.
#   tiny.en: 4 threads is optimal — model is small enough that more threads
#            add scheduling overhead without improving throughput.
#   small:   8 threads — larger model benefits from more parallelism.
#            On a 4-core CPU this uses hyperthreading; on 8-core it's ideal.
# num_workers: parallel audio-chunk decode workers (PyAV demuxing).
_model_en    = WhisperModel("tiny.en", device="cpu", compute_type="int8", num_workers=2, cpu_threads=4)
_model_multi = WhisperModel("small",   device="cpu", compute_type="int8", num_workers=2, cpu_threads=8)

# ── Warm up both Whisper models at startup ────────────────────────────────────
# faster-whisper downloads/compiles weights lazily on the first .transcribe() call.
# Without warmup the first WS session blocks for minutes while small (~460MB) loads,
# causing "Server not connected". We trigger the load now using a valid silent WAV.
import io as _io, struct as _struct, threading as _threading, wave as _wave

def _make_silent_wav(duration_ms: int = 100) -> bytes:
    """Return a valid mono 16kHz 16-bit WAV file containing silence."""
    sample_rate  = 16000
    n_samples    = sample_rate * duration_ms // 1000
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()

_SILENT_WAV = _make_silent_wav(100)   # 100ms silence, created once

def _warmup_model(m, label):
    try:
        segs, _ = m.transcribe(
            _io.BytesIO(_SILENT_WAV),
            language="en", beam_size=1,
            vad_filter=True,
        )
        list(segs)   # consume iterator to force full model load
        print(f"[Warmup] {label} ready")
    except Exception as e:
        print(f"[Warmup] {label} warmup error (will retry on first use): {e}")

# Daemon threads — uvicorn starts immediately; models load in background.
# tiny.en finishes in ~1-2s. small may take longer on first run (downloads ~460MB).
# Wait for "[Warmup] small ready" in the log before using the extension.
_threading.Thread(target=_warmup_model, args=(_model_en,    "tiny.en"), daemon=True).start()
_threading.Thread(target=_warmup_model, args=(_model_multi, "small"),   daemon=True).start()

http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10))

# Dedicated client for Gemini STT.
# Read/write timeouts scale with audio size at call time via the per-request override
# (see transcribe_with_gemini). Base timeout is generous so large payloads are not
# cut short; Gemini always falls back to Whisper on any error.
_gemini_stt_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5, read=60, write=60, pool=5),
)

# Dedicated executor for Whisper transcription.
# max_workers=2: allows a second WS session to start pass-1 while the first is
# running pass-2, preventing head-of-line blocking between sessions.
# Each worker is a dedicated OS thread; two threads share CPU time but don't
# contend on Python's GIL since faster-whisper releases it during C inference.
import concurrent.futures as _cf
_whisper_executor = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")

# ── Keyword sets for fast O(1) single-word lookups ───────────────────────────
# Multi-word phrases (e.g. "difference between") still need substring search.
# Single-word entries are split out into frozensets for faster membership tests.
def _split_kw_list(lst):
    """Split keyword list into (single_words_set, multi_word_phrases_list)."""
    singles = frozenset(w for w in lst if " " not in w)
    multis  = [w for w in lst if " " in w]
    return singles, multis

# (populated below after word-list constants are fully defined)

def _fast_match(transcript_lower: str, singles: frozenset, multis: list) -> bool:
    """Check transcript against a precompiled keyword set + phrase list."""
    words = transcript_lower.split()
    if any(w in singles for w in words):
        return True
    return any(phrase in transcript_lower for phrase in multis)

import urllib.parse

# ── Language detection ────────────────────────────────────────────────────────
URDU_UNICODE_RANGE  = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
HINDI_UNICODE_RANGE = re.compile(r'[\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF]')
ROMAN_SOUTH_ASIAN_WORDS = {
    # Core question words
    "kya","kyun","kaise","kab","kahan","kaun","kitna","konsa","kitne",
    # Pronouns / address
    "aap","ap","mujhe","mera","meri","mere","yeh","woh","hum","tum","unka",
    # Common verbs / verb forms
    "hai","hain","tha","thi","the","ho","hoga","hogi","hote","hoti",
    "bata","batao","bataiye","samajh","samjhao","samjhaiye","pata","maloom",
    "karo","karna","karta","karti","karein","karte","kiya","kiye","karo",
    "saktay","sakta","sakti","saken","chahiye","chahie",
    "use","istemaal","lagao","likhain","dikhao","dekho",
    # Particles / connectors
    "toh","phir","lekin","aur","ya","ke","ka","ki","ko","se","par","mein",
    "ek","isko","usko","inhe","unhe","jo","jab","jaise","warna","phir","matlab",
    # Responses / affirmations
    "ji","jee","haan","nahi","nahin","bilkul","zaroor","theek","achha","accha",
    "shayad","waise","isliye","abhi","pehle","baad","saath",
    # Interview-specific Roman Urdu
    "farq","fark","mukhtalif","complexity","mushkil","asaan","behtar","zyada",
    "thoda","bahut","zyada","kam","bari","choti","puri","poori",
    "interview","sawaal","jawab","baat","kaam","karo","lagta","lagti",
    # Ownership / identity
    "apna","apni","apne","humara","hamara","tumhara","mera","tera","unka",
    "shukriya","meherbani","please","acha",
}

# Universal terms that are commonly misheared regardless of tech stack
UNIVERSAL_KEYWORDS = [
    "API", "REST", "GraphQL", "WebSocket", "HTTP", "HTTPS",
    "JWT", "OAuth", "CORS", "CI/CD", "Docker", "Kubernetes",
    "microservices", "SQL", "NoSQL", "Redis", "MongoDB",
    "Git", "GitHub", "GitLab", "Agile", "Scrum", "TDD",
    "async", "await", "concurrency", "thread", "deadlock",
    "dependency injection", "design pattern", "SOLID",
    "load balancer", "cache", "CDN", "cloud", "serverless",
]

# Pre-compiled regex for multi-concept detection — compiled once at startup,
# reused on every PROCESS call instead of being compiled inside the hot loop.
_CONCEPT_SPLIT_RE = re.compile(r',\s*|\s+and\s+|\s+or\s+')


def detect_language(text: str) -> str:
    if not text:
        return "english"
    total_chars = len(text.replace(" ", ""))
    urdu_chars  = len(URDU_UNICODE_RANGE.findall(text))
    if total_chars > 0 and urdu_chars / total_chars > 0.2:
        return "foreign_script"
    hindi_chars = len(HINDI_UNICODE_RANGE.findall(text))
    if total_chars > 0 and hindi_chars / total_chars > 0.2:
        return "foreign_script"
    words_lower = text.lower().split()
    hits = sum(1 for w in words_lower if w in ROMAN_SOUTH_ASIAN_WORDS)
    if hits >= 2 or (hits >= 1 and len(words_lower) <= 5):
        return "roman_south_asian"
    return "english"


# ── Session context extraction ────────────────────────────────────────────────
# Called ONCE per session when job context is received.
# Extracts: tech stack, programming language, domain keywords — all dynamically.
# Result is cached on the session object so no repeated LLM calls.

async def extract_session_context(job_title: str, job_desc: str, cv: str, api_key: str) -> dict:
    """
    Uses a fast LLM to extract structured context from the job posting.
    Returns:
      - tech_keywords: list of terms to boost in STT (Deepgram keyterms / Whisper prompt)
      - primary_language: main programming language (Python, JavaScript, C#, Java, etc.)
      - domain: short domain label (backend, frontend, fullstack, data, devops, mobile, etc.)
      - stack_summary: one-line summary of the tech stack for use in prompts
    """
    if not job_title and not job_desc:
        return {
            "tech_keywords": [],
            "primary_language": "the relevant language",
            "domain": "software engineering",
            "stack_summary": "a general software engineering role",
        }

    input_text = f"Job Title: {job_title}\n"
    if job_desc:
        input_text += f"Job Description:\n{job_desc}"
    if cv:
        input_text += f"\n\nCandidate CV / Resume:\n{cv}"

    try:
        resp = await http_client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_FAST_MODEL,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Extract structured info from this job posting and CV. "
                        "Respond with ONLY valid JSON, no explanation, no markdown.\n\n"
                        "JSON format:\n"
                        "{\n"
                        '  "tech_keywords": ["list", "of", "40", "domain-specific", "terms"],\n'
                        '  "primary_language": "for tech roles: main language e.g. Python/C#/JS. For non-tech: write \'none\'",\n'
                        '  "domain": "one of: backend, frontend, fullstack, mobile, data, devops, ml, embedded, qa, seo, marketing, sales, hr, finance, design, content, product, operations, customer_success",\n'
                        '  "domain_type": "one of: technical OR non_technical",\n'
                        '  "role_category": "one descriptive phrase e.g. \'SEO & Content Strategist\' or \'Senior .NET Developer\'",\n'
                        '  "stack_summary": "one sentence summarising the core skills/tools e.g. \'SEO strategy with Ahrefs, Semrush, Google Analytics and content marketing\' or \'Python Django REST API with PostgreSQL and AWS\'"\n'
                        "}\n\n"
                        "Rules for tech_keywords — CRITICAL:\n"
                        "- For TECHNICAL roles: include every framework, library, ORM, tool, service, database, pattern\n"
                        "- For NON-TECHNICAL roles (SEO, marketing, sales, HR, finance, design, content, product):\n"
                        "  include every tool (Ahrefs, Semrush, Google Analytics, HubSpot, Salesforce, Figma etc.)\n"
                        "  every methodology (on-page SEO, link building, A/B testing, content strategy etc.)\n"
                        "  every platform (Google Ads, Meta Ads, LinkedIn, Shopify etc.)\n"
                        "  every metric (CTR, CPC, ROAS, DA, DR, conversion rate, bounce rate etc.)\n"
                        "- Include terms a speech-to-text model might mishear (proper nouns, acronyms, brand names)\n"
                        "- Max 40 items, most important first\n"
                        "- Do NOT include generic words like 'experience', 'team', 'skills'\n\n"
                        f"{input_text}"
                    )
                }],
                "max_tokens": 600,
                "temperature": 0.0,
            }
        )
        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        # Strip any markdown fences if the model adds them despite instructions
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        print(f"[Session] Stack: {parsed.get('stack_summary','?')} | Lang: {parsed.get('primary_language','?')} | Domain: {parsed.get('domain','?')}")
        print(f"[Session] Keywords extracted: {parsed.get('tech_keywords',[])}")
        return parsed
    except Exception as e:
        print(f"[Session extract error] {e}")
        # Fallback: extract capitalised tokens from job desc directly
        fallback_keywords = []
        if job_title:
            fallback_keywords += [w.strip("().,") for w in job_title.split() if len(w) > 2]
        if job_desc:
            tokens = re.findall(r'\b[A-Z][A-Za-z.]*[A-Z]\w*|\b[A-Z]{2,}\b', job_desc[:500])
            fallback_keywords += tokens[:15]
        return {
            "tech_keywords": list(dict.fromkeys(fallback_keywords))[:20],
            "primary_language": "the relevant language",
            "domain": "software engineering",
            "stack_summary": job_title or "a technical role",
        }


# ── STT ───────────────────────────────────────────────────────────────────────

async def transcribe_with_deepgram(
    audio_blob: bytes,
    session_ctx: dict,
    lang_mode:   str = "auto",
) -> str:
    dg_language = "ur" if lang_mode == "urdu" else "en"

    # Combine universal terms + session-specific tech keywords
    tech_kw  = session_ctx.get("tech_keywords", [])
    keywords = list(dict.fromkeys(UNIVERSAL_KEYWORDS + tech_kw))[:40]

    # nova-3 base params — no 'keywords' param (that's nova-2 only)
    # nova-3 uses 'keyterm' for boosting specific terms
    params = {
        "model":            DEEPGRAM_MODEL,
        "language":         dg_language,
        "smart_format":     "true",
        "punctuate":        "true",
        "filler_words":     "false",
        "diarize":          "false",
        "profanity_filter": "false",
    }

    url = DEEPGRAM_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    if keywords:
        # nova-3 uses 'keyterm=term' (no boost value, just the term)
        url += "&" + "&".join(
            "keyterm=" + urllib.parse.quote(kw) for kw in keywords
        )

    blob_mb = len(audio_blob) / (1024 * 1024)
    print(f"[Deepgram] {blob_mb:.1f} MB | Lang: {dg_language} | Keyterms: {len(keywords)}")

    try:
        resp = await http_client.post(
            url,
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/webm"},
            content=audio_blob,
        )
        if resp.status_code != 200:
            print(f"[Deepgram] HTTP {resp.status_code}: {resp.text[:200]}")
            return ""

        transcript = (
            resp.json()
                .get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
                .strip()
        )
        print(f"[Deepgram] {len(transcript)} chars")
        return transcript

    except httpx.TimeoutException as e:
        print(f"[Deepgram timeout] {e}")
        return ""
    except Exception as e:
        print(f"[Deepgram error] {e}")
        return ""


async def transcribe_with_whisper(
    audio_blob:  bytes,
    session_ctx: dict,
    lang_mode:   str = "auto",
) -> str:
    # Pass audio as BytesIO — no temp file, no disk I/O.
    # faster-whisper's decode_audio() accepts BinaryIO via PyAV's av.open(),
    # which handles webm/opus in-memory exactly like a file path.
    # On Windows this saves ~150-400ms of NTFS write + fsync latency per call.

    # ── Model + parameter selection ───────────────────────────────────────────
    #
    # english → tiny.en, forced "en", English tech-keyword prompt.
    #           Fastest possible path — no language detection overhead.
    #
    # urdu    → base multilingual, forced "ur", Urdu-script hint prompt.
    #           Always uses the multilingual model since input is Urdu only.
    #
    # auto    → TWO-PASS strategy for best speed + accuracy:
    #   Pass 1: tiny.en (forced "en") — runs in ~2s, identical to English mode.
    #           We check the result with detect_language().
    #           • If English  → done. Return immediately (same speed as english mode).
    #           • If South Asian (Roman Urdu / Urdu script) → run Pass 2.
    #   Pass 2: base multilingual (forced "ur") — only runs when Urdu detected.
    #           Produces accurate Roman Urdu transcription.
    #           ~4-5s total (2s pass-1 + 2-3s pass-2), only paid when needed.
    #
    # This means auto mode is IDENTICAL in speed to english mode for English
    # speech, and only incurs the multilingual overhead when Urdu is actually spoken.

    loop = asyncio.get_event_loop()
    _estimated_duration_s = len(audio_blob) / 3_000
    _en_prompt  = build_whisper_prompt(session_ctx)
    # Rich bilingual prompt — Urdu script sets the domain register; Roman Urdu
    # phrases prime the decoder for the exact vocabulary used in Pakistani tech
    # interviews. Whisper uses this as prior context before hearing any audio,
    # dramatically reducing phonetic substitution errors on words like "kya",
    # "bata", "samjhao", "farq", "kaise", "hota", "use", "karna", "matlab".
    # Roman Urdu prompt: starts with English characters so Whisper's decoder
    # initialises in Latin-script mode, strongly biasing it to output Roman Urdu
    # (Urdu words spelled with English letters) rather than native Urdu script.
    # The Urdu script phrase at the end anchors the language identity.
    _ur_prompt  = (
        "Roman Urdu interview. Kya aap bata saktay hain? Farq kya hai? "
        "Kaise kaam karta hai? Kab use karna chahiye? Samjhao, explain karo. "
        "Main Python use karta hoon. Entity Framework mein complexity aa sakti hai. "
        "Performance, architecture, deployment. "
        "یہ ایک جاب انٹرویو ہے۔"
    )

    def _run_whisper(wmodel, wlang, wprompt):
        """Run Whisper synchronously — called via executor."""
        import io as _io
        _long = _estimated_duration_s > 25
        # is_multi: True when this is the small multilingual model (pass-2 / urdu mode).
        # The small model gets tighter VAD to strip silence faster — the main
        # speed cost on CPU is encoder passes on silent frames, not decoder steps.
        _is_multi = (wmodel is _model_multi)
        kwargs = dict(
            beam_size=1,          # greedy — fastest; accuracy gap vs beam=5 is minimal
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=150 if _is_multi else 250,  # trim silence faster
                speech_pad_ms=200         if _is_multi else 300,    # less padding
                threshold=0.50            if _is_multi else 0.45,   # stricter speech gate
            ),
            no_speech_threshold=0.65,
            compression_ratio_threshold=2.4,  # was 2.8 — reject more repetition faster
            word_timestamps=False,
            condition_on_previous_text=False,  # disable — saves ~15% time, not needed for Q&A
            temperature=0.0,
            initial_prompt=wprompt,
        )
        if wlang:
            kwargs["language"] = wlang
        return wmodel.transcribe(_io.BytesIO(audio_blob), **kwargs)

    def _is_hallucination(text: str, detected_lang: str, scope_lang_mode: str) -> bool:
        """Return True if the transcript looks like multilingual-model hallucination."""
        if not text:
            return False
        words = text.lower().split()
        if len(words) >= 4:
            from collections import Counter as _Counter
            top_count = _Counter(words).most_common(1)[0][1]
            if top_count / len(words) > 0.5:
                return True
        # Unexpected language detection (Arabic, Chinese, etc.) = noise hallucination
        _out_of_scope = {"ar", "zh", "ja", "ko", "he", "fa", "hi"}
        if scope_lang_mode in ("auto", "urdu") and detected_lang in _out_of_scope:
            return True
        return False

    # ── English mode ──────────────────────────────────────────────────────────
    if lang_mode == "english":
        segs, info = await loop.run_in_executor(
            _whisper_executor, _run_whisper, _model_en, "en", _en_prompt
        )
        transcript = " ".join(s.text for s in segs).strip()
        print(f"[Whisper/en] Lang: {getattr(info,'language','?')} | {len(transcript)} chars | {len(audio_blob)//1024}KB")
        return transcript

    # ── Urdu mode ─────────────────────────────────────────────────────────────
    if lang_mode == "urdu":
        segs, info = await loop.run_in_executor(
            _whisper_executor, _run_whisper, _model_multi, "ur", _ur_prompt
        )
        transcript = " ".join(s.text for s in segs).strip()
        detected_wlang = getattr(info, 'language', '?')
        print(f"[Whisper/ur] Lang: {detected_wlang} | {len(transcript)} chars | {len(audio_blob)//1024}KB")
        if _is_hallucination(transcript, detected_wlang, "urdu"):
            print(f"[Whisper/ur] Hallucination — discarding")
            return ""
        return transcript

    # ── Auto mode — two-pass (English-first, Urdu fallback) ──────────────────
    # Pass 1: tiny.en — fast English transcription
    segs1, info1 = await loop.run_in_executor(
        _whisper_executor, _run_whisper, _model_en, "en", _en_prompt
    )
    transcript1    = " ".join(s.text for s in segs1).strip()
    detected_lang1 = getattr(info1, 'language', 'en')
    print(f"[Whisper/auto-pass1] Lang: {detected_lang1} | {len(transcript1)} chars | {len(audio_blob)//1024}KB")

    # Detect whether audio contains South Asian (Urdu/Roman-Urdu) speech.
    #
    # Strategy: run detect_language() on pass-1 output first (catches clean Roman Urdu).
    # Additionally: pass the RAW audio through the small multilingual model's own
    # language-detect path by checking what language "small" would assign — this catches
    # cases where tiny.en garbled the Urdu words beyond recognition (e.g. "kya" → "chia").
    # We do this lightweight check by running a tiny segment of the multilingual model's
    # language-id step, which is far cheaper than a full transcription.
    #
    # Practical trigger rules (any one is sufficient to run pass-2):
    #   1. detect_language() on pass-1 text returns roman_south_asian / foreign_script.
    #   2. The pass-1 transcript looks suspiciously garbled for its length
    #      (many short non-English-looking tokens — heuristic for Urdu being forced to "en").

    lang_check = detect_language(transcript1)
    is_south_asian = lang_check in ("roman_south_asian", "foreign_script")

    # ── Pass-2 trigger heuristics ─────────────────────────────────────────────
    # When tiny.en is forced to transcribe Urdu audio it does one of three things:
    #   A) Produces recognisable Roman Urdu words  → caught by detect_language()
    #   B) Produces phonetic English garbage       → caught by _GARBLED_URDU_SIGNALS
    #   C) Hallucinates a repeated English phrase  → caught by _is_pass1_repetitive
    #      e.g. "I'm going to show you how to do it. I'm going to show you..."
    #      This is the most common failure mode for Urdu audio through tiny.en.
    # Any one trigger is sufficient to run pass-2.

    # Trigger A: detect_language found Roman Urdu vocabulary (already computed above)

    # Trigger B: transcript contains known phonetic distortions tiny.en makes for Urdu
    _GARBLED_URDU_SIGNALS = {
        "chia", "chya", "kya", "ap", "aap", "hai", "hain", "toh",
        "samjh", "bata", "karo", "lekin", "matlab", "theek", "bilkul",
        "zaroor", "farq", "mushkil", "puri", "bahut", "zyada", "abhi",
    }
    words1 = transcript1.lower().split()
    _garbled_hits = sum(1 for w in words1 if w.strip(".,?!") in _GARBLED_URDU_SIGNALS)

    # Trigger C: pass-1 output is repetitive — tiny.en hallucinating on Urdu audio.
    # "I'm going to show you how to do it. I'm going to show you how to do it."
    # Detect by checking if any 4-gram (sequence of 4 words) appears 2+ times.
    _is_pass1_repetitive = False
    if len(words1) >= 8:
        from collections import Counter as _Counter2
        _ngrams = [" ".join(words1[i:i+4]) for i in range(len(words1) - 3)]
        _top_ngram_count = _Counter2(_ngrams).most_common(1)[0][1] if _ngrams else 0
        if _top_ngram_count >= 2:
            _is_pass1_repetitive = True
            print(f"[Whisper/auto] Pass-1 repetition detected — Urdu audio forced through tiny.en")

    # Trigger D: audio long enough for a question but pass-1 returned almost nothing
    _short_for_audio = len(words1) < 4 and len(audio_blob) > 30_000

    if not is_south_asian and _garbled_hits < 1 and not _is_pass1_repetitive and not _short_for_audio:
        # Confident English — return pass-1 result immediately, zero extra latency.
        print(f"[Whisper/auto] English detected — returning pass-1 result")
        return transcript1

    if is_south_asian:       reason = lang_check
    elif _is_pass1_repetitive: reason = "pass1_repetitive"
    elif _garbled_hits:      reason = f"garbled_hits={_garbled_hits}"
    else:                    reason = "short_for_audio"

    # Pass 2: small multilingual — accurate Urdu-mode re-transcription
    print(f"[Whisper/auto] Urdu detected (reason='{reason}') — running Urdu pass")
    segs2, info2 = await loop.run_in_executor(
        _whisper_executor, _run_whisper, _model_multi, "ur", _ur_prompt
    )
    transcript2    = " ".join(s.text for s in segs2).strip()
    detected_lang2 = getattr(info2, 'language', '?')
    print(f"[Whisper/auto-pass2] Lang: {detected_lang2} | {len(transcript2)} chars | {len(audio_blob)//1024}KB")

    if _is_hallucination(transcript2, detected_lang2, "auto"):
        print(f"[Whisper/auto] Pass-2 hallucination — falling back to pass-1")
        return transcript1  # prefer imperfect English over hallucinated nonsense

    # ── Urdu script → Roman Urdu conversion ──────────────────────────────────
    # The "small" model forced to "ur" often outputs native Urdu script (Arabic
    # letters) instead of Roman Urdu. Detect this and transliterate via a fast
    # lookup table so the AI receives readable Roman Urdu text.
    # If the transcript is mostly Latin already, this is a no-op.
    _urdu_char_count = sum(1 for c in transcript2 if '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF')
    if _urdu_char_count > len(transcript2) * 0.25:
        print(f"[Whisper/auto] Urdu script detected in pass-2 ({_urdu_char_count} chars) — keeping as-is for AI")
        # Keep the Urdu script — translate_to_english will handle it in the PROCESS pipeline
        # (it checks detected_lang and translates foreign_script to English before AI call)
    return transcript2


async def transcribe_with_gemini(
    audio_blob:  bytes,
    session_ctx: dict,
    lang_mode:   str = "auto",
    gemini_key:  str = "",
    stt_model:   str = "",
) -> str:
    """
    Transcribe audio using Google Gemini multimodal API.
    Uses stt_model if provided (e.g. the user's selected ai_model when provider=gemini),
    otherwise falls back to gemini-2.0-flash which has native audio understanding.
    Falls back to Whisper if no API key or on any error.
    """
    if not gemini_key:
        print("[Gemini STT] No API key — falling back to Whisper")
        return await transcribe_with_whisper(audio_blob, session_ctx, lang_mode)

    import base64
    # Prefer the user's selected model; fall back to gemini-2.0-flash (strong audio model)
    _stt_gemini_model = stt_model if stt_model else "gemini-2.0-flash"

    # Keep STT prompt minimal — fewer input tokens = lower TTFT from Gemini.
    # Only 8 top keywords (enough for proper-noun bias), no stack summary.
    tech_kw = session_ctx.get("tech_keywords", [])
    kw_hint = (", ".join(tech_kw[:8]) + ". ") if tech_kw else ""

    if lang_mode == "urdu":
        lang_hint = "Urdu/Roman-Urdu audio. "
    elif lang_mode == "english":
        lang_hint = "English audio. "
    else:
        lang_hint = ""

    prompt = (
        f"Transcribe verbatim. {lang_hint}{kw_hint}"
        "Return ONLY the transcript text."
    )

    blob_mb = len(audio_blob) / (1024 * 1024)
    print(f"[Gemini STT] {blob_mb:.1f} MB | Lang: {lang_mode}")

    # Scale maxOutputTokens with audio size so long recordings aren't truncated.
    # webm/opus at ~32kbps ≈ 4KB/s → ~150 tokens/min of speech.
    # We estimate generously (5 tokens/s) and add a 200-token headroom buffer.
    # Clamped between 300 (minimum useful) and 4096 (Gemini output cap).
    _estimated_duration_s = len(audio_blob) / 4_000   # conservative: 4KB/s
    _max_output_tokens    = max(300, min(4096, int(_estimated_duration_s * 5) + 200))

    # Offload base64 encoding + JSON serialisation to thread executor.
    # b64encode on 150KB + json.dumps of the full payload blocks the event loop
    # for 10-30ms — offloading keeps the loop free for other coroutines.
    import base64 as _b64mod
    def _build_stt_payload():
        _audio_b64 = _b64mod.b64encode(audio_blob).decode("utf-8")
        return {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "audio/webm", "data": _audio_b64}},
                    {"text": prompt},
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": _max_output_tokens},
        }

    _loop_stt = asyncio.get_event_loop()
    stt_payload = await _loop_stt.run_in_executor(None, _build_stt_payload)

    # Dynamic per-request timeout: 10s base + 1s per estimated 10s of audio.
    # This lets short clips fail fast to Whisper while long recordings get the time
    # they need. Capped at 55s so we always respond before the client gives up.
    _dyn_read_timeout = max(10, min(55, 10 + int(_estimated_duration_s / 10)))

    try:
        resp = await _gemini_stt_client.post(
            f"{GEMINI_BASE_URL}/v1beta/models/{_stt_gemini_model}:generateContent?key={gemini_key}",
            headers={"Content-Type": "application/json"},
            json=stt_payload,
            timeout=httpx.Timeout(connect=5, read=_dyn_read_timeout, write=60, pool=5),
        )
        if resp.status_code != 200:
            print(f"[Gemini STT] HTTP {resp.status_code}: {resp.text[:200]} — falling back to Whisper")
            return await transcribe_with_whisper(audio_blob, session_ctx, lang_mode)

        transcript = (
            resp.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
        )
        print(f"[Gemini STT] {len(transcript)} chars")
        return transcript

    except httpx.TimeoutException as e:
        print(f"[Gemini STT timeout] {e} — falling back to Whisper")
        return await transcribe_with_whisper(audio_blob, session_ctx, lang_mode)
    except Exception as e:
        print(f"[Gemini STT error] {e} — falling back to Whisper")
        return await transcribe_with_whisper(audio_blob, session_ctx, lang_mode)


def build_whisper_prompt(session_ctx: dict) -> str:
    """
    Build a Whisper initial_prompt from extracted session context.

    Whisper's initial_prompt is prepended to the audio as fake prior context.
    The decoder uses it to bias token probabilities before hearing a single frame.
    A prompt that reads like natural interview dialogue containing the domain's
    technical terms is dramatically more effective than a bare keyword list:

      - A keyword list ("ASP.NET, Singleton, Scoped, Transient") gives each term
        equal weight with no surrounding grammar — the decoder can still drift to
        phonetically similar common words ("simpleton", "school", "trust").
      - A sentence like "Can you explain the difference between Singleton, Scoped,
        and Transient lifetimes in ASP.NET Core dependency injection?" makes the
        decoder expect those exact tokens in an interview question frame, making
        substitution far less likely.

    We build two sentences:
      1. A role/stack sentence that sets the domain register.
      2. A sample question sentence that naturally embeds the top keywords —
         the most likely interview question forms for this role.
    """
    stack   = session_ctx.get("stack_summary", "")
    kw      = session_ctx.get("tech_keywords", [])
    primary = session_ctx.get("primary_language", "")
    domain  = session_ctx.get("domain", "")

    # Sentence 1: domain register
    if stack:
        sentence1 = f"This is a job interview for a {stack} role."
    else:
        sentence1 = "This is a technical job interview."

    # Sentence 2: embed top keywords in a natural question frame so the decoder
    # sees them as expected tokens rather than out-of-vocabulary surprises.
    # Use the first 12 keywords — enough to cover the core vocabulary without
    # making the prompt so long it pushes real audio context out of the window.
    if kw:
        # Split into two natural groups: first 6 as a "can you explain" question,
        # next 6 as a "what is the difference" question — covers both question forms.
        group_a = ", ".join(kw[:6])
        group_b = ", ".join(kw[6:12]) if len(kw) > 6 else ""
        sentence2 = f"Can you explain {group_a}?"
        if group_b:
            sentence2 += f" What is the difference between {group_b}?"
    elif primary:
        sentence2 = f"Can you explain dependency injection, design patterns, and architecture in {primary}?"
    else:
        sentence2 = "Can you explain the architecture, design patterns, and deployment approach you used?"

    return f"{sentence1} {sentence2}"


# ── Rule-based STT corrections (instant, zero latency) ────────────────────────
# These catch the most common Deepgram/Whisper mishears for .NET stack
# without needing an LLM call. Runs BEFORE the AI correction step.

RULE_FIXES = [
    # ── Deepgram nova-2/3 mishears "can you" as "Cache you" or "James" ────────
    # These were observed directly in logs: "Cache you tell me" / "Hey, James. Cache you"
    (r"(?i)\bcache\s+you\b",              "can you"),
    (r"(?i)\bhey[,\s]+james[,\s]+",       "Hey, "),   # "Hey, James. Can you" → "Hey, can you"
    (r"(?i)\bspeed\s+up\b",              "faster"),   # "more speed up" → "faster"
    # ── .NET / .NET Core ──────────────────────────────────────────────────────
    (r"(?i)\bdot\s*net\s+cors\b",        ".NET Core"),
    (r"(?i)\bdot\s*net\s+core\b",        ".NET Core"),
    (r"(?i)\bdot\s*net\s+framework\b",   ".NET Framework"),
    (r"(?i)\bdot\s*net\b",                ".NET"),
    # ASP.NET
    (r"(?i)\basp\s+dot\s+net\b",         "ASP.NET"),
    # ── Common phonetic mishears ──────────────────────────────────────────────
    (r"(?i)\bpost\s*grace\b",             "Postgres"),
    (r"(?i)\bpost\s*gress\b",             "Postgres"),
    (r"(?i)\bmy\s+sequel\b",              "MySQL"),
    (r"(?i)\bno\s+sequel\b",              "NoSQL"),
    (r"(?i)\bpie\s+torch\b",              "PyTorch"),
    (r"(?i)\bget\s+hub\b",               "GitHub"),
    (r"(?i)\bget\s+lab\b",               "GitLab"),
    (r"(?i)\bjay\s+son\b",               "JSON"),
    (r"(?i)\brest\s+full\b",             "RESTful"),
    (r"(?i)\brest\s+ful\b",              "RESTful"),
    (r"(?i)\bseek\s+well\b",             "sequel"),
    (r"(?i)\bci\s+cd\b",                 "CI/CD"),
    (r"(?i)\bkubernetes\b",               "Kubernetes"),
    # ── C / C++ / C# mishears ─────────────────────────────────────────────────
    (r"\bc\s+plus\s+plus\b",             "C++"),
    (r"\bc\s*#",                          "C#"),
    (r"(?i)\bc\s+sharp\b",               "C#"),
]

_RULE_FIXES_COMPILED = None

def _get_compiled_rules():
    global _RULE_FIXES_COMPILED
    if _RULE_FIXES_COMPILED is None:
        import re as _re
        compiled = []
        for pattern, replacement in RULE_FIXES:
            try:
                compiled.append((_re.compile(pattern), replacement))
            except Exception as e:
                print(f"[RuleFix] Bad regex skipped: {pattern!r} — {e}")
        _RULE_FIXES_COMPILED = compiled
    return _RULE_FIXES_COMPILED

def apply_rule_based_fixes(transcript: str, session_ctx: dict) -> str:
    """
    Apply fast regex-based corrections for the most common STT mishears.
    These fire before the AI correction step — zero latency.
    One bad pattern cannot crash the server — each is try/catched at compile time.
    """
    result = transcript
    try:
        for pattern, replacement in _get_compiled_rules():
            result = pattern.sub(replacement, result)
    except Exception as e:
        print(f"[RuleFix] Error applying fixes: {e}")
        return transcript
    return result


async def fix_transcript_with_ai(
    transcript:  str,
    session_ctx: dict,
    api_key:     str,
    job_desc:    str = "",
    cv:          str = "",
) -> str:
    """
    Minimal LLM pass: fix ONLY clear phonetic mishears — nothing else.
    The model MUST NOT change the question topic, add content, or rewrite meaning.
    If there is any doubt, return the original unchanged.
    """
    kw = session_ctx.get("tech_keywords", [])

    # Keep context short — the model only needs to know valid tech terms,
    # not the full job desc (which previously caused it to "redirect" questions)
    known_terms = ", ".join(kw[:25]) if kw else "none"

    prompt = (
        "You are a spell-checker for speech-to-text output in a technical interview. "
        "Your job is to fix words that were phonetically mis-transcribed.\n\n"
        "STRICT RULES:\n"
        "1. NEVER change the topic, subject, or meaning of the sentence.\n"
        "2. NEVER add, remove, or reorder words unless it is a clear phonetic mistake.\n"
        "3. NEVER replace a correctly-spelled technical term with a different one.\n"
        "   EXCEPTION: if a word looks like a phonetic mishear of a well-known tech term, fix it.\n"
        "   Example: 'Depper' → 'Dapper' (micro-ORM for .NET, sounds identical)\n"
        "   Example: 'Sequelize' heard as 'Sequel Eyes' → 'Sequelize'\n"
        "4. NEVER rephrase or improve the sentence.\n"
        "5. If the sentence makes sense as-is with no obvious mishear, return it EXACTLY.\n"
        "6. Known valid tech terms (do NOT change these): " + known_terms + "\n\n"
        "Examples of what you SHOULD fix:\n"
        "  'Cache you tell me' → 'Can you tell me'\n"
        "  'post grace sequel' → 'PostgreSQL'\n"
        "  'pie torch' → 'PyTorch'\n"
        "  'Depper' → 'Dapper'  (sounds like 'Dapper', the .NET micro-ORM)\n\n"
        "Return ONLY the corrected text. No explanation. No quotes. No preamble.\n\n"
        f"Transcript: {transcript}"
    )

    try:
        resp = await http_client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_FAST_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            }
        )
        fixed = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not fixed:
            return transcript

        # ── Safety gates — if ANY of these fail, discard the fix ─────────────

        orig_words  = transcript.split()
        fixed_words = fixed.split()

        # 1. Length ratio: fixed must be within 20% of original word count
        #    (previous 0.6–1.8 was far too loose — allowed full rewrites)
        ratio = len(fixed_words) / max(len(orig_words), 1)
        if not (0.80 <= ratio <= 1.20):
            print(f"[STT Fix] Rejected (word ratio {ratio:.2f}): '{fixed[:60]}'")
            return transcript

        # 2. First content word must be preserved (prevents topic substitution)
        #    Compare lowercased, stripped of punctuation
        import string
        def first_word(s):
            for w in s.lower().split():
                w = w.strip(string.punctuation)
                if len(w) > 2:  # skip short filler like "a", "is"
                    return w
            return ""

        # Allow first-word change only if it looks like a clear mishear fix
        # (e.g. "Cache" → "Can"). Block if it's a completely different word.
        orig_fw  = first_word(transcript)
        fixed_fw = first_word(fixed)
        if orig_fw and fixed_fw and orig_fw != fixed_fw:
            # Allow only if the fixed version is clearly a phonetic cousin
            # (share at least 2 chars at start, or original was a known bad word)
            known_bad_first = {"cache", "james", "hey james"}
            if orig_fw not in known_bad_first and not (
                len(orig_fw) > 2 and len(fixed_fw) > 2 and
                (orig_fw[:2] == fixed_fw[:2] or orig_fw[-2:] == fixed_fw[-2:])
            ):
                print(f"[STT Fix] Rejected (first-word change '{orig_fw}'→'{fixed_fw}'): '{fixed[:60]}'")
                return transcript

        # 3. Key nouns/verbs from original must still appear in fixed
        #    (prevents replacing the entire question subject)
        orig_lower  = transcript.lower()
        fixed_lower = fixed.lower()
        # Check that programming languages / key subjects are preserved
        for lang_term in ["c++", "c#", "c ", "python", "java", "javascript", "sql",
                          "react", "node", "angular", "vue", "docker", "kubernetes"]:
            if lang_term in orig_lower and lang_term not in fixed_lower:
                print(f"[STT Fix] Rejected (key term '{lang_term}' removed): '{fixed[:60]}'")
                return transcript

        if fixed != transcript:
            print(f"[STT Fix] '{transcript[:80]}' → '{fixed[:80]}'")
        return fixed

    except Exception:
        return transcript


async def translate_to_english(text: str, api_key: str, source_lang: str) -> str:
    lang_label = (
        "Hindi or Urdu (non-Latin script)"
        if source_lang == "foreign_script"
        else "Roman Urdu or Roman Hindi"
    )
    try:
        resp = await http_client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_FAST_MODEL,
                "messages": [{"role": "user", "content": (
                    f"Translate the following {lang_label} text to English.\n"
                    "Context: This is a job interview question.\n"
                    "Keep all technical terms in English as-is.\n"
                    "Return ONLY the English translation. No explanation. No quotes.\n\n"
                    f"Text: {text}"
                )}],
                "max_tokens": 300,
                "temperature": 0.1,
            }
        )
        translated = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return translated if translated else text
    except Exception as e:
        print(f"[Translation error] {e}")
        return text


# ── Intent classification ─────────────────────────────────────────────────────

FOLLOWUP_WORDS = [
    "explain more","more detail","elaborate","tell me more","can you explain",
    "what do you mean","give example","give me example","expand","continue",
    "go on","more about","in detail","detail me","explain further","deeper",
    "dig deeper","please explain","each layer","explain each","purpose of each",
    "what about","and also","what else","go deeper","more on that",
    "elaborate more","keep going","aur batao","aur bataiye","mazeed batao",
    "thoda aur","example do","example dein","samjhao","samjhaiye","detail mein",
    "what do you mean by","what is meant by","what does that mean",
    "what is a","what are","what is the","what is an",
]

CODE_WORDS = [
    "write a","write me","write simple","write function","write code",
    "show me code","give code","code example","implement","create a function",
    "create a class","write a query","write a method","show implementation",
    "code likho","code likhein","code dikhao","function likho",
]

GREETING_WORDS = [
    "how are you","how do you do","good morning","good afternoon","good evening",
    "how is it going","what's up","hey how are","hi how are","hello how are",
    "nice to meet","pleased to meet","how have you been","hope you are",
    "hope you're","can you hear me","are you there","can you hear","hello hello",
    "assalam","salam","walaikum","assalamualaikum","asslam walaikum",
    "kaise hain","kaise ho","theek ho","aap kaise","ap kaise",
    "subah bakhair","shab bakhair","khuda hafiz","allah hafiz","aadaab","namaste",
]

TECHNICAL_WORDS = [
    # generic question signals (apply to all domains)
    "difference","explain","what is","how does","how do","why","when","which",
    "compare","define","example","describe","tell me about","have you","did you",
    "experience","used","worked","built","designed","implemented","developed",
    "managed","led","deployed","integrated","architecture","pattern","performance",
    "security","testing","ci/cd","agile","scrum","api","rest","database","cloud",
    "devops","docker","microservice","frontend","backend","fullstack","framework",
    # SEO / digital marketing
    "seo","sem","ppc","ctr","cpc","cpm","roas","roi","serp","keyword","backlink",
    "on-page","off-page","technical seo","link building","anchor text","domain authority",
    "page rank","crawl","index","sitemap","robots","canonical","redirect","meta",
    "ahrefs","semrush","moz","google analytics","search console","google ads",
    "content strategy","content marketing","organic","paid","conversion","bounce rate",
    # marketing / social / ads
    "campaign","funnel","lead generation","email marketing","a/b test","ab test",
    "facebook ads","meta ads","linkedin ads","instagram","tiktok","influencer",
    "brand","copywriting","landing page","hubspot","mailchimp","salesforce",
    # sales / business
    "pipeline","quota","revenue","crm","cold call","outreach","proposal","objection",
    "closing","upsell","cross-sell","account management","b2b","b2c","saas",
    # HR / people
    "recruitment","talent","onboarding","performance review","kpi","okr","culture",
    "retention","engagement","compensation","payroll","hris","ats",
    # finance / ops
    "budget","forecast","p&l","cash flow","financial model","excel","reporting",
    # design / product
    "ux","ui","figma","user research","wireframe","prototype","roadmap","sprint",
    "product strategy","stakeholder","requirement","user story",
    # urdu/roman equivalents
    "kya hai","kya hota","farq kya","kaise kaam","kaise hota",
    "explain karo","samjhao","kyon use","kab use",
]

EXPERIENCE_WORDS = [
    "tell me about yourself","introduce yourself","your background",
    "walk me through","your experience","your projects","have you worked",
    "your role","your responsibilities","your achievements","your strengths",
    "your weaknesses","why should we hire","why do you want","what motivates",
    # behavioural / situational questions
    "tell me about a time","describe a situation","give me an example",
    "what problems","problems you faced","challenges you faced","biggest challenge",
    "difficult situation","how did you handle","how did you deal","what went wrong",
    "what would you do","how do you approach","what was the hardest","toughest project",
    "what did you learn","what would you do differently","how did you overcome",
    "conflict with","disagreement","mistake you made","failure","what failed",
    "under pressure","tight deadline","how do you prioritize","team conflict",
    "worked with difficult","worked under pressure","what have you built",
    "most proud of","biggest accomplishment","proudest moment",
    # urdu equivalents
    "apne baare mein","apna background","apna tajurba","apna experience",
    "aap ne kya kiya","aap ki strengths","aap ki weaknesses",
    "kyun hire karein","apna kaam","projects batao",
    "kya problems aayi","kya challenges the","mushkil situation",
    "kaise handle kiya","kya seekha","galti kya hui",
]


# ── Build fast lookup structures after all word-list constants are defined ────
_FOLLOWUP_SINGLES,   _FOLLOWUP_MULTI   = _split_kw_list(FOLLOWUP_WORDS)
_CODE_SINGLES,       _CODE_MULTI       = _split_kw_list(CODE_WORDS)
_GREETING_SINGLES,   _GREETING_MULTI   = _split_kw_list(GREETING_WORDS)
_TECHNICAL_SINGLES,  _TECHNICAL_MULTI  = _split_kw_list(TECHNICAL_WORDS)
_EXPERIENCE_SINGLES, _EXPERIENCE_MULTI = _split_kw_list(EXPERIENCE_WORDS)


def strip_numbers_from_cv(cv: str) -> str:
    cv = re.sub(r'\b\d+(\.\d+)?%', '', cv)
    cv = re.sub(r'\b\d+x\b', '', cv)
    cv = re.sub(r'\$[\d,]+', '', cv)
    return cv


def build_response_lang_instruction(lang_mode: str, detected_lang: str = "english") -> str:
    """
    Return a strong response-language instruction block for the AI prompt.

    english → empty string (model responds in English by default).
    urdu    → Roman Urdu always, regardless of input.
    auto    → Roman Urdu always (user chose auto = mixed Urdu+English is expected;
              the interviewer may speak English but the candidate answers in Roman Urdu
              so both sides understand naturally).
              Technical terms, code, and library/framework names stay in English.
    """
    if lang_mode == "english":
        return ""

    # Both "urdu" and "auto" require Roman Urdu responses.
    # The instruction is deliberately explicit and multi-line so it overrides
    # any conflicting "respond in English" rules cached in context_block.
    roman_urdu_instruction = (
        "=== ZABAAN KA RULE (LANGUAGE RULE — HIGHEST PRIORITY) ===\n"
        "Yeh interview Roman Urdu mein ho raha hai.\n"
        "HAMESHA Roman Urdu mein jawab do — yaani Urdu alfaaz ko English haroof mein likho.\n"
        "Misal ke taur par: 'Main pehle data pipeline check karta hoon. "
        "Phir consistency validate karta hoon using KS tests.'\n"
        "RULES:\n"
        "• Jawab Roman Urdu mein hona chahiye — pure English NAHI.\n"
        "• Technical terms, library names, code (jaise: Entity Framework, REST API, Docker, "
        "  Python, SQL) English mein rakhein — unka Urdu translation MAT karo.\n"
        "• Codeblocks ya pseudocode English mein likhein — sirf explanation Roman Urdu mein.\n"
        "• Pehla lafz ya sentence Urdu mein shuru karo, English se nahi.\n"
        "• Poora jawab complete karo — beech mein mat choro.\n"
        "=== END LANGUAGE RULE ===\n"
    )
    return roman_urdu_instruction


def build_system_context(profile: str, job_title: str, job_desc: str, cv: str, session_ctx: dict,
                          lang_mode: str = "english") -> str:
    stack_summary = session_ctx.get("stack_summary", "")

    # Language rule — injected here so it is part of the cached context_block.
    # For English mode: explicit English-only rule.
    # For Urdu/Auto: NO English-only rule (the PERSONA block carries the Roman Urdu instruction).
    if lang_mode == "english":
        lang_rule = "1. Always respond in English only.\n"
    else:
        lang_rule = (
            "1. LANGUAGE: Jawab hamesha Roman Urdu mein do "
            "(Urdu words written in English/Latin letters). "
            "Technical terms aur code English mein rakhein.\n"
        )

    parts = [
        "CRITICAL RULES:\n"
        + lang_rule +
        "2. NEVER comment on the question. Never say 'It seems like', 'The original question was', 'I notice'. Just answer.\n"
        "3. ANSWER STRUCTURE — depends on the question type:\n"
        "   CONCEPTUAL (what is X, why does X work this way, how does X behave):\n"
        "     a) State the fact, reason, or answer directly — not what you check or do to find it\n"
        "     b) Explain the mechanism or reason behind it\n"
        "     c) Short natural closing or practical nuance\n"
        "   EXPERIENCE (tell me about a time, what have you built, have you worked with X):\n"
        "     a) State your action or approach directly\n"
        "     b) Brief reasoning or methodology\n"
        "     c) Outcome or result\n"
        "4. OPENER RULE — match the question, not a template:\n"
        "   Conceptual: GOOD: 'REST is synchronous because the client blocks waiting for the response.'\n"
        "   Experience: GOOD: 'I start by validating the pipeline consistency end to end.'\n"
        "   NEVER open with: a company name, employer, project name, or 'I check X first' for a conceptual question.\n"
        "   NEVER narrate your thought process to answer a factual question — just answer it.\n"
        "5. COMPANY/PROJECT/TOOL NAMES — may only appear mid-answer as brief supporting evidence:\n"
        "   Use natural phrases like 'for example', 'in one project', 'on a recent deployment'.\n"
        "   Never let company name, project name, or tool name be the subject of the opening sentence.\n"
        "6. EXPERIENCE REFERENCES — only use CV details when the question genuinely calls for it:\n"
        "   • Conceptual or technical questions (what is X, how does X work, why is X used): answer the concept directly. Do NOT default to pulling in past roles, company names, or project names.\n"
        "   • Experience or behavioural questions (tell me about a time, have you worked with, what have you built): then draw from the CV — but keep company/project references generic ('in one project', 'on a recent system') unless the interviewer specifically asks for company names.\n"
        "   • NEVER mention a company name or specific organisation unprompted. If experience is relevant, describe it generically.\n"
        "   • Do NOT open with or default to 'in my previous company' or 'at [Company]' as a habit — only reference past work when the question explicitly calls for it.\n"
        "7. ALWAYS finish the answer completely — never trail off or end mid-sentence.\n"
        "8. SPEAKING REGISTER — you are talking in a live interview, not writing documentation:\n"
        "   BANNED anywhere in the answer:\n"
        "   • 'I determine...' / 'I would determine...' → say 'I check' / 'I look at' / 'it depends on'\n"
        "   • 'Understanding X is crucial' / 'It is important to understand X' → skip the preamble, just say it\n"
        "   • 'One must consider...' → say 'you have to think about' or just state it directly\n"
        "   • 'This approach/solution/architecture/method enables/ensures/allows...' → say what it does plainly\n"
        "   • Any sentence that sounds like a textbook definition or blog post conclusion\n"
        "   • 'By doing so,...' / 'In this way,...' / 'As a result of this approach,...' → use 'so', 'which', 'that way'\n"
        "   Every sentence must sound like something a real engineer would say out loud in an interview.\n"
        "9. STT MISHEAR TOLERANCE — the question came from speech-to-text and may contain typos or near-miss spellings:\n"
        "   • Always interpret the question charitably. If a term looks like a phonetic near-miss of a known technology, library, tool, or concept, answer for the most likely intended term.\n"
        "   • Do NOT say a term is 'not recognized', 'doesn't exist', or 'not a real framework' when it is plausibly a mishear of something real.\n"
        "   • If genuinely ambiguous between two real things, briefly note the ambiguity and answer both, or pick the most contextually likely one.\n"
        "   • The interviewer's intent is always more important than the exact spelling of the transcript."
    ]
    if profile:
        parts.append(f"CANDIDATE PROFILE (use this to personalise every answer):\n{profile}")
    if job_title:
        line = f"APPLYING FOR: {job_title}"
        if stack_summary:
            line += f"\nROLE CONTEXT: {stack_summary}"
        parts.append(line)
    if job_desc:
        parts.append(f"JOB DESCRIPTION:\n{job_desc.strip()}")
    if cv:
        clean_cv = strip_numbers_from_cv(cv)
        parts.append(f"CANDIDATE CV / RESUME (ground all experience answers in this):\n{clean_cv.strip()}")
    return "\n\n".join(parts)


def build_history_block(conversation_history: list, current_transcript: str = "") -> str:
    if not conversation_history:
        return ""
    recent = conversation_history[-2:]

    # When a current transcript is provided, filter to only history turns that
    # share meaningful topic overlap with the current question, so unrelated
    # prior exchanges do not pollute the prompt context.
    if current_transcript:
        current_words = set(w.lower().strip(".,?!") for w in current_transcript.split() if len(w) > 3)
        filtered = []
        for q, a in recent:
            prior_words = set(w.lower().strip(".,?!") for w in q.split() if len(w) > 3)
            # Include turn if at least 2 content words overlap, or if the prior
            # question is very short (likely a follow-up like "elaborate" / "more").
            overlap = current_words & prior_words
            if len(overlap) >= 2 or len(q.split()) <= 6:
                filtered.append((q, a))
        if not filtered:
            return ""
        recent = filtered

    lines = ["RECENT INTERVIEW CONVERSATION (last 1-2 relevant exchanges):"]
    for i, (q, a) in enumerate(recent, 1):
        lines.append(f"[Turn {i}] Interviewer: {q}")
        lines.append(f"[Turn {i}] Your answer:  {a}")
    lines.append("")
    return "\n".join(lines)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global current_key_index, current_cerebras_key_index
    await ws.accept()

    audio_chunks         = []
    conversation_history = []
    _session_id          = ""   # set from CONTEXT; used for cross-reconnect store
    profile   = ""
    job_title = ""
    job_desc  = ""
    cv        = ""
    stt_mode  = "whisper"
    lang_mode = "auto"
    ai_provider  = "groq"
    ai_model     = GROQ_MODEL
    cerebras_key = ""
    groq_key     = ""
    gemini_key   = ""
    qwen_key     = ""

    # Cached per-session extracted context.
    # We kick off extraction as a background task when CONTEXT arrives,
    # so it's ready by the time the first PROCESS comes in.
    session_ctx: dict | None = None
    _ctx_container = {
        "ctx":          None,   # extracted session context (keywords, stack, domain)
        "context_block": None,  # pre-built system context string — rebuilt only when ctx changes
        "persona":       None,  # PERSONA string — constant per session
        "style":         None,  # STYLE string — constant per session
    }

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                audio_chunks.append(msg["bytes"])
                # Keep session store alive while chunks are flowing
                if _session_id and _session_id in _SESSION_STORE:
                    _SESSION_STORE[_session_id]["_ts"] = _time.monotonic()

            elif "text" in msg:
                text = msg["text"]

                if text == "PING":
                    await ws.send_text("PONG")
                    continue

                if text.startswith("CONTEXT:"):
                    try:
                        ctx       = json.loads(text[8:])
                        profile   = ctx.get("profile",   "")
                        job_title = ctx.get("jobTitle",  "")
                        job_desc  = ctx.get("jobDesc",   "")
                        cv        = ctx.get("cv",        "")
                        stt_mode  = ctx.get("sttMode",   "whisper")
                        lang_mode = ctx.get("langMode",  "auto")
                        ai_provider  = ctx.get("provider",    "groq")
                        ai_model     = ctx.get("aiModel",     GROQ_MODEL)
                        cerebras_key = ctx.get("cerebrasKey", "")
                        groq_key     = ctx.get("groqKey",     "")
                        gemini_key   = ctx.get("geminiKey",   "")
                        qwen_key     = ctx.get("qwenKey",     "")
                        # If user supplied their own Groq key, use it; else fall back to built-in
                        if groq_key:
                            GROQ_API_KEYS[0] = groq_key

                        # ── Reconnect-safe session store ──────────────────────
                        # Chrome closes the offscreen WS after ~30s; the client
                        # reconnects and re-sends CONTEXT with the same sessionId.
                        # If we recognise the ID we restore the audio buffer so
                        # chunks recorded before the drop are not lost.
                        _session_id = ctx.get("sessionId", "")
                        _evict_stale_sessions()
                        if _session_id and _session_id in _SESSION_STORE:
                            # Reconnect: restore accumulated audio buffer and history.
                            # audio_chunks is reassigned to the stored list so new chunks
                            # from flushPending() go straight into the same object.
                            _stored              = _SESSION_STORE[_session_id]
                            audio_chunks         = _stored["chunks"]
                            conversation_history = _stored["history"]
                            _ctx_container.update({
                                k: _stored.get(k) for k in ("ctx","context_block","persona","style")
                            })
                            _SESSION_STORE[_session_id]["_ts"] = _time.monotonic()
                            _kb = sum(len(c) for c in audio_chunks) // 1024
                            print(f"[Reconnect] Session {_session_id[:8]}… resumed — {len(audio_chunks)} chunks ({_kb} KB) already buffered")
                        elif _session_id:
                            # New session: register the live lists in the store.
                            # Storing references (not copies) means any append to
                            # audio_chunks is immediately visible via _SESSION_STORE.
                            _SESSION_STORE[_session_id] = {
                                "chunks":        audio_chunks,        # live reference
                                "history":       conversation_history, # live reference
                                "ctx":           None,
                                "context_block": None,
                                "persona":       None,
                                "style":         None,
                                "_ts":           _time.monotonic(),
                            }
                            print(f"[Session] New session {_session_id[:8]}… registered")

                        print(f"[Context] Job: '{job_title}' | STT: {stt_mode} | Lang: {lang_mode} | AI: {ai_provider}/{ai_model}")

                        # Extract session context in background — ready before first PROCESS.
                        # We also pre-build context_block, PERSONA, and STYLE here so that
                        # every PROCESS call just reads cached strings instead of rebuilding them.
                        _api = GROQ_API_KEYS[current_key_index]
                        _container   = _ctx_container
                        _profile_s   = profile
                        _job_title_s = job_title
                        _job_desc_s  = job_desc
                        _cv_s        = cv
                        _lang_mode_s = lang_mode  # capture for _bg_extract closure

                        async def _bg_extract():
                            # Check cross-session cache first — avoids redundant Groq call
                            _cache_key = hashlib.sha1(
                                (_job_title_s + _job_desc_s + _cv_s).encode()
                            ).hexdigest()
                            if _cache_key in _SESSION_CTX_CACHE:
                                extracted = _SESSION_CTX_CACHE[_cache_key]
                                print("[Cache] Session context HIT — skipping extraction")
                            else:
                                extracted = await extract_session_context(
                                    _job_title_s, _job_desc_s, _cv_s, _api
                                )
                                if len(_SESSION_CTX_CACHE) >= _SESSION_CTX_CACHE_MAX:
                                    _SESSION_CTX_CACHE.pop(next(iter(_SESSION_CTX_CACHE)))
                                _SESSION_CTX_CACHE[_cache_key] = extracted
                            _container["ctx"] = extracted

                            # Pre-build and cache the system context string — unchanged per session
                            # Pass lang_mode so the English-only rule is replaced with Roman Urdu
                            # rule for urdu/auto modes, avoiding conflict with PERSONA instruction.
                            _container["context_block"] = build_system_context(
                                _profile_s, _job_title_s, _job_desc_s, _cv_s, extracted,
                                lang_mode=_lang_mode_s
                            )

                            # Pre-build PERSONA and STYLE — built once per session, reused every call.
                            # For Urdu/Auto modes the persona itself is in Roman Urdu so the AI
                            # has a bilingual frame of reference for how to structure the answer.
                            if _lang_mode_s in ("urdu", "auto"):
                                _container["persona"] = (
                                    "Aap ek candidate hain jo live interview mein jawab de rahe hain. "
                                    "Ek real engineer ki tarah baat karo — casual, confident, seedha. "
                                    "Conversation ho rahi hai, recitation nahi.\n\n"
                                    "JAWAB KAISE DEIN — question ki type ke hisaab se:\n"
                                    "  CONCEPTUAL sawaal (X kya hai, X kyun aisa kaam karta hai, X ka behavior):\n"
                                    "    → Seedha concept ka jawab do. Fact ya reason se shuru karo, apni checking se nahi.\n"
                                    "    → SAHI: 'REST isliye synchronous hai kyunke client response aane tak block rehta hai.'\n"
                                    "    → GALAT: 'Main pehle API design check karta hoon.' (kisne poochha aap kya check karte hain)\n"
                                    "    → GALAT: 'Main transport layer dekhta hoon determine karne ke liye...' (thought process narrate karna)\n"
                                    "  EXPERIENCE sawaal (koi example batao, aap ne kya banaya, kaise handle kiya):\n"
                                    "    → Phir apna action ya approach pehle batao.\n"
                                    "    → SAHI: 'Main pipeline consistency pehle validate karta hoon.'\n\n"
                                    "TONE — casual aur professional, jaise kisi senior colleague se baat ho:\n"
                                    "  • Chhote sentences. Plain words. Filler nahi.\n"
                                    "  • Factual sawaal ke liye thought process narrate mat karo.\n"
                                    "  • Jawab kaise socha — yeh mat batao, seedha jawab do.\n"
                                    "  BANNED:\n"
                                    "    - 'Main pehle X check karta hoon' jab sawaal conceptual ho\n"
                                    "    - 'Main determine karta hoon...'\n"
                                    "    - 'X samajhna zaroori hai'\n"
                                    "    - 'Yeh approach enable/ensure karta hai...'\n"
                                    "    - Koi textbook definition ya blog conclusion"
                                )
                                _container["style"] = (
                                    "STYLE RULES — har sentence par laagu hota hai, sirf pehle par nahi:\n"
                                    "• Jawab Roman Urdu mein — Urdu alfaaz English haroof mein likhain.\n"
                                    "• Technical terms (Entity Framework, REST API, Docker, SQL, Python) English mein rakhain.\n"
                                    "• Active voice. Chhote sentences. Filler words nahi.\n"
                                    "• Baat karte hue sound karo — har sentence ka test: 'Kya koi engineer yeh out loud kahega?'\n"
                                    "• KABHI NAHI: company name, tool name, ya project name se jawab shuru karo.\n"
                                    "• KABHI DEFAULT MAT KARO past role ya company mention karne par — sirf tab jab question explicitly maange.\n"
                                    "• Jab experience relevant ho, generic rakho: 'ek project mein', 'ek purane system mein' — company ka naam mat lo.\n"
                                    "• Metrics jawab dene ke baad aate hain, pehle nahi.\n"
                                    "• Har jawab poora karo — beech mein mat choro.\n"
                                    "BANNED PATTERNS — ye AI ya documentation jaisi lagti hain:\n"
                                    "• 'Main determine karta hoon...' → 'Main check karta hoon' / 'Yeh depend karta hai'\n"
                                    "• 'X samajhna zaroori hai' → seedha bolo\n"
                                    "• 'Yeh approach/solution enable/ensure karta hai...' → seedha kaho kya hota hai\n"
                                    "• 'Is tarah se,...' / 'Aisa karne se,...' → 'toh', 'jo', 'matlab'\n"
                                    "• 'Mere purane company mein...' / '[Company] mein...' — default ki tarah nahi, sirf agar question maange\n"
                                    "• Koi bhi textbook-jaisi definition\n"
                                    "PREFERRED:\n"
                                    "• 'Yeh depend karta hai...' / 'Asli baat yeh hai...' / 'Practice mein...'\n"
                                    "• 'Key cheez yeh hai...' / 'Matlab yeh ke...' / 'Basically...'"
                                )
                            else:
                                _container["persona"] = (
                                    "You are a candidate in a live job interview. "
                                    "Talk like a real engineer — casual, confident, direct. "
                                    "You are having a conversation, not reciting or explaining from a script.\n\n"
                                    "HOW TO ANSWER — depends on the question type:\n"
                                    "  CONCEPTUAL questions (what is X, why does X work this way, how does X behave):\n"
                                    "    → Answer the concept directly. Start with the fact, not with what you do.\n"
                                    "    → GOOD: 'REST is synchronous because every request blocks until the server responds.'\n"
                                    "    → GOOD: 'The limit depends on the database — SQL Server allows up to 999 non-clustered.'\n"
                                    "    → BAD: 'I check the API design first.' (nobody asked what you check)\n"
                                    "    → BAD: 'I look at the transport layer to determine...' (narrating your thought process)\n"
                                    "  EXPERIENCE questions (tell me about a time, have you worked with, what have you built):\n"
                                    "    → Then lead with what you did, your approach, your decision.\n"
                                    "    → GOOD: 'I start by validating the pipeline consistency end to end.'\n\n"
                                    "TONE — casual and professional, like talking to a senior colleague:\n"
                                    "  • Short sentences. Plain words. No filler.\n"
                                    "  • Never narrate your thought process for a factual question.\n"
                                    "  • Never explain how you figured out the answer — just give the answer.\n"
                                    "  BANNED anywhere:\n"
                                    "    - 'I check X first' / 'I look at X first' when the question is conceptual\n"
                                    "    - 'I determine...' / 'I would determine...'\n"
                                    "    - 'Understanding X is crucial' / 'It is important to understand'\n"
                                    "    - 'One must consider...'\n"
                                    "    - 'This approach/solution/architecture enables/ensures/allows...'\n"
                                    "    - Any textbook definition or blog-post conclusion"
                                )
                                _container["style"] = (
                                    "STYLE RULES — apply to EVERY sentence, not just the first:\n"
                                    "• Active voice throughout. Short sentences. No filler.\n"
                                    "• Sound like you're talking to the interviewer, not writing for a reader.\n"
                                    "• Every sentence must pass: 'would a real engineer say this out loud?' If not, rewrite it.\n"
                                    "• NEVER open with a company name, tool name, project name, or employer.\n"
                                    "• NEVER default to mentioning a past role, company, or employer — only reference experience when the question explicitly asks for it.\n"
                                    "• When experience IS relevant, keep it generic: 'in one project', 'in a previous role', 'on a recent system' — never name the company or organisation.\n"
                                    "• Metrics and outcomes welcome AFTER the answer is given, not before.\n"
                                    "• Complete every answer — never trail off.\n"
                                    "BANNED PATTERNS — these make responses sound like AI output or documentation:\n"
                                    "• 'I determine...' / 'I would determine...' — say 'I check' / 'I look at' / 'it depends on'\n"
                                    "• 'Understanding X is crucial' / 'It is important to understand' — skip the preamble, just say it\n"
                                    "• 'One must consider...' — say 'you have to think about' or just state it\n"
                                    "• 'This approach/solution/method/architecture enables/ensures/allows...' — say what it does plainly\n"
                                    "• 'By doing so,...' / 'In this way,...' / 'As a result of this approach,...' — use 'so', 'which', 'that way'\n"
                                    "• 'In my previous company...' / 'At [Company]...' as a default habit — only when explicitly asked\n"
                                    "• Any sentence that sounds like a textbook definition\n"
                                    "• Any closing sentence that sounds like a blog post conclusion\n"
                                    "PREFERRED PATTERNS — natural spoken engineering language:\n"
                                    "• 'It depends on...' / 'The thing to watch is...' / 'In practice...'\n"
                                    "• 'The key thing is...' / 'What that means is...' / 'Basically...'\n"
                                    "• 'In most cases...' / 'The real limit is...' / 'What I'd do is...'"
                                )
                            print("[Cache] context_block + PERSONA + STYLE pre-built and cached")
                            # Sync extracted context back into session store so a
                            # subsequent reconnect can restore the full cache.
                            if _session_id and _session_id in _SESSION_STORE:
                                _SESSION_STORE[_session_id].update({
                                    "ctx":           _container["ctx"],
                                    "context_block": _container["context_block"],
                                    "persona":       _container["persona"],
                                    "style":         _container["style"],
                                })

                        asyncio.create_task(_bg_extract())

                    except Exception as e:
                        print(f"[Context parse error] {e}")

                elif text.startswith("PROFILE:"):
                    profile = text[8:].strip()

                elif text == "PROCESS":
                    if not audio_chunks:
                        await ws.send_text("[no audio recorded — make sure tab audio is playing and try again]\n\n---\n\n")
                        continue

                    # Inline blob assembly — b"".join on <2MB is ~50µs, far cheaper
                    # than run_in_executor's thread scheduling + asyncio overhead (~1-5ms).
                    # Only offload to executor for very large recordings (>5MB).
                    _chunks_ref  = audio_chunks
                    audio_chunks = []
                    # Processed — remove from session store so a future recording
                    # starts with a clean buffer (not leftover chunks from this one).
                    if _session_id and _session_id in _SESSION_STORE:
                        _SESSION_STORE[_session_id]["chunks"] = audio_chunks
                    if sum(len(c) for c in _chunks_ref) > 5_000_000:
                        audio_blob = await asyncio.get_event_loop().run_in_executor(
                            None, b"".join, _chunks_ref
                        )
                    else:
                        audio_blob = b"".join(_chunks_ref)
                    blob_mb = len(audio_blob) / (1024 * 1024)
                    print(f"[Process] {len(audio_blob)} bytes ({blob_mb:.1f} MB) | STT: {stt_mode}")

                    # Guard: webm files under ~8KB are just the container header with no
                    # real audio payload. This happens when stop is pressed within the first
                    # 500ms before MediaRecorder fires its first chunk. Give a clear message.
                    MIN_AUDIO_BYTES = 8_000
                    if len(audio_blob) < MIN_AUDIO_BYTES:
                        print(f"[Process] Audio too short ({len(audio_blob)} bytes) — skipping")
                        await ws.send_text("[recording too short — hold longer to capture audio]\n\n---\n\n")
                        continue

                    if blob_mb > 50:
                        await ws.send_text(f"[Warning: audio is {blob_mb:.0f} MB — transcription may be slow]\n\n")

                    # ── Context + STT in parallel when bg_extract is still running ──
                    # If the background extraction task hasn't finished yet, don't block STT.
                    # STT only needs ctx for keyword-hint prompts (minor quality gain).
                    # We start STT immediately with whatever ctx is ready (possibly empty {}),
                    # then await the real ctx before building the AI prompt.
                    ctx = _ctx_container.get("ctx") or {}

                    # ── Transcribe + ctx (parallel when ctx not yet ready) ────
                    # Cache loop.time() once — avoids repeated syscall overhead
                    _loop       = asyncio.get_event_loop()
                    t0          = _loop.time()

                    async def _do_stt():
                        if stt_mode == "deepgram":
                            return await transcribe_with_deepgram(audio_blob, ctx, lang_mode)
                        elif stt_mode == "gemini":
                            # For short clips (<80KB ≈ <8s speech), Whisper local is faster
                            # than the Gemini network round-trip (~2.5s vs ~1.5s local).
                            # Only use Gemini STT for longer recordings where accuracy gains matter.
                            if len(audio_blob) < 80_000:
                                print("[Gemini STT] Short clip — routing to Whisper for speed")
                                return await transcribe_with_whisper(audio_blob, ctx, lang_mode)
                            _stt_model_hint = ai_model if ai_provider == "gemini" else ""
                            return await transcribe_with_gemini(audio_blob, ctx, lang_mode, gemini_key, _stt_model_hint)
                        else:
                            return await transcribe_with_whisper(audio_blob, ctx, lang_mode)

                    # Run STT and a concurrent WS keepalive ping so the client
                    # doesn't time out during long (5-10s) Whisper processing.
                    # The client ignores PONG messages so this is transparent.
                    async def _stt_with_keepalive():
                        """Run STT while sending periodic PONG to keep WS alive."""
                        stt_task = asyncio.create_task(_do_stt())
                        while not stt_task.done():
                            try:
                                await asyncio.wait_for(asyncio.shield(stt_task), timeout=4.0)
                            except asyncio.TimeoutError:
                                # STT still running — ping the client to keep WS alive
                                try:
                                    await ws.send_text("PONG")
                                except Exception:
                                    pass
                        return await stt_task

                    try:
                        transcript = await _stt_with_keepalive()
                    except Exception as e:
                        await ws.send_text(f"[Transcription error: {e}]")
                        continue

                    stt_time = _loop.time() - t0
                    print(f"[Timing] STT: {stt_time:.2f}s")
                    print(f"[Transcript RAW]: {repr(transcript[:200])}")

                    if not transcript:
                        await ws.send_text("[no speech detected — check audio and try again]")
                        continue

                    api_key = GROQ_API_KEYS[current_key_index]

                    # ── Transcript correction ────────────────────────────────
                    # Rule-based fixes only — zero latency, zero hallucination.
                    # AI fix was causing "await"→"async", "Cache"→"Kinza" — removed.
                    transcript = apply_rule_based_fixes(transcript, ctx)

                    # ── Language detection & translation ─────────────────────
                    original_transcript     = transcript
                    translated_from_foreign = False

                    detected_lang = detect_language(transcript)

                    if detected_lang in ("foreign_script", "roman_south_asian") and lang_mode != "english":
                        transcript              = await translate_to_english(transcript, api_key, detected_lang)
                        translated_from_foreign = True

                    transcript_lower = transcript.lower()
                    has_technical    = _fast_match(transcript_lower, _TECHNICAL_SINGLES,  _TECHNICAL_MULTI)
                    has_experience   = _fast_match(transcript_lower, _EXPERIENCE_SINGLES, _EXPERIENCE_MULTI)
                    is_greeting      = _fast_match(transcript_lower, _GREETING_SINGLES,   _GREETING_MULTI)

                    # Compute all intent flags before sending [Q] so prompt is ready
                    # the instant the WebSocket send returns — minimises dead time.
                    is_followup     = _fast_match(transcript_lower, _FOLLOWUP_SINGLES,   _FOLLOWUP_MULTI)
                    is_code_request = _fast_match(transcript_lower, _CODE_SINGLES,       _CODE_MULTI)
                    is_experience_q = _fast_match(transcript_lower, _EXPERIENCE_SINGLES, _EXPERIENCE_MULTI)
                    wants_code      = is_code_request  # same check — reuse result

                    # ── Response language instruction ─────────────────────────
                    # For auto/urdu modes, always inject the Roman Urdu instruction.
                    # We pass detected_lang only as context — the function now always
                    # fires for auto mode regardless of what language was detected,
                    # because the user explicitly selected auto = Roman Urdu responses.
                    _resp_lang_instr = build_response_lang_instruction(lang_mode, detected_lang)

                    if translated_from_foreign:
                        await ws.send_text(f"[Q - Original]: {original_transcript}\n")
                        await ws.send_text(f"[Translated to EN]: {transcript}\n\n")
                    else:
                        await ws.send_text(f"[Q]: {transcript}\n\n")

                    if is_greeting and not has_technical and not has_experience and len(transcript.split()) < 15:
                        await ws.send_text("[Greeting detected — waiting for a question...]\n\n---\n\n")
                        continue

                    # Ensure ctx is fully populated before prompt building.
                    # By the time STT finishes (~2-3s), bg_extract has almost always completed.
                    # If not, we do a single quick synchronous check — no extra await needed.
                    if not _ctx_container.get("ctx"):
                        _ctx_container["ctx"] = ctx  # keep empty dict as fallback
                    ctx = _ctx_container["ctx"] or ctx

                    # Use pre-built cached strings — built once in _bg_extract, reused every call
                    context_block = _ctx_container["context_block"] or build_system_context(profile, job_title, job_desc, cv, ctx, lang_mode=lang_mode)
                    history_block = build_history_block(conversation_history, transcript)
                    # Prepend response-language directive to PERSONA so it applies to every
                    # prompt branch below without duplicating the instruction in each template.
                    _base_persona = _ctx_container["persona"] or ""
                    PERSONA       = (_resp_lang_instr + _base_persona) if _resp_lang_instr else _base_persona
                    STYLE         = _ctx_container["style"]   or ""

                    # Dynamic values from session context (read from already-cached ctx)
                    primary_lang  = ctx.get("primary_language", "the relevant language")
                    domain        = ctx.get("domain", "software engineering")
                    stack_summary = ctx.get("stack_summary", job_title or "a technical role")
                    _domain_type  = ctx.get("domain_type", "technical")

                    # Detect multi-concept questions (e.g. "explain X, Y and Z")
                    # _CONCEPT_SPLIT_RE compiled once at module level — zero overhead here
                    _concept_split = _CONCEPT_SPLIT_RE.split(transcript_lower)
                    _concept_count = len([c for c in _concept_split if len(c.strip()) > 3])
                    is_multi_concept = (
                        _concept_count >= 3 or
                        any(w in transcript_lower for w in [
                            "difference between", "compare", "differences", "distinguish",
                            "contrast", "vs", "versus",
                        ]) and _concept_count >= 2
                    )

                    if is_followup and conversation_history:
                        last_q, last_a = conversation_history[-1]
                        prompt = (
                            f"{context_block}\n\n"
                            f"{history_block}"
                            f"{PERSONA}\n{STYLE}\n\n"
                            "This is a follow-up to what you just said. Go one level deeper — add a real example, "
                            "a specific detail, or expand on a point you mentioned. Don't repeat yourself. "
                            "Keep it natural and conversational, like continuing a thought.\n"
                            f"What you said before: {last_a[:200]}\n"
                            f"Follow-up: {transcript}\nAnswer:"
                        )
                        num_predict = 250

                    elif is_code_request:
                        prompt = (
                            f"{context_block}\n\n"
                            f"{history_block}"
                            f"{PERSONA}\n{STYLE}\n\n"
                            f"Write the {primary_lang} code. 1 sentence intro, then clean pseudocode. No markdown. Max 8 lines.\n\n"
                            f"Question: {transcript}\nAnswer:"
                        )
                        num_predict = 220

                    elif is_experience_q:
                        prompt = (
                            f"{context_block}\n\n"
                            f"{history_block}"
                            f"{PERSONA}\n\n"
                            "ANSWERING A BEHAVIOURAL / EXPERIENCE QUESTION:\n"
                            "Follow this exact structure:\n"
                            "  Sentence 1: State the approach, action, or decision directly. "
                            "Must state WHAT you do or did — not where, not which company.\n"
                            "  Sentences 2-3: Explain the methodology and reasoning.\n"
                            "  Sentences 4-5: Add one specific supporting detail from your experience if genuinely relevant — "
                            "describe it generically: 'in one project', 'on a recent system', 'in a previous role'. "
                            "Do NOT name any company, organisation, or employer.\n"
                            "  Final sentence: State the outcome or conclusion.\n\n"
                            "Rules:\n"
                            "• Write flowing prose — no bullets.\n"
                            "• Use 'I' throughout — only use 'we' for team actions.\n"
                            "• Do not invent details not in the CV.\n"
                            "• NEVER mention a company name or specific organisation — keep all experience references generic.\n"
                            "• Only include an experience reference if it genuinely supports the answer — do not force it.\n"
                            "• 4-6 sentences total. Always finish completely.\n\n"
                            f"Question: {transcript}\nAnswer:"
                        )
                        num_predict = 360

                    elif is_multi_concept:
                        prompt = (
                            f"{context_block}\n\n"
                            f"{history_block}"
                            f"{PERSONA}\n{STYLE}\n\n"
                            "STT NOTE: The question came from speech-to-text — interpret any near-miss spellings as their most likely intended tech term. Never say a term is unrecognized.\n"
                            "Multiple concepts — one bullet each. First character MUST be '•'. No intro line.\n"
                            "Each bullet: '• Name — what it is. Real example.' (2 sentences max per bullet)\n"
                            "Cover every concept. Complete every bullet. Never cut off.\n\n"
                            f"Question: {transcript}\nAnswer:"
                        )
                        num_predict = 280

                    else:
                        # Detect if they want a list/types
                        asks_for_list = any(w in transcript_lower for w in [
                            "types", "type of", "kinds", "causes", "reasons", "steps",
                            "ways", "options", "examples", "list", "name some", "what are",
                            "which are", "different", "categories", "methods", "how many",
                            "name the", "tell me the", "what are the",
                        ])

                        if asks_for_list:
                            prompt = (
                                f"{context_block}\n\n"
                                f"{history_block}"
                                f"{PERSONA}\n{STYLE}\n\n"
                                "STT NOTE: The question came from speech-to-text — interpret any near-miss spellings as their most likely intended tech term. Never say a term is unrecognized.\n"
                                "Bullet list. Each bullet: '• Name — one crisp sentence.' No intro, no closing. Complete all bullets.\n\n"
                                f"Question: {transcript}\nAnswer:"
                            )
                            num_predict = 220
                        else:
                            prompt = (
                                f"{context_block}\n\n"
                                f"{history_block}"
                                f"{PERSONA}\n{STYLE}\n\n"
                                "STT NOTE: The question came from speech-to-text and may contain near-miss spellings. Always interpret charitably — map to the most likely intended tech term and answer it directly. Never say a term doesn't exist.\n"
                                "Answer conversationally in plain prose — 3 to 4 sentences max.\n"
                                "  Sentence 1: Answer the actual question. For conceptual questions, state the fact or reason directly.\n"
                                "     GOOD: 'REST is synchronous because the client blocks until the server responds.'\n"
                                "     GOOD: 'It depends on the database — SQL Server caps it at 999 non-clustered indexes.'\n"
                                "     BAD: 'I check the design first.' / 'I look at the transport layer to determine...'\n"
                                "  Sentence 2: One sentence explaining the mechanism or reason behind it.\n"
                                "  Sentence 3: A practical nuance, edge case, or natural closing — keep it short.\n"
                                "No bullets. No intro. No thought-process narration. Complete the final sentence.\n\n"
                                f"Question: {transcript}\nAnswer:"
                            )
                            num_predict = 200

                    # ── Stream answer ─────────────────────────────────────────
                    t1          = _loop.time()
                    full_answer = ""

                    if ai_provider == "cerebras" and cerebras_key:
                        # ── Cerebras streaming (with retry + Groq fallback) ───
                        cerebras_ok = False

                        # qwen-3 models have a thinking/reasoning mode that can produce
                        # an empty content stream. Use enable_thinking=False to suppress it.
                        # We build two payloads: one with the flag, one plain fallback.
                        _is_qwen3 = "qwen-3" in ai_model.lower() or "qwen3" in ai_model.lower()

                        def _make_cb_payload(suppress_thinking: bool):
                            p = {
                                "model":       ai_model,
                                "messages":    [{"role": "user", "content": prompt}],
                                "max_tokens":  num_predict,
                                "temperature": 0.3,
                                "stream":      True,
                            }
                            if suppress_thinking and _is_qwen3:
                                p["enable_thinking"] = False
                            return p

                        # If this model previously rejected enable_thinking, skip the flag immediately
                        _cb_use_thinking_flag = _is_qwen3 and ai_model not in _CEREBRAS_NO_THINKING
                        _cb_400_retried       = False       # only retry once after a 400

                        for cb_attempt in range(3):  # retry up to 3x on 429
                            try:
                                async with http_client.stream(
                                    "POST",
                                    CEREBRAS_URL,
                                    headers={
                                        "Authorization": f"Bearer {cerebras_key}",
                                        "Content-Type":  "application/json",
                                    },
                                    json=_make_cb_payload(_cb_use_thinking_flag),
                                ) as resp:
                                    if resp.status_code == 429:
                                        if cb_attempt == 0:
                                            # One short retry before falling back to Groq
                                            print(f"[Cerebras] 429 rate limit, retrying once (attempt {cb_attempt+1}/3)")
                                            await asyncio.sleep(0.5)
                                            continue
                                        # Second 429 — don't keep waiting, fall through to Groq immediately
                                        print(f"[Cerebras] 429 repeated — falling through to Groq immediately")
                                        break
                                    elif resp.status_code == 400 and _cb_use_thinking_flag and not _cb_400_retried:
                                        # enable_thinking not supported — remember this model, retry without flag
                                        body = await resp.aread()
                                        print(f"[Cerebras] 400 with thinking flag — model {ai_model} added to no-thinking list")
                                        _CEREBRAS_NO_THINKING.add(ai_model)
                                        _cb_use_thinking_flag = False
                                        _cb_400_retried       = True
                                        continue
                                    elif resp.status_code == 404:
                                        body = await resp.aread()
                                        print(f"[Cerebras] 404 model not found: {ai_model} — falling back to Groq")
                                        await ws.send_text(f"[Cerebras model '{ai_model}' not found — using Groq instead]\n")
                                        break  # break to Groq fallback
                                    elif resp.status_code != 200:
                                        body = await resp.aread()
                                        print(f"[Cerebras] Error {resp.status_code}: {body[:200]}")
                                        await ws.send_text(f"[Cerebras error {resp.status_code} — using Groq instead]\n")
                                        break  # break to Groq fallback
                                    else:
                                        first_token = True
                                        async def _stream_cerebras():
                                            nonlocal full_answer, first_token
                                            async for line in resp.aiter_lines():
                                                line = line.strip()
                                                if not line or line == "data: [DONE]":
                                                    continue
                                                if line.startswith("data: "):
                                                    try:
                                                        data  = json.loads(line[6:])
                                                        delta = data.get("choices", [{}])[0].get("delta", {})
                                                        token = delta.get("content", "")
                                                        if token:
                                                            if first_token:
                                                                elapsed = _loop.time()
                                                                print(f"[Timing] Cerebras first token: {elapsed-t1:.2f}s | Total: {elapsed-t0:.2f}s")
                                                                first_token = False
                                                            full_answer += token
                                                            await ws.send_text(token)
                                                    except (json.JSONDecodeError, KeyError, IndexError):
                                                        continue
                                        try:
                                            await asyncio.wait_for(_stream_cerebras(), timeout=20.0)
                                        except asyncio.TimeoutError:
                                            print(f"[Cerebras] Stream timeout after 8s — sending partial answer")
                                            if full_answer:
                                                cerebras_ok = True  # partial is fine
                                            break
                                        # If stream completed but nothing was emitted, fall through to Groq
                                        if not full_answer:
                                            print(f"[Cerebras] Empty response for model {ai_model} — falling back to Groq")
                                            break
                                        cerebras_ok = True
                                        break  # success
                            except Exception as e:
                                print(f"[Cerebras] Connection error: {e}")
                                break  # fall through to Groq

                        if not cerebras_ok and not full_answer:
                            # Groq fallback when Cerebras fails
                            print("[Cerebras] Falling back to Groq llama-3.3-70b-versatile")
                            _fb_key = GROQ_API_KEYS[current_key_index]
                            try:
                                async with http_client.stream(
                                    "POST",
                                    "https://api.groq.com/openai/v1/chat/completions",
                                    headers={"Authorization": f"Bearer {_fb_key}", "Content-Type": "application/json"},
                                    json={
                                        "model": GROQ_MODEL,
                                        "messages": [{"role": "user", "content": prompt}],
                                        "max_tokens": num_predict,
                                        "temperature": 0.3,
                                        "stream": True,
                                    }
                                ) as resp:
                                    if resp.status_code == 200:
                                        first_token = True
                                        async for line in resp.aiter_lines():
                                            if not line or line == "data: [DONE]":
                                                continue
                                            if line.startswith("data: "):
                                                try:
                                                    data  = json.loads(line[6:])
                                                    token = data.get("choices",[{}])[0].get("delta",{}).get("content","")
                                                    if token:
                                                        if first_token:
                                                            elapsed = _loop.time()
                                                            print(f"[Timing] Groq fallback first token: {elapsed-t1:.2f}s")
                                                            first_token = False
                                                        full_answer += token
                                                        await ws.send_text(token)
                                                except (json.JSONDecodeError, KeyError, IndexError):
                                                    continue
                            except Exception as e:
                                await ws.send_text(f"\n[Groq fallback also failed: {e}]\n")

                    elif ai_provider == "gemini" and gemini_key:
                        # ── Gemini streaming ─────────────────────────────────
                        # streamGenerateContent with alt=sse delivers SSE chunks.
                        # Each chunk is a JSON object (NOT an array) with shape:
                        #   {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
                        # Gemini 2.5 models also emit "thought" chunks first with
                        # an empty parts[0].text — we skip those and only forward
                        # non-empty text tokens. thinkingBudget=0 disables thinking
                        # mode entirely for lowest latency.
                        # The stream ends by closing — there is NO "data: [DONE]".
                        print(f"[Gemini] model={ai_model}")
                        _gemini_model = ai_model  # e.g. "gemini-2.5-flash"
                        _gemini_ok = False
                        try:
                            _gemini_url = (
                                f"{GEMINI_BASE_URL}/v1beta/models/{_gemini_model}"
                                f":streamGenerateContent?alt=sse&key={gemini_key}"
                            )
                            # thinkingConfig is only supported on gemini-2.5-flash (not -lite, not 2.0).
                            # Sending it to an unsupported model causes a 400 error + silent retry.
                            # Only add it when the model name contains "flash" but NOT "lite".
                            _supports_thinking = (
                                "flash" in _gemini_model and "lite" not in _gemini_model
                                and "2.5" in _gemini_model
                            )
                            _gen_cfg: dict = {
                                "temperature": 0.4,
                                "maxOutputTokens": num_predict,
                            }
                            if _supports_thinking:
                                _gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

                            _gemini_payload = {
                                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                "generationConfig": _gen_cfg,
                            }

                            async def _stream_gemini_inner():
                                nonlocal full_answer, first_token, _gemini_ok
                                first_token = True

                                # Strategy: for short answers use non-streaming generateContent —
                                # one round-trip, lower TTFT than SSE chunking overhead.
                                # For longer answers use SSE streaming for progressive display.
                                _use_nonstream = num_predict <= 160

                                if _use_nonstream:
                                    # Non-streaming: single POST, get full answer, fake-stream words
                                    _ns_url = (
                                        f"{GEMINI_BASE_URL}/v1beta/models/{_gemini_model}"
                                        f":generateContent?key={gemini_key}"
                                    )
                                    _ns_resp = await http_client.post(
                                        _ns_url,
                                        headers={"Content-Type": "application/json"},
                                        json=_gemini_payload,
                                    )
                                    if _ns_resp.status_code != 200:
                                        print(f"[Gemini] Non-stream HTTP {_ns_resp.status_code}: {_ns_resp.text[:200]}")
                                        raise Exception(f"Gemini HTTP {_ns_resp.status_code}")
                                    _ns_data = _ns_resp.json()
                                    _ns_text = ""
                                    for _c in _ns_data.get("candidates", []):
                                        for _p in _c.get("content", {}).get("parts", []):
                                            _ns_text += _p.get("text", "")
                                    if not _ns_text:
                                        raise Exception("Gemini empty response")
                                    # Log timing at point we have the full answer
                                    _el = _loop.time()
                                    print(f"[Timing] Gemini first token: {_el-t1:.2f}s | Total: {_el-t0:.2f}s")
                                    # Fake-stream: send in word-sized chunks for smooth UI
                                    _words = _ns_text.split(" ")
                                    _buf = ""
                                    for _wi, _w in enumerate(_words):
                                        _buf += ("" if _wi == 0 else " ") + _w
                                        if len(_buf) >= 12 or _wi == len(_words) - 1:
                                            full_answer += _buf
                                            await ws.send_text(_buf)
                                            _buf = ""
                                            await asyncio.sleep(0)  # yield to event loop
                                    _gemini_ok = True
                                else:
                                    # SSE streaming for longer answers
                                    async with http_client.stream(
                                        "POST",
                                        _gemini_url,
                                        headers={"Content-Type": "application/json"},
                                        json=_gemini_payload,
                                    ) as _gresp:
                                        if _gresp.status_code != 200:
                                            _err = await _gresp.aread()
                                            print(f"[Gemini] HTTP {_gresp.status_code}: {_err[:200]}")
                                            raise Exception(f"Gemini HTTP {_gresp.status_code}")

                                        async for _line in _gresp.aiter_lines():
                                            if not _line or not _line.startswith("data: "):
                                                continue
                                            _raw = _line[6:].strip()
                                            if not _raw or _raw == "[DONE]":
                                                continue
                                            try:
                                                _chunk = json.loads(_raw)
                                                for _cand in _chunk.get("candidates", []):
                                                    for _part in _cand.get("content", {}).get("parts", []):
                                                        _tok = _part.get("text", "")
                                                        if _tok:
                                                            if first_token:
                                                                _el = _loop.time()
                                                                print(f"[Timing] Gemini first token: {_el-t1:.2f}s | Total: {_el-t0:.2f}s")
                                                                first_token = False
                                                            full_answer += _tok
                                                            await ws.send_text(_tok)
                                            except (json.JSONDecodeError, KeyError, IndexError):
                                                continue

                                    if full_answer:
                                        _gemini_ok = True
                                    else:
                                        raise Exception("Gemini empty response")

                            try:
                                await asyncio.wait_for(_stream_gemini_inner(), timeout=20.0)
                            except asyncio.TimeoutError:
                                print("[Gemini] Stream timeout after 12s — using partial answer")
                                if full_answer:
                                    _gemini_ok = True

                        except Exception as _gemini_err:
                            print(f"[Gemini] Error: {_gemini_err} — falling back to Groq")

                        if not _gemini_ok and not full_answer:
                            # Groq fallback when Gemini fails
                            print("[Gemini] Falling back to Groq")
                            _fb_key = GROQ_API_KEYS[current_key_index]
                            try:
                                async with http_client.stream(
                                    "POST",
                                    "https://api.groq.com/openai/v1/chat/completions",
                                    headers={"Authorization": f"Bearer {_fb_key}", "Content-Type": "application/json"},
                                    json={
                                        "model": GROQ_MODEL,
                                        "messages": [{"role": "user", "content": prompt}],
                                        "max_tokens": num_predict,
                                        "temperature": 0.3,
                                        "stream": True,
                                    }
                                ) as resp:
                                    if resp.status_code == 200:
                                        first_token = True
                                        async for line in resp.aiter_lines():
                                            if not line or line == "data: [DONE]":
                                                continue
                                            if line.startswith("data: "):
                                                try:
                                                    data  = json.loads(line[6:])
                                                    token = data.get("choices",[{}])[0].get("delta",{}).get("content","")
                                                    if token:
                                                        if first_token:
                                                            elapsed = _loop.time()
                                                            print(f"[Timing] Groq fallback (Gemini) first token: {elapsed-t1:.2f}s")
                                                            first_token = False
                                                        full_answer += token
                                                        await ws.send_text(token)
                                                except (json.JSONDecodeError, KeyError, IndexError):
                                                    continue
                            except Exception as _fb_e:
                                await ws.send_text(f"\n[Gemini and Groq fallback both failed: {_fb_e}]\n")

                    elif ai_provider == "qwen":
                        # ── Qwen (Alibaba DashScope) streaming ────────────────
                        # DashScope exposes an OpenAI-compatible endpoint:
                        # https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                        # Models: qwen-plus, qwen-turbo, qwen-max, qwen-long,
                        #         qwen-flash, qwen-coder-turbo, etc.
                        # Falls back to Groq if no key or on error.
                        DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
                        _qwen_ok = False

                        if qwen_key and ai_provider == "qwen":
                            _qwen_model = ai_model  # e.g. "qwen-coder-turbo"
                            print(f"[Qwen] model={_qwen_model}")
                            try:
                                async with http_client.stream(
                                    "POST",
                                    DASHSCOPE_URL,
                                    headers={
                                        "Authorization": f"Bearer {qwen_key}",
                                        "Content-Type":  "application/json",
                                    },
                                    json={
                                        "model":       _qwen_model,
                                        "messages":    [{"role": "user", "content": prompt}],
                                        "max_tokens":  num_predict,
                                        "temperature": 0.4,
                                        "stream":      True,
                                    }
                                ) as resp:
                                    if resp.status_code == 200:
                                        first_token = True

                                        async def _stream_qwen():
                                            nonlocal full_answer, first_token
                                            async for line in resp.aiter_lines():
                                                if not line or line == "data: [DONE]":
                                                    continue
                                                if line.startswith("data: "):
                                                    try:
                                                        data  = json.loads(line[6:])
                                                        token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                                        if token:
                                                            if first_token:
                                                                elapsed = _loop.time()
                                                                print(f"[Timing] Qwen first token: {elapsed-t1:.2f}s | Total: {elapsed-t0:.2f}s")
                                                                first_token = False
                                                            full_answer += token
                                                            await ws.send_text(token)
                                                    except (json.JSONDecodeError, KeyError, IndexError):
                                                        continue

                                        try:
                                            await asyncio.wait_for(_stream_qwen(), timeout=20.0)
                                            if full_answer:
                                                _qwen_ok = True
                                        except asyncio.TimeoutError:
                                            print("[Qwen] Stream timeout after 20s — using partial answer")
                                            if full_answer:
                                                _qwen_ok = True
                                    else:
                                        _err_body = await resp.aread()
                                        print(f"[Qwen] HTTP {resp.status_code}: {_err_body[:200]} — falling back to Groq")
                                        await ws.send_text(f"[Qwen error {resp.status_code} — using Groq instead]\n")

                            except Exception as _qwen_err:
                                print(f"[Qwen] Error: {_qwen_err} — falling back to Groq")

                        if not _qwen_ok and not full_answer:
                            # ── Groq fallback (when Qwen fails or no key) ─────
                            if ai_provider == "qwen":
                                print("[Qwen] Falling back to Groq")
                            attempts = 0
                            _groq_model = GROQ_MODEL
                            while attempts < len(GROQ_API_KEYS):
                                api_key = GROQ_API_KEYS[current_key_index]
                                try:
                                    async with http_client.stream(
                                        "POST",
                                        "https://api.groq.com/openai/v1/chat/completions",
                                        headers={
                                            "Authorization": f"Bearer {api_key}",
                                            "Content-Type":  "application/json",
                                        },
                                        json={
                                            "model":       _groq_model,
                                            "messages":    [{"role": "user", "content": prompt}],
                                            "max_tokens":  num_predict,
                                            "temperature": 0.4,
                                            "stream":      True,
                                        }
                                    ) as resp:
                                        if resp.status_code == 429:
                                            current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                                            attempts += 1
                                            continue
                                        first_token = True
                                        async def _stream_qwen_groq_fallback():
                                            nonlocal full_answer, first_token
                                            async for line in resp.aiter_lines():
                                                if not line or line == "data: [DONE]":
                                                    continue
                                                if line.startswith("data: "):
                                                    try:
                                                        data  = json.loads(line[6:])
                                                        token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                                        if token:
                                                            if first_token:
                                                                elapsed = _loop.time()
                                                                print(f"[Timing] Groq fallback (Qwen) first token: {elapsed-t1:.2f}s")
                                                                first_token = False
                                                            full_answer += token
                                                            await ws.send_text(token)
                                                    except (json.JSONDecodeError, KeyError, IndexError):
                                                        continue
                                        try:
                                            await asyncio.wait_for(_stream_qwen_groq_fallback(), timeout=20.0)
                                        except asyncio.TimeoutError:
                                            print("[Groq fallback] Timeout after 20s")
                                        break
                                except Exception as e:
                                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                                    attempts += 1
                                    if attempts >= len(GROQ_API_KEYS):
                                        await ws.send_text(f"\n[All Groq keys exhausted: {e}]")

                    else:
                        # ── Groq streaming ────────────────────────────────────
                        attempts = 0
                        _groq_model = ai_model if ai_model else GROQ_MODEL
                        while attempts < len(GROQ_API_KEYS):
                            api_key = GROQ_API_KEYS[current_key_index]
                            try:
                                async with http_client.stream(
                                    "POST",
                                    "https://api.groq.com/openai/v1/chat/completions",
                                    headers={
                                        "Authorization": f"Bearer {api_key}",
                                        "Content-Type":  "application/json",
                                    },
                                    json={
                                        "model":       _groq_model,
                                        "messages":    [{"role": "user", "content": prompt}],
                                        "max_tokens":  num_predict,
                                        "temperature": 0.4,
                                        "stream":      True,
                                    }
                                ) as resp:
                                    if resp.status_code == 429:
                                        current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                                        attempts += 1
                                        await ws.send_text(f"[Switching to backup key {current_key_index+1}...]\n")
                                        continue

                                    first_token = True
                                    async def _stream_groq():
                                        nonlocal full_answer, first_token
                                        async for line in resp.aiter_lines():
                                            if not line or line == "data: [DONE]":
                                                continue
                                            if line.startswith("data: "):
                                                try:
                                                    data  = json.loads(line[6:])
                                                    token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                                    if token:
                                                        if first_token:
                                                            elapsed = _loop.time()
                                                            print(f"[Timing] Groq first token: {elapsed-t1:.2f}s | Total: {elapsed-t0:.2f}s")
                                                            first_token = False
                                                        full_answer += token
                                                        await ws.send_text(token)
                                                except (json.JSONDecodeError, KeyError, IndexError):
                                                    continue
                                    try:
                                        await asyncio.wait_for(_stream_groq(), timeout=20.0)
                                    except asyncio.TimeoutError:
                                        print(f"[Groq] Stream timeout after 20s — sending partial answer")
                                    break

                            except Exception as e:
                                current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                                attempts += 1
                                if attempts >= len(GROQ_API_KEYS):
                                    await ws.send_text(f"\n[All Groq keys exhausted: {e}]")

                    if full_answer:
                        conversation_history.append((transcript, full_answer))
                        if len(conversation_history) > 5:
                            conversation_history.pop(0)

                    await ws.send_text("\n\n---\n\n")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")