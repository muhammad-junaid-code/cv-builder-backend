"""
CV Builder AI — UI2 PDF Builder
Modern Sidebar layout: teal left column, white right column (two-column ReportLab table).
"""

import io
import re as _contact_re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

# ── Import shared helpers from main ──────────────────────────────────────────
from UI._shared import _normalise_edu_entry, _infer_degree_duration, _contact_href

# ==============================================================================
# PDF BUILDER — UI2 (Modern Sidebar: teal left column, white right column)
# Distinct from UI1: two-column ReportLab table, teal header block, no uppercase sections
# ==============================================================================
def build_cv_pdf_ui2(cv: dict, profile_data: dict = None) -> bytes:
    """UI2 — Modern two-column sidebar layout rendered with ReportLab."""
    from reportlab.platypus import KeepTogether

    _pd       = profile_data or {}
    p_name    = (_pd.get("name") or "").strip() or "CANDIDATE"
    p_links   = _pd.get("links") or []
    p_work    = _pd.get("work")  or []
    p_edu     = [_normalise_edu_entry(e) for e in (_pd.get("edu") or [])]

    TEAL      = colors.HexColor("#1a5276")
    LTEAL     = colors.HexColor("#d6eaf8")
    TEAL_DARK = colors.HexColor("#154360")
    SIDEBAR_W_MM = 62 * mm   # sidebar column width

    buf  = io.BytesIO()
    PAGE_W, _ = A4
    ML, MR, MT, MB = 0, 0, 0, 0
    # 4.0× A4 height — tall enough that Frame.addFromList() never silently
    # drops overflowing content (projects, skills, final sections).
    PAGE_H_SINGLE = 841.89 * 4.0

    doc = SimpleDocTemplate(
        buf, pagesize=(PAGE_W, PAGE_H_SINGLE),
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title=f"{p_name} CV", author=p_name,
    )
    TW = PAGE_W  # full width; columns handled via Table

    def ps2(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0,
                 spaceBefore=0, textColor=colors.HexColor("#111111"))
        d.update(kw)
        return ParagraphStyle(name, **d)

    # ── Styles ────────────────────────────────────────────────────────────────
    S = {
        # Sidebar styles (white/light text on teal)
        "sb_name_first": ps2("s_nf", fontName="Helvetica", fontSize=16, leading=20,
                              textColor=colors.HexColor("#cce5f5"), spaceBefore=0),
        "sb_name_last":  ps2("s_nl", fontName="Helvetica-Bold", fontSize=18, leading=22,
                              textColor=colors.white, spaceBefore=0),
        "sb_jobtitle":   ps2("s_jt", fontName="Helvetica", fontSize=8, leading=11,
                              textColor=colors.HexColor("#7fb3d3"), spaceBefore=4),
        "sb_sec":        ps2("s_sec", fontName="Helvetica-Bold", fontSize=7.5, leading=11,
                              textColor=colors.HexColor("#7fb3d3"), spaceBefore=14, spaceAfter=4),
        "sb_lbl":        ps2("s_lbl", fontName="Helvetica-Bold", fontSize=7, leading=9,
                              textColor=colors.HexColor("#7fb3d3")),
        "sb_val":        ps2("s_val", fontName="Helvetica", fontSize=8.5, leading=12,
                              textColor=colors.white),
        "sb_skill":      ps2("s_sk",  fontName="Helvetica", fontSize=8, leading=11.5,
                              textColor=colors.HexColor("#ddeeff")),
        "sb_uni":        ps2("s_uni", fontName="Helvetica-Bold", fontSize=9, leading=12,
                              textColor=colors.white, spaceBefore=6),
        "sb_deg":        ps2("s_deg", fontName="Helvetica", fontSize=8, leading=11,
                              textColor=colors.HexColor("#a8d8ea")),
        "sb_note":       ps2("s_nt",  fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
                              textColor=colors.HexColor("#7fb3d3")),
        # Main panel styles — badge-style section headers, distinct from UI1
        "m_sec":         ps2("m_sec", fontName="Helvetica-Bold", fontSize=8.5, leading=12,
                              textColor=colors.white, spaceBefore=14, spaceAfter=6),
        "m_company":     ps2("m_co",  fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                              textColor=colors.HexColor("#0d2b45"), spaceBefore=4),
        "m_role":        ps2("m_rl",  fontName="Helvetica", fontSize=9.5, leading=13,
                              textColor=TEAL, spaceAfter=3),
        "m_date":        ps2("m_dt",  fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                              textColor=TEAL_DARK, alignment=TA_RIGHT),
        "m_bullet":      ps2("m_bul", fontName="Helvetica", fontSize=9.5, leading=13.5,
                              leftIndent=10, spaceAfter=2,
                              textColor=colors.HexColor("#222222")),
        "m_tech":        ps2("m_tch", fontName="Helvetica-Bold", fontSize=8, leading=11,
                              leftIndent=10, textColor=TEAL),
        "m_summary":     ps2("m_sum", fontName="Helvetica", fontSize=9.5, leading=15,
                              textColor=colors.HexColor("#2c2c2c")),
        "m_proj_name":   ps2("m_pn",  fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                              textColor=colors.HexColor("#0d2b45"), spaceBefore=4),
        "m_proj_body":   ps2("m_pb",  fontName="Helvetica", fontSize=9, leading=13,
                              textColor=colors.HexColor("#444444")),
        "m_proj_bullet": ps2("m_pbl", fontName="Helvetica", fontSize=9, leading=12.5,
                              leftIndent=10, spaceAfter=2),
        "m_proj_stack":  ps2("m_ps",  fontName="Helvetica-Bold", fontSize=8, leading=10,
                              textColor=TEAL_DARK),
        "m_comp":        ps2("m_cp",  fontName="Helvetica", fontSize=9, leading=13,
                              textColor=colors.HexColor("#333333")),
    }

    def esc(s):
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    # ── Build sidebar flowable list ────────────────────────────────────────────
    name_parts = p_name.strip().split()
    first_name = " ".join(name_parts[:-1]) if len(name_parts) > 1 else p_name
    last_name  = name_parts[-1] if len(name_parts) > 1 else ""

    sidebar_items = [
        Paragraph(esc(first_name), S["sb_name_first"]),
        Paragraph(esc(last_name),  S["sb_name_last"]),
        Paragraph(esc(cv.get("title","") or ""), S["sb_jobtitle"]),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#2e6a9e"),
                   spaceBefore=10, spaceAfter=4),
    ]

    # Contact — phone numbers (>4 digits) become tel: links, emails → mailto:, URLs → https:
    sidebar_items.append(Paragraph("CONTACT", S["sb_sec"]))
    for lnk in p_links:
        sidebar_items.append(Paragraph(esc(lnk.get("label","")), S["sb_lbl"]))
        _cv = (lnk.get("value") or "").strip()
        _href = _contact_href(_cv)
        _safe = esc(_cv)
        if _href and lnk.get("label","").strip().lower() != "location":
            _val_para = Paragraph(f'<a href="{_href}" color="#cce5f5">{_safe}</a>', S["sb_val"])
        else:
            _val_para = Paragraph(_safe, S["sb_val"])
        sidebar_items.append(_val_para)
        sidebar_items.append(Spacer(1, 4))

    # Core Competencies (in sidebar; Skills already shown in main panel)
    comp_sidebar = (cv.get("competencies") or "").strip()
    if comp_sidebar:
        sidebar_items.append(Paragraph("CORE COMPETENCIES", S["sb_sec"]))
        # Split on * separator (same as competencies format) and render each pill
        comp_pills = [c.strip() for c in comp_sidebar.replace(" * ", "*").split("*") if c.strip()]
        for pill in comp_pills:
            sidebar_items.append(Paragraph(f"• {esc(pill)}", S["sb_skill"]))
            sidebar_items.append(Spacer(1, 2))

    # Education — prefer profile edu; fall back to cv["education"].
    # Pre-merge cv["education"] years into profile entries that lack from/to
    # (same logic as UI3) so dates always display correctly.
    _ui2_cv_edu = cv.get("education") or []
    if isinstance(_ui2_cv_edu, dict):
        _ui2_cv_edu = [_ui2_cv_edu]

    if p_edu:
        _ui2_edu_list = []
        for _i2, _pe2 in enumerate(p_edu):
            _e2 = dict(_pe2)
            _ef2 = str(_pe2.get("from") or "").strip()
            _et2 = str(_pe2.get("to")   or "").strip()
            if not _ef2 and not _et2 and _i2 < len(_ui2_cv_edu):
                _yr2 = str(_ui2_cv_edu[_i2].get("years") or "").strip()
                if _yr2:
                    _e2["years"] = _yr2
            _ui2_edu_list.append(_e2)
    else:
        _ui2_edu_list = []
        for _uce in _ui2_cv_edu:
            _yr_raw = str(_uce.get("years","") or "").strip()
            _ef2 = _yr_raw.split("-")[0].strip() if "-" in _yr_raw else ""
            _et2 = _yr_raw.split("-")[-1].strip() if "-" in _yr_raw else ""
            _ui2_edu_list.append({
                "institution": (_uce.get("university") or _uce.get("institution") or "").strip(),
                "degree":      (_uce.get("degree") or "").strip(),
                "cgpa":        (_uce.get("cgpa") or "").strip(),
                "from": _ef2, "to": _et2,
                "note":        (_uce.get("achievement") or "").strip(),
            })

    sidebar_items.append(Paragraph("EDUCATION", S["sb_sec"]))
    for e in _ui2_edu_list:
        ef = str(e.get("from") or "").strip()
        et = str(e.get("to")   or "").strip()
        # Fall back to years field when from/to absent
        if not ef and not et:
            _yr_fb = str(e.get("years") or "").strip()
            _sep_fb = "–" if "–" in _yr_fb else "-"
            if _yr_fb and _sep_fb in _yr_fb:
                _pts = [p.strip() for p in _yr_fb.split(_sep_fb, 1)]
                ef, et = _pts[0], _pts[-1]
        dr = f"{ef}–{et}" if ef and et else (ef + "–Present" if ef else et)
        sidebar_items.append(Paragraph(esc(e.get("institution") or ""), S["sb_uni"]))
        sidebar_items.append(Paragraph(esc(e.get("degree") or ""), S["sb_deg"]))
        _note = (e.get("note") or e.get("cgpa") or "").strip()
        if _note:
            sidebar_items.append(Paragraph(esc(_note), S["sb_note"]))
        if dr:
            sidebar_items.append(Paragraph(dr, S["sb_note"]))
        sidebar_items.append(Spacer(1, 4))

    # ── Build main panel flowable list ────────────────────────────────────────
    main_items = [Spacer(1, 8)]
    MAIN_PAD = 20  # define here so main_sec closure can reference it

    def main_sec(title):
        # Badge-style: full-width teal rectangle with white uppercase text
        badge_w = PAGE_W - SIDEBAR_W_MM - 2 * MAIN_PAD
        badge = Table(
            [[Paragraph(esc(title.upper()), S["m_sec"])]],
            colWidths=[badge_w],
            style=TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), TEAL_DARK),
                ("LEFTPADDING",   (0,0),(-1,-1), 8),
                ("RIGHTPADDING",  (0,0),(-1,-1), 8),
                ("TOPPADDING",    (0,0),(-1,-1), 4),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ])
        )
        main_items.append(Spacer(1, 6))
        main_items.append(badge)
        main_items.append(Spacer(1, 5))

    # Summary
    main_sec("Professional Summary")
    main_items.append(Paragraph(esc(cv.get("summary") or ""), S["m_summary"]))
    main_items.append(Spacer(1, 8))

    # Experience — unified loop: merges profile work + AI companies by index
    # so ALL companies always render with their date range and bullets.
    main_sec("Work Experience")
    ai_cos = cv.get("companies") or []
    # Cap to ai_cos count when it is non-empty — ai_cos has already been
    # hard-truncated by fix_companies() to the correct count for the selected
    # years_exp value (1 yr → 1, 2 yrs → 2, etc.).  Using max() would let
    # extra p_work entries bleed through as bare company names with no bullets.
    # Fall back to max() only when ai_cos is empty (profile-only render path).
    _ui2_num_entries = len(ai_cos) if ai_cos else max(len(ai_cos), len(p_work))
    for i in range(_ui2_num_entries):
        w  = p_work[i]  if i < len(p_work)  else {}
        ai = ai_cos[i]  if i < len(ai_cos)  else {}
        company = (w.get("company") or "").strip() or ai.get("company","")
        role    = (w.get("role")    or "").strip() or ai.get("role","")
        wf, wt  = str(w.get("from") or "").strip(), str(w.get("to") or "").strip()
        if wf and wt: dr = f"{wf} – {wt}"
        elif wf:      dr = f"{wf} – Present"
        else:         dr = ai.get("dateRange","")
        # Use absolute widths — percentage strings can be unreliable inside
        # a Frame context and the 30% col (~114pt) is too narrow for long dates
        # like "January 2023 - September 2024" (~127pt). Fixed 140pt date col.
        _mn_w = PAGE_W - SIDEBAR_W_MM - 2 * MAIN_PAD
        main_items.append(Table(
            [[Paragraph(esc(company), S["m_company"]), Paragraph(esc(dr), S["m_date"])]],
            colWidths=[_mn_w - 140, 140],
            style=TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                              ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                              ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),2),
                              ("ALIGN",(1,0),(1,-1),"RIGHT")])
        ))
        main_items.append(Paragraph(esc(role), S["m_role"]))
        bullets = ai.get("bullets") or []
        if not bullets and w.get("bullets"):
            bullets = [b.strip() for b in str(w.get("bullets","")).split("\n") if b.strip()]
        for b in bullets:
            b_clean = b.lstrip("•·▸–▪●◦ ").strip()
            main_items.append(Paragraph('<font size="7">▸</font> ' + esc(b_clean), S["m_bullet"]))
        tech_raw = ai.get("tech","")
        if tech_raw:
            sep = "|" if "|" in tech_raw else ","
            tags = " · ".join(t.strip() for t in tech_raw.split(sep) if t.strip())
            main_items.append(Paragraph(tags, S["m_tech"]))
        main_items.append(Spacer(1, 8))

    # Projects
    # ── Technical Skills ─────────────────────────────────────────────────────
    ui2_skills = cv.get("skills") or []
    if ui2_skills:
        main_sec("Technical Skills")
        # Add two styles needed for the skills cards
        sk_cat_s = ps2("u2_skc", fontName="Helvetica-Bold", fontSize=8.5, leading=12,
                        textColor=colors.white)
        sk_val_s = ps2("u2_skv", fontName="Helvetica", fontSize=9, leading=13,
                        textColor=colors.HexColor("#2c2c2c"))
        main_col_inner = PAGE_W - SIDEBAR_W_MM - 2 * MAIN_PAD
        cat_w = main_col_inner * 0.30
        val_w = main_col_inner * 0.70
        for idx, sk in enumerate(ui2_skills):
            colon = sk.find(":")
            if colon > 0:
                cat = sk[:colon].strip()
                val = sk[colon+1:].strip()
            else:
                cat, val = "Skills", sk
            # Alternating row tint for visual rhythm
            row_bg = colors.HexColor("#f0f5fb") if idx % 2 == 0 else colors.white
            skill_row = Table(
                [[Paragraph(esc(cat.upper()), sk_cat_s),
                  Paragraph(esc(val), sk_val_s)]],
                colWidths=[cat_w, val_w],
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (0, -1), TEAL_DARK),
                    ("BACKGROUND",    (1, 0), (1, -1), row_bg),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING",   (0, 0), (0, -1), 6),
                    ("RIGHTPADDING",  (0, 0), (0, -1), 4),
                    ("LEFTPADDING",   (1, 0), (1, -1), 8),
                    ("RIGHTPADDING",  (1, 0), (1, -1), 4),
                    ("TOPPADDING",    (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW",     (0, 0), (-1, -1), 0.5, colors.HexColor("#d0dce8")),
                ])
            )
            main_items.append(skill_row)
        main_items.append(Spacer(1, 8))

    # Projects
    main_sec("Key Projects")
    for p in (cv.get("projects") or []):
        raw_name = (p.get("name") or "").strip()
        import re as _re2
        name = _re2.sub(r'\s*\[[^\]]*\]\s*$', '', raw_name)
        name = _re2.sub(r'^[A-Z][A-Z\s&\-]{2,}:\s*', '', name).strip()
        main_items.append(Paragraph(esc(name), S["m_proj_name"]))
        if p.get("overview"):
            main_items.append(Paragraph(esc(p["overview"]), S["m_proj_body"]))
        for b in (p.get("bullets") or []):
            b_clean = b.lstrip("•·▸– ").strip()
            main_items.append(Paragraph('<font size="7">▸</font> ' + esc(b_clean), S["m_proj_bullet"]))
        tech_t = p.get("techTags") or []
        if not tech_t and p.get("tech"):
            sep = "|" if "|" in p["tech"] else ","
            tech_t = [t.strip() for t in p["tech"].split(sep) if t.strip()]
        if tech_t:
            main_items.append(Paragraph(" · ".join(esc(t) for t in tech_t), S["m_proj_stack"]))
        main_items.append(Spacer(1, 6))

    # Core Competencies moved to sidebar — no duplicate in main panel

    # ── Render UI2 two-column layout via low-level Frame/Canvas drawing ──────
    # A single-row ReportLab Table whose cell content is taller than the page
    # frame raises "Flowable … too large on page".  Fix: draw each column
    # directly into its own Frame on a raw canvas — no Table flowable needed.
    SIDEBAR_PAD = 16
    sidebar_col_w = SIDEBAR_W_MM
    main_col_w    = PAGE_W - sidebar_col_w

    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.platypus.frames import Frame as _Frame

    buf2 = io.BytesIO()
    c2   = _rl_canvas.Canvas(buf2, pagesize=(PAGE_W, PAGE_H_SINGLE))

    # Background fills (draw bottom→top so teal sidebar is underneath content)
    c2.setFillColor(TEAL)
    c2.rect(0, 0, sidebar_col_w, PAGE_H_SINGLE, fill=1, stroke=0)
    c2.setFillColor(colors.white)
    c2.rect(sidebar_col_w, 0, main_col_w, PAGE_H_SINGLE, fill=1, stroke=0)

    # Sidebar Frame — padded inside the teal column, top-aligned
    sb_inner_w = sidebar_col_w - 2 * SIDEBAR_PAD
    sb_inner_h = PAGE_H_SINGLE - 2 * SIDEBAR_PAD
    # ReportLab y-origin is bottom-left; content starts near the top.
    sb_frame = _Frame(
        SIDEBAR_PAD,                        # x (from left)
        SIDEBAR_PAD,                        # y (from bottom) — content grows upward
        sb_inner_w, sb_inner_h,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0,
    )
    sb_frame.addFromList(list(sidebar_items), c2)

    # Main-panel Frame — padded inside the right column
    mn_inner_w = main_col_w - 2 * MAIN_PAD
    mn_inner_h = PAGE_H_SINGLE - 8 - MAIN_PAD   # 8pt top gap, MAIN_PAD bottom gap
    mn_frame = _Frame(
        sidebar_col_w + MAIN_PAD,           # x
        MAIN_PAD,                           # y
        mn_inner_w, mn_inner_h,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0,
    )
    mn_frame.addFromList(list(main_items), c2)

    c2.save()
    buf2.seek(0)

    # Crop to actual content — use the lowest frame cursor as the content bottom
    lowest_y = min(
        sb_frame._y if hasattr(sb_frame, '_y') else 0,
        mn_frame._y if hasattr(mn_frame, '_y') else 0,
    )
    tight_h = PAGE_H_SINGLE - lowest_y + 8 * mm
    tight_h = max(tight_h, 100 * mm)
    crop_bottom = PAGE_H_SINGLE - tight_h

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "--quiet"])
        from pypdf import PdfReader, PdfWriter

    reader = PdfReader(buf2)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    page.mediabox.lower_left  = (0, crop_bottom)
    page.mediabox.upper_right = (PAGE_W, PAGE_H_SINGLE)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()
