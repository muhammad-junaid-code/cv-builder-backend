"""
CV Builder AI — UI6 PDF Builder
Bold Split layout: deep indigo full-bleed header band, then two-column body below.
Left column: skills + education + competencies. Right column: experience + projects.
Sharp contrast, bold typography, vibrant coral/crimson accent on dark indigo.
"""

import io
import math
import re as _re6
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

from UI._shared import _normalise_edu_entry, _infer_degree_duration, _contact_href


def build_cv_pdf_ui6(cv: dict, profile_data: dict = None) -> bytes:
    """UI6 — Bold Split: indigo header band, two-column body, coral accent."""
    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.platypus.frames import Frame as _Frame

    _pd     = profile_data or {}
    p_name  = (_pd.get("name") or "").strip() or "CANDIDATE"
    p_links = _pd.get("links") or []
    p_work  = _pd.get("work")  or []
    p_edu   = [_normalise_edu_entry(e) for e in (_pd.get("edu") or [])]

    # ── Palette ──────────────────────────────────────────────────────────────
    INDIGO    = colors.HexColor("#1e1b4b")   # deep indigo header bg
    INDIGO_M  = colors.HexColor("#312e81")   # mid indigo for left column bg
    CORAL     = colors.HexColor("#f43f5e")   # vibrant coral/rose accent
    CORAL_L   = colors.HexColor("#ffe4e8")   # pale coral for chips
    LAVENDER  = colors.HexColor("#c4b5fd")   # lavender for header text accents
    WHITE     = colors.white
    OFF_WHITE = colors.HexColor("#f8f7ff")   # left column bg
    BODY_BG   = colors.HexColor("#ffffff")   # right column bg
    INK       = colors.HexColor("#1e1b4b")   # heading ink (reuse indigo)
    BODY_TXT  = colors.HexColor("#374151")
    MUTED     = colors.HexColor("#9ca3af")
    RULE      = colors.HexColor("#e5e7eb")

    PAGE_W, _ = A4
    HEADER_H  = 30 * mm     # full-bleed indigo header height
    PAGE_H_SINGLE = 841.89 * 4.5
    LEFT_W    = 63 * mm     # left column width
    RIGHT_W   = PAGE_W - LEFT_W
    L_PAD     = 14
    R_PAD     = 16
    BODY_TOP  = PAGE_H_SINGLE - HEADER_H   # canvas y where body columns start

    def ps6(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0,
                 spaceBefore=0, textColor=BODY_TXT)
        d.update(kw)
        return ParagraphStyle(name, **d)

    S = {
        # ── Header (rendered directly on canvas) ──────────────────────────────
        "hd_name":   ps6("u6_hn", fontName="Helvetica-Bold", fontSize=22, leading=28,
                         textColor=WHITE, alignment=TA_CENTER),
        "hd_title":  ps6("u6_ht", fontName="Helvetica", fontSize=9, leading=13,
                         textColor=LAVENDER, alignment=TA_CENTER),
        "hd_contact":ps6("u6_hc", fontName="Helvetica", fontSize=8, leading=11,
                         textColor=colors.HexColor("#e0e7ff"), alignment=TA_CENTER),
        # ── Left column ───────────────────────────────────────────────────────
        "lc_sec":    ps6("u6_ls", fontName="Helvetica-Bold", fontSize=7, leading=10,
                         textColor=CORAL, spaceBefore=14, spaceAfter=4,
                         letterSpacing=1.5),
        "lc_skill_c":ps6("u6_lsc", fontName="Helvetica-Bold", fontSize=9, leading=12,
                         textColor=INK),
        "lc_skill_v":ps6("u6_lsv", fontName="Helvetica", fontSize=8.5, leading=13,
                         textColor=BODY_TXT),
        "lc_edu_uni":ps6("u6_leu", fontName="Helvetica-Bold", fontSize=9, leading=12,
                         textColor=INK, spaceBefore=6),
        "lc_edu_deg":ps6("u6_led", fontName="Helvetica", fontSize=8, leading=11,
                         textColor=BODY_TXT),
        "lc_edu_yr": ps6("u6_ley", fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
                         textColor=CORAL),
        "lc_comp":   ps6("u6_lcp", fontName="Helvetica", fontSize=8.5, leading=13,
                         textColor=BODY_TXT),
        # ── Right column ──────────────────────────────────────────────────────
        "rc_sec":    ps6("u6_rs", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                         textColor=CORAL, spaceBefore=14, spaceAfter=4,
                         letterSpacing=1.5),
        "rc_company":ps6("u6_rco", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                         textColor=INK, spaceBefore=2),
        "rc_role":   ps6("u6_rrl", fontName="Helvetica-Oblique", fontSize=9.5, leading=13,
                         textColor=colors.HexColor("#4338ca"), spaceAfter=3),
        "rc_date":   ps6("u6_rdt", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         textColor=MUTED, alignment=TA_RIGHT),
        "rc_bullet": ps6("u6_rbl", fontName="Helvetica", fontSize=9.5, leading=14,
                         leftIndent=10, textColor=BODY_TXT, spaceAfter=2),
        "rc_tech":   ps6("u6_rtch", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                         leftIndent=10, textColor=CORAL),
        "rc_summary":ps6("u6_rsum", fontName="Helvetica", fontSize=9.5, leading=15,
                         textColor=BODY_TXT),
        "rc_proj_nm":ps6("u6_rpn", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                         textColor=INK, spaceBefore=4),
        "rc_proj_bd":ps6("u6_rpb", fontName="Helvetica", fontSize=9, leading=13,
                         textColor=BODY_TXT),
        "rc_proj_bl":ps6("u6_rpbl", fontName="Helvetica", fontSize=9, leading=12.5,
                         leftIndent=10, textColor=BODY_TXT, spaceAfter=2),
        "rc_proj_tk":ps6("u6_rpt", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                         textColor=CORAL),
    }

    def esc(s):
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    # ── Left column items ─────────────────────────────────────────────────────
    left_items = [Spacer(1, 10)]
    lc_inner_w = LEFT_W - 2 * L_PAD

    def lc_sec(title):
        left_items.append(Paragraph(title.upper(), S["lc_sec"]))
        left_items.append(HRFlowable(width="100%", thickness=1, color=CORAL,
                                     spaceBefore=2, spaceAfter=6))

    # Skills
    skills = cv.get("skills") or []
    if skills:
        lc_sec("Skills")
        for sk in skills[:6]:
            colon = sk.find(":")
            if colon > 0:
                cat = sk[:colon].strip()
                val = sk[colon+1:].strip()
                left_items.append(Paragraph(f"<b>{esc(cat)}</b>", S["lc_skill_c"]))
                left_items.append(Paragraph(esc(val), S["lc_skill_v"]))
            else:
                left_items.append(Paragraph(esc(sk), S["lc_skill_v"]))
            left_items.append(HRFlowable(width="100%", thickness=0.3, color=RULE,
                                         spaceBefore=3, spaceAfter=3))

    # Competencies
    comp_str = (cv.get("competencies") or "").strip()
    if comp_str:
        lc_sec("Competencies")
        pills = [c.strip() for c in comp_str.replace("*","•").split("•") if c.strip()]
        for pill in pills:
            left_items.append(Paragraph(f"▸ {esc(pill)}", S["lc_comp"]))
            left_items.append(Spacer(1, 3))

    # Education
    _cv_edu = cv.get("education") or []
    if isinstance(_cv_edu, dict): _cv_edu = [_cv_edu]
    if p_edu:
        _edu_rl = []
        for _i, _pe in enumerate(p_edu):
            _e = dict(_pe)
            _ef = str(_pe.get("from") or "").strip()
            _et = str(_pe.get("to")   or "").strip()
            if not _ef and not _et and _i < len(_cv_edu):
                _yr = str(_cv_edu[_i].get("years") or "").strip()
                if _yr: _e["years"] = _yr
            _edu_rl.append(_e)
    else:
        _edu_rl = []
        for _ce in _cv_edu:
            _yr_r = str(_ce.get("years","") or "").strip()
            _sep  = "–" if "–" in _yr_r else "-"
            _pts  = [p.strip() for p in _yr_r.split(_sep,1)] if _sep in _yr_r else ["",""]
            _edu_rl.append({
                "institution": (_ce.get("university") or _ce.get("institution") or "").strip(),
                "degree": (_ce.get("degree") or "").strip(),
                "cgpa":   (_ce.get("cgpa")   or "").strip(),
                "from": _pts[0], "to": _pts[-1],
                "achievement": (_ce.get("achievement") or "").strip(),
            })

    if _edu_rl:
        lc_sec("Education")
        _u6_prev_yr = None
        for e in _edu_rl:
            ef  = str(e.get("from") or "").strip()
            et  = str(e.get("to")   or "").strip()
            deg = (e.get("degree") or "").strip()
            uni = (e.get("institution") or "").strip()
            cgpa = (e.get("cgpa") or "").strip()
            ach  = (e.get("achievement") or "").strip()

            if ef and et:
                dr = f"{ef}–{et}"
            else:
                _yr_r = str(e.get("years") or "").strip()
                _sep2 = "–" if "–" in _yr_r else "-"
                if _yr_r and _sep2 in _yr_r:
                    _yp = [p.strip() for p in _yr_r.split(_sep2,1)]
                    ef, et = _yp[0], _yp[-1]; dr = f"{ef}–{et}"
                elif et and not ef:
                    _dur = _infer_degree_duration(deg)
                    try: ef = str(int(et[:4]) - _dur)
                    except: pass
                    dr = f"{ef}–{et}" if ef else et
                elif ef and not et:
                    _dur = _infer_degree_duration(deg)
                    try: et = str(int(ef[:4]) + _dur)
                    except: pass
                    dr = f"{ef}–{et}" if et else ef
                elif _u6_prev_yr is not None:
                    _dur = _infer_degree_duration(deg)
                    et = str(_u6_prev_yr); ef = str(_u6_prev_yr - _dur)
                    dr = f"{ef}–{et}"
                else:
                    dr = ""
            try: _u6_prev_yr = int(str(ef)[:4]) if ef else _u6_prev_yr
            except: pass

            left_items.append(Paragraph(esc(uni), S["lc_edu_uni"]))
            left_items.append(Paragraph(esc(deg), S["lc_edu_deg"]))
            _note = (e.get("achievement") or e.get("cgpa") or "").strip()
            if _note: left_items.append(Paragraph(esc(_note), S["lc_edu_yr"]))
            if dr:    left_items.append(Paragraph(dr, S["lc_edu_yr"]))
            left_items.append(Spacer(1, 5))

    # ── Right column items ────────────────────────────────────────────────────
    right_items = [Spacer(1, 10)]
    rc_inner_w = RIGHT_W - 2 * R_PAD

    def rc_sec(title):
        right_items.append(Paragraph(title.upper(), S["rc_sec"]))
        right_items.append(HRFlowable(width="100%", thickness=1, color=CORAL,
                                      spaceBefore=2, spaceAfter=6))

    # Summary
    summary_text = (cv.get("summary") or "").strip()
    if summary_text:
        rc_sec("Professional Summary")
        right_items.append(Paragraph(esc(summary_text), S["rc_summary"]))

    # Experience
    ai_cos = cv.get("companies") or []
    num_entries = len(ai_cos) if ai_cos else max(len(ai_cos), len(p_work))
    if num_entries > 0:
        rc_sec("Experience")
        for i in range(num_entries):
            w  = p_work[i] if i < len(p_work) else {}
            ai = ai_cos[i] if i < len(ai_cos) else {}
            company = (w.get("company") or "").strip() or ai.get("company","")
            role    = (w.get("role")    or "").strip() or ai.get("role","")
            wf = str(w.get("from") or "").strip()
            wt = str(w.get("to")   or "").strip()
            if wf and wt:  dr = f"{wf} – {wt}"
            elif wf:       dr = f"{wf} – Present"
            else:          dr = ai.get("dateRange","")

            right_items.append(Table(
                [[Paragraph(esc(company), S["rc_company"]),
                  Paragraph(esc(dr), S["rc_date"])]],
                colWidths=[rc_inner_w - 120, 120],
                style=TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
                    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),2),
                    ("ALIGN",(1,0),(1,-1),"RIGHT"),
                ])
            ))
            if role:
                right_items.append(Paragraph(esc(role), S["rc_role"]))
            bullets = ai.get("bullets") or []
            if not bullets and w.get("bullets"):
                bullets = [b.strip() for b in str(w["bullets"]).split("\n") if b.strip()]
            for b in bullets:
                b_clean = b.lstrip("•·▸–▪● ").strip()
                right_items.append(Paragraph(
                    '<font color="#f43f5e" size="9">◉</font> ' + esc(b_clean),
                    S["rc_bullet"]))
            tech_raw = ai.get("tech","")
            if tech_raw:
                sep  = "|" if "|" in tech_raw else ","
                tags = "  ·  ".join(t.strip() for t in tech_raw.split(sep) if t.strip())
                right_items.append(Paragraph(esc(tags), S["rc_tech"]))
            right_items.append(HRFlowable(width="100%", thickness=0.4, color=RULE,
                                          spaceBefore=6, spaceAfter=2))

    # Projects
    projects = cv.get("projects") or []
    if projects:
        rc_sec("Projects")
        for p in projects:
            raw_name = (p.get("name") or "").strip()
            name = _re6.sub(r'\s*\[[^\]]*\]\s*$', '', raw_name)
            name = _re6.sub(r'^[A-Z][A-Z\s&\-]{2,}:\s*', '', name).strip()
            if name:
                right_items.append(Paragraph(esc(name), S["rc_proj_nm"]))
            if p.get("overview"):
                right_items.append(Paragraph(esc(p["overview"]), S["rc_proj_bd"]))
            for b in (p.get("bullets") or []):
                b_clean = b.lstrip("•·▸– ").strip()
                right_items.append(Paragraph(
                    '<font color="#f43f5e" size="9">◉</font> ' + esc(b_clean),
                    S["rc_proj_bl"]))
            tech_t = p.get("techTags") or []
            if not tech_t and p.get("tech"):
                sep = "|" if "|" in p["tech"] else ","
                tech_t = [t.strip() for t in p["tech"].split(sep) if t.strip()]
            if tech_t:
                right_items.append(Paragraph(
                    "  ·  ".join(esc(t) for t in tech_t), S["rc_proj_tk"]))
            right_items.append(Spacer(1, 7))

    # Certifications (optional) — tabular card design
    _certs6 = cv.get("certifications") or []
    if _certs6:
        rc_sec("Certifications")
        _cert_note6 = (cv.get("cert_note") or "").strip()
        if _cert_note6:
            _cnote6_s = ps6("u6_cnote", fontName="Helvetica-Oblique", fontSize=9, leading=13,
                             textColor=colors.HexColor("#444444"), spaceAfter=4)
            right_items.append(Paragraph(esc(_cert_note6), _cnote6_s))
        _c6_name_s = ps6("u6_cn",  fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=colors.HexColor("#111111"))
        _c6_meta_s = ps6("u6_cm",  fontName="Helvetica",         fontSize=8.5, leading=11, textColor=colors.HexColor("#9d174d"))
        _c6_desc_s = ps6("u6_cd",  fontName="Helvetica-Oblique", fontSize=8,   leading=11, textColor=colors.HexColor("#444444"))
        _c6_num_s  = ps6("u6_cnum",fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=CORAL, alignment=1)
        _c6_hdr_s  = ps6("u6_chdr",fontName="Helvetica-Bold",    fontSize=8,   leading=10, textColor=colors.HexColor("#ffffff"), alignment=1)
        _CC6_1 = rc_inner_w * 0.05
        _CC6_2 = rc_inner_w * 0.30
        _CC6_3 = rc_inner_w * 0.22
        _CC6_4 = rc_inner_w * 0.43
        _c6_hdr_row = [
            Paragraph("#",               _c6_hdr_s),
            Paragraph("Certificate",     _c6_hdr_s),
            Paragraph("Issuer",          _c6_hdr_s),
            Paragraph("Credential Link", _c6_hdr_s),
        ]
        _c6_data   = [_c6_hdr_row]
        _c6_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#881337")),
            ("TOPPADDING",    (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#fda4af")),
            ("LINEBELOW",     (0, 0), (-1, 0),  1.0, colors.HexColor("#f43f5e")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#fff1f2"), colors.white]),
        ]
        for _i6, _cert6 in enumerate(_certs6):
            _cn6  = (_cert6.get("name")        or "").strip()
            _cl6  = (_cert6.get("link")        or "").strip()
            _cis6 = (_cert6.get("issuer")      or "").strip()
            _cde6 = (_cert6.get("description") or "").strip()
            if not any([_cn6, _cl6, _cis6, _cde6]):
                continue
            _safe6 = _cl6.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _name_cell6 = [Paragraph(f"<b>{esc(_cn6)}</b>" if _cn6 else "—", _c6_name_s)]
            if _cde6:
                _name_cell6.append(Paragraph(esc(_cde6), _c6_desc_s))
            _c6_data.append([
                Paragraph(str(_i6 + 1), _c6_num_s),
                _name_cell6,
                Paragraph(esc(_cis6) if _cis6 else "—", _c6_meta_s),
                Paragraph(f'<a href="{_safe6}" color="#f43f5e">{_safe6}</a>' if _cl6 else "—", _c6_meta_s),
            ])
            _c6_styles.append(("TOPPADDING",    (0, _i6+1), (-1, _i6+1), 5))
            _c6_styles.append(("BOTTOMPADDING", (0, _i6+1), (-1, _i6+1), 5))
        _c6_tbl = Table(_c6_data, colWidths=[_CC6_1, _CC6_2, _CC6_3, _CC6_4], repeatRows=1)
        _c6_tbl.setStyle(TableStyle(_c6_styles))
        right_items.append(_c6_tbl)
        right_items.append(Spacer(1, 6))

    # ── Render via Canvas + Frames ────────────────────────────────────────────
    buf2 = io.BytesIO()
    c2   = _rl_canvas.Canvas(buf2, pagesize=(PAGE_W, PAGE_H_SINGLE))

    # Full-bleed indigo header band
    c2.setFillColor(INDIGO)
    c2.rect(0, PAGE_H_SINGLE - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)

    # Coral accent strip at header bottom
    c2.setFillColor(CORAL)
    c2.rect(0, PAGE_H_SINGLE - HEADER_H - 3, PAGE_W, 3, fill=1, stroke=0)

    # Left column background (off-white / very pale)
    c2.setFillColor(OFF_WHITE)
    c2.rect(0, 0, LEFT_W, PAGE_H_SINGLE - HEADER_H - 3, fill=1, stroke=0)

    # Right column background (pure white)
    c2.setFillColor(BODY_BG)
    c2.rect(LEFT_W, 0, RIGHT_W, PAGE_H_SINGLE - HEADER_H - 3, fill=1, stroke=0)

    # Subtle left-right divider
    c2.setFillColor(colors.HexColor("#e5e7eb"))
    c2.rect(LEFT_W, 0, 1, PAGE_H_SINGLE - HEADER_H - 3, fill=1, stroke=0)

    # ── Header Frame (name + title + contact) ─────────────────────────────────
    H_PAD = 18
    hd_items = []
    hd_items.append(Paragraph(esc(p_name), S["hd_name"]))
    title_str = (cv.get("title") or "").strip()
    if title_str:
        hd_items.append(Paragraph(esc(title_str), S["hd_title"]))

    if p_links:
        contact_parts = []
        for lnk in p_links:
            v    = (lnk.get("value") or "").strip()
            lbl  = (lnk.get("label") or "").strip().lower()
            href = "" if lbl == "location" else _contact_href(v)
            _sv  = esc(v)
            if href:
                contact_parts.append(f'<a href="{esc(href)}" color="#e0e7ff">{_sv}</a>')
            else:
                contact_parts.append(_sv)
        if contact_parts:
            hd_items.append(Spacer(1, 4))
            hd_items.append(Paragraph(
                '  <font color="#6d6a9c">|</font>  '.join(contact_parts),
                S["hd_contact"]))

    hd_frame = _Frame(
        H_PAD,
        PAGE_H_SINGLE - HEADER_H + 4,
        PAGE_W - 2*H_PAD,
        HEADER_H - 8,
        leftPadding=0, rightPadding=0, topPadding=4, bottomPadding=0,
        showBoundary=0,
    )
    hd_frame.addFromList(list(hd_items), c2)

    # ── Left body Frame ────────────────────────────────────────────────────────
    body_h = PAGE_H_SINGLE - HEADER_H - 6
    lc_frame = _Frame(
        L_PAD, 0,
        LEFT_W - 2*L_PAD, body_h - L_PAD,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0,
    )
    lc_frame.addFromList(list(left_items), c2)

    # ── Right body Frame ───────────────────────────────────────────────────────
    rc_frame = _Frame(
        LEFT_W + R_PAD, 0,
        RIGHT_W - 2*R_PAD, body_h - R_PAD,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        showBoundary=0,
    )
    rc_frame.addFromList(list(right_items), c2)

    c2.save()
    buf2.seek(0)

    # Crop to content
    lowest_y = min(
        lc_frame._y if hasattr(lc_frame, '_y') else 0,
        rc_frame._y if hasattr(rc_frame, '_y') else 0,
    )
    tight_h = PAGE_H_SINGLE - lowest_y + 10 * mm
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
