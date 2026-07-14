"""
CV Builder AI — UI4 PDF Builder
Executive Dark layout: dark charcoal sidebar (left), warm cream main panel (right).
Premium feel with gold accents, refined typography, and subtle texture via rules.
"""

import io
import math
import re as _re4
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

from UI._shared import _normalise_edu_entry, _infer_degree_duration, _contact_href


def build_cv_pdf_ui4(cv: dict, profile_data: dict = None) -> bytes:
    """UI4 — Executive Dark: charcoal sidebar, cream main panel, gold accents."""
    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.platypus.frames import Frame as _Frame

    _pd     = profile_data or {}
    p_name  = (_pd.get("name") or "").strip() or "CANDIDATE"
    p_links = _pd.get("links") or []
    p_work  = _pd.get("work")  or []
    p_edu   = [_normalise_edu_entry(e) for e in (_pd.get("edu") or [])]

    # ── Palette ──────────────────────────────────────────────────────────────
    CHARCOAL  = colors.HexColor("#1e1e24")   # deep dark sidebar bg
    CHARCOAL2 = colors.HexColor("#2a2a32")   # slightly lighter for dividers
    GOLD      = colors.HexColor("#c9a84c")   # warm gold accent
    GOLD_PALE = colors.HexColor("#f5e6c0")   # pale gold for subtle accents
    CREAM     = colors.HexColor("#faf8f3")   # warm cream main bg
    DARK_TXT  = colors.HexColor("#1a1a22")   # near-black for headings
    MID_TXT   = colors.HexColor("#4a4a58")   # body text
    PALE_TXT  = colors.HexColor("#888898")   # metadata / dates
    WHITE     = colors.white

    PAGE_W, _ = A4
    ML, MR, MT, MB = 0, 0, 0, 0
    PAGE_H_SINGLE = 841.89 * 4.5
    SIDEBAR_W = 62 * mm
    MAIN_W    = PAGE_W - SIDEBAR_W
    SB_PAD    = 15   # sidebar inner horizontal padding
    MN_PAD    = 18   # main panel inner horizontal padding

    def ps4(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0,
                 spaceBefore=0, textColor=DARK_TXT)
        d.update(kw)
        return ParagraphStyle(name, **d)

    S = {
        # ── Sidebar ───────────────────────────────────────────────────────────
        "sb_name":   ps4("s4_nm", fontName="Helvetica-Bold", fontSize=19, leading=24,
                         textColor=WHITE, spaceBefore=0),
        "sb_title":  ps4("s4_tt", fontName="Helvetica", fontSize=8.5, leading=12,
                         textColor=GOLD, spaceBefore=5, letterSpacing=0.5),
        "sb_sec":    ps4("s4_sc", fontName="Helvetica-Bold", fontSize=7, leading=10,
                         textColor=GOLD, spaceBefore=16, spaceAfter=5,
                         letterSpacing=1.2),
        "sb_lbl":    ps4("s4_lb", fontName="Helvetica-Bold", fontSize=7, leading=9,
                         textColor=colors.HexColor("#777788")),
        "sb_val":    ps4("s4_vl", fontName="Helvetica", fontSize=8.5, leading=13,
                         textColor=colors.HexColor("#d8d8e8")),
        "sb_skill":  ps4("s4_sk", fontName="Helvetica", fontSize=8, leading=12,
                         textColor=colors.HexColor("#c8c8d8")),
        "sb_edu_uni":ps4("s4_eu", fontName="Helvetica-Bold", fontSize=9, leading=12,
                         textColor=WHITE, spaceBefore=8),
        "sb_edu_deg":ps4("s4_ed", fontName="Helvetica", fontSize=8, leading=11,
                         textColor=colors.HexColor("#aaaacc")),
        "sb_edu_yr": ps4("s4_ey", fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
                         textColor=GOLD),
        # ── Main panel ────────────────────────────────────────────────────────
        "mn_sec":    ps4("s4_ms", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                         textColor=GOLD, spaceBefore=16, spaceAfter=4,
                         letterSpacing=1.5),
        "mn_company":ps4("s4_mc", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                         textColor=DARK_TXT, spaceBefore=3),
        "mn_role":   ps4("s4_mr", fontName="Helvetica-Oblique", fontSize=9.5, leading=13,
                         textColor=colors.HexColor("#5a5a78"), spaceAfter=3),
        "mn_date":   ps4("s4_md", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         textColor=PALE_TXT, alignment=TA_RIGHT),
        "mn_bullet": ps4("s4_mb", fontName="Helvetica", fontSize=9.5, leading=14,
                         leftIndent=10, textColor=MID_TXT, spaceAfter=2),
        "mn_tech":   ps4("s4_mt", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                         leftIndent=10, textColor=GOLD),
        "mn_summary":ps4("s4_msum", fontName="Helvetica", fontSize=9.5, leading=15,
                         textColor=MID_TXT),
        "mn_proj_nm":ps4("s4_mpn", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                         textColor=DARK_TXT, spaceBefore=4),
        "mn_proj_bd":ps4("s4_mpb", fontName="Helvetica", fontSize=9, leading=13,
                         textColor=MID_TXT),
        "mn_proj_bl":ps4("s4_mpl", fontName="Helvetica", fontSize=9, leading=12.5,
                         leftIndent=10, textColor=MID_TXT, spaceAfter=2),
        "mn_proj_tk":ps4("s4_mpt", fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                         textColor=GOLD),
        "mn_comp":   ps4("s4_mcp", fontName="Helvetica", fontSize=9.5, leading=14,
                         textColor=MID_TXT),
    }

    def esc(s):
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    # ── Sidebar items ─────────────────────────────────────────────────────────
    sidebar_items = []

    # Name block with gold bar above it
    sidebar_items.append(Spacer(1, 4))
    # Gold accent bar (simulated via Table with gold bg)
    gold_bar = Table([[""]], colWidths=[SIDEBAR_W - 2*SB_PAD],
                     style=TableStyle([
                         ("BACKGROUND",    (0,0),(-1,-1), GOLD),
                         ("TOPPADDING",    (0,0),(-1,-1), 2),
                         ("BOTTOMPADDING", (0,0),(-1,-1), 2),
                     ]))
    sidebar_items.append(gold_bar)
    sidebar_items.append(Spacer(1, 8))

    # Split name: first name lighter, last name bolder (done via two paragraphs)
    name_parts = p_name.strip().split()
    first_part = " ".join(name_parts[:-1]) if len(name_parts) > 1 else p_name
    last_part  = name_parts[-1] if len(name_parts) > 1 else ""
    if first_part:
        sidebar_items.append(Paragraph(esc(first_part.upper()),
                             ps4("s4_fn", fontName="Helvetica", fontSize=14, leading=18,
                                 textColor=colors.HexColor("#aaaacc"), letterSpacing=1)))
    sidebar_items.append(Paragraph(esc(last_part.upper() if last_part else p_name.upper()),
                         ps4("s4_ln", fontName="Helvetica-Bold", fontSize=19, leading=24,
                             textColor=WHITE, letterSpacing=0.5)))
    sidebar_items.append(Paragraph(esc(cv.get("title","") or ""),
                         ps4("s4_jt", fontName="Helvetica", fontSize=8.5, leading=12,
                             textColor=GOLD, spaceBefore=4)))
    sidebar_items.append(Spacer(1, 12))

    # ── Contact ───────────────────────────────────────────────────────────────
    divider_style = TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), CHARCOAL2),
        ("TOPPADDING",    (0,0),(-1,-1), 0), ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ])
    def sb_divider():
        return Table([[""]], colWidths=[SIDEBAR_W - 2*SB_PAD],
                     style=TableStyle([
                         ("LINEABOVE", (0,0),(-1,-1), 0.5, colors.HexColor("#3a3a48")),
                         ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
                     ]))

    sidebar_items.append(Paragraph("CONTACT", S["sb_sec"]))
    sidebar_items.append(sb_divider())
    for lnk in p_links:
        _cv = (lnk.get("value") or "").strip()
        _lbl = (lnk.get("label") or "").strip()
        _href = _contact_href(_cv)
        _safe = esc(_cv)
        sidebar_items.append(Paragraph(esc(_lbl), S["sb_lbl"]))
        if _href and _lbl.lower() != "location":
            sidebar_items.append(Paragraph(
                f'<a href="{_href}" color="#c8c8d8">{_safe}</a>', S["sb_val"]))
        else:
            sidebar_items.append(Paragraph(_safe, S["sb_val"]))
        sidebar_items.append(Spacer(1, 5))

    # ── Competencies in sidebar ───────────────────────────────────────────────
    comp_str = (cv.get("competencies") or "").strip()
    if comp_str:
        sidebar_items.append(Paragraph("COMPETENCIES", S["sb_sec"]))
        sidebar_items.append(sb_divider())
        pills = [c.strip() for c in comp_str.replace(" * ","*").split("*") if c.strip()]
        for pill in pills:
            sidebar_items.append(Paragraph(f"▸  {esc(pill)}", S["sb_skill"]))
            sidebar_items.append(Spacer(1, 3))

    # ── Education in sidebar ──────────────────────────────────────────────────
    _cv_edu = cv.get("education") or []
    if isinstance(_cv_edu, dict): _cv_edu = [_cv_edu]
    if p_edu:
        _edu_list = []
        for _i, _pe in enumerate(p_edu):
            _e = dict(_pe)
            _ef = str(_pe.get("from") or "").strip()
            _et = str(_pe.get("to")   or "").strip()
            if not _ef and not _et and _i < len(_cv_edu):
                _yr = str(_cv_edu[_i].get("years") or "").strip()
                if _yr: _e["years"] = _yr
            _edu_list.append(_e)
    else:
        _edu_list = []
        for _ce in _cv_edu:
            _yr_r = str(_ce.get("years","") or "").strip()
            _sep  = "–" if "–" in _yr_r else "-"
            _pts  = [p.strip() for p in _yr_r.split(_sep,1)] if _sep in _yr_r else ["",""]
            _edu_list.append({
                "institution": (_ce.get("university") or _ce.get("institution") or "").strip(),
                "degree": (_ce.get("degree") or "").strip(),
                "cgpa":   (_ce.get("cgpa")   or "").strip(),
                "from": _pts[0], "to": _pts[-1],
                "note": (_ce.get("achievement") or "").strip(),
            })

    sidebar_items.append(Paragraph("EDUCATION", S["sb_sec"]))
    sidebar_items.append(sb_divider())
    for e in _edu_list:
        ef = str(e.get("from") or "").strip()
        et = str(e.get("to")   or "").strip()
        if not ef and not et:
            _yr_fb = str(e.get("years") or "").strip()
            _sp = "–" if "–" in _yr_fb else "-"
            if _yr_fb and _sp in _yr_fb:
                _pp = [p.strip() for p in _yr_fb.split(_sp,1)]
                ef, et = _pp[0], _pp[-1]
        dr = f"{ef}–{et}" if ef and et else (f"{ef}–Present" if ef else et)
        sidebar_items.append(Paragraph(esc(e.get("institution") or ""), S["sb_edu_uni"]))
        sidebar_items.append(Paragraph(esc(e.get("degree") or ""), S["sb_edu_deg"]))
        _note = (e.get("note") or e.get("cgpa") or "").strip()
        if _note: sidebar_items.append(Paragraph(esc(_note), S["sb_edu_yr"]))
        if dr:    sidebar_items.append(Paragraph(dr, S["sb_edu_yr"]))
        sidebar_items.append(Spacer(1, 5))

    # ── Main panel items ──────────────────────────────────────────────────────
    main_items = [Spacer(1, 6)]
    mn_inner_w = MAIN_W - 2 * MN_PAD

    def mn_section(title):
        main_items.append(Paragraph(title.upper(), S["mn_sec"]))
        main_items.append(HRFlowable(width="100%", thickness=0.75, color=GOLD,
                                     spaceBefore=2, spaceAfter=6))

    # Summary
    mn_section("Professional Summary")
    main_items.append(Paragraph(esc(cv.get("summary") or ""), S["mn_summary"]))
    main_items.append(Spacer(1, 6))

    # Experience
    ai_cos = cv.get("companies") or []
    num_entries = len(ai_cos) if ai_cos else max(len(ai_cos), len(p_work))
    if num_entries > 0:
        mn_section("Work Experience")
        for i in range(num_entries):
            w  = p_work[i]  if i < len(p_work)  else {}
            ai = ai_cos[i]  if i < len(ai_cos)  else {}
            company = (w.get("company") or "").strip() or ai.get("company","")
            role    = (w.get("role")    or "").strip() or ai.get("role","")
            wf      = str(w.get("from") or "").strip()
            wt      = str(w.get("to")   or "").strip()
            if wf and wt:  dr = f"{wf} – {wt}"
            elif wf:       dr = f"{wf} – Present"
            else:          dr = ai.get("dateRange","")

            main_items.append(Table(
                [[Paragraph(esc(company), S["mn_company"]),
                  Paragraph(esc(dr), S["mn_date"])]],
                colWidths=[mn_inner_w - 130, 130],
                style=TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
                    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),2),
                    ("ALIGN",(1,0),(1,-1),"RIGHT"),
                ])
            ))
            if role:
                main_items.append(Paragraph(esc(role), S["mn_role"]))
            bullets = ai.get("bullets") or []
            if not bullets and w.get("bullets"):
                bullets = [b.strip() for b in str(w["bullets"]).split("\n") if b.strip()]
            for b in bullets:
                b_clean = b.lstrip("•·▸–▪● ").strip()
                main_items.append(Paragraph(
                    '<font color="#c9a84c" size="9">◆</font> ' + esc(b_clean),
                    S["mn_bullet"]))
            tech_raw = ai.get("tech","")
            if tech_raw:
                sep  = "|" if "|" in tech_raw else ","
                tags = "  ·  ".join(t.strip() for t in tech_raw.split(sep) if t.strip())
                main_items.append(Paragraph(esc(tags), S["mn_tech"]))
            main_items.append(Spacer(1, 8))

    # Skills
    skills = cv.get("skills") or []
    if skills:
        mn_section("Technical Skills")
        for sk in skills[:5]:
            colon = sk.find(":")
            if colon > 0:
                cat = sk[:colon].strip()
                val = sk[colon+1:].strip()
                main_items.append(Paragraph(
                    f'<b><font color="#1a1a22">{esc(cat)}: </font></b>'
                    f'<font color="#4a4a58">{esc(val)}</font>',
                    ps4("s4_sk_row", fontName="Helvetica", fontSize=9.5, leading=14,
                        textColor=MID_TXT)))
            else:
                main_items.append(Paragraph(esc(sk), ps4("s4_sk_plain", fontName="Helvetica",
                                    fontSize=9.5, leading=14, textColor=MID_TXT)))
            main_items.append(HRFlowable(width="100%", thickness=0.3,
                                         color=colors.HexColor("#e0ddd5"),
                                         spaceBefore=3, spaceAfter=3))

    # Projects
    projects = cv.get("projects") or []
    if projects:
        mn_section("Key Projects")
        for p in projects:
            raw_name = (p.get("name") or "").strip()
            name = _re4.sub(r'\s*\[[^\]]*\]\s*$', '', raw_name)
            name = _re4.sub(r'^[A-Z][A-Z\s&\-]{2,}:\s*', '', name).strip()
            if name:
                main_items.append(Paragraph(esc(name), S["mn_proj_nm"]))
            if p.get("overview"):
                main_items.append(Paragraph(esc(p["overview"]), S["mn_proj_bd"]))
            for b in (p.get("bullets") or []):
                b_clean = b.lstrip("•·▸– ").strip()
                main_items.append(Paragraph(
                    '<font color="#c9a84c" size="9">◆</font> ' + esc(b_clean),
                    S["mn_proj_bl"]))
            tech_t = p.get("techTags") or []
            if not tech_t and p.get("tech"):
                sep = "|" if "|" in p["tech"] else ","
                tech_t = [t.strip() for t in p["tech"].split(sep) if t.strip()]
            if tech_t:
                main_items.append(Paragraph(
                    "  ·  ".join(esc(t) for t in tech_t), S["mn_proj_tk"]))
            main_items.append(Spacer(1, 6))

    # Certifications (optional) — tabular card design
    _certs4 = cv.get("certifications") or []
    if _certs4:
        mn_section("Certifications")
        _cert_note4 = (cv.get("cert_note") or "").strip()
        if _cert_note4:
            _cnote4_s = ps4("u4_cnote", fontName="Helvetica-Oblique", fontSize=9, leading=13,
                             textColor=colors.HexColor("#444444"), spaceAfter=4)
            main_items.append(Paragraph(esc(_cert_note4), _cnote4_s))
        _c4_name_s = ps4("u4_cn",  fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=colors.HexColor("#111111"))
        _c4_meta_s = ps4("u4_cm",  fontName="Helvetica",         fontSize=8.5, leading=11, textColor=colors.HexColor("#7a5c00"))
        _c4_desc_s = ps4("u4_cd",  fontName="Helvetica-Oblique", fontSize=8,   leading=11, textColor=colors.HexColor("#444444"))
        _c4_num_s  = ps4("u4_cnum",fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=GOLD, alignment=1)
        _c4_hdr_s  = ps4("u4_chdr",fontName="Helvetica-Bold",    fontSize=8,   leading=10, textColor=colors.HexColor("#ffffff"), alignment=1)
        _mn_cw4 = mn_inner_w
        _CC4_1 = _mn_cw4 * 0.05
        _CC4_2 = _mn_cw4 * 0.30
        _CC4_3 = _mn_cw4 * 0.22
        _CC4_4 = _mn_cw4 * 0.43
        _c4_hdr_row = [
            Paragraph("#",               _c4_hdr_s),
            Paragraph("Certificate",     _c4_hdr_s),
            Paragraph("Issuer",          _c4_hdr_s),
            Paragraph("Credential Link", _c4_hdr_s),
        ]
        _c4_data   = [_c4_hdr_row]
        _c4_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#5a4000")),
            ("TOPPADDING",    (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#d4b870")),
            ("LINEBELOW",     (0, 0), (-1, 0),  1.0, colors.HexColor("#c9a84c")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#fdf8ed"), colors.white]),
        ]
        for _i4, _cert4 in enumerate(_certs4):
            _cn4  = (_cert4.get("name")        or "").strip()
            _cl4  = (_cert4.get("link")        or "").strip()
            _cis4 = (_cert4.get("issuer")      or "").strip()
            _cde4 = (_cert4.get("description") or "").strip()
            if not any([_cn4, _cl4, _cis4, _cde4]):
                continue
            _safe4 = _cl4.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _name_cell4 = [Paragraph(f"<b>{esc(_cn4)}</b>" if _cn4 else "—", _c4_name_s)]
            if _cde4:
                _name_cell4.append(Paragraph(esc(_cde4), _c4_desc_s))
            _c4_data.append([
                Paragraph(str(_i4 + 1), _c4_num_s),
                _name_cell4,
                Paragraph(esc(_cis4) if _cis4 else "—", _c4_meta_s),
                Paragraph(f'<a href="{_safe4}" color="#c9a84c">{_safe4}</a>' if _cl4 else "—", _c4_meta_s),
            ])
            _c4_styles.append(("TOPPADDING",    (0, _i4+1), (-1, _i4+1), 5))
            _c4_styles.append(("BOTTOMPADDING", (0, _i4+1), (-1, _i4+1), 5))
        _c4_tbl = Table(_c4_data, colWidths=[_CC4_1, _CC4_2, _CC4_3, _CC4_4], repeatRows=1)
        _c4_tbl.setStyle(TableStyle(_c4_styles))
        main_items.append(_c4_tbl)
        main_items.append(Spacer(1, 6))

    # ── Render via raw Canvas + Frames ────────────────────────────────────────
    buf2 = io.BytesIO()
    c2   = _rl_canvas.Canvas(buf2, pagesize=(PAGE_W, PAGE_H_SINGLE))

    # Backgrounds — mirrored from UI2's sidebar-on-left skeleton: the dark
    # charcoal panel sits on the RIGHT here so the two sidebar templates read
    # as opposite silhouettes at a glance, not the same layout re-skinned.
    c2.setFillColor(CREAM)
    c2.rect(0, 0, MAIN_W, PAGE_H_SINGLE, fill=1, stroke=0)
    c2.setFillColor(CHARCOAL)
    c2.rect(MAIN_W, 0, SIDEBAR_W, PAGE_H_SINGLE, fill=1, stroke=0)
    # Thin gold strip between columns
    c2.setFillColor(GOLD)
    c2.rect(MAIN_W - 2, 0, 2, PAGE_H_SINGLE, fill=1, stroke=0)

    mn_inner_h = PAGE_H_SINGLE - 8 - MN_PAD
    mn_frame = _Frame(
        MN_PAD, MN_PAD,
        MAIN_W - 2*MN_PAD - 2, mn_inner_h,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0,
    )
    mn_frame.addFromList(list(main_items), c2)

    sb_inner_w = SIDEBAR_W - 2 * SB_PAD
    sb_frame = _Frame(
        MAIN_W + SB_PAD, SB_PAD, sb_inner_w, PAGE_H_SINGLE - 2*SB_PAD,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0,
    )
    sb_frame.addFromList(list(sidebar_items), c2)

    c2.save()
    buf2.seek(0)

    # Crop
    lowest_y = min(
        sb_frame._y if hasattr(sb_frame, '_y') else 0,
        mn_frame._y if hasattr(mn_frame, '_y') else 0,
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
