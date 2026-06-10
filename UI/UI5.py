"""
CV Builder AI — UI5 PDF Builder
Clean Minimal layout: pure white canvas, ultra-thin rules, emerald green accents.
Generous whitespace, no sidebar — everything breathes. Very modern editorial feel.
"""

import io
import math
import re as _re5
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

from UI._shared import _normalise_edu_entry, _infer_degree_duration, _contact_href


def build_cv_pdf_ui5(cv: dict, profile_data: dict = None) -> bytes:
    """UI5 — Clean Minimal: white canvas, emerald accents, editorial spacing."""

    _pd     = profile_data or {}
    p_name  = (_pd.get("name") or "").strip() or "CANDIDATE"
    p_links = _pd.get("links") or []
    p_work  = _pd.get("work")  or []
    p_edu   = [_normalise_edu_entry(e) for e in (_pd.get("edu") or [])]

    # ── Palette ──────────────────────────────────────────────────────────────
    EMERALD   = colors.HexColor("#059669")   # vibrant emerald green
    EMERALD_D = colors.HexColor("#065f46")   # dark emerald for headings
    EMERALD_L = colors.HexColor("#d1fae5")   # pale emerald for bg chips
    INK       = colors.HexColor("#0f172a")   # near-black heading ink
    BODY      = colors.HexColor("#374151")   # comfortable body text
    MUTED     = colors.HexColor("#9ca3af")   # muted labels / dates
    RULE      = colors.HexColor("#e5e7eb")   # ultra-light rule lines
    WHITE     = colors.white

    buf  = io.BytesIO()
    PAGE_W, _ = A4
    ML, MR, MT, MB = 16*mm, 16*mm, 14*mm, 14*mm
    PAGE_H_SINGLE = 841.89 * 4.5

    doc = SimpleDocTemplate(
        buf, pagesize=(PAGE_W, PAGE_H_SINGLE),
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title=f"{p_name} CV", author=p_name,
    )
    TW = PAGE_W - ML - MR

    def ps5(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0,
                 spaceBefore=0, textColor=BODY)
        d.update(kw)
        return ParagraphStyle(name, **d)

    S = {
        "name":      ps5("u5_nm", fontName="Helvetica-Bold", fontSize=26, leading=32,
                         textColor=INK, alignment=TA_CENTER, spaceBefore=0),
        "subtitle":  ps5("u5_st", fontName="Helvetica", fontSize=10, leading=14,
                         textColor=EMERALD, spaceAfter=4, alignment=TA_CENTER),
        "contact":   ps5("u5_ct", fontName="Helvetica", fontSize=8.5, leading=12,
                         textColor=colors.HexColor("#0057a8"), alignment=TA_CENTER),
        # Section title: left-aligned, emerald color, small uppercase label
        "sec_label": ps5("u5_sl", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         textColor=EMERALD, spaceBefore=18, spaceAfter=2,
                         letterSpacing=1.8),
        "company":   ps5("u5_co", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                         textColor=INK, spaceBefore=4),
        "role":      ps5("u5_rl", fontName="Helvetica", fontSize=9.5, leading=13,
                         textColor=EMERALD_D, spaceAfter=3),
        "date":      ps5("u5_dt", fontName="Helvetica", fontSize=8.5, leading=12,
                         textColor=MUTED, alignment=TA_RIGHT),
        "bullet":    ps5("u5_bul", fontName="Helvetica", fontSize=9.5, leading=14,
                         leftIndent=12, textColor=BODY, spaceAfter=2),
        "tech":      ps5("u5_tch", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         leftIndent=12, textColor=EMERALD),
        "summary":   ps5("u5_sum", fontName="Helvetica", fontSize=9.5, leading=16,
                         textColor=BODY),
        "sk_cat":    ps5("u5_skc", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
                         textColor=INK),
        "sk_val":    ps5("u5_skv", fontName="Helvetica", fontSize=9.5, leading=13,
                         textColor=BODY),
        "proj_nm":   ps5("u5_pn", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                         textColor=INK, spaceBefore=4),
        "proj_bd":   ps5("u5_pb", fontName="Helvetica", fontSize=9, leading=13,
                         textColor=BODY),
        "proj_bl":   ps5("u5_pbl", fontName="Helvetica", fontSize=9, leading=12.5,
                         leftIndent=10, textColor=BODY, spaceAfter=2),
        "proj_tk":   ps5("u5_pt", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         textColor=EMERALD),
        "comp":      ps5("u5_cmp", fontName="Helvetica", fontSize=9.5, leading=14,
                         textColor=BODY),
        "edu_uni":   ps5("u5_eu", fontName="Helvetica-Bold", fontSize=11, leading=14,
                         textColor=INK, spaceBefore=4),
        "edu_deg":   ps5("u5_ed", fontName="Helvetica", fontSize=9.5, leading=13,
                         textColor=BODY),
        "edu_date":  ps5("u5_edt", fontName="Helvetica", fontSize=8.5, leading=12,
                         textColor=MUTED, alignment=TA_RIGHT),
        "edu_ach":   ps5("u5_ea", fontName="Helvetica-Oblique", fontSize=9, leading=12,
                         textColor=EMERALD),
    }

    def esc(s):
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def section(title):
        story.append(Paragraph(title.upper(), S["sec_label"]))
        story.append(HRFlowable(width="100%", thickness=1.2, color=EMERALD,
                                spaceBefore=2, spaceAfter=8))

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    # Left-aligned name + title; contact strip on same row right-aligned
    story.append(Paragraph(esc(p_name), S["name"]))
    title_str = (cv.get("title") or "").strip()
    if title_str:
        story.append(Paragraph(esc(title_str), S["subtitle"]))

    if p_links:
        contact_parts = []
        for lnk in p_links:
            v    = (lnk.get("value") or "").strip()
            lbl  = (lnk.get("label") or "").strip().lower()
            href = "" if lbl == "location" else _contact_href(v)
            _sv  = esc(v)
            if href:
                contact_parts.append(f'<a href="{esc(href)}" color="#0057a8">{_sv}</a>')
            else:
                contact_parts.append(_sv)
        if contact_parts:
            story.append(Paragraph(
                '  <font color="#9ca3af">|</font>  '.join(contact_parts),
                S["contact"]))

    story.append(HRFlowable(width="100%", thickness=2, color=EMERALD,
                             spaceBefore=8, spaceAfter=4))

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_text = (cv.get("summary") or "").strip()
    if summary_text:
        section("Professional Summary")
        story.append(Paragraph(esc(summary_text), S["summary"]))

    # ── Experience ────────────────────────────────────────────────────────────
    ai_cos = cv.get("companies") or []
    num_entries = len(ai_cos) if ai_cos else max(len(ai_cos), len(p_work))
    if num_entries > 0:
        section("Experience")
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

            story.append(Table(
                [[Paragraph(esc(company), S["company"]),
                  Paragraph(esc(dr), S["date"])]],
                colWidths=[TW * 0.68, TW * 0.32],
                style=TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
                    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),2),
                    ("ALIGN",(1,0),(1,-1),"RIGHT"),
                ])
            ))
            if role:
                story.append(Paragraph(esc(role), S["role"]))

            bullets = ai.get("bullets") or []
            if not bullets and w.get("bullets"):
                bullets = [b.strip() for b in str(w["bullets"]).split("\n") if b.strip()]
            for b in bullets:
                b_clean = b.lstrip("•·▸–▪● ").strip()
                story.append(Paragraph(
                    f'<font color="#059669" size="9">—</font> {esc(b_clean)}',
                    S["bullet"]))
            tech_raw = ai.get("tech","")
            if tech_raw:
                sep  = "|" if "|" in tech_raw else ","
                tags = "  ·  ".join(t.strip() for t in tech_raw.split(sep) if t.strip())
                story.append(Paragraph(esc(tags), S["tech"]))
            story.append(HRFlowable(width="100%", thickness=0.4, color=RULE,
                                    spaceBefore=6, spaceAfter=2))

    # ── Skills — two-column ───────────────────────────────────────────────────
    skills = cv.get("skills") or []
    if skills:
        section("Skills")
        col_data = []
        for sk in skills[:6]:
            colon = sk.find(":")
            cat = sk[:colon].strip() if colon > 0 else ""
            val = sk[colon+1:].strip() if colon > 0 else sk
            col_data.append((cat, val))

        half = math.ceil(len(col_data) / 2)
        left_col  = col_data[:half]
        right_col = col_data[half:]
        while len(right_col) < len(left_col):
            right_col.append(("",""))

        col_w = (TW - 12) / 2

        def _sk_cell(cat, val):
            items = []
            if cat:
                items.append(Paragraph(f"<b>{esc(cat)}</b>", S["sk_cat"]))
            if val:
                items.append(Paragraph(esc(val), S["sk_val"]))
            return items

        for (lc, lv), (rc, rv) in zip(left_col, right_col):
            row = Table(
                [[_sk_cell(lc, lv), _sk_cell(rc, rv)]],
                colWidths=[col_w, col_w],
                style=TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("LEFTPADDING",(0,0),(-1,-1),0),
                    ("RIGHTPADDING",(0,0),(-1,-1),8),
                    ("TOPPADDING",(0,0),(-1,-1),3),
                    ("BOTTOMPADDING",(0,0),(-1,-1),5),
                    ("LINEBELOW",(0,0),(-1,-1),0.4,RULE),
                ])
            )
            story.append(row)

    # ── Projects ──────────────────────────────────────────────────────────────
    projects = cv.get("projects") or []
    if projects:
        section("Projects")
        for p in projects:
            raw_name = (p.get("name") or "").strip()
            name = _re5.sub(r'\s*\[[^\]]*\]\s*$', '', raw_name)
            name = _re5.sub(r'^[A-Z][A-Z\s&\-]{2,}:\s*', '', name).strip()
            # Emerald left-border chip via a table
            nm_chip = Table(
                [[Paragraph(esc(name), S["proj_nm"])]],
                colWidths=[TW - 6],
                style=TableStyle([
                    ("LINEBEFORE", (0,0),(0,-1), 3, EMERALD),
                    ("LEFTPADDING",(0,0),(-1,-1),8),
                    ("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),2),
                    ("BOTTOMPADDING",(0,0),(-1,-1),2),
                ])
            )
            story.append(nm_chip)
            if p.get("overview"):
                story.append(Paragraph(esc(p["overview"]), S["proj_bd"]))
            for b in (p.get("bullets") or []):
                b_clean = b.lstrip("•·▸– ").strip()
                story.append(Paragraph(
                    f'<font color="#059669" size="9">—</font> {esc(b_clean)}',
                    S["proj_bl"]))
            tech_t = p.get("techTags") or []
            if not tech_t and p.get("tech"):
                sep = "|" if "|" in p["tech"] else ","
                tech_t = [t.strip() for t in p["tech"].split(sep) if t.strip()]
            if tech_t:
                story.append(Paragraph(
                    "  ·  ".join(esc(t) for t in tech_t), S["proj_tk"]))
            story.append(Spacer(1, 8))

    # ── Competencies ─────────────────────────────────────────────────────────
    comp_str = (cv.get("competencies") or "").strip()
    if comp_str:
        section("Core Competencies")
        pills = [c.strip() for c in comp_str.replace("*","•").split("•") if c.strip()]
        story.append(Paragraph(
            "  ·  ".join(esc(c) for c in pills), S["comp"]))
        story.append(Spacer(1, 4))

    # ── Education ─────────────────────────────────────────────────────────────
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
        section("Education")
        _u5_prev_yr = None
        for e in _edu_rl:
            ef  = str(e.get("from") or "").strip()
            et  = str(e.get("to")   or "").strip()
            deg = (e.get("degree") or "").strip()
            uni = (e.get("institution") or "").strip()
            cgpa = (e.get("cgpa") or "").strip()
            ach  = (e.get("achievement") or "").strip()

            if ef and et:
                dr = f"{ef} – {et}"
            else:
                _yr_r = str(e.get("years") or "").strip()
                _sep2 = "–" if "–" in _yr_r else "-"
                if _yr_r and _sep2 in _yr_r:
                    _yp = [p.strip() for p in _yr_r.split(_sep2,1)]
                    ef, et = _yp[0], _yp[-1]
                    dr = f"{ef} – {et}"
                elif et and not ef:
                    _dur = _infer_degree_duration(deg)
                    try: ef = str(int(et[:4]) - _dur)
                    except: pass
                    dr = f"{ef} – {et}" if ef else et
                elif ef and not et:
                    _dur = _infer_degree_duration(deg)
                    try: et = str(int(ef[:4]) + _dur)
                    except: pass
                    dr = f"{ef} – {et}" if et else ef
                elif _u5_prev_yr is not None:
                    _dur = _infer_degree_duration(deg)
                    et = str(_u5_prev_yr)
                    ef = str(_u5_prev_yr - _dur)
                    dr = f"{ef} – {et}"
                else:
                    dr = ""
            try: _u5_prev_yr = int(str(ef)[:4]) if ef else _u5_prev_yr
            except: pass

            story.append(Table(
                [[Paragraph(esc(uni), S["edu_uni"]),
                  Paragraph(esc(dr),  S["edu_date"])]],
                colWidths=[TW * 0.68, TW * 0.32],
                style=TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
                    ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                    ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),2),
                    ("ALIGN",(1,0),(1,-1),"RIGHT"),
                ])
            ))
            deg_parts = [deg] if deg else []
            if cgpa: deg_parts.append(f"CGPA: {cgpa}")
            if deg_parts:
                story.append(Paragraph(esc(" | ".join(deg_parts)), S["edu_deg"]))
            if ach:
                prefix = "★ " if "gold" in ach.lower() else "✓ "
                story.append(Paragraph(prefix + esc(ach), S["edu_ach"]))
            story.append(Spacer(1, 6))

    # Certifications (optional) — tabular card design
    _certs5 = cv.get("certifications") or []
    if _certs5:
        section("Certifications")
        _cert_note5 = (cv.get("cert_note") or "").strip()
        if _cert_note5:
            _cnote5_s = ps5("u5_cnote", fontName="Helvetica-Oblique", fontSize=9, leading=13,
                             textColor=colors.HexColor("#444444"), spaceAfter=4)
            story.append(Paragraph(esc(_cert_note5), _cnote5_s))
        _c5_name_s = ps5("u5_cn",  fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=colors.HexColor("#111111"))
        _c5_meta_s = ps5("u5_cm",  fontName="Helvetica",         fontSize=8.5, leading=11, textColor=colors.HexColor("#065f46"))
        _c5_desc_s = ps5("u5_cd",  fontName="Helvetica-Oblique", fontSize=8,   leading=11, textColor=colors.HexColor("#444444"))
        _c5_num_s  = ps5("u5_cnum",fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=colors.HexColor("#059669"), alignment=1)
        _c5_hdr_s  = ps5("u5_chdr",fontName="Helvetica-Bold",    fontSize=8,   leading=10, textColor=colors.HexColor("#ffffff"), alignment=1)
        _CC5_1 = TW * 0.05
        _CC5_2 = TW * 0.30
        _CC5_3 = TW * 0.22
        _CC5_4 = TW * 0.43
        _c5_hdr_row = [
            Paragraph("#",               _c5_hdr_s),
            Paragraph("Certificate",     _c5_hdr_s),
            Paragraph("Issuer",          _c5_hdr_s),
            Paragraph("Credential Link", _c5_hdr_s),
        ]
        _c5_data   = [_c5_hdr_row]
        _c5_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#064e3b")),
            ("TOPPADDING",    (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#6ee7b7")),
            ("LINEBELOW",     (0, 0), (-1, 0),  1.0, colors.HexColor("#059669")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#ecfdf5"), colors.white]),
        ]
        for _i5, _cert5 in enumerate(_certs5):
            _cn5  = (_cert5.get("name")        or "").strip()
            _cl5  = (_cert5.get("link")        or "").strip()
            _cis5 = (_cert5.get("issuer")      or "").strip()
            _cde5 = (_cert5.get("description") or "").strip()
            if not any([_cn5, _cl5, _cis5, _cde5]):
                continue
            _safe5 = _cl5.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _name_cell5 = [Paragraph(f"<b>{esc(_cn5)}</b>" if _cn5 else "—", _c5_name_s)]
            if _cde5:
                _name_cell5.append(Paragraph(esc(_cde5), _c5_desc_s))
            _c5_data.append([
                Paragraph(str(_i5 + 1), _c5_num_s),
                _name_cell5,
                Paragraph(esc(_cis5) if _cis5 else "—", _c5_meta_s),
                Paragraph(f'<a href="{_safe5}" color="#059669">{_safe5}</a>' if _cl5 else "—", _c5_meta_s),
            ])
            _c5_styles.append(("TOPPADDING",    (0, _i5+1), (-1, _i5+1), 5))
            _c5_styles.append(("BOTTOMPADDING", (0, _i5+1), (-1, _i5+1), 5))
        _c5_tbl = Table(_c5_data, colWidths=[_CC5_1, _CC5_2, _CC5_3, _CC5_4], repeatRows=1)
        _c5_tbl.setStyle(TableStyle(_c5_styles))
        story.append(_c5_tbl)
        story.append(Spacer(1, 6))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    _page_count = [0]
    def _count_page(canvas, doc):
        _page_count[0] += 1

    doc.build(story, onFirstPage=_count_page, onLaterPages=_count_page)

    last_y = doc.frame._y if hasattr(doc, 'frame') and doc.frame else MB
    tight_h = PAGE_H_SINGLE - last_y + MB + 1 * mm
    tight_h = max(tight_h, 60 * mm)
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
