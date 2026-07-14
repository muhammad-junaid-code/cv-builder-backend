"""
CV Builder AI — UI3 PDF Builder
Contemporary Card layout: centered header, icon-style sections, warm slate-blue/gold palette.
"""

import io
import math
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
# PDF BUILDER — UI3 (Contemporary Card: centered header, icon-style sections, warm palette)
# Distinct from UI1 (classic centered) and UI2 (sidebar teal):
#   • Header: CENTERED name + role + contact
#   • Section titles: colored filled square icon + bold text (no HR lines)
#   • Dates positioned LEFT of company name (reversed from UI1/UI2)
#   • Bullets: em-dash (–) style, no large black dots
#   • Skills: two-column grid layout
#   • Accent color: deep slate-blue (#2c3e6b) with warm gold (#c8962a)
# ==============================================================================
def build_cv_pdf_ui3(cv: dict, profile_data: dict = None) -> bytes:
    """UI3 — Contemporary card layout: centered header, icon sections, reversed date layout."""
    import re as _re3

    _pd     = profile_data or {}
    p_name  = (_pd.get("name") or "").strip() or "CANDIDATE"
    p_links = _pd.get("links") or []
    p_work  = _pd.get("work")  or []
    p_edu   = [_normalise_edu_entry(e) for e in (_pd.get("edu") or [])]

    NAVY     = colors.HexColor("#2c3e6b")   # deep slate-blue
    GOLD     = colors.HexColor("#c8962a")   # warm gold accent
    DARK     = colors.HexColor("#1c1c1c")
    MID      = colors.HexColor("#4a4a4a")
    LIGHT    = colors.HexColor("#888888")
    BG_RULE  = colors.HexColor("#e4e8f0")   # light blue-grey for dividers

    buf = io.BytesIO()
    PAGE_W, _ = A4
    ML, MR, MT, MB = 15*mm, 15*mm, 14*mm, 14*mm
    # 5.0× A4 height — tall enough that no realistic CV ever triggers a page
    # break on this single canvas. A page break would reset frame._y and cause
    # later content to overwrite earlier content at the same canvas coordinates,
    # making sections like Core Competencies and Education disappear.
    PAGE_H_SINGLE = 841.89 * 5.0

    doc = SimpleDocTemplate(
        buf, pagesize=(PAGE_W, PAGE_H_SINGLE),
        leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
        title=f"{p_name} CV", author=p_name,
    )
    TW = PAGE_W - ML - MR

    def ps3(name, **kw):
        d = dict(fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0,
                 spaceBefore=0, textColor=DARK)
        d.update(kw)
        return ParagraphStyle(name, **d)

    S = {
        # ── Centered header block ─────────────────────────────────────────────
        "name":      ps3("u3_nm", fontName="Helvetica-Bold", fontSize=22, leading=28,
                         textColor=NAVY, alignment=TA_LEFT, spaceBefore=0, spaceAfter=2),
        "subtitle":  ps3("u3_st", fontName="Helvetica", fontSize=9.5, leading=13,
                         textColor=MID, alignment=TA_CENTER, spaceAfter=3),
        "subtitle_left": ps3("u3_stl", fontName="Helvetica", fontSize=10, leading=14,
                         textColor=GOLD, alignment=TA_LEFT, spaceBefore=2, spaceAfter=0),
        "contact":   ps3("u3_ct", fontName="Helvetica", fontSize=8.5, leading=12,
                         textColor=colors.HexColor("#0057a8"), alignment=TA_CENTER),
        "contact_right": ps3("u3_ctr", fontName="Helvetica", fontSize=8.5, leading=13,
                         textColor=MID, alignment=TA_RIGHT, spaceAfter=0),
        # ── Section icon-style title ──────────────────────────────────────────
        "sec_icon":  ps3("u3_si", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                         textColor=colors.white, spaceBefore=0, spaceAfter=0),
        "sec_label": ps3("u3_sl", fontName="Helvetica-Bold", fontSize=11, leading=15,
                         textColor=NAVY, spaceBefore=12, spaceAfter=5),
        # ── Experience entries ────────────────────────────────────────────────
        "date":      ps3("u3_dt", fontName="Helvetica-Bold", fontSize=8.5, leading=12,
                         textColor=GOLD, alignment=TA_RIGHT),
        "company":   ps3("u3_co", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                         textColor=DARK),
        "role":      ps3("u3_rl", fontName="Helvetica-Oblique", fontSize=10, leading=13,
                         textColor=colors.HexColor("#3a5080"), spaceAfter=4),
        "bullet":    ps3("u3_bul", fontName="Helvetica", fontSize=9.5, leading=14,
                         leftIndent=12, textColor=MID, spaceAfter=3),
        "tech":      ps3("u3_tch", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         leftIndent=12, textColor=GOLD),
        # ── Skills two-column ────────────────────────────────────────────────
        "sk_cat":    ps3("u3_skc", fontName="Helvetica-Bold", fontSize=9.5, leading=13,
                         textColor=NAVY),
        "sk_val":    ps3("u3_skv", fontName="Helvetica", fontSize=9.5, leading=13,
                         textColor=MID),
        # ── Projects ─────────────────────────────────────────────────────────
        "proj_name": ps3("u3_pn", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                         textColor=DARK, spaceBefore=3),
        "proj_body": ps3("u3_pb", fontName="Helvetica", fontSize=9, leading=13,
                         textColor=MID),
        "proj_bul":  ps3("u3_pbl", fontName="Helvetica", fontSize=9, leading=12.5,
                         leftIndent=10, textColor=MID, spaceAfter=2),
        "proj_tech": ps3("u3_pt", fontName="Helvetica-Bold", fontSize=8, leading=11,
                         textColor=GOLD),
        # ── Competencies + Education ─────────────────────────────────────────
        "comp":      ps3("u3_cmp", fontName="Helvetica", fontSize=9.5, leading=13.5,
                         textColor=MID),
        "edu_uni":   ps3("u3_eu", fontName="Helvetica-Bold", fontSize=11, leading=14,
                         textColor=DARK),
        "edu_deg":   ps3("u3_ed", fontName="Helvetica", fontSize=9.5, leading=13,
                         textColor=MID),
        "edu_note":  ps3("u3_en", fontName="Helvetica-Oblique", fontSize=9, leading=12,
                         textColor=GOLD),
        "edu_date":  ps3("u3_edt", fontName="Helvetica-Bold", fontSize=8.5, leading=12,
                         textColor=GOLD, alignment=TA_RIGHT),
        # ── Summary ──────────────────────────────────────────────────────────
        "summary":   ps3("u3_sum", fontName="Helvetica", fontSize=9.5, leading=15.5,
                         textColor=MID, spaceAfter=2),
    }

    def esc(s):
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def section_title(text):
        """Icon-style header: filled navy square + bold label on same line via Table."""
        icon_cell  = Table(
            [[Paragraph(esc("  "), S["sec_icon"])]],
            colWidths=[10],
            style=TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), NAVY),
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 0),
                ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ])
        )
        label_cell = Paragraph(esc(text.upper()), S["sec_label"])
        row = Table(
            [[icon_cell, label_cell]],
            colWidths=[12, TW - 12],
            style=TableStyle([
                ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
                ("LEFTPADDING",  (0,0),(-1,-1), 0),
                ("RIGHTPADDING", (0,0),(-1,-1), 0),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 0),
            ])
        )
        divider = HRFlowable(width="100%", thickness=1, color=BG_RULE,
                             spaceBefore=3, spaceAfter=6)
        return [row, divider]

    story = []

    # ── Asymmetric left-aligned header ──────────────────────────────────────────
    # Deliberately NOT a centered stack (that skeleton belongs to UI1): name/title
    # sit left-aligned in a wide column, contact details stack right-aligned in a
    # narrow column beside them, so the whole block reads as one asymmetric band.
    name_cell = [Paragraph(esc(p_name), S["name"])]
    title_str = (cv.get("title") or "").strip()
    if title_str:
        name_cell.append(Paragraph(esc(title_str), S["subtitle_left"]))

    contact_cell = []
    if p_links:
        for lnk in p_links:
            v    = (lnk.get("value") or "").strip()
            lbl  = (lnk.get("label") or "").strip().lower()
            if not v:
                continue
            href = "" if lbl == "location" else _contact_href(v)
            _sv  = esc(v)
            text = f'<a href="{esc(href)}" color="#0057a8">{_sv}</a>' if href else _sv
            contact_cell.append(Paragraph(text, S["contact_right"]))

    header_tbl = Table(
        [[name_cell, contact_cell]],
        colWidths=[TW * 0.62, TW * 0.38],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(header_tbl)

    story.append(HRFlowable(width="100%", thickness=3, color=NAVY,
                             spaceBefore=8, spaceAfter=4))
    story.append(HRFlowable(width="100%", thickness=1, color=GOLD,
                             spaceBefore=2, spaceAfter=10))

    # ── Profile Summary ────────────────────────────────────────────────────────
    summary_text = (cv.get("summary") or "").strip()
    if summary_text:
        story += section_title("Professional Summary")
        story.append(Paragraph(esc(summary_text), S["summary"]))
        story.append(Spacer(1, 4))

    # ── Work Experience ────────────────────────────────────────────────────────
    ai_cos = cv.get("companies") or []
    # Build a unified list: for each AI company entry, merge in the matching
    # profile work entry (by index) so dates/company names from the profile
    # always take precedence while AI-generated bullets are always used.
    # Cap to ai_cos count when it is non-empty — ai_cos has already been
    # hard-truncated by fix_companies() to the correct count for the selected
    # years_exp value (1 yr → 1, 2 yrs → 2, etc.).  Using max() would let
    # extra p_work entries bleed through as bare company names with no bullets.
    # Fall back to max() only when ai_cos is empty (profile-only render path).
    num_entries = len(ai_cos) if ai_cos else max(len(ai_cos), len(p_work))
    if num_entries > 0:
        story += section_title("Work Experience")
        for i in range(num_entries):
            w  = p_work[i]   if i < len(p_work)  else {}
            ai = ai_cos[i]   if i < len(ai_cos)   else {}

            company = (w.get("company") or "").strip() or ai.get("company","")
            role    = (w.get("role")    or "").strip() or ai.get("role","")
            wf      = str(w.get("from") or "").strip()
            wt      = str(w.get("to")   or "").strip()
            if wf and wt:
                dr = f"{wf} – {wt}"
            elif wf:
                dr = f"{wf} – Present"
            else:
                dr = ai.get("dateRange","")

            # ── Timeline-card entry: gold accent bar on the left, date folded
            # inline next to the role (NOT mirrored to the right like UI1) ──────
            card_content = [Paragraph(esc(company), S["company"])]
            role_date_bits = []
            if role:
                role_date_bits.append(esc(role))
            if dr:
                role_date_bits.append(f'<font color="#c8962a"><b>{esc(dr)}</b></font>')
            if role_date_bits:
                card_content.append(Paragraph("   /   ".join(role_date_bits), S["role"]))

            # Prefer AI bullets (rich, JD-tailored); fall back to profile bullets
            bullets = ai.get("bullets") or []
            if not bullets and w.get("bullets"):
                bullets = [b.strip() for b in str(w["bullets"]).split("\n") if b.strip()]
            for b in bullets:
                b_clean = b.lstrip("•·▸–▪● ").strip()
                card_content.append(Paragraph('<font size="7">•</font> ' + esc(b_clean), S["bullet"]))

            tech_raw = ai.get("tech","")
            if tech_raw:
                sep  = "|" if "|" in tech_raw else ","
                tags = "  ·  ".join(t.strip() for t in tech_raw.split(sep) if t.strip())
                card_content.append(Paragraph(esc(tags), S["tech"]))

            story.append(Table(
                [[Paragraph("", S["role"]), card_content]],
                colWidths=[2.5*mm, TW - 2.5*mm - 3*mm],
                style=TableStyle([
                    ("BACKGROUND",    (0,0), (0,0), GOLD),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING",   (0,0), (0,0), 0),
                    ("RIGHTPADDING",  (0,0), (0,0), 0),
                    ("LEFTPADDING",   (1,0), (1,0), 8),
                    ("RIGHTPADDING",  (1,0), (1,0), 0),
                    ("TOPPADDING",    (0,0), (-1,-1), 6),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                ])
            ))
            story.append(Spacer(1, 10))

    # ── Technical Skills — two-column grid ────────────────────────────────────
    skills = cv.get("skills") or []
    if skills:
        story += section_title("Technical Skills")
        col_data = []
        for sk in skills[:5]:   # UI3 shows up to 5 skill categories
            colon = sk.find(":")
            if colon > 0:
                cat = sk[:colon].strip()
                val = sk[colon+1:].strip()
            else:
                cat, val = "", sk
            col_data.append((cat, val))
        # Arrange in two columns
        half = math.ceil(len(col_data) / 2)
        left_col  = col_data[:half]
        right_col = col_data[half:]
        while len(right_col) < len(left_col):
            right_col.append(("",""))

        def _sk_cell(cat, val):
            items = []
            if cat:
                items.append(Paragraph(f"<b>{esc(cat)}</b>", S["sk_cat"]))
            if val:
                items.append(Paragraph(esc(val), S["sk_val"]))
            return items

        col_w = (TW - 8) / 2
        for (lc, lv), (rc, rv) in zip(left_col, right_col):
            grid = Table(
                [[_sk_cell(lc, lv), _sk_cell(rc, rv)]],
                colWidths=[col_w, col_w],
                style=TableStyle([
                    ("VALIGN",       (0,0),(-1,-1),"TOP"),
                    ("LEFTPADDING",  (0,0),(-1,-1),0),
                    ("RIGHTPADDING", (0,0),(-1,-1),4),
                    ("TOPPADDING",   (0,0),(-1,-1),2),
                    ("BOTTOMPADDING",(0,0),(-1,-1),4),
                    ("LINEBELOW",    (0,0),(-1,-1),0.4,BG_RULE),
                ])
            )
            story.append(grid)
        story.append(Spacer(1, 4))

    # ── Selected Projects ──────────────────────────────────────────────────────
    projects = cv.get("projects") or []
    if projects:
        story += section_title("Projects")
        for p in projects:
            raw_name = (p.get("name") or "").strip()
            name = _re3.sub(r'\s*\[[^\]]*\]\s*$', '', raw_name)
            name = _re3.sub(r'^[A-Z][A-Z\s&\-]{2,}:\s*', '', name).strip()
            if name:
                story.append(Paragraph(esc(name), S["proj_name"]))
            if p.get("overview"):
                story.append(Paragraph(esc(p["overview"]), S["proj_body"]))
            for b in (p.get("bullets") or []):
                b_clean = b.lstrip("•·▸–▪● ").strip()
                story.append(Paragraph('<font size="7">•</font> ' + esc(b_clean), S["proj_bul"]))
            tech_t = p.get("techTags") or []
            if not tech_t and p.get("tech"):
                sep = "|" if "|" in p["tech"] else ","
                tech_t = [t.strip() for t in p["tech"].split(sep) if t.strip()]
            if tech_t:
                story.append(Paragraph("  ·  ".join(esc(t) for t in tech_t), S["proj_tech"]))
            story.append(Spacer(1, 7))

    # ── Core Competencies ─────────────────────────────────────────────────────
    comp_str = (cv.get("competencies") or "").strip()
    if comp_str:
        story += section_title("Core Competencies")
        # Display as comma-separated inline list
        pills = [c.strip() for c in comp_str.replace("*","•").split("•") if c.strip()]
        story.append(Paragraph("  ·  ".join(esc(c) for c in pills), S["comp"]))
        story.append(Spacer(1, 4))

    # ── Education ─────────────────────────────────────────────────────────────
    # Build the render list: p_edu (profile) is authoritative for names/degree/cgpa.
    # cv["education"] (AI-merged) is authoritative for the "years" field when the
    # profile entry has no from/to — this mirrors UI1's resolution logic exactly.
    _cv_edu_raw = cv.get("education") or []
    if isinstance(_cv_edu_raw, dict):
        _cv_edu_raw = [_cv_edu_raw]

    if p_edu:
        _edu_render_list = []
        for _i, _pe in enumerate(p_edu):
            _entry = dict(_pe)   # copy — never mutate original
            # If this profile entry lacks from/to, pull years from cv["education"]
            _ef = str(_pe.get("from") or "").strip()
            _et = str(_pe.get("to")   or "").strip()
            if not _ef and not _et and _i < len(_cv_edu_raw):
                _yr = str(_cv_edu_raw[_i].get("years") or "").strip()
                if _yr:
                    _entry["years"] = _yr
            _edu_render_list.append(_entry)
    else:
        # No profile edu — use cv["education"] entirely
        _edu_render_list = []
        for _ce in _cv_edu_raw:
            _edu_render_list.append({
                "institution": (_ce.get("university") or _ce.get("institution") or "").strip(),
                "degree":      (_ce.get("degree") or "").strip(),
                "cgpa":        (_ce.get("cgpa") or "").strip(),
                "years":       str(_ce.get("years") or "").strip(),
                "achievement": (_ce.get("achievement") or "").strip(),
            })

    if _edu_render_list:
        story += section_title("Education")
        _u3_prev_start_yr = None   # anchor for auto-sequencing multiple qualifications
        for e in _edu_render_list:
            ef  = str(e.get("from") or "").strip()
            et  = str(e.get("to")   or "").strip()
            deg = (e.get("degree") or "").strip()
            uni = (e.get("institution") or "").strip()
            cgpa = (e.get("cgpa") or "").strip()
            ach  = (e.get("achievement") or "").strip()

            # ── Resolve date range — same priority chain as UI1 ────────────────
            # 1. Explicit from + to in entry
            if ef and et:
                dr = f"{ef}–{et}"
            else:
                # 2. "years" field (e.g. "2016-2020" stored by cv["education"])
                _yr_raw = str(e.get("years") or "").strip()
                _sep = "–" if "–" in _yr_raw else "-"
                if _yr_raw and _sep in _yr_raw:
                    _yp = [p.strip() for p in _yr_raw.split(_sep, 1)]
                    ef, et = _yp[0], _yp[-1]
                    dr = f"{ef}–{et}"
                elif et and not ef:
                    # 3a. Only end year — infer start from degree duration
                    _dur = _infer_degree_duration(deg)
                    try: ef = str(int(et[:4]) - _dur)
                    except (ValueError, TypeError): pass
                    dr = f"{ef}–{et}" if ef else et
                elif ef and not et:
                    # 3b. Only start year — infer end from degree duration
                    _dur = _infer_degree_duration(deg)
                    try: et = str(int(ef[:4]) + _dur)
                    except (ValueError, TypeError): pass
                    dr = f"{ef}–{et}" if et else ef
                elif _u3_prev_start_yr is not None:
                    # 4. No dates at all — sequence backwards from previous entry
                    _dur = _infer_degree_duration(deg)
                    et = str(_u3_prev_start_yr)
                    ef = str(_u3_prev_start_yr - _dur)
                    dr = f"{ef}–{et}"
                else:
                    dr = ""
            # Update anchor for the next entry
            try: _u3_prev_start_yr = int(str(ef)[:4]) if ef else _u3_prev_start_yr
            except (ValueError, TypeError): pass

            if uni:
                # Use absolute colWidths (same unit as work experience) for
                # consistent right-alignment regardless of institution name length.
                story.append(Table(
                    [[Paragraph(esc(uni), S["edu_uni"]),
                      Paragraph(esc(dr),  S["edu_date"])]],
                    colWidths=[TW * 0.68, TW * 0.32],
                    style=TableStyle([("VALIGN",        (0,0),(-1,-1),"BOTTOM"),
                                      ("LEFTPADDING",   (0,0),(-1,-1),0),
                                      ("RIGHTPADDING",  (0,0),(-1,-1),0),
                                      ("TOPPADDING",    (0,0),(-1,-1),0),
                                      ("BOTTOMPADDING", (0,0),(-1,-1),2),
                                      ("ALIGN",         (1,0),(1,-1),"RIGHT")])
                ))
            deg_parts = [deg] if deg else []
            if cgpa:
                deg_parts.append(f"CGPA: {cgpa}")
            if deg_parts:
                story.append(Paragraph(esc(" | ".join(deg_parts)), S["edu_deg"]))
            if ach:
                # Use only Helvetica-supported characters — emoji glyphs render
                # as black squares (■) in standard PDF fonts.
                prefix = "★ " if "gold" in ach.lower() else "✓ "
                story.append(Paragraph(prefix + esc(ach), S["edu_note"]))
            story.append(Spacer(1, 6))

    # Certifications (optional) — tabular card design
    _certs3 = cv.get("certifications") or []
    if _certs3:
        story += section_title("Certifications")
        _cert_note3 = (cv.get("cert_note") or "").strip()
        if _cert_note3:
            _cnote3_s = ps3("u3_cnote", fontName="Helvetica-Oblique", fontSize=9, leading=13,
                             textColor=colors.HexColor("#444444"), spaceAfter=4)
            story.append(Paragraph(esc(_cert_note3), _cnote3_s))
        _c3_name_s = ps3("u3_cn",  fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=colors.HexColor("#111111"))
        _c3_meta_s = ps3("u3_cm",  fontName="Helvetica",         fontSize=8.5, leading=11, textColor=colors.HexColor("#2c3e6b"))
        _c3_desc_s = ps3("u3_cd",  fontName="Helvetica-Oblique", fontSize=8,   leading=11, textColor=colors.HexColor("#444444"))
        _c3_num_s  = ps3("u3_cnum",fontName="Helvetica-Bold",    fontSize=9,   leading=12, textColor=GOLD, alignment=1)
        _c3_hdr_s  = ps3("u3_chdr",fontName="Helvetica-Bold",    fontSize=8,   leading=10, textColor=colors.HexColor("#ffffff"), alignment=1)
        _CC3_1 = TW * 0.05
        _CC3_2 = TW * 0.30
        _CC3_3 = TW * 0.22
        _CC3_4 = TW * 0.43
        _c3_hdr_row = [
            Paragraph("#",               _c3_hdr_s),
            Paragraph("Certificate",     _c3_hdr_s),
            Paragraph("Issuer",          _c3_hdr_s),
            Paragraph("Credential Link", _c3_hdr_s),
        ]
        _c3_data   = [_c3_hdr_row]
        _c3_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e6b")),
            ("TOPPADDING",    (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#c8b87a")),
            ("LINEBELOW",     (0, 0), (-1, 0),  1.0, colors.HexColor("#c8962a")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f3ec"), colors.white]),
        ]
        for _i3, _cert3 in enumerate(_certs3):
            _cn3  = (_cert3.get("name")        or "").strip()
            _cl3  = (_cert3.get("link")        or "").strip()
            _cis3 = (_cert3.get("issuer")      or "").strip()
            _cde3 = (_cert3.get("description") or "").strip()
            if not any([_cn3, _cl3, _cis3, _cde3]):
                continue
            _safe3 = _cl3.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _name_cell3 = [Paragraph(f"<b>{esc(_cn3)}</b>" if _cn3 else "—", _c3_name_s)]
            if _cde3:
                _name_cell3.append(Paragraph(esc(_cde3), _c3_desc_s))
            _c3_data.append([
                Paragraph(str(_i3 + 1), _c3_num_s),
                _name_cell3,
                Paragraph(esc(_cis3) if _cis3 else "—", _c3_meta_s),
                Paragraph(f'<a href="{_safe3}" color="#2c3e6b">{_safe3}</a>' if _cl3 else "—", _c3_meta_s),
            ])
            _c3_styles.append(("TOPPADDING",    (0, _i3+1), (-1, _i3+1), 5))
            _c3_styles.append(("BOTTOMPADDING", (0, _i3+1), (-1, _i3+1), 5))
        _c3_tbl = Table(_c3_data, colWidths=[_CC3_1, _CC3_2, _CC3_3, _CC3_4], repeatRows=1)
        _c3_tbl.setStyle(TableStyle(_c3_styles))
        story.append(_c3_tbl)
        story.append(Spacer(1, 6))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    # Track page count via onPage callback so we can compute the true content
    # height even when content overflows onto a second (or third) "page" of the
    # tall single-page canvas.
    _page_count = [0]

    def _count_page(canvas, doc):
        _page_count[0] += 1

    doc.build(story, onFirstPage=_count_page, onLaterPages=_count_page)

    # Crop the canvas down to actual content height.
    # With PAGE_H_SINGLE = 841.89 * 5.0, no realistic CV triggers a page break,
    # so frame._y is always the absolute canvas y where the last content ended.
    # If somehow a page break did occur (extremely dense CV), fall back to showing
    # the full canvas (crop_bottom = 0) to ensure no content is ever hidden.
    last_y = doc.frame._y if hasattr(doc, 'frame') and doc.frame else MB
    n_pages = max(_page_count[0], 1)

    if n_pages == 1:
        # Normal case: frame._y is the absolute bottom of content on the canvas.
        tight_h = PAGE_H_SINGLE - last_y + MB + 1 * mm
    else:
        # Fallback: page break occurred (unexpectedly dense content).
        # Show the full canvas to guarantee nothing is cropped out.
        tight_h = PAGE_H_SINGLE

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
