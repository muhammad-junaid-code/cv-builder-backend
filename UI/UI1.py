"""
CV Builder AI — UI1 PDF Builder
Classic Executive layout (original): centered header, full-width sections, green medal.
"""

import io
import re as _contact_re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib import colors

# ── Import shared helpers from main ──────────────────────────────────────────
# These are defined in main.py and injected at import time via the UI package.
# Do NOT redefine them here; always use the ones from main.
from UI._shared import _normalise_edu_entry, _infer_degree_duration, _contact_href

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
            """Delegates to the shared _contact_href helper (defined above all builders).
            Consistent phone/email/URL detection across UI1, UI2, and UI3."""
            return _contact_href(val)

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
    # ── Education section — render ALL qualifications from the list ─────────────
    # cv["education"] is now always a list (sanitise_cv guarantees this).
    # p_edu (from profile_data) is the authoritative source; cv["education"] is
    # the AI-passed-through copy — we prefer p_edu when available.
    _edu_list_raw = cv.get("education") or []
    # Normalise: if somehow still a plain dict, wrap it
    if isinstance(_edu_list_raw, dict):
        _edu_list_raw = [_edu_list_raw]

    # Merge with p_edu: p_edu entries are authoritative for their index.
    # If p_edu has more entries than the AI returned, use p_edu as the master list.
    # Years are resolved using the same UI-first, auto-sequencing logic as the
    # AI merge path — _infer_degree_duration() drives all duration inference.
    _n_edu = max(len(_edu_list_raw), len(p_edu))
    _edu_entries = []
    _pdf_prev_start_yr = None   # start year of the entry above (for sequencing)
    for _ei in range(_n_edu):
        _cv_e  = _edu_list_raw[_ei] if _ei < len(_edu_list_raw) else {}
        _pr_e  = p_edu[_ei]         if _ei < len(p_edu)         else {}
        _uni   = (_pr_e.get("institution") or _cv_e.get("university") or "").strip()
        _deg   = (_pr_e.get("degree")      or _cv_e.get("degree")     or "").strip()
        _cgpa  = (_pr_e.get("cgpa")        or _cv_e.get("cgpa")       or "").strip()
        _ach   = (_pr_e.get("achievement") or "").strip()   # never AI-invented

        # Priority: years already resolved in cv["education"] (set by AI merge path)
        # → UI from/to → infer from degree duration → auto-sequence from anchor
        _yr = (_cv_e.get("years") or "").strip()
        if not _yr:
            _ef  = str(_pr_e.get("from") or "").strip()
            _et  = str(_pr_e.get("to")   or "").strip()
            _dur = _infer_degree_duration(_deg)
            if _ef and _et:
                _yr = f"{_ef} - {_et}"
            elif _et and not _ef:
                try:
                    _yr = f"{int(_et[:4]) - _dur} - {_et}"
                except (ValueError, TypeError):
                    _yr = ""
            elif _ef and not _et:
                try:
                    _yr = f"{_ef} - {int(_ef[:4]) + _dur}"
                except (ValueError, TypeError):
                    _yr = ""
            elif _pdf_prev_start_yr is not None:
                _yr = f"{_pdf_prev_start_yr - _dur} - {_pdf_prev_start_yr}"

        # Update anchor for the next entry
        try:
            _pdf_prev_start_yr = int(_yr.split("-")[0].strip()[:4])
        except (ValueError, TypeError, IndexError):
            pass

        _edu_entries.append({
            "university": _uni, "degree": _deg,
            "cgpa": _cgpa, "years": _yr, "achievement": _ach,
        })

    _has_any_edu = any(
        any([e["university"], e["degree"], e["years"], e["cgpa"]])
        for e in _edu_entries
    )
    if _has_any_edu:
        story.append(Paragraph("EDUCATION", S["sec_title_center"]))

        edu_date_style = ps("edu_dr",
            fontName="Helvetica", fontSize=10,
            alignment=TA_RIGHT,
            textColor=colors.HexColor("#666666")
        )

        for _ei, _entry in enumerate(_edu_entries):
            uni         = _entry["university"]
            degree      = _entry["degree"]
            years       = _entry["years"]
            cgpa        = _entry["cgpa"]
            achievement = _entry["achievement"]

            if not any([uni, degree, years, cgpa]):
                continue  # skip completely empty entries

            # Small gap between qualifications (not before the first)
            if _ei > 0:
                story.append(Spacer(1, 3 * mm))

            # ── University name (left) with years aligned to the right ───────
            if uni:
                uni_para   = Paragraph(uni.upper(), S["edu_uni"])
                years_para = Paragraph(years, edu_date_style) if years else Paragraph("", edu_date_style)
                edu_header_tbl = Table([[uni_para, years_para]], colWidths=[TW * 0.65, TW * 0.35])
                edu_header_tbl.setStyle(TableStyle([
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                    ("TOPPADDING",    (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]))
                story.append(edu_header_tbl)
            elif years:
                story.append(Paragraph(years, S["edu_deg"]))

            # ── Degree + CGPA line ───────────────────────────────────────────
            deg_parts = [x for x in [degree] if x]
            if cgpa:
                deg_parts.append(f"CGPA: {cgpa}")
            deg_text = " | ".join(deg_parts)
            if deg_text:
                story.append(Paragraph(deg_text, S["edu_deg"]))

            # ── Achievement — only from profile, never AI-invented ───────────
            if achievement:
                if "gold" in achievement.lower():
                    story.append(Paragraph(f"🏅 {achievement}", S["edu_medal"]))
                else:
                    story.append(Paragraph(f"✓ {achievement}", S["edu_deg"]))

    # Certifications (optional) — tabular card design matching UI1 blue palette
    _certs = cv.get("certifications") or []
    if _certs:
        story.append(Paragraph("CERTIFICATIONS", S["sec_title_center"]))

        # Section-level note (optional, set by user in the extension)
        _cert_note = (cv.get("cert_note") or "").strip()
        if _cert_note:
            story.append(Paragraph(
                _cert_note,
                ps("cnote", fontName="Helvetica-Oblique", fontSize=9, leading=13,
                   textColor=colors.HexColor("#444444"), spaceAfter=4)
            ))

        # ── Paragraph styles — all inherit UI1's #0057A8 blue ────────────────
        _BLUE   = "#0057A8"
        _DKBLUE = "#003d7a"   # header background — one shade darker than accent

        _c_hdr_s  = ps("c_hdr",  fontName="Helvetica-Bold",    fontSize=8.5, leading=11,
                        textColor=colors.white, alignment=TA_CENTER)
        _c_num_s  = ps("c_num",  fontName="Helvetica-Bold",    fontSize=9,   leading=12,
                        textColor=colors.HexColor(_BLUE), alignment=TA_CENTER)
        _c_name_s = ps("c_name", fontName="Helvetica-Bold",    fontSize=9.5, leading=13,
                        textColor=colors.HexColor("#111111"))
        _c_desc_s = ps("c_desc", fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
                        textColor=colors.HexColor("#555555"))
        _c_meta_s = ps("c_meta", fontName="Helvetica",         fontSize=8.5, leading=12,
                        textColor=colors.HexColor(_BLUE))

        # ── Column widths — #col, Certificate, Issuer, Certificate Link ──────
        _CW = [TW * 0.04, TW * 0.32, TW * 0.22, TW * 0.42]

        _tbl_data = [[
            Paragraph("#",                  _c_hdr_s),
            Paragraph("Certificate",        _c_hdr_s),
            Paragraph("Issuer",             _c_hdr_s),
            Paragraph("Certificate Link",   _c_hdr_s),   # ← "Certificate Link"
        ]]
        _tbl_styles = [
            # Header row
            ("BACKGROUND",    (0, 0), (-1,  0), colors.HexColor(_DKBLUE)),
            ("LINEBELOW",     (0, 0), (-1,  0), 1.2, colors.HexColor(_BLUE)),
            ("TOPPADDING",    (0, 0), (-1,  0), 5),
            ("BOTTOMPADDING", (0, 0), (-1,  0), 5),
            # All cells
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            # Subtle grid matching UI1's #cccccc dividers
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            # Alternating row tint — very light blue to match UI1's #0057A8 palette
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#eef4fb"), colors.white]),
        ]

        for _i, _cert in enumerate(_certs):
            _cn  = (_cert.get("name")        or "").strip()
            _cl  = (_cert.get("link")        or "").strip()
            _cis = (_cert.get("issuer")      or "").strip()
            _cde = (_cert.get("description") or "").strip()
            if not any([_cn, _cl, _cis, _cde]):
                continue

            # Certificate cell: bold name + optional italic description below
            _name_parts = []
            if _cn:
                _name_parts.append(Paragraph(_cn, _c_name_s))
            if _cde:
                _name_parts.append(Paragraph(_cde, _c_desc_s))
            _name_cell = _name_parts if _name_parts else [Paragraph("—", _c_meta_s)]

            _safe_cl = _cl.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            _link_para = Paragraph(
                f'<a href="{_safe_cl}" color="{_BLUE}">{_safe_cl}</a>' if _cl else "—",
                _c_meta_s
            )

            _tbl_data.append([
                Paragraph(str(_i + 1), _c_num_s),
                _name_cell,
                Paragraph(_cis if _cis else "—", _c_meta_s),
                _link_para,
            ])

        _cert_tbl = Table(_tbl_data, colWidths=_CW, repeatRows=1)
        _cert_tbl.setStyle(TableStyle(_tbl_styles))
        story.append(_cert_tbl)
        story.append(Spacer(1, 4 * mm))

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

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()
