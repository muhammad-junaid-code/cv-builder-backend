"""
CV Builder AI — UI package
Each sub-module contains exactly one PDF builder function:

    UI1.build_cv_pdf       — Classic Executive
    UI2.build_cv_pdf_ui2   — Modern Sidebar (teal two-column)
    UI3.build_cv_pdf_ui3   — Contemporary Card (slate-blue / gold)

Shared helpers (_normalise_edu_entry, _infer_degree_duration, _contact_href)
live in main.py and are injected into UI._shared by main.py at startup so
that the builders can import them without circular dependencies.
"""
