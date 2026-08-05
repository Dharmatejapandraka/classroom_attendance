"""
pdf_report.py
Generates a per-class attendance PDF report using reportlab.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

import database


def build_class_attendance_pdf(class_section, target_date=None):
    """
    Builds a PDF attendance report for a single class/section on a given date
    (defaults to today). Returns raw PDF bytes, ready for st.download_button.
    """
    if target_date is None:
        target_date = datetime.now().date().isoformat()

    members = database.get_all_members(class_section=class_section)
    present_map = database.get_attendance_on_date(target_date, class_section=class_section)

    total = len(members)
    present_count = len(present_map)
    absent_count = total - present_count
    pct = f"{(present_count / total * 100):.1f}%" if total else "0.0%"

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#0F172A"),
    )
    sub_style = ParagraphStyle(
        "ReportSub", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#475569"),
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "SectionHeader", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#0F172A"),
        spaceBefore=14, spaceAfter=8,
    )

    story = []
    story.append(Paragraph("FaceTrack Classroom Attendance Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Class / Section: <b>{class_section}</b>", sub_style))
    story.append(Paragraph(f"Date: <b>{target_date}</b>", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style))
    story.append(Spacer(1, 10))

    # Summary table
    summary_data = [
        ["Total Students", "Present", "Absent", "Attendance %"],
        [str(total), str(present_count), str(absent_count), pct],
    ]
    summary_table = Table(summary_data, colWidths=[1.6 * inch] * 4)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, 1), [colors.HexColor("#EEF2FF")]),
    ]))
    story.append(summary_table)

    # Present students
    story.append(Paragraph("Present Students", section_style))
    if present_map:
        rows = [["Name", "Phone", "Check-in Time"]]
        for mid, name, phone, _cls in members:
            if mid in present_map:
                rows.append([name, phone, present_map[mid]])
        story.append(_styled_table(rows, [2.6 * inch, 2.0 * inch, 1.6 * inch], colors.HexColor("#22C55E")))
    else:
        story.append(Paragraph("No students were marked present on this date.", styles["Normal"]))

    # Absent students
    story.append(Paragraph("Absent Students", section_style))
    absent_rows = [["Name", "Phone"]]
    for mid, name, phone, _cls in members:
        if mid not in present_map:
            absent_rows.append([name, phone])
    if len(absent_rows) > 1:
        story.append(_styled_table(absent_rows, [3.6 * inch, 2.6 * inch], colors.HexColor("#EF4444")))
    else:
        story.append(Paragraph("No absentees — full attendance!", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def _styled_table(rows, col_widths, header_color):
    table = Table(rows, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    return table
