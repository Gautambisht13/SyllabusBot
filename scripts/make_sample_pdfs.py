"""Generate realistic demo PDFs so the pipeline is runnable with zero setup.

    python scripts/make_sample_pdfs.py

Writes three documents under data/ — a course syllabus, an academic calendar
and a campus handbook — with the kind of content students actually ask about
(late work penalties, add/drop dates, attendance, quiet hours). Delete them and
drop in your real PDFs whenever you're ready.

Requires reportlab (in requirements.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# (relative path, title, [(heading, [paragraphs])])
DOCUMENTS: list[tuple[str, str, list[tuple[str, list[str]]]]] = [
    (
        "syllabi/CS101_Syllabus.pdf",
        "CS101 - Introduction to Computer Science - Fall 2026",
        [
            (
                "COURSE INFORMATION",
                [
                    "Course: CS101 Introduction to Computer Science (4 credits).",
                    "Instructor: Dr. Amara Osei. Office: Turing Hall 214.",
                    "Office hours: Tuesdays and Thursdays, 14:00-16:00, or by appointment.",
                    "Email: a.osei@example.edu. Allow up to 48 hours for a reply on weekdays.",
                    "Lectures: Mon/Wed 10:00-11:20, Turing Hall 105. Lab: Fri 13:00-14:50, Lab B.",
                ],
            ),
            (
                "GRADING",
                [
                    "Final grades in CS101 are calculated as follows: weekly programming "
                    "assignments 30%, two midterm exams 30% (15% each), final project 20%, "
                    "final exam 15%, lab participation 5%.",
                    "Letter grades: A 93-100, A- 90-92, B+ 87-89, B 83-86, B- 80-82, "
                    "C+ 77-79, C 73-76, C- 70-72, D 60-69, F below 60.",
                    "There is no curve and no extra credit in CS101.",
                ],
            ),
            (
                "LATE SUBMISSION POLICY",
                [
                    "Assignments in CS101 are due at 23:59 on the posted due date via the "
                    "course portal. Late work is accepted for up to 72 hours after the "
                    "deadline with a penalty of 10% of the earned score per 24-hour period "
                    "or part thereof.",
                    "After 72 hours, late submissions receive a grade of zero and are not "
                    "eligible for feedback.",
                    "Each student has three penalty-free late days per semester, which may "
                    "be applied to programming assignments only. Late days must be claimed "
                    "in the course portal before the deadline passes; they cannot be applied "
                    "retroactively.",
                    "The final project and both midterm exams are excluded from the late "
                    "policy and from penalty-free late days. No late final projects are "
                    "accepted under any circumstances.",
                    "Extensions beyond the late policy require documented medical or family "
                    "emergency and must be requested in writing to Dr. Osei within seven "
                    "calendar days of the deadline.",
                ],
            ),
            (
                "ATTENDANCE AND PARTICIPATION",
                [
                    "Lab attendance is mandatory. Students may miss two labs without "
                    "penalty; each additional unexcused absence reduces the lab "
                    "participation component by two percentage points.",
                    "Lecture attendance is not graded, but material presented in lecture and "
                    "not in the readings is examinable.",
                ],
            ),
            (
                "ACADEMIC INTEGRITY",
                [
                    "Programming assignments in CS101 are individual work. Discussing "
                    "approaches is permitted; sharing or copying code is not.",
                    "Generative AI tools may be used to explain concepts but not to produce "
                    "submitted code. Any AI assistance must be disclosed in the submission "
                    "comments.",
                    "A first violation results in a zero on the assignment and a report to "
                    "the Office of Academic Integrity. A second violation results in failure "
                    "of the course.",
                ],
            ),
        ],
    ),
    (
        "calendar/Academic_Calendar_2026_2027.pdf",
        "University Academic Calendar 2026-2027",
        [
            (
                "FALL SEMESTER 2026",
                [
                    "August 24, 2026 (Monday): Fall semester classes begin.",
                    "September 4, 2026 (Friday): Last day to add a course without "
                    "instructor permission. This is the end of the add/drop period.",
                    "September 7, 2026 (Monday): Labor Day, no classes, offices closed.",
                    "October 12-13, 2026: Fall reading days, no classes.",
                    "October 30, 2026 (Friday): Last day to withdraw from a course with a "
                    "grade of W. Withdrawals after this date require a petition to the "
                    "Dean's office.",
                    "November 25-27, 2026: Thanksgiving recess, no classes.",
                    "December 7, 2026 (Monday): Last day of Fall classes.",
                    "December 9-16, 2026: Fall final examination period.",
                    "December 21, 2026: Fall semester grades due from instructors at 17:00.",
                ],
            ),
            (
                "SPRING SEMESTER 2027",
                [
                    "January 19, 2027 (Tuesday): Spring semester classes begin.",
                    "January 29, 2027 (Friday): End of Spring add/drop period.",
                    "March 15-19, 2027: Spring break, no classes.",
                    "March 26, 2027 (Friday): Last day to withdraw with a grade of W.",
                    "May 3, 2027 (Monday): Last day of Spring classes.",
                    "May 5-12, 2027: Spring final examination period.",
                    "May 22, 2027 (Saturday): Commencement.",
                ],
            ),
            (
                "REGISTRATION AND TUITION DEADLINES",
                [
                    "Priority registration for Spring 2027 opens November 2, 2026 and is "
                    "assigned by earned credit hours.",
                    "Fall 2026 tuition payment deadline: August 14, 2026. A late payment fee "
                    "of $150 applies after this date.",
                    "Tuition refunds: 100% through the end of the add/drop period, 50% "
                    "through the third week of classes, none thereafter.",
                ],
            ),
        ],
    ),
    (
        "handbook/Campus_Student_Handbook.pdf",
        "Campus Student Handbook - Residence Life and Student Services",
        [
            (
                "RESIDENCE HALL POLICIES",
                [
                    "Quiet hours in all residence halls are 22:00-08:00 Sunday through "
                    "Thursday and 24:00-09:00 Friday and Saturday. During final examination "
                    "periods, 24-hour quiet hours are in effect.",
                    "Overnight guests are permitted for a maximum of three consecutive "
                    "nights and must be registered with the residence hall front desk.",
                    "Cooking appliances with exposed heating elements are prohibited in "
                    "student rooms. Microwaves under 1000 watts and mini-fridges under 4.5 "
                    "cubic feet are permitted.",
                ],
            ),
            (
                "STUDENT SERVICES AND HOURS",
                [
                    "The Library is open 07:30-24:00 Monday through Thursday, 07:30-20:00 "
                    "Friday, 10:00-20:00 Saturday, and 12:00-24:00 Sunday.",
                    "The Writing Center offers 45-minute appointments and is located on the "
                    "second floor of the Library. Walk-ins are available 13:00-16:00 on "
                    "weekdays.",
                    "Counseling and Psychological Services: Wellness Building, first floor. "
                    "Same-day crisis appointments are available 09:00-16:00 on weekdays; a "
                    "24/7 crisis line is staffed at all other times.",
                    "The Dining Hall serves breakfast 07:00-10:00, lunch 11:00-14:30, and "
                    "dinner 17:00-20:30 during the academic term.",
                ],
            ),
            (
                "STUDENT CONDUCT AND GRIEVANCES",
                [
                    "Students may appeal a final course grade by submitting a written appeal "
                    "to the department chair within 30 calendar days of the grade posting. "
                    "The appeal must state the grounds and include supporting materials.",
                    "The department chair issues a written decision within 15 business days. "
                    "Decisions may be appealed once more to the Dean of the college.",
                    "Reports of harassment or discrimination should be directed to the Office "
                    "of Equity and Title IX, Administration Building room 120.",
                ],
            ),
            (
                "TECHNOLOGY AND ACCESSIBILITY",
                [
                    "Every enrolled student receives 100 free print pages per semester; "
                    "additional pages are billed at $0.08 per page for black and white.",
                    "Students requiring accommodations must register with the Office of "
                    "Accessibility Services at least two weeks before the accommodation is "
                    "needed. Exam accommodations require notice 10 business days before the "
                    "exam date.",
                ],
            ),
        ],
    ),
]


def build_pdf(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        spaceBefore=14,
        spaceAfter=8,
        fontSize=12,
    )
    body = ParagraphStyle("Body", parent=styles["BodyText"], leading=15, spaceAfter=8)

    story: list = [Paragraph(title, styles["Title"]), Spacer(1, 0.2 * inch)]
    for index, (section_title, paragraphs) in enumerate(sections):
        # Force a page break every other section so the demo index spans
        # several pages and page-level citations are meaningful.
        if index and index % 2 == 0:
            story.append(PageBreak())
        story.append(Paragraph(section_title, heading))
        story.extend(Paragraph(text, body) for text in paragraphs)

    SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title=title,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    ).build(story)


def main() -> int:
    try:
        import reportlab  # noqa: F401
    except ModuleNotFoundError:
        print("reportlab is required: pip install reportlab", file=sys.stderr)
        return 1

    for relative, title, sections in DOCUMENTS:
        target = DATA_DIR / relative
        build_pdf(target, title, sections)
        print(f"wrote {target.relative_to(PROJECT_ROOT)}")

    print("\nNext:\n  python -m syllabusbot.ingest\n  python -m syllabusbot.cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
