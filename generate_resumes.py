"""Write 40+ resumes, six job descriptions, and the evaluation labels.

Run from the project root: python generate_resumes.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docx import Document
from fpdf import FPDF

from dataset import JOBS, PEOPLE
from fs_tools import read_file, write_file
from resume_rag import chunk_by_section, extract_metadata

EXTENSIONS = (".txt", ".docx", ".pdf")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def render(person: dict) -> str:
    skills = ", ".join(person["skills"])
    start = 2026 - int(person["years"])
    email = "{0}@email.com".format(person["name"].lower().replace(" ", "."))
    summary = "{0} with {1} years of experience in {2}. {3}".format(
        person["title"],
        person["years"],
        " and ".join(person["skills"][:2]),
        person["highlight"],
    )
    experience = "\n".join(
        [
            "{0} - {1} ({2}-present)".format(person["company"], person["title"], start),
            person["highlight"],
            "Daily work uses {0}.".format(skills),
        ]
    )
    return (
        "{name}\n"
        "{title} | {email}\n"
        "\n"
        "Summary\n"
        "{summary}\n"
        "\n"
        "Skills\n"
        "{skills}\n"
        "\n"
        "Experience\n"
        "{experience}\n"
        "\n"
        "Education\n"
        "{education}\n"
    ).format(
        name=person["name"],
        title=person["title"],
        email=email,
        summary=summary,
        skills=skills,
        experience=experience,
        education=person["education"],
    )


def write_docx(path: Path, text: str) -> None:
    document = Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    document.save(str(path))


def write_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.multi_cell(0, 7, line if line else " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def resume_filename(index: int, person: dict) -> str:
    return "resume_{0}{1}".format(slug(person["name"]), EXTENSIONS[index % 3])


def skill_tokens(skills_text: str) -> set:
    return {part.strip().lower() for part in skills_text.split(",") if part.strip()}


def check_text(person: dict, text: str, resume_path: str) -> None:
    sections = {chunk["section"] for chunk in chunk_by_section(text)}
    missing = {"summary", "skills", "experience", "education"} - sections
    if missing:
        raise RuntimeError(
            "{0} is missing sections after read: {1}\n{2}".format(
                person["name"], ", ".join(sorted(missing)), text
            )
        )
    meta = extract_metadata(text, resume_path)
    if meta["candidate_name"] != person["name"]:
        raise RuntimeError(
            "Name mismatch for {0}: got {1}\n{2}".format(person["name"], meta["candidate_name"], text)
        )
    if meta["experience_years"] != person["years"]:
        raise RuntimeError(
            "Years mismatch for {0}: got {1}".format(person["name"], meta["experience_years"])
        )
    have = skill_tokens(meta["skills"])
    for skill in person["skills"]:
        if skill.lower() not in have:
            raise RuntimeError("Skill {0} missing for {1}. Skills text: {2}".format(skill, person["name"], meta["skills"]))


def main() -> None:
    job_ids = {job["file"].replace(".txt", "") for job in JOBS}
    for person in PEOPLE:
        unknown = [job_id for job_id in person["jobs"] if job_id not in job_ids]
        if unknown:
            raise SystemExit("Unknown job id on {0}: {1}".format(person["name"], ", ".join(unknown)))
    if len(PEOPLE) < 30:
        raise SystemExit("Need at least 30 resumes, found {0}.".format(len(PEOPLE)))
    if len(JOBS) < 5:
        raise SystemExit("Need at least 5 job descriptions, found {0}.".format(len(JOBS)))

    resume_dir = Path("resumes")
    job_dir = Path("jobs")
    resume_dir.mkdir(exist_ok=True)
    job_dir.mkdir(exist_ok=True)

    written = {}
    labels = {job["file"]: [] for job in JOBS}
    keep = set()
    for index, person in enumerate(PEOPLE):
        filename = resume_filename(index, person)
        keep.add(filename)
        text = render(person)
        try:
            text.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise SystemExit("Resume text for {0} is not PDF-safe: {1}".format(person["name"], exc))
        path = resume_dir / filename
        check_text(person, text, "resumes/{0}".format(filename))
        if path.suffix == ".txt":
            result = write_file(str(path), text)
            if not result.get("success"):
                raise RuntimeError(result.get("error"))
        elif path.suffix == ".docx":
            write_docx(path, text)
        else:
            write_pdf(path, text)
        written[filename] = person
        for job_id in person["jobs"]:
            labels["{0}.txt".format(job_id)].append("resumes/{0}".format(filename))
        print("wrote", path.as_posix())

    for old in resume_dir.iterdir():
        if old.is_file() and old.name not in keep:
            old.unlink()
            print("removed", old.as_posix())

    for job in JOBS:
        body = (
            "Title: {title}\n"
            "Must-have skills: {skills}\n"
            "Minimum years: {years}\n"
            "\n"
            "{description}\n"
        ).format(
            title=job["title"],
            skills=", ".join(job["skills"]),
            years=job["min_years"],
            description=job["description"].strip(),
        )
        target = job_dir / job["file"]
        target.write_text(body, encoding="utf-8")
        print("wrote", target.as_posix())

    for filename, paths in labels.items():
        if not paths:
            raise SystemExit("Job {0} has no labeled resumes.".format(filename))

    Path("eval_labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    print("wrote eval_labels.json")

    for filename, person in written.items():
        loaded = read_file(str(resume_dir / filename))
        if not loaded.get("success"):
            raise RuntimeError("Could not re-read {0}: {1}".format(filename, loaded.get("error")))
        check_text(person, loaded["content"], "resumes/{0}".format(filename))

    print("checked {0} resumes and {1} jobs".format(len(written), len(JOBS)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc)
        sys.exit(1)
