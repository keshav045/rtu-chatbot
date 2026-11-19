"""
Seed Full 4-Year RTU Curriculum (Year 1, Year 2, Year 3, Year 4)
Ensures exact semester, branch, and year mappings for all engineering subjects.
"""

import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal
from app.database.models import Subject, SubjectOffering, Branch, Scheme

CURRICULUM_DATA = [
    # --- YEAR 1 (Common First Year across engineering branches) ---
    {
        "id": "SUBJ_EM1", "code": "1FY2-01", "short_name": "EM-1",
        "name": "Engineering Mathematics-I", "semester": 1, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_EM2", "code": "2FY2-01", "short_name": "EM-2",
        "name": "Engineering Mathematics-II", "semester": 2, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_PHY", "code": "1FY2-02", "short_name": "PHY",
        "name": "Engineering Physics", "semester": 1, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_CHEM", "code": "2FY2-03", "short_name": "CHEM",
        "name": "Engineering Chemistry", "semester": 2, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_BEE", "code": "1FY3-08", "short_name": "BEE",
        "name": "Basic Electrical Engineering", "semester": 1, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_BCE", "code": "1FY3-09", "short_name": "BCE",
        "name": "Basic Civil Engineering", "semester": 2, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_BME", "code": "2FY3-07", "short_name": "BME",
        "name": "Basic Mechanical Engineering", "semester": 2, "tier": "Tier B",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_CS_ENG", "code": "1FY1-04", "short_name": "CS-ENG",
        "name": "Communication Skills", "semester": 1, "tier": "Tier C",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },
    {
        "id": "SUBJ_HV", "code": "2FY1-05", "short_name": "HV",
        "name": "Human Values & Professional Ethics", "semester": 2, "tier": "Tier C",
        "branches": ["CSE", "IT", "ECE", "EE", "ME", "CE"]
    },

    # --- YEAR 2 (Semesters 3 & 4) ---
    {
        "id": "SUBJ_AEM", "code": "3CS2-01", "short_name": "AEM",
        "name": "Advanced Engineering Mathematics", "semester": 3, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DSA", "code": "3CS4-05", "short_name": "DSA",
        "name": "Data Structures and Algorithms", "semester": 3, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DE", "code": "3CS4-04", "short_name": "DE",
        "name": "Digital Electronics", "semester": 3, "tier": "Tier A",
        "branches": ["CSE", "IT", "ECE", "EE"]
    },
    {
        "id": "SUBJ_OOP", "code": "3CS4-06", "short_name": "OOP",
        "name": "Object Oriented Programming (C++)", "semester": 3, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_SE", "code": "3CS4-07", "short_name": "SE",
        "name": "Software Engineering", "semester": 3, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_MEFA", "code": "3CS1-03", "short_name": "MEFA",
        "name": "Managerial Economics and Financial Accounting", "semester": 3, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DMS", "code": "4CS2-01", "short_name": "DMS",
        "name": "Discrete Mathematics Structure", "semester": 4, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DBMS", "code": "4CS4-05", "short_name": "DBMS",
        "name": "Database Management System", "semester": 4, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_TOC", "code": "4CS4-06", "short_name": "TOC",
        "name": "Theory of Computation", "semester": 4, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DCCN", "code": "4CS4-07", "short_name": "DCCN",
        "name": "Data Communication and Computer Networks", "semester": 4, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_POC", "code": "4CS4-04", "short_name": "POC",
        "name": "Principles of Communication", "semester": 4, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_TC", "code": "4CS1-03", "short_name": "TC",
        "name": "Technical Communication", "semester": 4, "tier": "Tier C",
        "branches": ["CSE", "IT"]
    },

    # --- YEAR 3 (Semesters 5 & 6) ---
    {
        "id": "SUBJ_OS", "code": "5CS4-04", "short_name": "OS",
        "name": "Operating Systems", "semester": 5, "tier": "Tier A",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_AOA", "code": "5CS4-02", "short_name": "AOA",
        "name": "Analysis of Algorithms", "semester": 5, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_CN", "code": "5CS4-05", "short_name": "CN",
        "name": "Computer Networks", "semester": 5, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_MI", "code": "5CS4-03", "short_name": "MI",
        "name": "Microprocessor and Interfaces", "semester": 5, "tier": "Tier A",
        "branches": ["CSE", "IT", "ECE", "EE"]
    },
    {
        "id": "SUBJ_ITC", "code": "5CS4-01", "short_name": "ITC",
        "name": "Information Theory and Coding", "semester": 5, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_CD", "code": "6CS4-01", "short_name": "CD",
        "name": "Compiler Design", "semester": 6, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_CG", "code": "6CS4-02", "short_name": "CG",
        "name": "Computer Graphics & Multimedia", "semester": 6, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_AI", "code": "6CS4-03", "short_name": "AI",
        "name": "Artificial Intelligence", "semester": 6, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DSR", "code": "6CS4-04", "short_name": "DSR",
        "name": "Data Science with R", "semester": 6, "tier": "Tier C",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_CC", "code": "6CS4-05", "short_name": "CC",
        "name": "Cloud Computing", "semester": 6, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },

    # --- YEAR 4 (Semesters 7 & 8) ---
    {
        "id": "SUBJ_BDA", "code": "7CS4-01", "short_name": "BDA",
        "name": "Big Data Analytics", "semester": 7, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_ML", "code": "7CS4-02", "short_name": "ML",
        "name": "Machine Learning", "semester": 7, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_ISS", "code": "7CS4-03", "short_name": "ISS",
        "name": "Information System Security", "semester": 7, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_IOT", "code": "7CS4-04", "short_name": "IOT",
        "name": "Internet of Things", "semester": 7, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DL", "code": "8CS4-01", "short_name": "DL",
        "name": "Deep Learning", "semester": 8, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DS", "code": "8CS4-02", "short_name": "DS",
        "name": "Distributed Systems", "semester": 8, "tier": "Tier B",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_DIP", "code": "8CS4-03", "short_name": "DIP",
        "name": "Digital Image Processing", "semester": 8, "tier": "Tier C",
        "branches": ["CSE", "IT"]
    },
    {
        "id": "SUBJ_BC", "code": "8CS4-04", "short_name": "BC",
        "name": "Blockchain Technologies", "semester": 8, "tier": "Tier C",
        "branches": ["CSE", "IT"]
    },
]

def seed_curriculum():
    db = SessionLocal()
    try:
        scheme = db.query(Scheme).first()
        scheme_id = scheme.id if scheme else "SCHEME_RTU_2018"

        branch_map = {b.code: b.id for b in db.query(Branch).all()}

        added_subjects = 0
        updated_offerings = 0

        for item in CURRICULUM_DATA:
            # Check if subject exists by short_name or code
            subj = db.query(Subject).filter(
                (Subject.short_name == item["short_name"]) |
                (Subject.code == item["code"])
            ).first()

            if not subj:
                subj = Subject(
                    id=item["id"],
                    code=item["code"],
                    name=item["name"],
                    short_name=item["short_name"],
                    data_tier=item["tier"]
                )
                db.add(subj)
                db.flush()
                added_subjects += 1
            else:
                # Update tier or name if needed
                subj.name = item["name"]

            # Ensure offerings for each branch
            for b_code in item["branches"]:
                b_id = branch_map.get(b_code)
                if not b_id:
                    continue

                existing_offering = db.query(SubjectOffering).filter(
                    SubjectOffering.subject_id == subj.id,
                    SubjectOffering.branch_id == b_id,
                    SubjectOffering.semester == item["semester"]
                ).first()

                if not existing_offering:
                    offering_id = f"OFF_{subj.id}_{b_code}_S{item['semester']}"
                    offering = SubjectOffering(
                        id=offering_id,
                        subject_id=subj.id,
                        branch_id=b_id,
                        semester=item["semester"],
                        scheme_id=scheme_id
                    )
                    db.add(offering)
                    updated_offerings += 1

        db.commit()
        print(f"Successfully seeded curriculum: {added_subjects} new subjects, {updated_offerings} new offerings.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding curriculum: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_curriculum()

