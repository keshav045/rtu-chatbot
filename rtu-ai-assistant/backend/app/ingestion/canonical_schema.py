"""
Canonical Schema Definitions for RTU AI Academic Assistant
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class CanonicalPaperRecord(BaseModel):
    paper_id: str = Field(..., description="Unique paper identifier, e.g. RTU_QP_001")
    university: str = Field(default="RTU", description="Name of the university")
    subject: str = Field(..., description="Full subject name")
    subject_code: str = Field(..., description="Official RTU syllabus code, e.g. 3E1202")
    subject_short: str = Field(..., description="Standard abbreviation, e.g. DSA")
    semester: int = Field(..., description="Semester number: 3, 4, or 5")
    year: int = Field(..., description="Examination year, e.g. 2023")
    exam_session: str = Field(default="Main", description="Main, Back, Jan, Feb, May, Dec")
    branch: str = Field(default="CS-IT", description="Engineering branch/curriculum")
    source_pdf: str = Field(..., description="Filename of the clean question paper PDF")
    relative_path: str = Field(..., description="Relative path within cleaned_dataset")
    page_count: int = Field(..., description="Number of pages in the PDF")
    file_size_kb: float = Field(..., description="Size in KB")
    md5_hash: str = Field(..., description="MD5 hash checksum")

class CanonicalQuestionRecord(BaseModel):
    question_id: str = Field(..., description="Unique question identifier, e.g. RTU_DSA_2022_Q3A")
    paper_id: str = Field(..., description="Parent paper ID")
    university: str = Field(default="RTU", description="University name")
    subject: str = Field(..., description="Subject name")
    subject_code: str = Field(..., description="Syllabus course code, e.g. 3E1202")
    subject_short: str = Field(..., description="Subject abbreviation, e.g. DSA")
    branch: str = Field(default="CS-IT", description="Branch/curriculum")
    semester: int = Field(..., description="Semester number: 3, 4, 5")
    year: int = Field(..., description="Examination year")
    section: str = Field(..., description="Part A, Part B, Part C, or General")
    unit: Optional[str] = Field(default=None, description="Syllabus unit if identified, e.g. Unit 3")
    question_number: str = Field(..., description="Question number label, e.g. Q1, Q2(a), 3(a)")
    marks: int = Field(default=2, description="Marks allocated: 2, 8, 15, etc.")
    question_text: str = Field(..., description="Normalized question text without watermark junk")
    source_pdf: str = Field(..., description="Source PDF filename")
    page_number: int = Field(default=1, description="1-indexed physical page where question appears")

class IngestionManifest(BaseModel):
    manifest_version: str = "1.0.0"
    created_at: str
    total_papers_cataloged: int
    total_papers_with_extracted_questions: int
    total_questions_ingested: int
    subject_breakdown: Dict[str, int]
    semester_breakdown: Dict[str, int]
    year_span: str
    database_path: str
    validation_status: str
    validation_errors: List[str]

