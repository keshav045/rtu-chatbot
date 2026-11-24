"""
Authoritative SQLAlchemy Database Models for RTU Exam Helper Chatbot
Implements Section 5 of Implementation Plan Version 2.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, Index, Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class University(Base):
    __tablename__ = "universities"

    id = Column(String(50), primary_key=True)
    name = Column(String(150), nullable=False)

    schemes = relationship("Scheme", back_populates="university", cascade="all, delete-orphan")

class Scheme(Base):
    __tablename__ = "schemes"

    id = Column(String(50), primary_key=True)
    university_id = Column(String(50), ForeignKey("universities.id"), nullable=False)
    name = Column(String(100), nullable=False)
    year_start = Column(Integer, default=2018)

    university = relationship("University", back_populates="schemes")
    subject_offerings = relationship("SubjectOffering", back_populates="scheme")

class Branch(Base):
    __tablename__ = "branches"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    code = Column(String(20), nullable=False, unique=True)  # CSE, IT, ECE, EE, ME, CE

    offerings = relationship("SubjectOffering", back_populates="branch")
    users = relationship("User", back_populates="branch")

class Subject(Base):
    __tablename__ = "subjects"

    id = Column(String(50), primary_key=True)
    code = Column(String(50), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    short_name = Column(String(20), nullable=False, index=True)
    data_tier = Column(String(10), default="Tier C")  # Tier A, Tier B, Tier C

    offerings = relationship("SubjectOffering", back_populates="subject", cascade="all, delete-orphan")
    units = relationship("Unit", back_populates="subject", cascade="all, delete-orphan")
    papers = relationship("Paper", back_populates="subject", cascade="all, delete-orphan")
    uploads = relationship("Upload", back_populates="subject")

class SubjectOffering(Base):
    __tablename__ = "subject_offerings"

    id = Column(String(50), primary_key=True)
    subject_id = Column(String(50), ForeignKey("subjects.id"), nullable=False)
    branch_id = Column(String(50), ForeignKey("branches.id"), nullable=False)
    semester = Column(Integer, nullable=False, index=True)
    scheme_id = Column(String(50), ForeignKey("schemes.id"), nullable=False)

    subject = relationship("Subject", back_populates="offerings")
    branch = relationship("Branch", back_populates="offerings")
    scheme = relationship("Scheme", back_populates="subject_offerings")

class Unit(Base):
    __tablename__ = "units"

    id = Column(String(50), primary_key=True)
    subject_id = Column(String(50), ForeignKey("subjects.id"), nullable=False, index=True)
    scheme_id = Column(String(50), default="SCHEME_RTU_2018")
    unit_no = Column(Integer, nullable=False)  # 1 to 5
    title = Column(String(150), nullable=False)

    subject = relationship("Subject", back_populates="units")
    topics = relationship("Topic", back_populates="unit", cascade="all, delete-orphan")

class Topic(Base):
    __tablename__ = "topics"

    id = Column(String(50), primary_key=True)
    unit_id = Column(String(50), ForeignKey("units.id"), nullable=False, index=True)
    name = Column(String(150), nullable=False, index=True)
    aliases = Column(Text, default="")  # Comma-separated or JSON string
    est_study_minutes = Column(Integer, default=45)

    unit = relationship("Unit", back_populates="topics")
    questions = relationship("Question", back_populates="topic")
    chunks = relationship("Chunk", back_populates="topic", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="topic")

class Paper(Base):
    __tablename__ = "papers"

    id = Column(String(50), primary_key=True)
    subject_id = Column(String(50), ForeignKey("subjects.id"), nullable=False, index=True)
    scheme_id = Column(String(50), default="SCHEME_RTU_2018")
    year = Column(Integer, nullable=False, index=True)
    session = Column(String(50), default="Main")
    source = Column(String(20), default="admin")  # admin | upload
    source_pdf = Column(String(255), nullable=False)
    relative_path = Column(String(255), nullable=False)
    page_count = Column(Integer, default=2)
    file_size_kb = Column(Float, default=0.0)
    file_hash = Column(String(64), nullable=True, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    subject = relationship("Subject", back_populates="papers")
    questions = relationship("Question", back_populates="paper", cascade="all, delete-orphan")

class Question(Base):
    __tablename__ = "questions"

    id = Column(String(100), primary_key=True)
    paper_id = Column(String(50), ForeignKey("papers.id"), nullable=False, index=True)
    topic_id = Column(String(50), ForeignKey("topics.id"), nullable=True, index=True)
    text = Column(Text, nullable=False)
    marks = Column(Integer, default=2)
    part = Column(String(20), default="Part A")  # Part A, Part B, Part C
    tag_confidence = Column(Float, default=0.85)
    status = Column(String(20), default="approved")  # approved | pending | rejected
    page_number = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="questions")
    topic = relationship("Topic", back_populates="questions")
    citations = relationship("Citation", back_populates="question")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(100), primary_key=True)
    topic_id = Column(String(50), ForeignKey("topics.id"), nullable=True, index=True)
    source = Column(String(100), default="syllabus")  # syllabus | notes
    content = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=True)  # JSON representation of embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)

    topic = relationship("Topic", back_populates="chunks")

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), nullable=True)
    file_hash = Column(String(64), nullable=False, unique=True)
    status = Column(String(20), default="queued")  # queued | processing | approved | rejected | review_queue
    subject_id = Column(String(50), ForeignKey("subjects.id"), nullable=True)
    year = Column(Integer, nullable=True)
    session = Column(String(50), default="Main")
    confidence = Column(Float, default=0.0)
    reviewer_id = Column(String(50), nullable=True)
    filename = Column(String(255), nullable=False)
    saved_path = Column(String(255), nullable=False)
    extracted_questions_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    subject = relationship("Subject", back_populates="uploads")

class User(Base):
    __tablename__ = "users"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True, index=True)
    roll_no = Column(String(50), nullable=True, index=True)
    password_hash = Column(String(100), nullable=True)
    college = Column(String(150), default="Rajasthan Technical University (RTU)")
    branch_id = Column(String(50), ForeignKey("branches.id"), nullable=True)
    year = Column(Integer, default=2)  # 1 to 4
    language_pref = Column(String(20), default="english")  # english | hinglish
    created_at = Column(DateTime, default=datetime.utcnow)

    branch = relationship("Branch", back_populates="users")
    sessions = relationship("SessionModel", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("StudentMemory", back_populates="user", cascade="all, delete-orphan")
    topic_masteries = relationship("TopicMastery", back_populates="user", cascade="all, delete-orphan")

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=True)
    subject_id = Column(String(50), ForeignKey("subjects.id"), nullable=True)
    year = Column(Integer, default=2)
    branch = Column(String(20), default="CSE")
    hours_left = Column(Float, default=5.0)
    mode = Column(String(30), default="explain_topic")  # crash_course | explain_topic | practice_quiz | important_questions
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")
    quiz_attempts = relationship("QuizAttempt", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(50), primary_key=True)
    session_id = Column(String(50), ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="messages")
    citations = relationship("Citation", back_populates="message", cascade="all, delete-orphan")

class Citation(Base):
    __tablename__ = "citations"

    id = Column(String(50), primary_key=True)
    message_id = Column(String(50), ForeignKey("messages.id"), nullable=False, index=True)
    question_id = Column(String(100), ForeignKey("questions.id"), nullable=False, index=True)
    source_pdf = Column(String(255), nullable=False)
    page_number = Column(Integer, default=1)

    message = relationship("Message", back_populates="citations")
    question = relationship("Question", back_populates="citations")

class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(String(50), primary_key=True)
    session_id = Column(String(50), ForeignKey("sessions.id"), nullable=False, index=True)
    topic_id = Column(String(50), ForeignKey("topics.id"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    score = Column(Float, default=0.0)  # 0 to 10
    feedback = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="quiz_attempts")
    topic = relationship("Topic", back_populates="quiz_attempts")

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(String(50), primary_key=True)
    session_id = Column(String(50), nullable=True)
    message_id = Column(String(50), nullable=True)
    rating = Column(String(10), nullable=False)  # up | down
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class StudentMemory(Base):
    """
    Long-Term Memory store for student facts, preferences, study goals,
    and cross-session episodic summaries.
    """
    __tablename__ = "student_memories"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    category = Column(String(50), default="academic_fact")  # academic_fact | study_goal | preference | episodic_summary
    content = Column(Text, nullable=False)
    importance_weight = Column(Float, default=1.0)  # 1.0 = standard, 2.0 = critical exam deadline/priority
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")

class TopicMastery(Base):
    """
    Tracks topic-level student competency, weak spots, practice frequency,
    and mastery scores across sessions.
    """
    __tablename__ = "topic_mastery"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    subject_code = Column(String(50), nullable=False, index=True)  # e.g. "3CS4-05", "DBMS", "OS"
    topic_name = Column(String(150), nullable=False, index=True)
    confidence_score = Column(Float, default=0.5)  # 0.0 (struggling) to 1.0 (mastered)
    attempts_count = Column(Integer, default=1)
    is_weak_area = Column(Boolean, default=False, index=True)
    notes = Column(Text, nullable=True)  # Specific struggle, e.g. "Confused by B-tree splitting"
    last_studied_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="topic_masteries")
