"""
Reclassify and Clean Dataset Catalog for RTU AI Academic Assistant
Updates dataset_catalog.csv and dataset_catalog.json with accurate subject mappings,
short codes, RTU subject codes, exam years (normalized <= 2024), and semesters.
"""

import re
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CLEAN_DIR = BASE_DIR / "cleaned_dataset"
CSV_PATH = CLEAN_DIR / "metadata" / "dataset_catalog.csv"
JSON_PATH = CLEAN_DIR / "metadata" / "dataset_catalog.json"

RULES = [
    # Non-CS / Chemical / Petroleum
    (r'chemical-reaction|plant-design|process-equipment|health-safety|petroleum-refinery', 'CH-PE', 'Chemical/Petroleum Engineering Paper', '6CH0-00'),

    # Foundational & Math
    (r'disaster-management', 'DM', 'Disaster Management', '8CS4-01'),
    (r'advance.*eng.*math.*1|mathematics-3|aem-1|aem1|\baem\b', 'AEM', 'Advanced Engineering Mathematics', '3CS2-01'),
    (r'advance.*eng.*math.*2|mathematics-4|aem-2|aem2', 'AEM-2', 'Advanced Engineering Mathematics-II', '4CS2-02'),
    (r'discrete-math|discrete.*structure|\bdms\b', 'DMS', 'Discrete Mathematics Structure', '4CS2-01'),
    (r'technical-communication|\btc\b', 'TC', 'Technical Communication', '4CS1-02'),
    (r'managerial-economics|financial-accounting|\bmefa\b', 'MEFA', 'Managerial Economics and Financial Accounting', '3CS1-03'),

    # Core CS Sem 3 & 4
    (r'data-structure|\bdsa\b', 'DSA', 'Data Structures and Algorithms', '3CS4-05'),
    (r'object-oriented|\boop\b|cpp', 'OOP', 'Object Oriented Programming (C++)', '3CS4-06'),
    (r'digital-electronics|digital-logic-design|\bde\b|\bdld\b', 'DE', 'Digital Electronics', '3CS3-04'),
    (r'electronic-device|electronics-device|edc', 'EDC', 'Electronic Devices and Circuits', '3CS3-05'),
    (r'circuit-analysis|circuit.*synthesis|\bcas\b', 'CAS', 'Circuit Analysis and Synthesis', '3CS3-06'),
    (r'linux|shell-programming', 'LSP', 'Linux and Shell Programming', '3CS4-08'),
    
    (r'data-?base-management|\bdbms\b', 'DBMS', 'Database Management System', '4CS4-05'),
    (r'advance.*data-?base|\badbms\b', 'ADBMS', 'Advance Database Management Systems', '7CS4-11'),
    (r'theory-of-computation|automata|\btoc\b', 'TOC', 'Theory of Computation', '4CS4-06'),
    (r'data-communication.*computer-network|data-communication-and-network|\bdccn\b', 'DCCN', 'Data Communication and Computer Networks', '4CS4-07'),
    (r'principles?-of-communication|fundamental-of-communication|principles-and-communication|analog-and-digital-communication|\bpoc\b', 'POC', 'Principles of Communication', '4CS3-04'),
    (r'microp|micro-p|\bmi\b', 'MI', 'Microprocessor and Interfaces', '5CS3-01'),
    (r'principles?-of-programming-language|\bppl\b', 'PPL', 'Principles of Programming Languages', '4CS4-08'),
    (r'statistics-and-probability|\bspt\b', 'SPT', 'Statistics and Probability Theory', '4CS4-09'),
    (r'software-engineering|\bse\b|software-project-management|software-testing', 'SE', 'Software Engineering', '3CS4-07'),

    # Core CS Sem 5 & 6
    (r'operating-system|\bos\b', 'OS', 'Operating Systems', '5CS4-03'),
    (r'design-and-analysis-of-al[rg]orithms|analysis-of-algorithms?|\bdaa\b|\baoa\b', 'DAA', 'Analysis of Algorithms', '5CS4-05'),
    (r'computer-network|\bcn\b', 'CN', 'Computer Networks', '5CS4-01'),
    (r'compiler-design|compiler-construction|\bcd\b', 'CD', 'Compiler Design', '6CS4-01'),
    (r'artificial-intelligence|intelligent-systems|\bai\b', 'AI', 'Artificial Intelligence', '6CS4-03'),
    (r'computer-architecture|\bca\b|advanced-computer-architecture|\baca\b', 'CA', 'Computer Architecture and Organization', '6CS4-04'),
    (r'machine-learning|\bml\b', 'ML', 'Machine Learning', '7CS4-02'),
    (r'cloud-computing|\bcc\b', 'CC', 'Cloud Computing', '6CS4-05'),
    (r'big-data|\bbda\b', 'BDA', 'Big Data Analytics', '7CS4-01'),
    (r'internet-of-things|\biot\b', 'IOT', 'Internet of Things', '7CS4-04'),
    (r'information-system.*securit|information-security|\biss\b', 'ISS', 'Information System Security', '7CS4-03'),
    (r'digital-image-processing|image-processing|\bdip\b', 'DIP', 'Digital Image Processing', '8CS4-03'),
    (r'distribut(ed|ion)-systems?|\bds\b', 'DS', 'Distributed Systems', '8CS4-02'),
    (r'data-mining|warehousing|\bdmw\b', 'DMW', 'Data Mining and Warehousing', '7CS4-06'),
    (r'computer-graphics|\bcg\b', 'CG', 'Computer Graphics & Multimedia', '6CS4-02'),
    (r'mobile-computing|\bmc\b', 'MC', 'Mobile Computing', '8CS4-04'),
    (r'real-time-systems|\brts\b', 'RTS', 'Real Time Systems', '8CS4-05'),
    (r'java', 'JAVA', 'Programming in Java', '6CS4-06'),
    (r'python', 'PYTHON', 'Python Programming', '4CS4-10'),
    (r'information-theory|\bitc\b', 'ITC', 'Information Theory and Coding', '5CS4-02'),
    (r'management-information-system', 'MIS', 'Management Information Systems', '5CS5-11'),
    (r'wireless-communication', 'WCN', 'Wireless Communication and Networks', '7CS4-12'),
    (r'data-compression', 'DCT', 'Data Compression Techniques', '8CS4-13'),
    (r'e-commerce', 'EC', 'E-Commerce and ERP', '6CS5-12'),
    (r'human-computer|factor-of-human', 'HCI', 'Human Computer Interface', '6CS5-13'),
    (r'vlsi', 'VLSI', 'CAD for VLSI Design', '7CS5-14'),
    (r'erp', 'ERP', 'ERP Systems', '8CS5-15'),
    (r'storage', 'SIM', 'Storage and Information Management', '8CS5-16'),
    (r'logic-and-functional', 'LFP', 'Logic and Functional Programming', '4CS5-17'),
    (r'data-science-with-r|r-programming', 'DSR', 'Data Science with R', '6CS4-04'),
    (r'system-software', 'SSE', 'System Software Engineering', '5CS5-18'),
    (r'telecommunication', 'TCS', 'Telecommunication Fundamentals', '5CS5-19'),
    (r'cyber-security', 'CS', 'Cyber Security', '7CS5-20'),
    (r'natural-language', 'NLP', 'Natural Language Processing', '7CS5-21'),
    (r'deep-learning', 'DL', 'Deep Learning', '8CS4-01'),
    (r'blockchain|block-chain', 'BC', 'Blockchain Technologies', '8CS4-04'),
    (r'embedded-system', 'ESD', 'Embedded System Design', '6CS5-22'),
    (r'digital-signal-processing', 'DSP', 'Digital Signal Processing', '5CS5-23'),
    (r'cyber-forensic|digital-forensic', 'DFIR', 'Digital Forensics and Cyber Crime', '6CS5-24'),
    (r'internet-programming', 'IP', 'Internet Programming', '7CS5-25'),
    (r'indian-constitution', 'IC', 'Indian Constitution', '5FY1-01'),
    (r'quality-management', 'QM', 'Quality Management', '7CS5-26'),
    (r'hardware-testing', 'HTFT', 'Hardware Testing and Fault Tolerance', '8CS5-27')
]

def reclassify_catalog():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} entries from {CSV_PATH}")

    # Reclassify entries from 123 onwards (the 601 newly downloaded ones)
    for idx in range(123, len(df)):
        orig = str(df.at[idx, "original_file"]).lower()
        matched = False
        for pat, short, full, code in RULES:
            if re.search(pat, orig):
                df.at[idx, "subject_short"] = short
                df.at[idx, "subject_name"] = full
                df.at[idx, "subject_code"] = code
                matched = True
                break

        # Normalize year
        year_match = re.search(r'\b(20[0-2]\d)\b', orig)
        if year_match:
            yr = int(year_match.group(1))
            df.at[idx, "year"] = min(yr, 2024)
        else:
            df.at[idx, "year"] = min(int(df.at[idx, "year"]), 2024)

    # Also normalize year for any entries 0..122 if year > 2024
    for idx in range(0, 123):
        if int(df.at[idx, "year"]) > 2024:
            df.at[idx, "year"] = 2024

    # Save to CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved updated catalog to {CSV_PATH}")

    # Save to JSON
    records = df.to_dict(orient="records")
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved updated catalog to {JSON_PATH}")

    print("\nSummary of Subject Counts across 724 Papers:")
    print(df["subject_short"].value_counts())

if __name__ == "__main__":
    reclassify_catalog()
