"""
Idempotent Additive Syllabus Seeder for RTU AI Academic Assistant.
Parses all 5 official RTU syllabus PDFs (First Year + CSE/IT + ECE + EE + ME + CE),
populates missing Unit, Topic (with keyword aliases), Subject, and SubjectOffering rows
for subjects that currently have zero topics, tags previously untagged Questions where
appropriate, and preserves all existing subjects/units/topics/tiers untouched.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import fitz

BACKEND_DIR = Path(__file__).resolve().parent.parent
SYLLABUS_DIR = BACKEND_DIR.parent / "syllabus"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal
from app.database.models import Subject, SubjectOffering, Unit, Topic, Branch, Scheme, Question, Paper


PDF_BRANCH_MAP = {
    "Syllabus_2018onwards.pdf": ["CSE", "IT"],
    "BTech_EC_2018_onwards.pdf": ["ECE"],
    "BTech_EE_2018.pdf": ["EE"],
    "BTech_ME_2018_onwards.pdf": ["ME"],
    "BTech_CE_2018_onwards.pdf": ["CE"],
}

# Canonical matching overrides from normalized PDF subject title / code -> existing DB short_name
TITLE_TO_EXISTING_SHORT = {
    "engineering mathematics-i": "EM-1",
    "engineering mathematics i": "EM-1",
    "engineering mathematics-ii": "EM-2",
    "engineering mathematics ii": "EM-2",
    "engineering physics": "PHY",
    "engineering chemistry": "CHEM",
    "communication skills": "CS-ENG",
    "human values": "HV",
    "programming for problem solving": "PPS",
    "basic mechanical engineering": "BME",
    "basic electrical engineering": "BEE",
    "basic civil engineering": "BCE",
    "advanced engineering mathematics": "AEM",
    "advance engineering mathematics-i": "AEM",
    "advance engineering mathematics-ii": "AEM-2",
    "advanced engineering mathematics-ii": "AEM-2",
    "digital electronics": "DE",
    "data structures and algorithms": "DSA",
    "object oriented programming": "OOP",
    "software engineering": "SE",
    "managerial economics and financial accounting": "MEFA",
    "technical communication": "TC",
    "discrete mathematics structure": "DMS",
    "microprocessor and interfaces": "MI",
    "compiler design": "CD",
    "operating system": "OS",
    "operating systems": "OS",
    "computer graphics & multimedia": "CG",
    "analysis of algorithms": "AOA",
    "computer network": "CN",
    "computer networks": "CN",
    "data communication and computer networks": "DCCN",
    "principles of communication": "POC",
    "digital image processing": "DIP",
    "machine learning": "ML",
    "information security system": "ISS",
    "information system security": "ISS",
    "artificial intelligence": "AI",
    "distributed system": "DS",
    "distributed systems": "DS",
    "information theory & coding": "ITC",
    "information theory and coding": "ITC",
    "cloud computing": "CC",
    "ecommerce & erp": "EC",
    "e-commerce and erp": "EC",
    "big data analytics": "BDA",
    "internet of things": "IOT",
    "computer architecture": "CA",
    "digital signal processing": "DSP",
    "embedded systems": "ESD",
    "embedded system design": "ESD",
    "wireless communication": "WCN",
}

# Supplementary CS/IT topics for cataloged subjects not parsed verbatim from PDF headers
SUPPLEMENTARY_SYLLABUS: Dict[str, List[Tuple[int, str, List[Tuple[str, str]]]]] = {
    "PHY": [
        (1, "Unit 1: Wave Optics, Interference & Diffraction", [
            ("Wave Optics: Newton's Rings, Michelson's Interferometer & Coherence", "wave optics, newton's rings, newtons rings, interference, michelson interferometer, thin film interference, diffraction, fraunhofer diffraction, single slit, diffraction grating, rayleigh criterion, resolving power"),
        ]),
    ],
    "SE": [
        (1, "Unit 1: Software Process Models & SDLC", [
            ("SDLC, Waterfall, Spiral, Incremental & Agile Models", "sdlc, waterfall, spiral model, prototyping, agile, scrum, extreme programming, software process"),
            ("Requirement Engineering, SRS & Feasibility Study", "requirement engineering, srs, software requirement specification, functional requirement, non functional, dfd, feasibility"),
        ]),
        (2, "Unit 2: Software Design, Architecture & Coupling/Cohesion", [
            ("Modularity, Cohesion, Coupling & DFD Design", "software design, cohesion, coupling, data flow diagram, dfd, modularity, structure chart"),
            ("Object-Oriented Design & UML Diagrams", "uml, class diagram, use case, sequence diagram, activity diagram, architecture design"),
        ]),
        (3, "Unit 3: Software Testing & Quality Assurance", [
            ("Black-Box vs White-Box Testing & Cyclomatic Complexity", "software testing, black box, white box, boundary value, equivalence partitioning, cyclomatic complexity, unit testing, integration testing"),
            ("Verification, Validation & Software Quality Assurance", "verification, validation, sqa, iso 9001, cmm, software quality, debugging"),
        ]),
        (4, "Unit 4: Software Metrics & Project Management", [
            ("COCOMO Cost Estimation & Function Point Analysis", "cocomo, effort estimation, function point, size metrics, project scheduling, gantt, pert, risk management"),
            ("Software Maintenance, Re-engineering & Configuration Management", "software maintenance, reverse engineering, re-engineering, configuration management, version control"),
        ]),
    ],
    "AEM-2": [
        (1, "Unit 1: Laplace & Fourier Transforms", [
            ("Laplace Transform, Inverse Laplace & Differential Equations", "laplace transform, inverse laplace, convolution theorem, differential equation, aem-2"),
            ("Fourier Series, Fourier Transform & Z-Transform", "fourier transform, fourier integral, z-transform, half range"),
        ]),
        (2, "Unit 2: Partial Differential Equations & Complex Variables", [
            ("Partial Differential Equations: Heat, Wave & Laplace", "partial differential equation, pde, charpit, wave equation, heat equation"),
            ("Complex Analysis: Analytic Functions & Cauchy-Riemann", "analytic function, cauchy riemann, residue theorem, contour integration"),
        ]),
    ],
    "BC": [
        (1, "Unit 1: Blockchain Fundamentals & Cryptography", [
            ("Distributed Ledger, Cryptographic Hash & Merkle Trees", "blockchain, distributed ledger, merkle tree, hash pointer, sha-256, public key cryptography, consensus"),
            ("Proof of Work (PoW), Proof of Stake (PoS) & Byzantine Fault Tolerance", "proof of work, pow, proof of stake, pos, pbft, mining, double spending"),
        ]),
        (2, "Unit 2: Smart Contracts, Ethereum & Hyperledger", [
            ("Ethereum Architecture, Solidity Smart Contracts & EVM", "ethereum, smart contract, solidity, evm, gas limit, dapp, web3"),
            ("Hyperledger Fabric, Permissioned Blockchains & Applications", "hyperledger fabric, permissioned blockchain, chaincode, supply chain blockchain"),
        ]),
    ],
    "HCI": [
        (1, "Unit 1: Human-Computer Interaction & Usability", [
            ("Usability Principles, Norman's Design & Nielsen Heuristics", "hci, human computer interaction, usability, heuristic evaluation, norman principles, affordance, user interface"),
            ("Cognitive Models, Fitts' Law & Interaction Design", "fitts law, goms, cognitive model, prototyping, wireframing, user experience"),
        ]),
    ],
    "CS": [
        (1, "Unit 1: Cyber Security & Threat Landscape", [
            ("Cyber Attacks, Malware, Phishing & OWASP Vulnerabilities", "cyber security, malware, ransomware, phishing, sql injection, xss, dos, ddos, owasp, zero day"),
            ("Authentication, IDS/IPS & Incident Response", "cyber defense, ids, ips, firewall, siem, incident response, vulnerability assessment"),
        ]),
    ],
    "DFIR": [
        (1, "Unit 1: Digital Forensics & Incident Investigation", [
            ("Chain of Custody, Disk Imaging & File System Forensics", "digital forensics, chain of custody, disk imaging, evidence acquisition, deleted file recovery"),
            ("Network Forensics, Memory Analysis & Log Investigation", "network forensics, memory dump, volatility, packet capture, cyber crime investigation"),
        ]),
    ],
    "DCT": [
        (1, "Unit 1: Lossless & Lossy Data Compression", [
            ("Huffman Coding, Arithmetic Coding & LZW Compression", "data compression, huffman coding, arithmetic coding, dictionary coding, lzw, entropy"),
            ("Transform Coding, JPEG & MPEG Compression", "jpeg compression, dct, discrete cosine transform, quantization, mpeg, run length encoding"),
        ]),
    ],
    "RTS": [
        (1, "Unit 1: Real-Time Task Scheduling & Kernels", [
            ("Hard vs Soft Real-Time Systems & Periodic Task Model", "real time system, hard real time, soft real time, deadline, jitter, periodic task"),
            ("Rate Monotonic (RMS) & Earliest Deadline First (EDF) Scheduling", "rate monotonic, rms, earliest deadline first, edf, priority inversion, priority inheritance, liu layland"),
        ]),
    ],
    "MC": [
        (1, "Unit 1: Cellular Architecture & Mobile Computing", [
            ("Frequency Reuse, Handoff & GSM Architecture", "mobile computing, frequency reuse, cell splitting, handoff, handover, gsm, hlr, vlr, bts, msc"),
            ("Mobile IP, CDMA, LTE & Wireless Protocols", "mobile ip, cdma, tdma, fdma, gprs, lte, agent discovery, tunneling"),
        ]),
    ],
    "PPL": [
        (1, "Unit 1: Programming Language Paradigms & Semantics", [
            ("Syntax, Operational Semantics, Scope & Binding", "programming languages, lexical scope, dynamic scope, binding, activation record, parameter passing"),
            ("Functional, Logic & Concurrent Programming Paradigms", "functional programming, lambda calculus, logic programming, prolog, polymorphism, type checking"),
        ]),
    ],
    "DMW": [
        (1, "Unit 1: Data Warehousing, OLAP & Data Mining", [
            ("Data Warehouse Architecture, Star/Snowflake Schema & OLAP", "data warehouse, olap, star schema, snowflake schema, etl, fact table, rollup, drill down"),
            ("Association Rules (Apriori), Clustering & Classification", "data mining, apriori algorithm, association rule, frequent itemset, k-means, decision tree"),
        ]),
    ],
    "IP": [
        (1, "Unit 1: Full-Stack Web Protocols & Server Programming", [
            ("HTTP, HTML5, CSS, JavaScript DOM & REST APIs", "internet programming, http, dom, javascript, rest api, json, session, cookies, mvc"),
            ("Server-Side Scripting, Servlets, JSP & Database Connectivity", "servlet, jsp, web server, jdbc, cgi, backend request response"),
        ]),
    ],
    "CAS": [
        (1, "Unit 1: Network Theorems & Transient Circuit Analysis", [
            ("Thevenin, Norton, Superposition & Maximum Power Transfer", "circuit analysis, thevenin, norton, superposition theorem, mesh analysis, nodal analysis"),
            ("Laplace Transient Analysis, Two-Port Networks & Resonance", "transient response, rl circuit, rc circuit, rlc, two port network, z parameters, y parameters, resonance"),
        ]),
    ],
    "EDC": [
        (1, "Unit 1: Semiconductor Physics, Diodes & Transistors", [
            ("PN Junction Diode, Zener Diode & Rectifier Circuits", "electronic devices, pn junction, diode, zener regulator, rectifier, clipper, clamper"),
            ("BJT, MOSFET, JFET Characteristics & Biasing", "bjt, transistor, common emitter, mosfet, jfet, load line, biasing, amplification"),
        ]),
    ],
    "SPT": [
        (1, "Unit 1: Probability Distributions & Stochastic Processes", [
            ("Bayes Theorem, Random Variables & Probability Distributions", "probability, bayes theorem, random variable, binomial, poisson, normal distribution, expectation, variance"),
            ("Correlation, Regression, Hypothesis Testing & Markov Chains", "correlation, regression, chi square, t-test, hypothesis testing, markov chain, stochastic process"),
        ]),
    ],
}


def parse_pdf_syllabus(pdf_path: Path) -> List[Dict]:
    """Extracts structured syllabus subjects & unit paragraphs from an RTU syllabus PDF."""
    if not pdf_path.exists():
        return []

    doc = fitz.open(str(pdf_path))
    found = []
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1], 1), round(b[0], 1)))
        full_txt = "\n".join(b[4].strip() for b in blocks if b[4].strip())

        m = re.search(
            r"(\d[A-Z]{2,3}\d-\d{2}(?:\s*/\s*\d[A-Z]{2,3}\d-\d{2})?)\s*:\s*([^\n\r]+)",
            full_txt,
        )
        if not m:
            continue

        code = re.sub(r"\s+", " ", m.group(1)).strip()
        title = re.sub(r"\s+", " ", m.group(2)).strip()
        if (
            any(k in title.lower() for k in ["lab", "practical", "seminar", "project", "activities", "workshop", "training"])
            and "management" not in title.lower()
        ):
            continue

        units: List[Tuple[int, str]] = []
        pending_no: Optional[int] = None
        for x0, _y0, _x1, _y1, txt, *_ in blocks:
            s = re.sub(r"\s+", " ", txt).strip()
            if not s or x0 > 480:
                continue
            if re.fullmatch(r"[1-6]", s):
                pending_no = int(s)
                continue
            um = re.match(r"^([1-6])\s+([A-Za-z].*)", s)
            if um:
                uno, utxt = int(um.group(1)), um.group(2).strip()
                if not any(k in utxt.lower() for k in ["credit:", "max. marks", "end term exam"]):
                    units.append((uno, utxt))
                pending_no = None
            elif pending_no is not None and not re.fullmatch(r"\d+", s):
                if "objective, scope" not in s.lower() and not s.lower().startswith("total"):
                    units.append((pending_no, s))
                pending_no = None

        clean_units: List[Tuple[int, str]] = []
        for uno, utxt in units:
            utxt = re.sub(r"\s+\d{1,2}\s*$", "", utxt).strip()
            if len(utxt) >= 15:
                clean_units.append((uno, utxt))
        if clean_units:
            found.append({"code": code.split("/")[0].strip(), "name": title, "units": clean_units})

    doc.close()
    return found


def _make_short_code(name: str, code: str) -> str:
    clean = re.sub(r"\([^)]*\)", "", name).strip()
    words = [w for w in re.findall(r"[A-Za-z0-9]+", clean) if w.lower() not in {"of", "and", "in", "for", "the", "to", "with"}]
    if len(words) == 1:
        return words[0][:6].upper()
    abbr = "".join(w[0].upper() for w in words[:4])
    if len(abbr) >= 2:
        return f"{abbr}_{code[:3]}" if len(abbr) <= 3 and code else abbr
    return code.replace("-", "")[:8].upper()


def _split_unit_into_topics(unit_no: int, unit_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Converts a dense RTU syllabus unit paragraph into clean (unit_title, [(topic_name, aliases), ...])."""
    cleaned = re.sub(r"\s+", " ", unit_text).strip()
    if ":" in cleaned[:85]:
        head, tail = cleaned.split(":", 1)
        unit_title = f"Unit {unit_no}: {head.strip()[:80]}"
        raw_parts = [p.strip() for p in re.split(r"[.;]", tail) if len(p.strip()) > 6]
    else:
        head_words = " ".join(cleaned.split()[:6]).rstrip(",;:")
        unit_title = f"Unit {unit_no}: {head_words[:80]}"
        raw_parts = [p.strip() for p in re.split(r"[.;]", cleaned) if len(p.strip()) > 6]

    if not raw_parts:
        raw_parts = [cleaned[:120]]

    # Group overly granular parts into 1-3 rich syllabus topics per unit
    topics: List[Tuple[str, str]] = []
    chunk: List[str] = []
    for part in raw_parts:
        chunk.append(part)
        if len(chunk) >= 2 or len(", ".join(chunk)) >= 75:
            joined = "; ".join(chunk)
            t_name = joined[:110].rstrip(" ,;:")
            kw_tokens = [k.strip().lower() for k in re.split(r"[,:;()/]+", joined) if len(k.strip()) >= 3]
            aliases = ", ".join(dict.fromkeys(kw_tokens))[:250]
            topics.append((t_name, aliases))
            chunk = []
    if chunk:
        joined = "; ".join(chunk)
        t_name = joined[:110].rstrip(" ,;:")
        kw_tokens = [k.strip().lower() for k in re.split(r"[,:;()/]+", joined) if len(k.strip()) >= 3]
        aliases = ", ".join(dict.fromkeys(kw_tokens))[:250]
        topics.append((t_name, aliases))

    return unit_title, topics[:3]


def _semester_from_code(code: str) -> int:
    m = re.match(r"^([1-8])", code.strip())
    return int(m.group(1)) if m else 3


def run_seed() -> None:
    db = SessionLocal()
    try:
        scheme = db.query(Scheme).first()
        scheme_id = scheme.id if scheme else "SCHEME_RTU_2018"
        branch_map = {b.code: b.id for b in db.query(Branch).all()}

        existing_subjects = db.query(Subject).all()
        by_short = {s.short_name.upper(): s for s in existing_subjects if s.short_name}
        by_code = {s.code.upper(): s for s in existing_subjects if s.code}
        by_name = {re.sub(r"\s+", " ", s.name.strip().lower()): s for s in existing_subjects if s.name}

        # Count topics per existing subject_id
        existing_topic_counts: Dict[str, int] = {}
        for u in db.query(Unit).all():
            cnt = db.query(Topic).filter(Topic.unit_id == u.id).count()
            existing_topic_counts[u.subject_id] = existing_topic_counts.get(u.subject_id, 0) + cnt

        added_subjects = 0
        added_offerings = 0
        added_units = 0
        added_topics = 0

        for pdf_name, branches in PDF_BRANCH_MAP.items():
            pdf_path = SYLLABUS_DIR / pdf_name
            modules = parse_pdf_syllabus(pdf_path)
            print(f"[SyllabusSeeder] Parsed {len(modules)} modules from {pdf_name}")

            for mod in modules:
                code = mod["code"]
                title = mod["name"]
                title_norm = re.sub(r"\s+", " ", title.lower().strip())
                sem = _semester_from_code(code)

                # Resolve or create Subject cleanly
                subj = None
                mapped_short = TITLE_TO_EXISTING_SHORT.get(title_norm)
                if mapped_short and mapped_short.upper() in by_short:
                    subj = by_short[mapped_short.upper()]
                elif code.upper() in by_code:
                    subj = by_code[code.upper()]
                elif title_norm in by_name:
                    subj = by_name[title_norm]

                if subj is None:
                    # Only add new branch subjects cleanly if non-first-year or missing core
                    short = mapped_short or _make_short_code(title, code)
                    if short.upper() in by_short:
                        short = f"{short}_{code.replace('-', '')[:5]}"
                    subj_id = f"SUBJ_{re.sub(r'[^A-Z0-9_]', '', short.upper())}"
                    if db.query(Subject).filter(Subject.id == subj_id).first():
                        subj_id = f"SUBJ_{re.sub(r'[^A-Z0-9]', '', code.upper())}"
                    subj = Subject(
                        id=subj_id,
                        code=code,
                        name=title,
                        short_name=short,
                        data_tier="Tier C",
                    )
                    db.add(subj)
                    db.flush()
                    by_short[short.upper()] = subj
                    by_code[code.upper()] = subj
                    by_name[title_norm] = subj
                    existing_topic_counts[subj.id] = 0
                    added_subjects += 1

                # Ensure SubjectOffering rows exist for branches of this PDF
                target_branches = list(branch_map.keys()) if code.startswith(("1FY", "2FY")) else branches
                for b_code in target_branches:
                    b_id = branch_map.get(b_code)
                    if not b_id:
                        continue
                    exists_off = db.query(SubjectOffering).filter(
                        SubjectOffering.subject_id == subj.id,
                        SubjectOffering.branch_id == b_id,
                    ).first()
                    if not exists_off:
                        off_id = f"OFF_{subj.id}_{b_code}_S{sem}"
                        if not db.query(SubjectOffering).filter(SubjectOffering.id == off_id).first():
                            db.add(SubjectOffering(
                                id=off_id,
                                subject_id=subj.id,
                                branch_id=b_id,
                                semester=sem,
                                scheme_id=scheme_id,
                            ))
                            added_offerings += 1

                # Seed Units & Topics ONLY if this subject currently has 0 topics
                if existing_topic_counts.get(subj.id, 0) == 0:
                    seen_units = set()
                    for uno, utext in mod["units"]:
                        if uno in seen_units:
                            continue
                        seen_units.add(uno)
                        unit_title, topic_pairs = _split_unit_into_topics(uno, utext)
                        unit_id = f"UNIT_{subj.id.replace('SUBJ_', '')}_U{uno}"
                        unit_obj = db.query(Unit).filter(Unit.id == unit_id).first()
                        if not unit_obj:
                            unit_obj = Unit(
                                id=unit_id,
                                subject_id=subj.id,
                                scheme_id=scheme_id,
                                unit_no=uno,
                                title=unit_title,
                            )
                            db.add(unit_obj)
                            db.flush()
                            added_units += 1

                        for t_idx, (t_name, t_aliases) in enumerate(topic_pairs, 1):
                            topic_id = f"TOPIC_{subj.id.replace('SUBJ_', '')}_U{uno}_{t_idx:02d}"
                            if not db.query(Topic).filter(Topic.id == topic_id).first():
                                db.add(Topic(
                                    id=topic_id,
                                    unit_id=unit_id,
                                    name=t_name,
                                    aliases=t_aliases,
                                    est_study_minutes=45,
                                ))
                                added_topics += 1
                                existing_topic_counts[subj.id] = existing_topic_counts.get(subj.id, 0) + 1

        # Also seed supplementary cataloged subjects (force PHY Wave Optics check cleanly too)
        for short_code, unit_specs in SUPPLEMENTARY_SYLLABUS.items():
            subj = by_short.get(short_code.upper())
            if not subj or (existing_topic_counts.get(subj.id, 0) > 0 and short_code != "PHY"):
                continue
            for uno, utitle, tlist in unit_specs:
                unit_id = f"UNIT_{subj.id.replace('SUBJ_', '')}_U{uno}"
                if short_code == "PHY" and uno == 1:
                    unit_id = "UNIT_PHY_U0_OPTICS"
                if not db.query(Unit).filter(Unit.id == unit_id).first():
                    db.add(Unit(
                        id=unit_id,
                        subject_id=subj.id,
                        scheme_id=scheme_id,
                        unit_no=uno,
                        title=utitle,
                    ))
                    db.flush()
                    added_units += 1
                for t_idx, (t_name, t_aliases) in enumerate(tlist, 1):
                    topic_id = (
                        f"TOPIC_PHY_U0_OPTICS_{t_idx:02d}"
                        if (short_code == "PHY" and uno == 1)
                        else f"TOPIC_{subj.id.replace('SUBJ_', '')}_U{uno}_{t_idx:02d}"
                    )
                    if not db.query(Topic).filter(Topic.id == topic_id).first():
                        db.add(Topic(
                            id=topic_id,
                            unit_id=unit_id,
                            name=t_name,
                            aliases=t_aliases,
                            est_study_minutes=45,
                        ))
                        added_topics += 1
                        existing_topic_counts[subj.id] = existing_topic_counts.get(subj.id, 0) + 1

        # Tag previously untagged questions (`topic_id IS NULL`) when matching subject topics now exist
        untagged = db.query(Question).filter(Question.topic_id.is_(None), Question.status == "approved").all()
        tagged_now = 0
        subj_topics_cache: Dict[str, List[Topic]] = {}
        for q in untagged:
            if not q.paper or not q.paper.subject_id:
                continue
            sid = q.paper.subject_id
            if sid not in subj_topics_cache:
                subj_topics_cache[sid] = (
                    db.query(Topic).join(Unit, Unit.id == Topic.unit_id).filter(Unit.subject_id == sid).all()
                )
            tlist = subj_topics_cache[sid]
            if not tlist:
                continue
            q_low = (q.text or "").lower()
            best_t = None
            best_hits = 0
            for t in tlist:
                kws = [k.strip() for k in (t.aliases or t.name).lower().split(",") if len(k.strip()) >= 3]
                hits = sum(1 for k in kws if k in q_low)
                if hits > best_hits:
                    best_hits = hits
                    best_t = t
            if best_t and best_hits > 0:
                q.topic_id = best_t.id
                q.tag_confidence = 0.78
                tagged_now += 1

        db.commit()
        total_subj = db.query(Subject).count()
        total_topics = db.query(Topic).count()
        print(
            f"[SyllabusSeeder] Added {added_subjects} subjects, {added_offerings} offerings, "
            f"{added_units} units, {added_topics} topics; newly tagged {tagged_now} questions. "
            f"DB totals -> subjects={total_subj}, topics={total_topics}."
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
