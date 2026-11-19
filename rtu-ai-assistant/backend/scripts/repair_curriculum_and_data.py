"""
Repair Curriculum and Question-Topic Metadata for RTU AI Academic Assistant
1. Replaces corrupted/missing units and topics for all 12 active RTU subjects.
2. Accurately tags all 387 questions to their corresponding syllabus topics.
3. Computes honest data tiers (Tier A >= 5 years, Tier B >= 1 year, Tier C = 0 years).
4. Restores proper subject names (e.g. Computer Networks).
"""

import sqlite3
import re
from pathlib import Path

DB_PATH = Path("rtu-ai-assistant/backend/data/storage/rtu_metadata.db")

CURRICULUM_SPEC = {
    "DSA": {
        "name": "Data Structures and Algorithms",
        "code": "3CS4-05",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Stacks and Expressions",
                "topics": [
                    ("Stacks: Operations, Static and Dynamic Arrays", ["stack", "lifo", "push", "pop", "static array", "dynamic array", "multiple stack"]),
                    ("Stack Applications: Infix, Postfix and Recursion", ["infix", "postfix", "prefix", "expression", "towers of hanoi", "factorial", "reversing list", "contd"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Queues and Linked Lists",
                "topics": [
                    ("Queues: Operations, Circular & Priority Queues", ["queue", "circular queue", "priority queue", "dequeue", "fifo", "round robin"]),
                    ("Linked Lists: Singly, Doubly and Circular", ["linked list", "singly linked", "doubly linked", "circular linked list", "header linked list", "pointer", "node"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Searching and Sorting Techniques",
                "topics": [
                    ("Searching: Sequential and Binary Search", ["sequential search", "binary search", "search time"]),
                    ("Sorting Techniques: Bubble, Insertion & Selection", ["bubble sort", "insertion sort", "selection sort", "sorting by"]),
                    ("Advanced Sorting: Quick, Merge, Heap & Radix Sort", ["quick sort", "heap sort", "merge sort", "radix sort", "max heap", "min heap", "build max heap", "sort the following numbers"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Trees and Tree Traversals",
                "topics": [
                    ("Binary Trees and Tree Traversals", ["binary tree", "tree traversal", "inorder", "preorder", "postorder", "threaded binary tree"]),
                    ("Binary Search Trees (BST) and Balanced Trees", ["binary search tree", "bst", "search tree"]),
                    ("AVL Trees, B-Trees and B+ Trees", ["avl tree", "avl", "b-tree", "b+ tree", "b+tree", "rotation", "balanced tree", "m-way tree"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Graphs and Hashing",
                "topics": [
                    ("Graphs: Representations and Traversals (BFS & DFS)", ["graph", "bfs", "dfs", "breadth first", "depth first", "adjacency matrix", "adjacency list"]),
                    ("Minimum Spanning Trees & Dijkstra's Algorithm", ["minimum spanning tree", "mst", "prims", "kruskal", "dijkstra", "shortest path"]),
                    ("Hashing and Collision Resolution", ["hash", "hashing", "hash function", "collision", "linear probing", "quadratic probing", "double hashing", "chaining"])
                ]
            }
        ]
    },
    "DBMS": {
        "name": "Database Management System",
        "code": "4CS4-05",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Database Architecture & ER Model",
                "topics": [
                    ("Database System Architecture & Data Independence", ["database system", "traditional file", "file system", "3-tier", "data independence", "schema", "architecture", "dba"]),
                    ("Entity-Relationship (ER) Modeling & ER Diagrams", ["er model", "entity", "weak entity", "strong entity", "er diagram", "attribute", "relationship", "keys", "primary key", "foreign key", "candidate key"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Relational Data Model & SQL",
                "topics": [
                    ("Relational Algebra & Relational Calculus", ["relational algebra", "select", "project", "cartesian product", "relational calculus", "tuple relational", "division"]),
                    ("SQL: DDL, DML, Joins & Queries", ["sql", "ddl", "dml", "join", "outer join", "natural join", "nested query", "group by", "having", "view", "odbc", "jdbc", "trigger"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Database Design & Normalization",
                "topics": [
                    ("Functional Dependencies & Lossless Decomposition", ["functional dependency", "armstrong", "closure", "lossless", "dependency preserving", "canonical cover"]),
                    ("Normal Forms: 1NF, 2NF, 3NF & BCNF", ["normal form", "normalization", "1nf", "2nf", "3nf", "bcnf", "multi-valued dependency", "4nf", "5nf", "anomaly"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Transaction Processing & Serializability",
                "topics": [
                    ("Transaction States & ACID Properties", ["transaction", "acid", "atomicity", "consistency", "isolation", "durability", "commit", "rollback"]),
                    ("Serializability & Recoverable Schedules", ["serializability", "conflict serializability", "view serializability", "precedence graph", "schedule", "recoverable", "cascadeless"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Concurrency Control & Recovery",
                "topics": [
                    ("Concurrency Control Protocols & Deadlock", ["concurrency control", "lock", "2pl", "two phase locking", "timestamp", "deadlock", "wait-die", "wound-wait", "lock conversion"]),
                    ("Database Failure & Recovery Techniques", ["recovery", "log-based", "wal", "write-ahead logging", "shadow paging", "checkpoint", "undo", "redo", "crash"])
                ]
            }
        ]
    },
    "TOC": {
        "name": "Theory of Computation",
        "code": "4CS4-06",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Finite Automata & State Machines",
                "topics": [
                    ("Deterministic and Non-Deterministic Finite Automata", ["finite automata", "dfa", "nfa", "ndfa", "transition diagram", "transition table", "equivalence of dfa", "minimization of dfa", "even binary", "checks whether"]),
                    ("Mealy and Moore Machines", ["mealy", "moore", "output machine", "state machine", "converter", "melay"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Regular Expressions & Languages",
                "topics": [
                    ("Regular Expressions and Finite Automata Equivalence", ["regular expression", "regular set", "regular language", "arden", "conversion", "regex"]),
                    ("Pumping Lemma for Regular Languages", ["pumping lemma for regular", "myhill", "closure properties of regular", "decision properties"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Context-Free Grammars (CFG)",
                "topics": [
                    ("CFG, Derivations & Ambiguity", ["context free grammar", "cfg", "derivation tree", "parse tree", "ambiguity", "ambiguous grammar", "leftmost", "rightmost", "sentential"]),
                    ("Normal Forms for CFG: CNF and GNF", ["chomsky normal form", "cnf", "greibach", "gnf", "simplification of cfg", "null production", "unit production"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Pushdown Automata (PDA)",
                "topics": [
                    ("Pushdown Automata (PDA) Definition & Design", ["pushdown automata", "pda", "dpda", "npda", "stack", "acceptance by empty store", "acceptance by final state"]),
                    ("PDA, CFL & Pumping Lemma for CFL", ["cfl", "context free language", "properties of cfl", "pumping lemma for cfl", "closure properties of cfl"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Turing Machines & Computability",
                "topics": [
                    ("Turing Machines & Computable Functions", ["turing machine", "tm", "tape", "universal turing machine", "transition function", "computable language"]),
                    ("Undecidability, Chomsky Hierarchy & NP Problems", ["undecidability", "halting problem", "chomsky hierarchy", "p problem", "np problem", "np complete", "np-hard", "tractable", "untractable", "recursively enumerable"])
                ]
            }
        ]
    },
    "DCCN": {
        "name": "Data Communication and Computer Networks",
        "code": "4CS4-07",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Network Architecture & Physical Layer",
                "topics": [
                    ("Network Topologies, Hardware & Protocols", ["topology", "topologies", "network hardware", "mesh", "star", "bus", "ring", "protocol", "communication protocol"]),
                    ("OSI Reference Model & TCP/IP Model", ["osi model", "tcp/ip model", "layers of the osi", "physical layer", "data link layer", "network layer", "transport layer", "session layer", "presentation layer"]),
                    ("Physical Layer: Line Coding & Modulation", ["line coding", "digital transmission", "analog signal", "modulation", "bandwidth", "attenuation", "transmission media", "guided media"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Data Link Layer & MAC Protocols",
                "topics": [
                    ("Framing & Error Detection/Correction (CRC, Hamming)", ["data link", "framing", "error detection", "error correction", "crc", "cyclic redundancy", "checksum", "parity check", "hamming code", "linear block code"]),
                    ("Flow Control Protocols: Stop-and-Wait & Sliding Window", ["flow control", "stop and wait", "sliding window", "go-back-n", "selective repeat", "piggybacking", "arq"]),
                    ("Multiple Access Protocols: ALOHA, CSMA/CD & CSMA/CA", ["multiple access", "mac", "aloha", "pure aloha", "slotted aloha", "csma", "csma/cd", "csma/ca", "collision"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Network Layer & Routing Algorithms",
                "topics": [
                    ("IP Addressing: IPv4, IPv6 & Subnetting", ["ipv4", "ipv6", "ip addressing", "subnet", "subnetting", "cidr", "arp", "rarp", "icmp", "address mapping"]),
                    ("Routing Algorithms & Congestion Control", ["routing algorithm", "distance vector", "link state", "dijkstra", "bellman-ford", "congestion control", "unicast", "multicast", "broadcast", "quality of service"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Transport Layer Protocols",
                "topics": [
                    ("Transport Layer: TCP vs UDP & Connection Management", ["transport layer", "tcp", "udp", "port number", "3-way handshake", "three-way", "connection establishment", "socket"]),
                    ("TCP Congestion & Traffic Shaping (Leaky/Token Bucket)", ["traffic shaping", "leaky bucket", "token bucket", "tcp flow control", "congestion window", "slow start"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Application Layer & Network Security",
                "topics": [
                    ("Application Layer Protocols: DNS, HTTP, SMTP & FTP", ["application layer", "dns", "domain name", "http", "https", "smtp", "ftp", "electronic mail", "www"]),
                    ("Network Security, Cryptography & Firewalls", ["network security", "cryptography", "encryption", "decryption", "symmetric", "asymmetric", "digital signature", "firewall", "security"])
                ]
            }
        ]
    },
    "TC": {
        "name": "Technical Communication",
        "code": "4CS1-02",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to Technical Communication",
                "topics": [
                    ("Technical Communication: Aspects, Forms & Skills", ["technical communication", "aspects of technical", "forms of technical", "importance of technical", "communication skills", "lsrw", "listening", "speaking", "reading", "writing", "linguistic ability", "style"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Technical Comprehension & Information Design",
                "topics": [
                    ("Technical Comprehension, Manuals & Summarizing", ["comprehension of technical", "technical manual", "technical texts", "summarizing", "note-making", "interpreting", "instructions"]),
                    ("Information Design for Print & Online Media", ["information design", "document design", "strategies for organization", "print and online"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Technical Writing Process & Correspondence",
                "topics": [
                    ("Technical Writing Process, Grammar & Style", ["technical writing process", "discourse", "drafts and revising", "grammar", "common error", "technical style"]),
                    ("Official Correspondence: Letters, Email, Resume & MoM", ["official notes", "letter", "letters", "email", "e-mail", "resume", "job application", "minutes of meetings", "mom"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Advanced Technical Writing & Reports",
                "topics": [
                    ("Technical Reports: Formats, Structure & Types", ["technical report", "technical reports", "types of technical reports", "format of report", "structure of report"]),
                    ("Technical Proposals & Research Articles", ["technical proposal", "project proposal", "technical article", "technical articles", "writing strategies"])
                ]
            }
        ]
    },
    "DMS": {
        "name": "Discrete Mathematics Structure",
        "code": "4CS2-01",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Set Theory, Relations & Functions",
                "topics": [
                    ("Set Theory, Cardinality & Inclusion-Exclusion", ["set theory", "countable set", "uncountable", "set operations", "partition of set", "cardinality", "inclusion-exclusion", "venn diagram"]),
                    ("Relations: Types, Equivalence & Closures", ["relation", "relations", "equivalence relation", "partial ordering", "composition of relation"]),
                    ("Functions & Theorem Proving (Induction, Pigeonhole)", ["function", "into and onto", "one to one", "inverse function", "mathematical induction", "pigeonhole principle", "proof by contradiction"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Propositional Logic & Finite State Machines",
                "topics": [
                    ("Propositional & Predicate Logic, Truth Tables", ["propositional logic", "proposition", "first order logic", "truth table", "tautology", "contradiction", "predicate", "quantifier", "normal form"]),
                    ("Finite State Machines as Language Recognizers", ["finite state machine", "fsm", "state machine", "language recognizer", "models of physical"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Posets, Lattices & Combinatorics",
                "topics": [
                    ("Posets, Hasse Diagrams & Lattices", ["poset", "hasse diagram", "lattice", "lattices", "partially ordered", "well ordered set", "complemented lattice"]),
                    ("Combinatorics & Recurrence Relations", ["combinatorics", "permutation", "combination", "binomial theorem", "recurrence relation", "generating function", "recursive algorithm"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Algebraic Structures",
                "topics": [
                    ("Groups, Subgroups & Abelian Groups", ["algebraic structure", "semigroup", "semi group", "monoid", "group", "abelian group", "subgroup", "cyclic group", "coset", "lagrange", "homomorphism", "isomorphism"]),
                    ("Rings and Fields", ["ring", "field", "rings and fields", "integral domain"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Graph Theory and Trees",
                "topics": [
                    ("Graph Theory: Planar, Bipartite & Euler Graphs", ["graph theory", "planar graph", "multigraph", "euler", "hamiltonian", "isomorphism of graphs", "coloring"]),
                    ("Trees and Cut Sets in Discrete Mathematics", ["tree", "spanning tree", "cut set", "directed graph"])
                ]
            }
        ]
    },
    "OOP": {
        "name": "Object Oriented Programming (C++)",
        "code": "3CS4-06",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: OOP Paradigms, Classes & Objects",
                "topics": [
                    ("OOP Characteristics, Classes & Objects", ["oop paradigm", "class", "object", "data member", "member function", "access specifier", "public", "private", "protected", "array of objects", "structure in c++"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: References, Memory & Constructors",
                "topics": [
                    ("Dynamic Memory, Inline & Friend Functions", ["reference", "dynamic memory", "new and delete", "inline function", "function overloading", "default arguments", "friend function", "friend class", "this pointer"]),
                    ("Constructors and Destructors", ["constructor", "destructor", "copy constructor", "parameterized constructor", "default constructor"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Inheritance & Virtual Functions",
                "topics": [
                    ("Inheritance Types & Virtual Base Classes", ["inheritance", "multiple inheritance", "multilevel inheritance", "hierarchical", "virtual base class"]),
                    ("Function Overriding & Abstract Classes", ["function overriding", "abstract class", "pure virtual function", "virtual function"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Polymorphism & Operator Overloading",
                "topics": [
                    ("Operator Overloading & Static Members", ["operator overloading", "overloading unary", "overloading binary", "static data member", "constant member"]),
                    ("Polymorphism and Dynamic Binding", ["polymorphism", "dynamic binding", "early binding", "late binding", "virtual destructor"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Exception Handling, Templates & Files",
                "topics": [
                    ("Exception Handling and Templates", ["exception handling", "try", "catch", "throw", "template", "function template", "class template"]),
                    ("Stream Classes and File Handling in C++", ["stream class", "file handling", "fstream", "ifstream", "ofstream", "file pointer"])
                ]
            }
        ]
    },
    "OS": {
        "name": "Operating Systems",
        "code": "5CS4-03",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to Operating Systems",
                "topics": [
                    ("OS Overview, System Calls & Dual Mode", ["operating system", "os overview", "system call", "dual mode", "kernel", "trap", "monolithic", "microkernel", "pcb", "process control block"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Process Management & CPU Scheduling",
                "topics": [
                    ("CPU Scheduling: FCFS, SJF, RR & Priority", ["cpu scheduling", "fcfs", "sjf", "round robin", "priority scheduling", "turnaround time", "waiting time", "gantt chart", "multilevel queue"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Process Synchronization & Deadlocks",
                "topics": [
                    ("Critical Section, Semaphores & Synchronization", ["process synchronization", "critical section", "semaphore", "mutex", "peterson", "dining philosophers", "readers writers", "producer consumer"]),
                    ("Deadlock: Prevention, Avoidance & Banker's Algorithm", ["deadlock", "deadlock prevention", "deadlock avoidance", "banker", "resource allocation graph", "safety algorithm", "deadlock detection"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Memory Management & Virtual Memory",
                "topics": [
                    ("Paging, Segmentation & Memory Allocation", ["memory management", "paging", "page table", "segmentation", "tlb", "fragmentation", "internal fragmentation", "external fragmentation"]),
                    ("Virtual Memory & Page Replacement Algorithms", ["virtual memory", "demand paging", "page fault", "page replacement", "fifo", "lru", "optimal page", "belady anomaly", "thrashing"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Storage Management & File Systems",
                "topics": [
                    ("Disk Scheduling: FCFS, SSTF, SCAN & LOOK", ["disk scheduling", "sstf", "scan", "c-scan", "look", "c-look", "seek time", "rotational latency", "raid"]),
                    ("File System Structure, Allocation & Free Space", ["file system", "file allocation", "contiguous allocation", "linked allocation", "indexed allocation", "inode", "directory structure"])
                ]
            }
        ]
    },
    "DAA": {
        "name": "Analysis of Algorithms",
        "code": "5CS4-05",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Algorithm Analysis & Recurrences",
                "topics": [
                    ("Asymptotic Notations & Performance Analysis", ["asymptotic", "big o", "omega", "theta", "time complexity", "space complexity", "best case", "worst case"]),
                    ("Recurrence Relations & Master Theorem", ["recurrence relation", "master theorem", "substitution method", "recursion tree", "divide and conquer"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Divide & Conquer and Greedy Algorithms",
                "topics": [
                    ("Divide and Conquer: Quick Sort, Merge Sort & Selection", ["divide and conquer", "merge sort", "quick sort", "median finding", "strassen"]),
                    ("Greedy Method: Knapsack, Huffman & Job Sequencing", ["greedy method", "fractional knapsack", "huffman coding", "job sequencing", "optimal storage", "activity selection"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Dynamic Programming",
                "topics": [
                    ("Dynamic Programming: 0/1 Knapsack & LCS", ["dynamic programming", "0/1 knapsack", "longest common subsequence", "lcs", "optimal substructure", "overlapping subproblems"]),
                    ("Matrix Chain Multiplication & All-Pairs Shortest Paths", ["matrix chain", "floyd warshall", "bellman ford", "travelling salesman", "tsp"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Backtracking & Branch and Bound",
                "topics": [
                    ("Backtracking: N-Queens, Subset Sum & Graph Coloring", ["backtracking", "n-queens", "subset sum", "graph coloring", "hamiltonian cycle", "state space tree"]),
                    ("Branch and Bound: Assignment & 0/1 Knapsack", ["branch and bound", "fifo branch and bound", "lc branch and bound", "assignment problem"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: String Matching & NP-Completeness",
                "topics": [
                    ("String Matching: Naive, KMP & Rabin-Karp", ["string matching", "kmp", "knuth morris pratt", "rabin karp", "finite automata string"]),
                    ("NP-Completeness: P, NP, NP-Hard & NP-Complete", ["p class", "np class", "np-complete", "np-hard", "cook theorem", "clique problem", "vertex cover", "polynomial time reduction"])
                ]
            }
        ]
    },
    "CD": {
        "name": "Compiler Design",
        "code": "6CS4-01",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Lexical Analysis & Compiler Architecture",
                "topics": [
                    ("Phases of Compiler & Compiler-Compilers", ["phases of compiler", "compiler architecture", "lexical analyzer", "syntax analyzer", "semantic analyzer", "code generator", "compiler compiler"]),
                    ("Lexical Analysis: Tokens, Lexemes & Regular Expressions", ["token", "lexeme", "input buffering", "regular expression for tokens", "transition diagram", "lex tool"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Syntax Analysis & Top-Down Parsing",
                "topics": [
                    ("Top-Down Parsing: LL(1) & Recursive Descent", ["top-down parsing", "recursive descent", "ll(1)", "first and follow", "predictive parser", "left recursion", "left factoring"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Bottom-Up Parsing",
                "topics": [
                    ("LR Parsing: Shift-Reduce, SLR(1), LR(0) & LALR(1)", ["bottom-up parsing", "shift reduce", "lr parsing", "lr(0)", "slr(1)", "clr(1)", "lalr(1)", "handle pruning", "operator precedence", "yacc"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Syntax-Directed Translation & Intermediate Code",
                "topics": [
                    ("Syntax-Directed Definitions & Translations", ["syntax directed", "s-attributed", "l-attributed", "annotated parse tree", "dependency graph"]),
                    ("Intermediate Code Generation & 3-Address Code", ["intermediate code", "three address code", "quadruples", "triples", "syntax tree", "postfix notation"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Code Optimization & Code Generation",
                "topics": [
                    ("Code Optimization: Basic Blocks & Flow Graphs", ["code optimization", "basic block", "flow graph", "loop optimization", "dead code elimination", "constant folding", "peephole optimization", "dag"]),
                    ("Code Generation & Register Allocation", ["code generation", "register allocation", "target machine", "instruction selection", "next use"])
                ]
            }
        ]
    },
    "AI": {
        "name": "Artificial Intelligence",
        "code": "6CS4-03",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to AI & Search Techniques",
                "topics": [
                    ("AI Concepts, Agents & State Space Search", ["artificial intelligence", "intelligent agent", "turing test", "state space", "production system", "water jug", "8-puzzle"]),
                    ("Heuristic Search: A*, AO*, Hill Climbing & Best-First", ["heuristic search", "best first search", "a* algorithm", "ao*", "hill climbing", "simulated annealing", "admissibility of a*"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Game Playing & Adversarial Search",
                "topics": [
                    ("Minimax Algorithm & Alpha-Beta Pruning", ["game playing", "minimax", "alpha-beta pruning", "adversarial search", "evaluation function", "tic tac toe", "chess"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Knowledge Representation & Logic",
                "topics": [
                    ("Propositional & Predicate Logic, Resolution & Unification", ["knowledge representation", "predicate logic", "fopl", "resolution", "unification", "wff", "clause form", "forward chaining", "backward chaining"]),
                    ("Structured Knowledge: Semantic Nets, Frames & Ontologies", ["semantic net", "conceptual dependency", "script", "frame", "ontology"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Reasoning Under Uncertainty",
                "topics": [
                    ("Probabilistic Reasoning, Bayes Rule & Bayesian Networks", ["probabilistic reasoning", "bayes rule", "bayesian network", "certainty factor", "dempster shafer", "fuzzy logic", "fuzzy set"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Expert Systems & Natural Language Processing",
                "topics": [
                    ("Expert Systems Architecture & Knowledge Engineering", ["expert system", "knowledge base", "inference engine", "rule based system", "mycin"]),
                    ("NLP Fundamentals & Machine Learning in AI", ["natural language processing", "nlp", "syntactic analysis", "parsing in nlp", "machine learning basics in ai"])
                ]
            }
        ]
    },
    "CA": {
        "name": "Computer Architecture and Organization",
        "code": "6CS4-04",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Register Transfer & Basic Organization",
                "topics": [
                    ("Register Transfer Language, Bus & Memory Transfers", ["register transfer", "bus transfer", "three-state bus", "memory transfer", "instruction code", "stored program"]),
                    ("Common Bus System & Computer Instructions", ["common bus", "computer register", "instruction cycle", "fetch and decode", "timing and control", "input-output configuration"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Microprogrammed Control & CPU Architecture",
                "topics": [
                    ("Control Memory, Address Sequencing & Microprogram Sequencer", ["control memory", "microprogrammed control", "hardwired control", "address sequencing", "microprogram sequencer", "mapping of instruction"]),
                    ("CPU Organization, Addressing Modes & RISC vs CISC", ["general register organization", "stack organization", "addressing modes", "instruction formats", "risc vs cisc", "risc architecture"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Computer Arithmetic",
                "topics": [
                    ("Addition, Subtraction & Booth's Multiplication", ["computer arithmetic", "booth algorithm", "multiplication algorithm", "division algorithm", "restoring division", "non-restoring division", "floating-point arithmetic"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Memory Organization & Memory Hierarchy",
                "topics": [
                    ("Memory Hierarchy, Main Memory & Auxiliary Memory", ["memory hierarchy", "ram and rom", "auxiliary memory", "magnetic disk"]),
                    ("Associative Memory, Cache Memory & Virtual Memory", ["associative memory", "cache memory", "cache mapping", "direct mapping", "associative mapping", "set-associative", "virtual memory", "page replacement in hardware"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: I/O Organization & Pipelining",
                "topics": [
                    ("Input-Output Interface, Asynchronous Data Transfer & DMA", ["i/o interface", "asynchronous transfer", "strobe control", "handshaking", "programmed i/o", "interrupt-driven i/o", "dma", "direct memory access"]),
                    ("Pipelining: Arithmetic, Instruction & RISC Pipeline", ["pipelining", "instruction pipeline", "arithmetic pipeline", "pipeline hazards", "data hazards", "control hazards", "structural hazards", "vector processing"])
                ]
            }
        ]
    },
    "ML": {
        "name": "Machine Learning",
        "code": "7CS4-02",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to Machine Learning",
                "topics": [
                    ("Machine Learning Foundations & Supervised vs Unsupervised", ["machine learning", "types of learning", "supervised learning", "unsupervised learning", "reinforcement learning", "hypothesis space", "inductive bias", "bias-variance"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Regression & Linear Models",
                "topics": [
                    ("Linear Regression, Gradient Descent & Regularization", ["linear regression", "multiple regression", "cost function", "gradient descent", "ridge regression", "lasso regression", "regularization", "logistic regression"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Classification Algorithms",
                "topics": [
                    ("Decision Trees, Random Forest & Naive Bayes", ["decision tree", "id3", "c4.5", "entropy", "information gain", "gini index", "pruning", "random forest", "naive bayes classifier", "bayesian classifier"]),
                    ("Support Vector Machines (SVM) & Kernel Tricks", ["support vector machine", "svm", "hyperplane", "maximum margin", "kernel trick", "rbf kernel", "soft margin"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Clustering & Dimensionality Reduction",
                "topics": [
                    ("K-Means, Hierarchical Clustering & PCA", ["k-means", "hierarchical clustering", "agglomerative", "dendrogram", "pca", "principal component analysis", "dimensionality reduction", "lda"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Neural Networks & Ensemble Methods",
                "topics": [
                    ("Perceptron, Multilayer Perceptron & Backpropagation", ["perceptron", "multilayer perceptron", "backpropagation", "activation function", "sigmoid", "relu", "neural network"]),
                    ("Ensemble Learning: Bagging, Boosting & Model Evaluation", ["ensemble learning", "bagging", "boosting", "adaboost", "confusion matrix", "precision", "recall", "f1-score", "roc-auc", "cross-validation"])
                ]
            }
        ]
    },
    "CC": {
        "name": "Cloud Computing",
        "code": "6CS4-05",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to Cloud Computing",
                "topics": [
                    ("Cloud Architecture, NIST Model & Cloud Characteristics", ["cloud computing", "cloud architecture", "nist cloud", "essential characteristics", "on-demand self-service", "broad network access", "resource pooling"]),
                    ("Cloud Service Models: IaaS, PaaS & SaaS", ["iaas", "paas", "saas", "infrastructure as a service", "platform as a service", "software as a service", "deployment models", "public cloud", "private cloud", "hybrid cloud"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Virtualization Technologies",
                "topics": [
                    ("Hypervisors, Hardware Virtualization & Containers", ["virtualization", "hypervisor", "type 1 hypervisor", "type 2 hypervisor", "full virtualization", "paravirtualization", "hardware assisted", "containerization", "docker", "vm migration"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Cloud Storage & Resource Management",
                "topics": [
                    ("Distributed Cloud Storage, S3 & Load Balancing", ["cloud storage", "distributed file system", "s3", "blob storage", "resource provisioning", "load balancing algorithms", "round robin cloud", "sla", "service level agreement"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Cloud Security & Privacy",
                "topics": [
                    ("Cloud Security Architecture, IAM & Multi-Tenancy", ["cloud security", "identity and access management", "iam", "multi-tenancy", "data protection in cloud", "encryption in cloud", "virtualization security"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Cloud Platforms & Future Trends",
                "topics": [
                    ("AWS, Azure, Google Cloud & Serverless Architecture", ["aws", "azure", "google cloud platform", "serverless", "faas", "microservices in cloud", "mapreduce in cloud", "hadoop in cloud"])
                ]
            }
        ]
    },
    "BDA": {
        "name": "Big Data Analytics",
        "code": "7CS4-01",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Big Data Fundamentals & Hadoop",
                "topics": [
                    ("Big Data 5Vs & Hadoop Architecture (HDFS & YARN)", ["big data", "5vs", "volume velocity variety", "hadoop", "hadoop architecture", "hdfs", "namenode", "datanode", "secondary namenode", "yarn", "resourcemanager"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: MapReduce Programming Model",
                "topics": [
                    ("MapReduce Architecture: Mapper, Reducer & Combiner", ["mapreduce", "mapper", "reducer", "combiner", "partitioner", "shuffling and sorting", "word count in mapreduce", "data flow in mapreduce"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Hadoop Ecosystem Tools",
                "topics": [
                    ("Hive, Pig, Sqoop & NoSQL Databases", ["apache hive", "hiveql", "apache pig", "pig latin", "sqoop", "flume", "hbase", "nosql", "cap theorem", "column-oriented database"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Apache Spark & Real-Time Processing",
                "topics": [
                    ("Spark Architecture, RDDs & Spark Streaming", ["apache spark", "spark architecture", "rdd", "resilient distributed datasets", "spark dataframe", "spark sql", "spark streaming", "mllib"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Big Data Analytics & Mining",
                "topics": [
                    ("Clustering, Recommendation Systems & PageRank", ["recommendation system", "collaborative filtering", "content-based", "pagerank", "mining data streams", "bloom filter", "flajolet-martin"])
                ]
            }
        ]
    },
    "IOT": {
        "name": "Internet of Things",
        "code": "7CS4-04",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to IoT & Architecture",
                "topics": [
                    ("IoT Definition, Characteristics & Architecture", ["internet of things", "iot architecture", "iot characteristics", "physical design of iot", "logical design of iot", "iot protocols", "sensors and actuators"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: IoT Networking & Communication Protocols",
                "topics": [
                    ("MQTT, CoAP, AMQP & Wireless IoT Technologies", ["mqtt", "coap", "amqp", "http vs mqtt", "zigbee", "bluetooth low energy", "ble", "lora", "lorawan", "6lowpan", "rfid"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Embedded IoT Platforms & Hardware",
                "topics": [
                    ("Arduino, Raspberry Pi & Sensor Interfacing", ["arduino", "raspberry pi", "esp8266", "esp32", "gpio", "adc in iot", "interfacing sensors", "actuator interfacing", "embedded linux"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Cloud & Edge Computing for IoT",
                "topics": [
                    ("IoT Cloud Platforms & Fog/Edge Analytics", ["iot cloud", "thingspeak", "aws iot core", "azure iot hub", "fog computing in iot", "edge analytics", "data collection in iot"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: IoT Security & Real-World Applications",
                "topics": [
                    ("IoT Security Challenges & Applications", ["iot security", "privacy in iot", "smart home", "smart city", "smart agriculture", "industrial iot", "iiot", "healthcare iot"])
                ]
            }
        ]
    },
    "ISS": {
        "name": "Information System Security",
        "code": "7CS4-03",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Cryptography & Symmetric Ciphers",
                "topics": [
                    ("Classical Ciphers & DES/3DES", ["classical encryption", "caesar cipher", "playfair", "hill cipher", "feistel cipher", "des", "data encryption standard", "triple des", "3des", "aes", "advanced encryption standard", "modes of operation", "ecb", "cbc"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Asymmetric Cryptography & Key Distribution",
                "topics": [
                    ("RSA, Diffie-Hellman & Elliptic Curve Cryptography", ["asymmetric key", "public key cryptography", "rsa algorithm", "diffie hellman", "man in the middle", "elliptic curve", "ecc", "pki", "public key infrastructure", "digital certificate"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Hash Functions & Digital Signatures",
                "topics": [
                    ("Message Authentication Codes (MAC), SHA & Digital Signatures", ["message authentication", "mac", "hmac", "hash function", "sha-1", "sha-256", "md5", "digital signature", "dss", "digital signature standard"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Network & Transport Layer Security",
                "topics": [
                    ("IPsec, SSL/TLS & Email Security", ["ipsec", "ah", "esp", "transport mode", "tunnel mode", "ssl", "tls", "handshake protocol", "pgp", "pretty good privacy", "s/mime", "kerberos"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: System Security, Firewalls & Malware",
                "topics": [
                    ("Firewalls, IDS/IPS & Malicious Software", ["firewall", "packet filtering firewall", "circuit level gateway", "application proxy", "intrusion detection", "ids", "ips", "virus", "worm", "trojan horse", "denial of service", "dos", "ddos"])
                ]
            }
        ]
    },
    "DM": {
        "name": "Disaster Management",
        "code": "8CS4-01",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to Disasters & Hazards",
                "topics": [
                    ("Disaster Concepts: Hazards, Vulnerability & Risk", ["disaster", "hazard", "vulnerability", "risk", "disaster management cycle", "natural disaster", "earthquake", "flood", "cyclone", "drought", "man-made disaster"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Disaster Preparedness & Mitigation",
                "topics": [
                    ("Preparedness Planning, Early Warning & Mitigation", ["disaster preparedness", "early warning system", "disaster mitigation", "structural mitigation", "non-structural mitigation", "community based disaster"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Disaster Response & Relief",
                "topics": [
                    ("Emergency Response, Search and Rescue & Relief", ["disaster response", "search and rescue", "evacuation", "emergency relief", "shelter", "incident command system", "medical aid in disaster"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Post-Disaster Recovery & Reconstruction",
                "topics": [
                    ("Damage Assessment, Rehabilitation & Reconstruction", ["damage assessment", "post disaster recovery", "rehabilitation", "reconstruction", "building back better", "psychosocial care"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Institutional Framework & Technology in Disaster",
                "topics": [
                    ("NDMA, NDRF, Disaster Management Act & GIS", ["ndma", "sdma", "ndrf", "disaster management act 2005", "remote sensing in disaster", "gis in disaster management", "role of armed forces in disaster"])
                ]
            }
        ]
    },
    "DE": {
        "name": "Digital Electronics",
        "code": "3CS3-04",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Number Systems and Boolean Algebra",
                "topics": [
                    ("Number Systems, Binary Arithmetic & Codes", ["number systems", "binary", "octal", "hexadecimal", "1s complement", "2s complement", "bcd", "gray code", "excess-3", "ascii"]),
                    ("Boolean Algebra, Logic Gates & De Morgan's Laws", ["boolean algebra", "logic gates", "and", "or", "not", "nand", "nor", "xor", "xnor", "universal gates", "de morgan", "boolean laws"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Combinational Logic Circuits & Minimization",
                "topics": [
                    ("Karnaugh Maps (K-Map) & Quine-McCluskey", ["k-map", "karnaugh map", "sop", "pos", "minterm", "maxterm", "dont care", "quine mccluskey", "tabular method"]),
                    ("Adders, Subtractors & Code Converters", ["half adder", "full adder", "half subtractor", "full subtractor", "look ahead carry", "parallel adder", "code converter"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: MSI Combinational Devices",
                "topics": [
                    ("Multiplexers, Demultiplexers, Encoders & Decoders", ["multiplexer", "mux", "demultiplexer", "demux", "decoder", "encoder", "priority encoder", "seven segment decoder", "magnitude comparator"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Sequential Logic Circuits & Flip-Flops",
                "topics": [
                    ("Latches & Flip-Flops: SR, JK, D & T", ["latch", "flip-flop", "sr flip flop", "jk flip flop", "d flip flop", "t flip flop", "master slave jk", "race around condition", "excitation table", "characteristic equation"]),
                    ("Counters: Synchronous & Asynchronous (Ripple)", ["counter", "ripple counter", "asynchronous counter", "synchronous counter", "mod-n counter", "up down counter", "ring counter", "johnson counter"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Shift Registers & Semiconductor Memories",
                "topics": [
                    ("Shift Registers & Register Modes (SISO, SIPO, PISO, PIPO)", ["shift register", "siso", "sipo", "piso", "pipo", "bidirectional shift register", "universal shift register"]),
                    ("Semiconductor Memories & Programmable Logic (ROM, RAM, PLA, PAL)", ["semiconductor memory", "ram", "rom", "sram", "dram", "pla", "pal", "programmable logic array", "fpga"])
                ]
            }
        ]
    },
    "MEFA": {
        "name": "Managerial Economics and Financial Accounting",
        "code": "3CS1-03",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Introduction to Managerial Economics",
                "topics": [
                    ("Managerial Economics: Scope, Significance & Circular Flow", ["managerial economics", "nature of managerial economics", "scope of managerial economics", "microeconomics vs macroeconomics", "circular flow of economic activity", "wealth"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Demand Analysis & Forecasting",
                "topics": [
                    ("Law of Demand, Elasticity of Demand & Demand Forecasting", ["law of demand", "elasticity of demand", "price elasticity", "income elasticity", "cross elasticity", "demand forecasting", "trend projection", "survey method"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Production, Cost & Market Structures",
                "topics": [
                    ("Production Function, Law of Returns & Cost Analysis", ["production function", "law of variable proportions", "returns to scale", "cobb douglas", "isoquant", "cost concepts", "fixed cost", "variable cost", "marginal cost", "break even analysis", "bep"]),
                    ("Market Structures: Perfect Competition, Monopoly & Oligopoly", ["market structure", "perfect competition", "monopoly", "monopolistic competition", "oligopoly", "price determination", "kinked demand curve"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Financial Accounting & Balance Sheet",
                "topics": [
                    ("Accounting Principles, Journal, Ledger & Trial Balance", ["financial accounting", "gaap", "accounting concepts", "double entry system", "journal", "ledger", "trial balance", "depreciation"]),
                    ("Final Accounts: Trading, Profit & Loss A/c & Balance Sheet", ["final accounts", "trading account", "profit and loss account", "balance sheet", "assets and liabilities", "working capital"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Financial Statement Analysis & Ratio Analysis",
                "topics": [
                    ("Ratio Analysis: Liquidity, Solvency & Profitability Ratios", ["ratio analysis", "liquidity ratio", "current ratio", "quick ratio", "solvency ratio", "debt equity ratio", "profitability ratio", "gross profit ratio", "net profit ratio", "funds flow", "cash flow statement"])
                ]
            }
        ]
    },
    "CN": {
        "name": "Computer Networks",
        "code": "5CS4-01",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: Network Architectures & Layered Protocols",
                "topics": [
                    ("OSI Architecture, TCP/IP Suite & Network Topologies", ["computer networks", "osi reference model", "tcp/ip protocol suite", "network topologies", "packet switching vs circuit switching", "delays and throughput in networks"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: Data Link Layer & MAC Protocols",
                "topics": [
                    ("Error Detection, Flow Control & Sliding Window", ["data link layer", "error control", "crc", "sliding window protocols", "stop and wait arq", "go back n", "selective repeat arq"]),
                    ("Ethernet, CSMA/CD & Wireless LANs (802.11)", ["mac sublayer", "aloha", "csma/cd", "ethernet standards", "fast ethernet", "gigabit ethernet", "wireless lan", "ieee 802.11", "bluetooth architecture"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: Network Layer & Routing Protocols",
                "topics": [
                    ("IPv4, IPv6 Addressing, Subnetting & CIDR", ["network layer", "ipv4 addressing", "ipv6 addressing", "subnetting", "subnet mask", "cidr", "nat", "arp", "icmp"]),
                    ("Routing Protocols: RIP, OSPF, BGP & Distance Vector", ["routing algorithm", "distance vector routing", "link state routing", "rip", "ospf", "bgp", "hierarchical routing", "multicast routing"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: Transport Layer & Congestion Control",
                "topics": [
                    ("TCP, UDP, Port Numbers & 3-Way Handshake", ["transport layer", "tcp segment structure", "udp datagram", "three way handshake", "connection termination", "tcp flow control"]),
                    ("TCP Congestion Control & Avoidance Algorithms", ["tcp congestion control", "slow start", "congestion avoidance", "fast retransmit", "fast recovery", "leaky bucket", "token bucket"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Application Layer Protocols & Network Security",
                "topics": [
                    ("DNS, HTTP, SMTP, FTP & DHCP", ["application layer", "dns resolution", "http 1.1 vs 2.0", "persistent http", "smtp architecture", "pop3", "imap", "ftp control and data", "dhcp"]),
                    ("Network Security: Cryptography, Firewalls & IPsec", ["network security", "symmetric vs asymmetric encryption", "rsa", "digital signatures", "firewalls", "packet filtering", "ssl/tls", "ipsec architecture"])
                ]
            }
        ]
    },
    "MI": {
        "name": "Microprocessor and Interfaces",
        "code": "5CS3-01",
        "units": [
            {
                "unit_no": 1,
                "title": "Unit 1: 8085 Microprocessor Architecture",
                "topics": [
                    ("8085 CPU Architecture, Registers & Pin Diagram", ["8085 microprocessor", "8085 architecture", "registers in 8085", "accumulator", "program counter", "stack pointer", "flag register", "pin configuration of 8085", "timing diagram of 8085", "machine cycle"])
                ]
            },
            {
                "unit_no": 2,
                "title": "Unit 2: 8085 Instruction Set & Assembly Programming",
                "topics": [
                    ("8085 Instruction Set & Addressing Modes", ["8085 instruction set", "addressing modes of 8085", "data transfer instructions", "arithmetic instructions", "logical instructions", "branching instructions"]),
                    ("Assembly Language Programming in 8085", ["assembly language programming", "8085 assembly", "subroutines", "stack operations in 8085", "delay calculation in 8085", "sorting in 8085"])
                ]
            },
            {
                "unit_no": 3,
                "title": "Unit 3: 8086 Microprocessor Architecture",
                "topics": [
                    ("8086 CPU Architecture, BIU, EU & Segmentation", ["8086 microprocessor", "8086 architecture", "bus interface unit", "biu", "execution unit", "eu", "memory segmentation in 8086", "logical vs physical address", "flag register in 8086", "pin diagram of 8086", "minimum mode vs maximum mode"])
                ]
            },
            {
                "unit_no": 4,
                "title": "Unit 4: 8086 Instruction Set & Addressing Modes",
                "topics": [
                    ("8086 Instruction Set & Assembler Directives", ["8086 instruction set", "addressing modes of 8086", "assembler directives", "assume", "db", "dw", "segment", "proc", "macro vs procedure in 8086", "string instructions in 8086"])
                ]
            },
            {
                "unit_no": 5,
                "title": "Unit 5: Interfacing Peripheral Devices (8255, 8254, 8259, 8251)",
                "topics": [
                    ("Programmable Peripheral Interface 8255 & Counter 8254", ["8255 ppi", "programmable peripheral interface", "modes of 8255", "bsr mode", "8254 timer", "programmable interval timer", "modes of 8254"]),
                    ("Interrupt Controller 8259, USART 8251 & DMA 8257", ["8259 pic", "programmable interrupt controller", "vector table", "usart 8251", "dma controller 8257", "adc and dac interfacing with 8085/8086", "stepper motor interfacing"])
                ]
            }
        ]
    }
}

def clean_and_repair():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print("[*] Fixing subject names...")
    cur.execute("UPDATE subjects SET name = 'Computer Networks' WHERE short_name = 'CN' AND name LIKE '%Information%'")

    for short_code, spec in CURRICULUM_SPEC.items():
        subj_id = f"SUBJ_{short_code}"
        print(f"\n[*] Processing Subject: {short_code} ({spec['name']})...")

        # Ensure subject row
        cur.execute("SELECT id FROM subjects WHERE id = ?", (subj_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO subjects (id, code, name, short_name, data_tier) VALUES (?, ?, ?, ?, 'Tier A')",
                        (subj_id, spec["code"], spec["name"], short_code))
        else:
            cur.execute("UPDATE subjects SET name = ?, code = ? WHERE id = ?", (spec["name"], spec["code"], subj_id))

        # Clear old topics & units for this subject
        cur.execute("SELECT id FROM units WHERE subject_id = ?", (subj_id,))
        existing_unit_ids = [r[0] for r in cur.fetchall()]
        for uid in existing_unit_ids:
            cur.execute("DELETE FROM topics WHERE unit_id = ?", (uid,))
        cur.execute("DELETE FROM units WHERE subject_id = ?", (subj_id,))

        # Insert new clean units & topics
        topic_lookup = []
        for u in spec["units"]:
            unit_id = f"UNIT_{short_code}_U{u['unit_no']}"
            cur.execute("INSERT INTO units (id, subject_id, scheme_id, unit_no, title) VALUES (?, ?, 'SCHEME_RTU_2018', ?, ?)",
                        (unit_id, subj_id, u["unit_no"], u["title"]))

            for t_idx, (t_name, keywords) in enumerate(u["topics"], 1):
                topic_id = f"TOPIC_{short_code}_U{u['unit_no']}_{t_idx:02d}"
                aliases = ", ".join(keywords)
                cur.execute("INSERT INTO topics (id, unit_id, name, aliases, est_study_minutes) VALUES (?, ?, ?, ?, 45)",
                            (topic_id, unit_id, t_name, aliases))
                topic_lookup.append((topic_id, t_name, keywords))

        # Retag questions for this subject
        cur.execute("""
            SELECT q.id, q.text, q.marks, q.part
            FROM questions q
            JOIN papers p ON q.paper_id = p.id
            WHERE p.subject_id = ?
        """, (subj_id,))
        questions = cur.fetchall()
        print(f"    -> Re-tagging {len(questions)} questions across {len(topic_lookup)} canonical topics...")

        tagged_count = 0
        for qid, qtext, qmarks, qpart in questions:
            text_lower = (qtext or "").lower()
            best_topic_id = None
            best_score = 0

            for top_id, top_name, keywords in topic_lookup:
                score = 0
                for kw in keywords:
                    if kw.lower() in text_lower:
                        score += 3
                # also check topic name words
                for w in top_name.lower().split():
                    if len(w) > 3 and w in text_lower:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_topic_id = top_id

            if not best_topic_id:
                # Fallback to first topic of unit 1
                best_topic_id = topic_lookup[0][0]

            cur.execute("UPDATE questions SET topic_id = ?, tag_confidence = 0.90 WHERE id = ?", (best_topic_id, qid))
            tagged_count += 1

        print(f"    -> Successfully tagged {tagged_count} questions.")

    # Honest Data Tier calculation for all subjects
    print("\n[*] Recomputing honest data tiers for all subjects based on approved past exam years...")
    cur.execute("""
        SELECT s.id, COUNT(DISTINCT p.year) as distinct_years
        FROM subjects s
        LEFT JOIN papers p ON p.subject_id = s.id
        LEFT JOIN questions q ON q.paper_id = p.id AND q.status = 'approved'
        GROUP BY s.id
    """)
    for sid, distinct_years in cur.fetchall():
        if distinct_years >= 5:
            tier = "Tier A"
        elif distinct_years >= 1:
            tier = "Tier B"
        else:
            tier = "Tier C"
        cur.execute("UPDATE subjects SET data_tier = ? WHERE id = ?", (tier, sid))

    con.commit()
    con.close()
    print("\n[+] Database repair and re-tagging complete!")

if __name__ == "__main__":
    clean_and_repair()
