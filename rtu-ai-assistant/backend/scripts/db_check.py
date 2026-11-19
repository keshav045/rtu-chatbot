import sqlite3, os

db = sqlite3.connect("rtu-ai-assistant/backend/data/storage/rtu_metadata.db")

null_q = db.execute("SELECT COUNT(*) FROM questions WHERE text IS NULL OR trim(text) = ''").fetchone()[0]
print(f"Questions with NULL/empty text: {null_q}")

orphan_q = db.execute("SELECT COUNT(*) FROM questions q LEFT JOIN papers p ON q.paper_id = p.id WHERE p.id IS NULL").fetchone()[0]
print(f"Orphan questions (no parent paper): {orphan_q}")

orphan_p = db.execute("SELECT COUNT(*) FROM papers p LEFT JOIN subjects s ON p.subject_id = s.id WHERE s.id IS NULL").fetchone()[0]
print(f"Orphan papers (no parent subject): {orphan_p}")

q_count = db.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
print(f"Total questions in DB: {q_count}")

dup_ids = db.execute("SELECT id, COUNT(*) c FROM questions GROUP BY id HAVING c > 1").fetchall()
print(f"Duplicate question IDs: {len(dup_ids)}")

idx_path = "rtu-ai-assistant/backend/data/indexes"
idx_files = os.listdir(idx_path) if os.path.exists(idx_path) else []
print(f"Index files: {idx_files}")

# Check env file location
for env_path in ["rtu-ai-assistant/backend/.env", ".env", "rtu-ai-assistant/.env"]:
    if os.path.exists(env_path):
        lines = open(env_path).readlines()
        print(f".env at {env_path}: {len(lines)} lines")
        for l in lines:
            if "=" in l and not l.strip().startswith("#"):
                key = l.split("=")[0].strip()
                val = l.split("=", 1)[1].strip()
                masked = val[:4] + "..." if len(val) > 4 else "(empty)"
                print(f"  {key}: {masked}")
        break
else:
    print(".env file: NOT FOUND in any expected location")

db.close()
print("DB check done.")
