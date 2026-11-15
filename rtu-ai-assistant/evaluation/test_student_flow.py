import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8000/api"

def api_call(path, data=None):
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None
    r = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers=headers)
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run():
    print("========================================")
    print("STEP 1: Student Registration")
    print("========================================")
    student_payload = {
        "name": "Keshav Sharma",
        "roll_no": "21EBKCS045",
        "email": "keshav@rtu.ac.in",
        "password": "rtupassword2026",
        "branch": "CSE",
        "year": 3,
        "college": "Rajasthan Technical University (RTU)"
    }
    
    # Clean previous if exists
    try:
        reg_resp = api_call("/auth/register", student_payload)
        user_id = reg_resp["user_id"]
        print(f"SUCCESS: Student registered! ID: {user_id}, Name: {reg_resp['name']}")
    except Exception as e:
        print(f"Registration notice: {e}, attempting login directly...")
        login_resp = api_call("/auth/login", {"roll_no_or_email": "21EBKCS045", "password": "rtupassword2026"})
        user_id = login_resp["user_id"]
        print(f"SUCCESS: Logged in! ID: {user_id}, Name: {login_resp['name']}")

    print("\n========================================")
    print("STEP 2: Student Login & Profile Me")
    print("========================================")
    login_resp = api_call("/auth/login", {"roll_no_or_email": "21EBKCS045", "password": "rtupassword2026"})
    assert login_resp["roll_no"] == "21EBKCS045"
    print(f"SUCCESS: Logged in student: {login_resp['name']} ({login_resp['roll_no']})")

    me_resp = api_call(f"/auth/me?user_id={user_id}")
    assert me_resp["name"] == "Keshav Sharma"
    print(f"SUCCESS: Profile verified: {me_resp['name']} | Branch: {me_resp['branch']} | Year: {me_resp['year']}")

    print("\n========================================")
    print("STEP 3: Student asks Question 1 (QuickSort)")
    print("========================================")
    chat1 = api_call("/chat", {
        "message": "Explain QuickSort algorithm and time complexity for 10 marks",
        "user_id": user_id,
        "language": "english"
    })
    sess_id_1 = chat1["session_id"]
    title_1 = chat1["title"]
    print(f"SUCCESS: Question 1 Answered!")
    print(f"Session ID: {sess_id_1}")
    print(f"Conversation Title: {title_1}")
    print(f"Citations returned: {len(chat1.get('citations', []))}")
    assert "QuickSort" in title_1

    print("\n========================================")
    print("STEP 4: Student asks Question 2 (Follow-up)")
    print("========================================")
    chat2 = api_call("/chat", {
        "message": "Worst case complexity kya hoti hai?",
        "session_id": sess_id_1,
        "user_id": user_id,
        "language": "hinglish"
    })
    print(f"SUCCESS: Follow-up answered! Response length: {len(chat2['assistant_message']['content'])} chars")

    print("\n========================================")
    print("STEP 5: Student starts New Conversation (AVL)")
    print("========================================")
    chat3 = api_call("/chat", {
        "message": "Define AVL tree balance factor for 2 marks",
        "user_id": user_id,
        "language": "english"
    })
    sess_id_2 = chat3["session_id"]
    title_2 = chat3["title"]
    print(f"SUCCESS: Question 3 Answered!")
    print(f"Session ID: {sess_id_2}")
    print(f"Conversation Title: {title_2}")
    assert sess_id_1 != sess_id_2

    print("\n========================================")
    print("STEP 6: Fetching Student Previous Conversations")
    print("========================================")
    sessions = api_call(f"/sessions?user_id={user_id}")
    print(f"SUCCESS: Found {len(sessions)} student conversation sessions:")
    for idx, s in enumerate(sessions):
        print(f"  [{idx+1}] ID: {s['conversation_id']}")
        print(f"      Title: '{s['title']}'")
        print(f"      Messages: {s['message_count']}")
        assert len(s["title"]) > 0, "Title must not be empty!"
        assert s["conversation_id"] is not None, "conversation_id must be populated!"

    print("\n========================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("========================================")

if __name__ == "__main__":
    run()

