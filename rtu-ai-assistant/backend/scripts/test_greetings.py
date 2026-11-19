import requests

for q in ['hello', 'hey', 'namaste']:
    r = requests.post('http://127.0.0.1:8000/api/chat', json={'message': q, 'filters': {'subject': 'DSA'}}, timeout=10)
    d = r.json()
    print(f"Query: '{q}'")
    print(f"  Status: {r.status_code}")
    print(f"  Latency: {d.get('latency_ms')} ms")
    print(f"  Response: {d.get('response', '')}")
    print()
