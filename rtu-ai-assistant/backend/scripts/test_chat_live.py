import requests, json

payload = {
    'message': 'Explain AVL tree rotations with an example',
    'filters': {'subject': 'DSA'}
}
r = requests.post('http://127.0.0.1:8000/api/chat', json=payload, timeout=60)
print('HTTP Status:', r.status_code)
d = r.json()
print('latency_ms:', d.get('latency_ms'))
citations = d.get('citations', [])
print('citations count:', len(citations))
for idx, c in enumerate(citations):
    print(f"  {idx+1}. Year: {c.get('year')} | Paper: {c.get('source_pdf')} | Subj: {c.get('subject')} | QID: {c.get('question_id')}")

ans = d.get('response', '') or d.get('assistant_reply', '')
print('\nAnswer preview (first 400 chars):')
print(ans[:400])
