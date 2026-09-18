#!/usr/bin/env python3
"""Capture AR-vs-spec outputs for byte-for-byte comparison.

Usage: ar_spec_diff.py capture <out.json>     # run the fixed suite, save
       ar_spec_diff.py diff <a.json> <b.json> # compare two captures

Greedy (temperature 0, fixed seed) so any difference is a real divergence,
not sampling noise. Covers text and image turns, with and without thinking,
so the same suite validates the methodology on text and then gates VL.
"""
import json, sys, urllib.request, base64, zlib, struct

URL = "http://100.115.45.41:11435/v1/chat/completions"
MODEL = "qwen3.8:27b"

W = H = 64
rows = b''.join(b'\x00' + b''.join(
    bytes((220, 40, 40)) if 16 <= x < 48 and 16 <= y < 48 else bytes((255, 255, 255))
    for x in range(W)) for y in range(H))
ch = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
png = (b'\x89PNG\r\n\x1a\n'
       + ch(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
       + ch(b'IDAT', zlib.compress(rows, 9)) + ch(b'IEND', b''))
IMG = "data:image/png;base64," + base64.b64encode(png).decode()

CASES = [
    ("text-short",   [{"role": "user", "content": "Count from 1 to 12, one number per line."}]),
    ("text-prose",   [{"role": "user", "content": "In exactly three sentences, explain what a prime number is."}]),
    ("text-multi",   [{"role": "user", "content": "My code is BLUE-9."},
                      {"role": "assistant", "content": "Noted, BLUE-9."},
                      {"role": "user", "content": "Repeat my code, then count 1 to 5."}]),
    ("image-describe", [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one sentence, then count from 1 to 8."},
        {"type": "image_url", "image_url": {"url": IMG}}]}]),
    ("image-multi", [{"role": "user", "content": "Remember the word ORCHID."},
                     {"role": "assistant", "content": "Noted."},
                     {"role": "user", "content": [
                         {"type": "text", "text": "What colour is the square, and what word did I ask you to remember?"},
                         {"type": "image_url", "image_url": {"url": IMG}}]}]),
]

def run(name, msgs):
    body = json.dumps({
        "model": MODEL, "messages": msgs,
        "max_tokens": 400, "temperature": 0, "seed": 12345,
    }).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=1200))
    m = d["choices"][0]["message"]
    t = d.get("timings", {})
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "finish": d["choices"][0].get("finish_reason"),
        "completion_tokens": d["usage"]["completion_tokens"],
        "dflash": t.get("dflash"),
        "decode_tok_s": t.get("decode_tok_s"),
    }

if sys.argv[1] == "capture":
    out = {}
    for name, msgs in CASES:
        out[name] = run(name, msgs)
        r = out[name]
        print(f"{name:16s} dflash={str(r['dflash']):5s} decode={r['decode_tok_s']} "
              f"tokens={r['completion_tokens']}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(f"saved -> {sys.argv[2]}")
else:
    a = json.load(open(sys.argv[2]))
    b = json.load(open(sys.argv[3]))
    bad = 0
    for k in a:
        ca, cb = a[k], b.get(k, {})
        same = (ca["content"] == cb.get("content")
                and ca["reasoning"] == cb.get("reasoning")
                and ca["finish"] == cb.get("finish"))
        print(f"{k:16s} {'IDENTICAL' if same else 'DIVERGED'}  "
              f"({a[k]['dflash']} vs {cb.get('dflash')})")
        if not same:
            bad += 1
            for field in ("content", "reasoning"):
                x, y = ca[field], cb.get(field, "")
                if x != y:
                    i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
                    print(f"    {field}: first diff at char {i}")
                    print(f"      A: {x[max(0,i-40):i+60]!r}")
                    print(f"      B: {y[max(0,i-40):i+60]!r}")
    print("RESULT:", "ALL IDENTICAL" if bad == 0 else f"{bad} DIVERGED")
    sys.exit(0 if bad == 0 else 1)
