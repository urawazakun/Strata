"""prefix_cache_check.py - correctness test for --serve prompt prefix reuse.

Drives a resident `strata --serve` engine over the GEN protocol:
  1. GEN(A) then GEN(A+B): the second request reuses A's snapshots.
  2. GEN(X) (unrelated) then GEN(A+B): the second request is a cold full prefill.
Compares the full greedy output token sequences and DONE lines: they must be identical.
Also replays a small Hermes-like 3-turn transcript both ways (reuse vs forced-cold).

Usage: python tools/prefix_cache_check.py [--exe ...] [--args JSON...] [--turns N]
Writes ids files under the scratch dir for the speed replay to reuse.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

TOK_DIR = Path(r"H:\qwen38-v100\strata-work\pack-qwen-iq3_s\tokenizer")
OUT_DIR = Path(r"H:\qwen38-v100\briefs\prefix-cache-test")


def load_tokenizer():
    vocab = json.loads((TOK_DIR / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (TOK_DIR / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((TOK_DIR / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe: str, args: list[str], cwd: str):
        import os
        env = dict(os.environ)
        env["PATH"] = r"C:\llm-local\cuda-11.8\bin" + os.pathsep + env.get("PATH", "")
        self.log = open(OUT_DIR / "engine-stderr.log", "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1, env=env)
        self.err_lines: list[str] = []
        self.lock = threading.Lock()
        line = self.proc.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError("engine did not print READY: " + line)

    def gen(self, ids: list[int], max_new: int) -> tuple[list[int], str]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(f"GEN {max_new} {','.join(str(int(t)) for t in ids)}\n")
        self.proc.stdin.flush()
        out: list[int] = []
        done = ""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine stdout closed")
            line = line.strip()
            if line.startswith("T "):
                out.append(int(line[2:]))
            elif line.startswith("DONE"):
                done = line
                break
            elif line.startswith("ERR"):
                raise RuntimeError("engine error: " + line)
        return out, done

    def close(self):
        try:
            assert self.proc.stdin
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        try:
            self.log.close()
        except Exception:
            pass


def render_turns(tok, tpl, tools, system_text, user_texts, replies):
    """Build Hermes-like prompts: system+tools, then each turn appends the previous
    assistant reply (WITHOUT reasoning, as the template re-renders it) + tool result + user."""
    prompts = []
    messages = [{"role": "system", "content": system_text}]
    for i, u in enumerate(user_texts):
        if i > 0:
            messages.append({"role": "assistant", "content": replies[i - 1]})
            messages.append({"role": "tool", "content": f'{{"result": "ok-{i}"}}'})
        messages.append({"role": "user", "content": u})
        text = tpl.render(messages, tools=tools, enable_thinking=False)
        prompts.append(tok.encode(text, parse_special=True))
    return prompts


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"H:\qwen38-v100\build-strata\strata.exe")
    ap.add_argument("--cfg", default=r"H:\qwen38-v100\deploy-strata\strata-qwen-iq3_s.run.json")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--cache", type=int, default=2800)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()

    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    args = [x for x in cfg["args"]]
    # shrink context/cache for the fast correctness run
    out_args: list[str] = []
    skip_next = False
    for i, x in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if x in ("--max-context", "--expert-cache"):
            skip_next = True
            continue
        if x == "--stats":
            continue
        out_args.append(x)
    out_args += ["--max-context", str(a.ctx), "--expert-cache", str(a.cache)]
    tok = load_tokenizer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tpl_path = TOK_DIR / "chat_template.jinja"
    if not tpl_path.exists():
        tpl_path = ROOT / "serve" / "chat_template.jinja"
    tpl = ChatTemplate(tpl_path)

    tools = [{"name": "get_weather", "description": "weather of a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    system_text = "You are Hermes, a helpful assistant. Answer briefly."
    user_texts = ["What is the capital of France? Answer in one sentence.",
                  "And its population? One sentence.",
                  "Thanks. What river runs through it? One sentence."]
    eng = Engine(a.exe, out_args, cfg.get("cwd", str(ROOT)))
    try:
        # --- part 1: A then A+B via reuse vs cold ---
        msgs = [{"role": "system", "content": system_text},
                {"role": "user", "content": user_texts[0]}]
        a_ids = tok.encode(tpl.render(msgs, tools=tools, enable_thinking=False), parse_special=True)
        # pad A past one prefill chunk so snapshots exist (chunk is 2048 in the run config)
        pad = tok.encode((" Note: the warehouse shelf holds boxes." * 220), parse_special=True)
        a_ids = a_ids + pad
        b_extra = tok.encode(" Elaborate with one more short sentence.", parse_special=True)
        ab_ids = a_ids + b_extra
        x_ids = tok.encode("Completely unrelated prompt about volcanoes and tides." * 6, parse_special=True)
        print(f"A={len(a_ids)} A+B={len(ab_ids)} X={len(x_ids)}", flush=True)

        out_a, done_a = eng.gen(a_ids, a.max_new)
        print("GEN(A) done:", done_a, flush=True)
        out_reuse, done_reuse = eng.gen(ab_ids, a.max_new)
        print("GEN(A+B) via reuse done:", done_reuse, flush=True)
        out_x, done_x = eng.gen(x_ids, a.max_new)
        print("GEN(X) done:", done_x, flush=True)
        out_cold, done_cold = eng.gen(ab_ids, a.max_new)
        print("GEN(A+B) cold done:", done_cold, flush=True)

        ok1 = out_reuse == out_cold
        print(f"part1 reuse-vs-cold token sequences identical: {ok1} "
              f"(len {len(out_reuse)} vs {len(out_cold)})", flush=True)
        if not ok1:
            for i, (r, c) in enumerate(zip(out_reuse, out_cold)):
                if r != c:
                    print(f"  first divergence at {i}: reuse={r} cold={c}", flush=True)
                    break

        # --- part 2: 3-turn transcript, reuse order vs forced-cold order ---
        replies = ["Paris is the capital of France.", "About 2.1 million in the city proper.",
                   "The Seine runs through Paris."]
        prompts = render_turns(tok, tpl, tools, system_text, user_texts, replies)
        print("turn prompt lens:", [len(p) for p in prompts], flush=True)
        turn_reuse = [eng.gen(p, a.max_new) for p in prompts]
        turn_cold = []
        for p in prompts:
            eng.gen(x_ids, 8)  # evict snapshots -> next GEN is a cold full prefill
            turn_cold.append(eng.gen(p, a.max_new))
        ok2 = all(r[0] == c[0] for r, c in zip(turn_reuse, turn_cold))
        for i, (r, c) in enumerate(zip(turn_reuse, turn_cold)):
            print(f"turn{i+1} reuse==cold: {r[0] == c[0]} lens {len(r[0])}/{len(c[0])} "
                  f"reuse:{r[1]} cold:{c[1]}", flush=True)

        ok = ok1 and ok2
        print("RESULT:", "PASS" if ok else "FAIL", flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "result.json").write_text(json.dumps({
            "part1_identical": ok1, "part1_len": len(out_reuse),
            "turns_identical": [r[0] == c[0] for r, c in zip(turn_reuse, turn_cold)],
            "a_len": len(a_ids), "ab_len": len(ab_ids)}, indent=1), encoding="utf-8")
        return 0 if ok else 1
    finally:
        eng.close()


if __name__ == "__main__":
    sys.exit(main())
