"""prefix_cache_check.py - correctness test for --serve multi-prompt prefix reuse.

Drives a resident `strata --serve` engine over the GEN protocol:
  part 1: GEN(A) then GEN(A+B) via reuse vs a cold engine: identical greedy outputs.
  part 2 (P8c): GEN(A), GEN(X) (short unrelated), GEN(A+B): must HIT (fast prompt +
    engine log shows a resume) and match the cold engine.
  part 3 (P8c): new session S2 sharing A's long system prefix: must HIT on the shared
    prefix and match the cold engine.
  part 4: a Hermes-like 3-turn transcript, reuse order vs a cold engine.

Usage: python tools/prefix_cache_check.py [--exe ...] [--args JSON...] [--turns N]
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
    def __init__(self, exe: str, args: list[str], cwd: str, log_name: str = "engine-stderr.log"):
        import os
        env = dict(os.environ)
        env["PATH"] = r"C:\llm-local\cuda-11.8\bin" + os.pathsep + env.get("PATH", "")
        self.log_path = OUT_DIR / log_name
        self.log = open(self.log_path, "w", encoding="utf-8")
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

    def last_reuse_line(self) -> str:
        self.log.flush()
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in text.splitlines() if "prefix reuse:" in l]
        return lines[-1] if lines else ""

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
    # LONG shared system prefix (> 1 prefill chunk of 2048) so a new session sharing it can hit
    system_text = ("You are Hermes, a helpful assistant. Answer briefly. " +
                   "Note: the warehouse shelf holds boxes." * 260)
    user_texts = ["What is the capital of France? Answer in one sentence.",
                  "And its population? One sentence.",
                  "Thanks. What river runs through it? One sentence."]
    cwd = cfg.get("cwd", str(ROOT))
    eng = Engine(a.exe, out_args, cwd)
    cold = None
    try:
        # --- part 1: A then A+B via reuse vs cold ---
        msgs = [{"role": "system", "content": system_text},
                {"role": "user", "content": user_texts[0]}]
        a_ids = tok.encode(tpl.render(msgs, tools=tools, enable_thinking=False), parse_special=True)
        b_extra = tok.encode(" Elaborate with one more short sentence.", parse_special=True)
        ab_ids = a_ids + b_extra
        x_ids = tok.encode("Completely unrelated prompt about volcanoes and tides." * 6, parse_special=True)
        print(f"A={len(a_ids)} A+B={len(ab_ids)} X={len(x_ids)}", flush=True)

        out_a, done_a = eng.gen(a_ids, a.max_new)
        print("GEN(A) done:", done_a, flush=True)
        print("  cache:", eng.last_reuse_line(), flush=True)
        out_reuse, done_reuse = eng.gen(ab_ids, a.max_new)
        print("GEN(A+B) via reuse done:", done_reuse, flush=True)
        print("  cache:", eng.last_reuse_line(), flush=True)
        eng.close()
        # TRUE cold reference: a fresh engine whose cache has never seen this prompt family
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        out_cold, done_cold = cold.gen(ab_ids, a.max_new)
        print("GEN(A+B) cold done:", done_cold, flush=True)
        cold.close()
        cold = None

        ok1 = out_reuse == out_cold
        print(f"part1 reuse-vs-cold token sequences identical: {ok1} "
              f"(len {len(out_reuse)} vs {len(out_cold)})", flush=True)
        if not ok1:
            for i, (r, c) in enumerate(zip(out_reuse, out_cold)):
                if r != c:
                    print(f"  first divergence at {i}: reuse={r} cold={c}", flush=True)
                    break

        # --- part 2 (P8c): A, then short unrelated X, then A+B must HIT and match cold ---
        # (fresh engine: A and X build entries, A+B must reuse A's entry despite X in between)
        eng = Engine(a.exe, out_args, cwd, log_name="engine-stderr2.log")
        eng.gen(a_ids, a.max_new)
        out_x, done_x = eng.gen(x_ids, a.max_new)
        print("GEN(X) done:", done_x, flush=True)
        out_ax, done_ax = eng.gen(ab_ids, a.max_new)
        print("GEN(A+B) after X done:", done_ax, flush=True)
        line2 = eng.last_reuse_line()
        print("  cache:", line2, flush=True)
        hit2 = "resume at" in line2
        ok2 = out_ax == out_cold
        print(f"part2 hit-after-X: hit={hit2} identical-to-cold={ok2} (len {len(out_ax)})", flush=True)
        if not ok2:
            for i, (r, c) in enumerate(zip(out_ax, out_cold)):
                if r != c:
                    print(f"  first divergence at {i}: repair={r} cold={c}", flush=True)
                    break

        # --- part 3 (P8c): new session S2 sharing A's system prefix must HIT and match cold ---
        msgs2 = [{"role": "system", "content": system_text},
                 {"role": "user", "content": "What is the capital of Japan? Answer in one sentence."}]
        s2_ids = tok.encode(tpl.render(msgs2, tools=tools, enable_thinking=False), parse_special=True)
        lcp = 0
        while lcp < min(len(s2_ids), len(a_ids)) and s2_ids[lcp] == a_ids[lcp]:
            lcp += 1
        print(f"S2={len(s2_ids)} shared-prefix LCP(S2,A)={lcp}", flush=True)
        out_s2, done_s2 = eng.gen(s2_ids, a.max_new)
        print("GEN(S2) done:", done_s2, flush=True)
        line3 = eng.last_reuse_line()
        print("  cache:", line3, flush=True)
        hit3 = "resume at" in line3
        eng.close()
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        out_s2c, done_s2c = cold.gen(s2_ids, a.max_new)
        print("GEN(S2) cold done:", done_s2c, flush=True)
        cold.close()
        cold = None
        ok3 = out_s2 == out_s2c
        print(f"part3 new-session hit: hit={hit3} identical-to-cold={ok3} (len {len(out_s2)})", flush=True)
        if not ok3:
            for i, (r, c) in enumerate(zip(out_s2, out_s2c)):
                if r != c:
                    print(f"  first divergence at {i}: s2={r} cold={c}", flush=True)
                    break

        # --- part 5 (P8d): entry from a LONGER prompt (A+B+C), then a SHORTER request
        # sharing only A's prefix -> must HIT (the old MTP skip guard blocked every shorter
        # prompt) and be identical to cold.  short5 is a strict truncation of long5, longer
        # than one prefill chunk (so a snapshot sits at or below it) but clearly shorter.
        filler = tok.encode((" The river runs through the city and the bridges cross it." * 300),
                            parse_special=True)
        long5_ids = a_ids + filler
        assert len(long5_ids) >= 3500, f"long5 too short for the P8d case: {len(long5_ids)}"
        k5 = len(long5_ids) - 1200
        assert k5 >= 2300, f"P8d cut {k5} leaves no snapshot below it"
        short5_ids = long5_ids[:k5]
        print(f"long5={len(long5_ids)} short5={len(short5_ids)}", flush=True)
        eng = Engine(a.exe, out_args, cwd, log_name="engine-stderr4.log")
        eng.gen(long5_ids, a.max_new)
        out_s5, done_s5 = eng.gen(short5_ids, a.max_new)
        print("GEN(short5) after long5 done:", done_s5, flush=True)
        line5 = eng.last_reuse_line()
        print("  cache:", line5, flush=True)
        hit5 = "resume at" in line5
        eng.close()
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        out_s5c, done_s5c = cold.gen(short5_ids, a.max_new)
        print("GEN(short5) cold done:", done_s5c, flush=True)
        cold.close()
        cold = None
        ok5 = out_s5 == out_s5c
        print(f"part5 longer->shorter hit: hit={hit5} identical-to-cold={ok5} "
              f"(len {len(out_s5)}/{len(out_s5c)})", flush=True)
        if not ok5:
            for i, (r, c) in enumerate(zip(out_s5, out_s5c)):
                if r != c:
                    print(f"  first divergence at {i}: s5={r} cold={c}", flush=True)
                    break

        # --- part 6 (P8e): the LCP lands just after an <|im_end|> far from the 2048
        # chunk grid -> the resume point must equal that message boundary exactly,
        # and the output must match cold.
        import re
        IM_END = 248046
        CHUNK = 2048

        def sys_end_for(repeat: int) -> tuple[list[int], int]:
            s = ("You are Hermes, a helpful assistant. Answer briefly. " +
                 "Note: the warehouse shelf holds boxes." * repeat)
            m = [{"role": "system", "content": s},
                 {"role": "user", "content": user_texts[0]}]
            p = tok.encode(tpl.render(m, tools=tools, enable_thinking=False), parse_special=True)
            return p, p.index(IM_END) + 1  # first <|im_end|> closes the system message

        base_ids, sys_end = sys_end_for(260)
        if sys_end <= 300 or min(sys_end % CHUNK, CHUNK - sys_end % CHUNK) <= 384:
            for r in (100, 140, 180, 220, 300):
                base_ids, sys_end = sys_end_for(r)
                if sys_end > 300 and min(sys_end % CHUNK, CHUNK - sys_end % CHUNK) > 384:
                    break
        grid_dist = min(sys_end % CHUNK, CHUNK - sys_end % CHUNK)
        print(f"part6 base={len(base_ids)} sys_end={sys_end} (grid dist {grid_dist})", flush=True)
        assert sys_end > 300 and grid_dist > 384, "no off-grid sys_end found"
        p1_ids = base_ids + filler[:1500]
        tail6 = tok.encode(" A completely different question about railways and harbors.",
                           parse_special=True)
        p2_ids = base_ids[:sys_end] + tail6
        lcp6 = 0
        while lcp6 < min(len(p1_ids), len(p2_ids)) and p1_ids[lcp6] == p2_ids[lcp6]:
            lcp6 += 1
        print(f"part6 p1={len(p1_ids)} p2={len(p2_ids)} LCP={lcp6}", flush=True)
        assert lcp6 == sys_end, f"LCP {lcp6} != sys_end {sys_end}"
        eng = Engine(a.exe, out_args, cwd, log_name="engine-stderr5.log")
        eng.gen(p1_ids, a.max_new)
        out_p6, done_p6 = eng.gen(p2_ids, a.max_new)
        print("GEN(p2) after p1 done:", done_p6, flush=True)
        line6 = eng.last_reuse_line()
        print("  cache:", line6, flush=True)
        m6 = re.search(r"resume at (\d+)", line6)
        resume6 = int(m6.group(1)) if m6 else -1
        hit6 = resume6 == sys_end
        print(f"part6 resume={resume6} want={sys_end} exact={hit6}", flush=True)
        eng.close()
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        out_p6c, done_p6c = cold.gen(p2_ids, a.max_new)
        print("GEN(p2) cold done:", done_p6c, flush=True)
        cold.close()
        cold = None
        ok6 = out_p6 == out_p6c
        print(f"part6 msg-boundary resume: exact={hit6} identical-to-cold={ok6} "
              f"(len {len(out_p6)}/{len(out_p6c)})", flush=True)
        if not ok6:
            for i, (r, c) in enumerate(zip(out_p6, out_p6c)):
                if r != c:
                    print(f"  first divergence at {i}: p6={r} cold={c}", flush=True)
                    break

        # --- part 7 (P8f): in-session growing turns (Hermes shape: each turn appends an
        # assistant tool_call + tool result, no new user text).  Each turn's resume point must be
        # >= the previous prompt length - small slack (BPE merge at the boundary costs ~1 token);
        # the old suffix-length commit gate stuck every turn at the first turn's prefix instead.
        msgs7 = [{"role": "system", "content": system_text},
                 {"role": "user", "content": user_texts[0]}]
        calls7 = [[{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
                  [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}}],
                  [{"function": {"name": "get_weather", "arguments": {"city": "Lyon"}}}]]
        results7 = ['{"temp": "18C"}', '{"temp": "19C"}', '{"temp": "20C"}']
        prompts7 = []
        for i in range(4):
            prompts7.append(tok.encode(tpl.render(msgs7, tools=tools, enable_thinking=False),
                                       parse_special=True))
            if i < 3:
                msgs7.append({"role": "assistant", "content": "", "tool_calls": calls7[i]})
                msgs7.append({"role": "tool", "content": results7[i]})
        print("part7 turn prompt lens:", [len(p) for p in prompts7], flush=True)
        eng = Engine(a.exe, out_args, cwd, log_name="engine-stderr6.log")
        turn7_reuse, lines7 = [], []
        for p in prompts7:
            turn7_reuse.append(eng.gen(p, a.max_new))
            lines7.append(eng.last_reuse_line())
        for i, l in enumerate(lines7):
            print(f"  turn{i+1} cache:", l, flush=True)
        resumes7 = [int(m.group(1)) if (m := re.search(r"resume at (\d+)", l)) else -1
                    for l in lines7]
        ok7 = True
        for i in range(1, 4):
            want, got = len(prompts7[i - 1]) - 8, resumes7[i]
            good = got >= want
            ok7 = ok7 and good
            print(f"part7 turn{i+1} resume={got} want>={want} (prev_len {len(prompts7[i-1])}): "
                  f"{'OK' if good else 'STUCK'}", flush=True)
        eng.close()
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        turn7_cold = [cold.gen(p, a.max_new) for p in prompts7]
        cold.close()
        cold = None
        ok7c = all(r[0] == c[0] for r, c in zip(turn7_reuse, turn7_cold))
        print(f"part7 growing-turns resume-ok={ok7} identical-to-cold={ok7c}", flush=True)
        ok7 = ok7 and ok7c

        # --- part 8 (P8g): >=8 growing turns, then a NEW session sharing only the system
        # prefix must still HIT (shared prefix segments survive long sessions: every descendant
        # entry references them, so eviction of middle turns cannot lose the system prefix).
        msgs8 = [{"role": "system", "content": system_text},
                 {"role": "user", "content": user_texts[0]}]
        prompts8 = []
        for i in range(10):
            prompts8.append(tok.encode(tpl.render(msgs8, tools=tools, enable_thinking=False),
                                       parse_special=True))
            msgs8.append({"role": "assistant", "content": "",
                          "tool_calls": [{"function": {"name": "get_weather",
                                                      "arguments": {"city": f"Paris-{i}"}}}]})
            msgs8.append({"role": "tool", "content": f'{{"result": "ok-{i}"}}'})
        print("part8 turn prompt lens:", [len(p) for p in prompts8], flush=True)
        eng = Engine(a.exe, out_args, cwd, log_name="engine-stderr7.log")
        turn8_reuse, lines8 = [], []
        for p in prompts8:
            turn8_reuse.append(eng.gen(p, a.max_new))
            lines8.append(eng.last_reuse_line())
        for i, l in enumerate(lines8):
            print(f"  turn{i+1} cache:", l, flush=True)
        resumes8 = [int(m.group(1)) if (m := re.search(r"resume at (\d+)", l)) else -1
                    for l in lines8]
        ok8 = True
        for i in range(1, 10):
            want, got = len(prompts8[i - 1]) - 8, resumes8[i]
            good = got >= want
            ok8 = ok8 and good
            print(f"part8 turn{i+1} resume={got} want>={want} (prev_len {len(prompts8[i-1])}): "
                  f"{'OK' if good else 'STUCK'}", flush=True)
        msgs8n = [{"role": "system", "content": system_text},
                  {"role": "user", "content": "What is the capital of Italy? Answer in one sentence."}]
        s8_ids = tok.encode(tpl.render(msgs8n, tools=tools, enable_thinking=False), parse_special=True)
        lcp8 = 0
        while lcp8 < min(len(s8_ids), len(prompts8[0])) and s8_ids[lcp8] == prompts8[0][lcp8]:
            lcp8 += 1
        print(f"S8={len(s8_ids)} shared-prefix LCP(S8,turn1)={lcp8}", flush=True)
        out_s8, done_s8 = eng.gen(s8_ids, a.max_new)
        print("GEN(S8) done:", done_s8, flush=True)
        line8 = eng.last_reuse_line()
        print("  cache:", line8, flush=True)
        hit8 = "resume at" in line8
        eng.close()
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        turn8_cold = [cold.gen(p, a.max_new) for p in prompts8]
        out_s8c, done_s8c = cold.gen(s8_ids, a.max_new)
        print("GEN(S8) cold done:", done_s8c, flush=True)
        cold.close()
        cold = None
        ok8c = all(r[0] == c[0] for r, c in zip(turn8_reuse, turn8_cold)) and out_s8 == out_s8c
        print(f"part8 10-turns resume-ok={ok8} new-session hit={hit8} identical-to-cold={ok8c}",
              flush=True)
        ok8 = ok8 and hit8 and ok8c

        # --- part 4: 3-turn transcript, reuse order vs the cold engine ---
        replies = ["Paris is the capital of France.", "About 2.1 million in the city proper.",
                   "The Seine runs through Paris."]
        prompts = render_turns(tok, tpl, tools, system_text, user_texts, replies)
        print("turn prompt lens:", [len(p) for p in prompts], flush=True)
        eng = Engine(a.exe, out_args, cwd, log_name="engine-stderr3.log")
        turn_reuse = [eng.gen(p, a.max_new) for p in prompts]
        eng.close()
        cold = Engine(a.exe, out_args, cwd, log_name="engine-cold-stderr.log")
        turn_cold = [cold.gen(p, a.max_new) for p in prompts]
        cold.close()
        cold = None
        ok4 = all(r[0] == c[0] for r, c in zip(turn_reuse, turn_cold))
        for i, (r, c) in enumerate(zip(turn_reuse, turn_cold)):
            print(f"turn{i+1} reuse==cold: {r[0] == c[0]} lens {len(r[0])}/{len(c[0])} "
                  f"reuse:{r[1]} cold:{c[1]}", flush=True)

        ok = ok1 and ok2 and hit2 and ok3 and hit3 and ok4 and ok5 and hit5 and ok6 and hit6 and ok7 and ok8
        print("RESULT:", "PASS" if ok else "FAIL", flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "result.json").write_text(json.dumps({
            "part1_identical": ok1, "part1_len": len(out_reuse),
            "part2_hit": hit2, "part2_identical": ok2,
            "part3_hit": hit3, "part3_identical": ok3, "part3_lcp": lcp,
            "part5_hit": hit5, "part5_identical": ok5,
            "part5_long": len(long5_ids), "part5_short": len(short5_ids),
            "part6_resume": resume6, "part6_sys_end": sys_end,
            "part6_exact": hit6, "part6_identical": ok6,
            "part7_resumes": resumes7, "part7_lens": [len(p) for p in prompts7],
            "part7_ok": ok7,
            "part8_resumes": resumes8, "part8_lens": [len(p) for p in prompts8],
            "part8_hit": hit8, "part8_lcp": lcp8, "part8_ok": ok8,
            "turns_identical": [r[0] == c[0] for r, c in zip(turn_reuse, turn_cold)],
            "a_len": len(a_ids), "ab_len": len(ab_ids)}, indent=1), encoding="utf-8")
        return 0 if ok else 1
    finally:
        try:
            eng.close()
        except Exception:
            pass
        if cold is not None:
            try:
                cold.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
