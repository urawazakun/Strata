"""prefix_cache_speed.py - 3-turn ~11.5K-prefix agent replay for the prefix-cache speed test.

Builds a Hermes-like transcript (system + big tool list, 3 turns with fixed assistant
replies + tool results), drives `strata --serve` directly, and prints DONE lines.
The engine's stderr (with the `strata serve:` timing lines) goes to the log file.
Saves turn prompts to OUT_DIR/turnN.ids for reruns.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from prefix_cache_check import Engine, load_tokenizer  # noqa: E402

TOK_DIR = Path(r"H:\qwen38-v100\strata-work\pack-qwen-iq3_s\tokenizer")
OUT_DIR = Path(r"H:\qwen38-v100\briefs\prefix-cache-test")


def make_tools(n: int):
    verbs = ["get", "list", "describe", "update", "delete", "create", "search", "fetch"]
    nouns = ["weather", "calendar", "ticket", "invoice", "shipment", "server", "database",
             "repository", "pipeline", "alert", "dashboard", "report", "user", "order"]
    tools = []
    for i in range(n):
        v, w = verbs[i % len(verbs)], nouns[(i * 7) % len(nouns)]
        tools.append({
            "name": f"{v}_{w}_{i}",
            "description": (f"Perform the {v} operation on {w} resource number {i}. Takes a location, "
                            f"a date range, an optional priority flag and a free-form query string. "
                            f"Returns a structured status object with the operation outcome."),
            "parameters": {"type": "object",
                           "properties": {"location": {"type": "string"},
                                          "date_from": {"type": "string"},
                                          "date_to": {"type": "string"},
                                          "priority": {"type": "integer"},
                                          "query": {"type": "string"}},
                           "required": ["location", "query"]}})
    return tools


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"H:\qwen38-v100\build-strata\strata.exe")
    ap.add_argument("--cfg", default=r"H:\qwen38-v100\deploy-strata\strata-qwen-iq3_s.run.json")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--target", type=int, default=11500)
    a = ap.parse_args()

    cfg = json.loads(Path(a.cfg).read_text(encoding="utf-8"))
    args = [x for x in cfg["args"] if x != "--stats"]
    tok = load_tokenizer()
    tpl_path = TOK_DIR / "chat_template.jinja"
    if not tpl_path.exists():
        tpl_path = ROOT / "serve" / "chat_template.jinja"
    tpl = ChatTemplate(tpl_path)

    system_text = ("You are Hermes, a helpful AI assistant. You answer concisely and call tools "
                   "when needed. Follow the conversation and use prior tool results.")
    # size the tool list so system+tools ~= target tokens
    tools = make_tools(8)
    probe = tok.encode(tpl.render([{"role": "system", "content": system_text}],
                                  tools=tools, enable_thinking=False), parse_special=True)
    per_tool = (len(probe) - 60) / 8
    n_tools = max(8, int((a.target - 60) / per_tool))
    tools = make_tools(n_tools)
    sys_ids = tok.encode(tpl.render([{"role": "system", "content": system_text}],
                                    tools=tools, enable_thinking=False), parse_special=True)
    print(f"system+tools: {len(sys_ids)} tokens with {n_tools} tools", flush=True)

    user_texts = ["Check the weather in Osaka and tell me in two sentences whether to bring an umbrella.",
                  "Now also check Tokyo and compare the two cities in two sentences.",
                  "Thanks. Which of the two is warmer right now? One sentence."]
    replies = ["I will check the weather in Osaka for you.",
               "Osaka first, then Tokyo, then the comparison.",
               "Comparing the two readings now."]
    tool_names = [tools[3]["name"], tools[11 % n_tools]["name"], tools[5]["name"]]

    messages: list[dict] = [{"role": "system", "content": system_text}]
    prompts = []
    for i, u in enumerate(user_texts):
        if i > 0:
            messages.append({"role": "assistant", "content": replies[i - 1]})
            messages.append({"role": "tool", "content": json.dumps(
                {"name": tool_names[i - 1], "result": f"sunny, 21C (reading {i})"})})
        messages.append({"role": "user", "content": u})
        text = tpl.render(messages, tools=tools, enable_thinking=False)
        prompts.append(tok.encode(text, parse_special=True))
    print("turn prompt lens:", [len(p) for p in prompts], flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(prompts):
        (OUT_DIR / f"turn{i + 1}.ids").write_text(",".join(map(str, p)), encoding="utf-8")

    eng = Engine(a.exe, args, cfg.get("cwd", str(ROOT)))
    try:
        t0 = time.time()
        for i, p in enumerate(prompts):
            out, done = eng.gen(p, a.max_new)
            print(f"turn{i + 1}: {done} (wall {time.time() - t0:.0f}s)", flush=True)
    finally:
        eng.close()
    print("engine stderr ->", OUT_DIR / "engine-stderr.log", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
