"""Ask the locally-running LM Studio model a question, optionally with file context.

Token-saver pattern for the orchestrating agent:
    Big inputs (long files, grep dumps, raw search hits) go to the local
    qwen3.6-27b once.  Only the small distilled answer comes back into
    the agent's conversation.  Net savings = (input tokens) - (answer
    tokens), which is usually 5-50× for summarization-style tasks.

Usage
-----
    # Inline question, no file context
    uv run python scripts/ask_local_llm.py "Summarise the rerank flow in 3 bullets."

    # With one or more files appended as context
    uv run python scripts/ask_local_llm.py \
        --file app/services/reranker.py \
        --file app/services/lm_studio.py \
        "Identify any duplicated logic between these two modules in <=80 words."

    # Read prompt from stdin (handy for piping grep output)
    grep -rn "TODO" docs/ | uv run python scripts/ask_local_llm.py --stdin \
        "Group these TODOs by topic, return a 2-line summary per group."

Notes
-----
* Honours the same ``LM_STUDIO_*`` env vars as the rest of the service
  (``LM_STUDIO_URL``, ``LM_STUDIO_RERANK_MODEL`` for the chat model id,
  ``LM_STUDIO_TIMEOUT``).
* ``/no_think`` user-message trailer + ``chat_template_kwargs`` are sent
  belt-and-suspenders style; for thinking-locked Qwen3 presets the answer
  ends up in ``reasoning_content`` and we surface it transparently.
* Exits non-zero on any failure (LM Studio unreachable, model not loaded,
  empty response) so shell pipelines fail loudly rather than silently
  feeding garbage downstream.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure ``app/`` is importable when the script is run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.services import lm_studio  # noqa: E402

_SYSTEM_PROMPT = (
    "You are a precise technical assistant helping an automated coding agent "
    "save context tokens.  Always:\n"
    "- Answer in the requested length / format exactly.\n"
    "- Use plain prose or short bullets; no preamble like 'Sure, here is…'.\n"
    "- If the input is too ambiguous to answer, say 'INSUFFICIENT' and stop."
)

_NO_THINK_TRAILER = "\n\n/no_think"


def _read_files(paths: list[str]) -> str:
    """Concatenate file contents with clear delimiters so the model can cite them."""
    chunks: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            sys.stderr.write(f"warning: file not found: {p}\n")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"--- BEGIN {path} ---\n{text}\n--- END {path} ---")
    return "\n\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("prompt", nargs="?", default="", help="The question or instruction")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Path to a file whose contents should be included as context (repeatable).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read additional context from stdin and append it to the prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Reply token budget (default 2048; thinking models need headroom).",
    )
    args = parser.parse_args()

    if not lm_studio.is_available():
        sys.stderr.write(
            "error: LM Studio not reachable at "
            f"{lm_studio.base_url() or '(unset LM_STUDIO_URL)'}\n"
        )
        return 2

    file_ctx = _read_files(args.file) if args.file else ""
    stdin_ctx = sys.stdin.read() if args.stdin else ""

    parts: list[str] = []
    if args.prompt:
        parts.append(args.prompt)
    if file_ctx:
        parts.append("Context files:\n" + file_ctx)
    if stdin_ctx:
        parts.append("Stdin:\n" + stdin_ctx)
    if not parts:
        sys.stderr.write("error: no prompt, no --file, and --stdin not set\n")
        return 2

    user_msg = "\n\n".join(parts) + _NO_THINK_TRAILER

    response = lm_studio.chat_complete(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=args.max_tokens,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False},
    )
    if not response:
        sys.stderr.write("error: empty response from LM Studio\n")
        return 1

    print(response.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
