"""Retrieval-Augmented Generation over the retrieved time-series context.

The retrieval half always runs (it is the point of the challenge). The
generation half is optional: if an OpenAI key is configured we ask a chat model
to synthesize an answer grounded in the retrieved rows; otherwise we return a
grounded extractive answer built from the top matches, so the endpoint works
with no external LLM.
"""

from __future__ import annotations

import os


def _context_block(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        ts = h["ts"].isoformat() if hasattr(h["ts"], "isoformat") else str(h["ts"])
        lines.append(
            f"[{i}] {ts} | {h['site']}/{h['device_id']} | {h['sensor_type']}"
            f"={h['value']}{h.get('unit') or ''} | {h['note']}"
        )
    return "\n".join(lines)


def _extractive(question: str, hits: list[dict]) -> str:
    if not hits:
        return "No matching readings were found in the selected time range and filters."
    top = hits[0]
    ts = top["ts"].isoformat() if hasattr(top["ts"], "isoformat") else str(top["ts"])
    parts = [
        f"Based on the {len(hits)} most similar readings, the closest match is "
        f"{top['device_id']} at {top['site']} on {ts}: {top['note']} "
        f"({top['sensor_type']}={top['value']}{top.get('unit') or ''}, "
        f"similarity distance {top['distance']:.3f})."
    ]
    other = hits[1:4]
    if other:
        parts.append("Related readings:")
        for h in other:
            hts = h["ts"].isoformat() if hasattr(h["ts"], "isoformat") else str(h["ts"])
            parts.append(f"- {hts} {h['device_id']} ({h['site']}): {h['note']}")
    return "\n".join(parts)


def synthesize(question: str, hits: list[dict]) -> tuple[str, str]:
    """Return (answer, mode)."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            system = (
                "You are a reliability engineer. Answer the question using ONLY the "
                "provided time-series readings. Cite readings as [n]. If the context "
                "does not contain the answer, say so."
            )
            user = f"Question: {question}\n\nRetrieved readings:\n{_context_block(hits)}"
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
            )
            return resp.choices[0].message.content or "", "llm"
        except Exception as exc:  # fall back to extractive on any failure
            text = _extractive(question, hits)
            return f"{text}\n\n(LLM synthesis unavailable: {exc})", "extractive-fallback"
    return _extractive(question, hits), "extractive"
