# SOLO — Self-Learning Chatbot: Project Context File

> **Purpose of this file:** Upload this at the start of any new Claude conversation to restore full project context. Last updated: 2026-07-27 (end of Phase 1+3, about to start Phase 2).

---

## 1. Who / What

- **Developer:** Vijay (goes by "vijayybhatt" on GitHub), data engineer, based in Pune, learning GenAI. Solo developer.
- **Project:** "SOLO" — a self-learning chatbot. The bot itself has been given the persona name **Solo** ("your name is solo, you work for vijay").
- **Repo:** https://github.com/vijayybhatt/SOLO (private repo assumed)
- **Communication style preference:** step-by-step, one step at a time, casual tone ("bro" is fine), wants to understand *why*, not just copy commands. Beginner at git/terminal/Python tooling but learns fast and asks sharp questions.

---

## 2. The Big Idea (Architecture — Hybrid Approach v2)

Vijay initially planned a memory+retrieval "self-learning" system (knowledge graph, no weight updates). He then pushed back: "ChatGPT/Gemini/Claude all do memory, that's not new, I don't want this." After discussing real options (continual LoRA / online learning / hybrid), **he chose the HYBRID approach**:

> **Graph learns WHAT'S TRUE (facts). Weights learn HOW TO BEHAVE (LoRA fine-tuning on corrections). Both update continuously via a nightly two-track job.**

```
User message
     │
     ▼
[1] Entity Resolution ──► "which Rahul?" (ask user when ambiguous)
     │
     ▼
[2] Filter Model (small local LLM) ──► KEEP / DISCARD / REVIEW
     │
     ▼
[3] Fact Extraction ──► triples into Knowledge Graph (Neo4j)
     │                        │
     ▼                        ▼
[4] GraphRAG Retrieval ◄── active facts only
     │
     ▼
[5] MAIN LOCAL MODEL (Llama 3.1 8B + current LoRA adapter) ──► response
     │
     ▼
[6] Feedback Capture (👍/👎/corrections → Postgres)
     │
     ▼
[7] NIGHTLY JOB — two tracks:
     ├─ Track A (facts): reconcile conflicts, expire, prune graph
     └─ Track B (behavior): curate corrections → train LoRA (MLX) →
        run golden test set → promote adapter ONLY if it passes
```

### Key design decisions (agreed & locked)
- **No API models. Fully local, zero cloud spend** (hard constraint from Vijay).
- **No live per-message weight updates** (poisoning/forgetting risk) — training is nightly batch, gated by a golden test set.
- Facts NEVER go into weights (stale/undeletable) — facts live in the graph with `timestamp`, `status` (active/expired), `source_message_id`. Newest-wins conflict rule; same-day contradictions escalate to the user.
- Entity resolution = ask the user when ambiguous, don't automate fully.
- Every fact edge should eventually carry `asserted_by: user_id` (multi-user scoping — Claude's suggestion, accepted).
- Log `facts_used` per response so bad answers can be traced to bad retrieval vs. bad generation.
- Sample some filter DISCARDs into review queue (catch silent false negatives) — planned, not built yet.
- End-state retrieval will be hybrid: graph (facts) + vector store (fuzzy/semantic recall).
- Privacy: disclaimer + delete-by-user-request flow (planned).

---

## 3. Hardware & Environment

- **Dev machine: MacBook Air M5, 16GB RAM** (chosen over his ASUS VivoBook GTX 1650 4GB/8GB RAM — too weak; VivoBook may later host the deployed bot).
- 16GB strategy: 8B chat model + 3B filter fit fine (~10-11GB worst case with Neo4j+Postgres). LoRA training targets the **3B model first** via **MLX (`mlx-lm`)**, run nightly when Docker services can be paused. QLoRA on the 8B is a stretch goal (Docker stopped, overnight).

### Installed & working
- Xcode CLT, Homebrew, Python 3.12 (brew), VS Code + Python extension
- Ollama (menu bar app) with models pulled:
  - `llama3.1:8b` — main chat model
  - `qwen2.5:3b` — filter model
  - `nomic-embed-text` — embeddings (added after discovering the 8B can't serve embeddings — got `501 server does not support embeddings`)
- Docker Desktop + running Postgres container:
  ```bash
  docker run -d --name solo-pg -e POSTGRES_PASSWORD=devpass -e POSTGRES_DB=solo \
    -p 5432:5432 -v solo_pgdata:/var/lib/postgresql/data postgres:16
  ```
- venv (Python 3.12) in project root. Packages: langchain, langchain-ollama, langchain-chroma, langgraph, chromadb, python-dotenv, psycopg2-binary, sqlalchemy
- `.gitignore` includes: `venv/`, `.env`, `__pycache__/`, `memory_db/`

### Gotchas Vijay already learned (don't re-explain unless asked)
- New terminal = venv off; `source venv/bin/activate` first. Only `pip`/`python` need venv; git/docker/brew/ollama don't.
- macOS has no bare `python` outside venv — only `python3`.
- First venv was accidentally created with system Python 3.9 → deleted, recreated with `python3.12 -m venv venv`.
- Don't paste `# comments` along with shell commands.

---

## 4. Git Workflow (agreed)

- `main` = always stable, never edited directly.
- One branch per phase: `feature/phaseN-description` → PR on GitHub → merge → delete branch → branch again.
- **Current branch: `feature/phase2-feedback`** (created, pushed, tracking origin). Old stray branch `features/phase2` on remote — may still need `git push origin --delete features/phase2` (was mid-cleanup).
- Phase 1+3 code is committed and pushed to `main`.

---

## 5. Build Order & Status

| Phase | What | Status |
|---|---|---|
| 1 | Chatbot on local model + Chroma vector memory | ✅ DONE |
| 2 | Feedback capture (rating/correction → Postgres) | 🔜 NEXT — Postgres running, code not written |
| 3 | Filter model gating memory writes | ✅ DONE (pulled forward, merged into chat.py) |
| 4 | Knowledge graph + timestamped facts (Neo4j) | ⬜ |
| 5 | Entity disambiguation ("which Rahul?") | ⬜ |
| 6 | GraphRAG retrieval | ⬜ |
| 7 | Nightly Track A: graph maintenance (conflicts, expire, prune) | ⬜ |
| 8 | Golden test set (~50–100 fixed Q&A) — MUST precede training | ⬜ |
| 9 | Nightly Track B: MLX LoRA pipeline (curate→train→eval→promote/reject) | ⬜ |
| 10 | Stretch: adapter versioning/rollback, drift monitoring | ⬜ |

### Phase 2 plan (next session)
1. Table in Postgres: `interactions(id, ts, user_msg, bot_reply, facts_used, feedback)`
2. After each bot reply: press Enter to skip, or `+` / `-` / free-text correction
3. Log facts_used (the retrieved memories injected into that response)
4. Negative feedback → flagged for nightly review (later phases consume this)

---

## 6. Current Code (`chat.py`, working, on main)

```python
"""Phase 1+3: Chatbot with vector memory + filter model.
Only stores actual user facts — no questions, no bot replies, no junk."""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

DEBUG = True  # set False once tuned

# --- Models ---
chat_model = ChatOllama(model="llama3.1:8b", temperature=0.7, keep_alive="30m")
filter_model = ChatOllama(model="qwen2.5:3b", temperature=0, keep_alive="30m")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Memory ---
memory = Chroma(
    collection_name="chat_memory",
    embedding_function=embeddings,
    persist_directory="./memory_db",
)

SYSTEM = (
    "You are a helpful personal assistant.\n"
    "Below are facts about this user from past conversations. Use them ONLY if "
    "directly relevant to the current question. NEVER say 'you mentioned earlier' "
    "or 'we've talked before' — just naturally use the information as if you "
    "simply know it. If the facts are not relevant, ignore them completely.\n"
    "Known facts:\n{memories}"
)

FILTER_PROMPT = (
    "You are a memory filter. Decide if this message contains a lasting personal "
    "fact about the user worth remembering long-term (their name, location, job, "
    "preferences, relationships, plans, possessions).\n"
    "Questions, greetings, small talk, and requests are NOT facts.\n"
    "Reply with exactly one word: KEEP or DISCARD.\n\n"
    "Message: {msg}"
)

def should_store(msg: str) -> bool:
    verdict = filter_model.invoke(FILTER_PROMPT.format(msg=msg)).content.strip().upper()
    if DEBUG:
        print(f"[debug] filter verdict: {verdict}")
    return verdict.startswith("KEEP")

def recall(query: str, k: int = 4, min_relevance: float = 0.25) -> str:
    results = memory.similarity_search_with_relevance_scores(query, k=k)
    if DEBUG:
        print("\n[debug] retrieval scores:")
        for doc, score in results:
            print(f"  {score:.3f} | {doc.page_content[:70]}")
    relevant = [doc.page_content for doc, score in results if score >= min_relevance]
    return "\n".join(f"- {m}" for m in relevant) if relevant else "(none)"

def main():
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower().strip(" .!") in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        memories = recall(user_input)

        print("\nBot: ", end="", flush=True)
        full_response = ""
        for chunk in chat_model.stream([
            SystemMessage(content=SYSTEM.format(memories=memories)),
            HumanMessage(content=user_input),
        ]):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
        print("\n")

        if should_store(user_input):
            memory.add_texts([f"User fact: {user_input}"])

if __name__ == "__main__":
    main()
```

---

## 7. Debugging History (lessons already learned — don't repeat)

1. **Embeddings 501 error** — `llama3.1:8b` can't serve embeddings on his Ollama setup → switched to `nomic-embed-text` (also faster/lighter). Must `rm -rf memory_db` when changing embedding model (dimension mismatch).
2. **"Slow responses"** — was mostly non-streaming (waiting for full response) vs `ollama run`'s streaming. Fixed with `.stream()` + `keep_alive="30m"`. `ollama ps` shows GPU/CPU split if slowness returns.
3. **"We've talked before" creepy narration** — fixed via system prompt instructing silent use of memories.
4. **Vector memory poison loop (important lesson):** storing raw exchanges (incl. questions & bot's "I don't know" replies) meant identity questions retrieved past failed identity questions above the actual fact ("who am i?" matched "who am i bro? → I don't know" at 0.495 vs the real fact at 0.254). No threshold fixes wrong ordering. Fix = filter model gates writes; never store bot replies; store only user facts. This empirically motivated the filter model and, ahead, the knowledge graph (exact entity lookup vs. fuzzy similarity).
5. **Threshold tuning:** first tried `max_distance 0.8` with raw distance scores (wrong for this embedder — filtered everything out, bot remembered nothing). Now using `similarity_search_with_relevance_scores` (0→1, higher=better) with `min_relevance=0.25` + DEBUG printing of scores. Working well in tests.
6. **Filter model quality:** qwen2.5:3b correctly KEEPs facts and DISCARDs greetings/questions/meta-requests ("kindly remember this info" → DISCARD, correct since it contains no fact).

---

## 8. Verified Working Behavior (end of last session)

- Facts persist across full process restarts (Chroma persist_directory).
- Bot answers "where does your boss live?" → Pune; "what is your name?" → Solo — from memory, naturally, without narrating that it remembers.
- Memory store stays clean: only real user facts get written.
- Persona assignment persists ("your name is solo, you work for vijay").

---

## 9. Next Session TODO (start here)

1. Confirm on branch `feature/phase2-feedback`; delete stray remote branch `features/phase2` if still present.
2. Write Phase 2: SQLAlchemy models + feedback prompt loop in chat.py (schema in §5).
3. Test: chat, give `+`/`-`/correction, verify rows in Postgres (`docker exec -it solo-pg psql -U postgres -d solo`).
4. Commit small, push branch, open PR, merge to main.
5. Then Phase 4: Neo4j container + fact extraction into graph.
