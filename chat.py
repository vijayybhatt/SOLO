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
    """Small model decides: is this a fact worth remembering?"""
    verdict = filter_model.invoke(FILTER_PROMPT.format(msg=msg)).content.strip().upper()
    if DEBUG:
        print(f"[debug] filter verdict: {verdict}")
    return verdict.startswith("KEEP")

def recall(query: str, k: int = 4, min_relevance: float = 0.25) -> str:
    """Pull stored facts relevant to the query."""
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

        # 1. Retrieve relevant facts
        memories = recall(user_input)

        # 2. Generate response (streaming)
        print("\nBot: ", end="", flush=True)
        full_response = ""
        for chunk in chat_model.stream([
            SystemMessage(content=SYSTEM.format(memories=memories)),
            HumanMessage(content=user_input),
        ]):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
        print("\n")

        # 3. Store ONLY filtered user facts — never bot replies
        if should_store(user_input):
            memory.add_texts([f"User fact: {user_input}"])

if __name__ == "__main__":
    main()