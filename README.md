# Simple AI Research Agent (LangGraph + LangChain + Streamlit)

**Architecture:**
`User Question → Web Search (DuckDuckGo) → Collect Results → LLM Summarization (LangGraph) → Final Answer`

Files in this project:
- `app.py` – the full agent + Streamlit UI (one file, beginner-friendly)
- `requirements.txt` – all dependencies
- `README.md` – this guide

---

## How the agent works

1. **You enter a question** in the Streamlit text box.
2. **`search_node`** runs `DuckDuckGoSearchRun()` (from LangChain) to fetch
   relevant text from the web for that question.
3. **`summarize_node`** sends the raw search results + your question to an
   LLM (`gpt-4o-mini` via `ChatOpenAI`), asking it to write a clear answer
   based only on that information.
4. These two steps are wired together with **LangGraph**: a tiny graph with
   two nodes (`search → summarize`) and a start/end point. LangGraph just
   manages "what runs next" — for a workflow this simple, it's basically a
   two-step pipeline, but it gives you an easy path to add more steps later
   (e.g. a second search, a fact-check node) without restructuring the code.
5. Streamlit displays the final answer, plus the raw search results in a
   collapsible section so you can double check the source material.

No databases, no multi-agent setup, no long-term memory — just the four
steps you asked for.

---

## Phase 1 — Build & Test in Google Colab

Google Colab notebooks can't render a live Streamlit app inline, so the
easiest and most reliable way to test is to **run the agent's logic
directly in notebook cells first**. This confirms your API key, search
tool, and LangGraph workflow all work before you touch any UI code.

### Step 1 — Install dependencies
In a Colab cell:
```python
!pip install -q langgraph langchain langchain-community langchain-openai duckduckgo-search streamlit
```

### Step 2 — Add your API key securely
Never hardcode your API key in the notebook. Use Colab's built-in **Secrets**
manager:
1. Click the 🔑 (key) icon in the left sidebar of Colab.
2. Add a new secret named `OPENAI_API_KEY` and paste your key as the value.
3. Toggle "Notebook access" on for that secret.
4. In a cell, load it into the environment:

```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

### Step 3 — Test the agent logic (no UI yet)
Paste the core logic (without the Streamlit part) into a cell and run it:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    question: str
    search_results: str
    answer: str

def search_node(state):
    tool = DuckDuckGoSearchRun()
    try:
        state["search_results"] = tool.run(state["question"])
    except Exception as e:
        state["search_results"] = f"[Search failed: {e}]"
    return state

def summarize_node(state):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Question: {state['question']}
Search results: {state['search_results']}
Write a clear, concise answer using only the info above."""
    try:
        state["answer"] = llm.invoke(prompt).content
    except Exception as e:
        state["answer"] = f"[Summarization failed: {e}]"
    return state

graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)
graph.set_entry_point("search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)
app = graph.compile()

result = app.invoke({"question": "What is LangGraph used for?", "search_results": "", "answer": ""})
print(result["answer"])
```

If this prints a sensible answer, your core agent works. ✅

### Step 4 (optional) — Preview the actual Streamlit UI from Colab
If you also want to see the real `app.py` UI while still in Colab, you can
tunnel it out with `localtunnel`:

```python
# 1. Upload app.py and requirements.txt to the Colab file browser
!pip install -q -r requirements.txt

# 2. Run Streamlit in the background
!streamlit run app.py &>/content/log.txt &

# 3. Expose it publicly with a tunnel
!npx localtunnel --port 8501
```
This prints a public URL — open it, and it will also show you a "tunnel
password" (it's the output of running `!curl ipv4.icanhazip.com` — run
that in another cell and paste the IP into the tunnel page).

This step is optional — most beginners can skip straight to Phase 2 once
Step 3 works, since Streamlit Cloud will run the exact same `app.py`.

---

## Phase 2 — Deploy to Streamlit Community Cloud via GitHub

### Step 1 — Create a GitHub repository
1. Go to [github.com](https://github.com) → **New repository**.
2. Name it, e.g., `ai-research-agent`.
3. Upload these three files to the repo (via the web UI "Add file → Upload
   files", or `git push` if you're comfortable with git):
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Step 2 — Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and set **Main file path** to
   `app.py`.
4. Click **Deploy**.

### Step 3 — Add your API key as a secret (do this before/after first deploy)
1. On your app's page in Streamlit Cloud, click **⋮ (menu) → Settings → Secrets**.
2. Add:
   ```toml
   OPENAI_API_KEY = "sk-your-real-key-here"
   ```
3. Save. The app will restart automatically and pick up the key
   (`app.py` already checks `st.secrets` first, so no code changes needed).

### Step 4 — Test the live app
Open the public URL Streamlit gives you, type a question, and confirm you
get a summarized answer with sources in the "raw search results" expander.

---

## Notes on error handling already built into `app.py`

- If the web search fails (rate limit, network issue), the app shows a
  placeholder message instead of crashing, and still attempts summarization.
- If the LLM call fails (bad/missing API key, quota issue), the error is
  caught and shown inline instead of crashing the whole app.
- If no API key is found anywhere (secrets, sidebar, environment), the app
  stops and shows a clear instruction before making any API calls.
- If the question field is empty, the app shows a warning instead of running.

## Extending later (optional, not required now)
Once this simple version works, natural next steps (not needed today) could
include: adding a "refine query" node before search, switching the search
tool, or adding simple caching — LangGraph makes it easy to insert new nodes
without rewriting the whole pipeline.
