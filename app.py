"""
Simple AI Research Agent
=========================
Architecture:  User Question -> Web Search (DuckDuckGo) -> Collect Results
               -> LLM Summarization (LangGraph) -> Final Answer

This single file contains:
  1. The LangGraph workflow (2 nodes: search -> summarize)
  2. The LangChain + DuckDuckGo web search integration
  3. The Streamlit user interface
  4. Basic error handling

Run locally / in Colab:
    streamlit run app.py
"""

import os
from typing import TypedDict

import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# 1. STATE DEFINITION
# ---------------------------------------------------------------------------
# LangGraph passes a "state" dict between nodes. We define exactly what
# fields that state will carry as the agent runs.
class AgentState(TypedDict):
    question: str          # the user's original question
    search_results: str    # raw text collected from the web search
    answer: str             # final summarized answer


# ---------------------------------------------------------------------------
# 2. HELPER: GET THE API KEY / LLM
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """
    Looks for the OpenAI API key in three places, in order of priority:
      1. Streamlit secrets (used on Streamlit Community Cloud)
      2. Streamlit session_state (used when the user types it in the sidebar)
      3. Environment variable (used in Colab / local terminal)
    """
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # st.secrets is not configured -> that's fine, just skip it

    if st.session_state.get("openai_api_key"):
        return st.session_state["openai_api_key"]

    return os.getenv("OPENAI_API_KEY")


def get_llm() -> ChatOpenAI:
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "No OpenAI API key found. Add it in the sidebar, in Streamlit "
            "secrets, or as the OPENAI_API_KEY environment variable."
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


# ---------------------------------------------------------------------------
# 3. LANGGRAPH NODES
# ---------------------------------------------------------------------------
def search_node(state: AgentState) -> AgentState:
    """Node 1: search the web with DuckDuckGo and store the raw results."""
    try:
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.run(state["question"])
        if not results:
            results = "No search results were found for this question."
    except Exception as e:
        # Never let the whole app crash because search failed.
        results = f"[Web search failed: {e}]"

    state["search_results"] = results
    return state


def summarize_node(state: AgentState) -> AgentState:
    """Node 2: ask the LLM to turn the raw search results into a clear answer."""
    prompt = f"""You are a helpful research assistant.

A user asked the following question:
"{state['question']}"

Here are some raw web search results related to the question:
---
{state['search_results']}
---

Using ONLY the information above, write a clear, concise, well-organized
answer to the user's question. If the search results don't contain enough
information, say so honestly instead of making things up.
"""
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        state["answer"] = response.content
    except Exception as e:
        state["answer"] = f"⚠️ Could not generate a summary: {e}"

    return state


# ---------------------------------------------------------------------------
# 4. BUILD THE LANGGRAPH WORKFLOW
# ---------------------------------------------------------------------------
# The graph is intentionally simple: search -> summarize -> end.
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Research Agent", page_icon="🔎")

    st.title("🔎 AI Research Agent")
    st.write(
        "Ask a question. The agent will search the web with DuckDuckGo, "
        "then use an LLM to summarize the findings into a clear answer."
    )

    # --- Sidebar: API key entry (handy for local/Colab testing) ---
    with st.sidebar:
        st.header("Settings")
        st.caption(
            "On Streamlit Community Cloud, set OPENAI_API_KEY in "
            "**Settings → Secrets** instead of typing it here."
        )
        key_input = st.text_input("OpenAI API Key", type="password")
        if key_input:
            st.session_state["openai_api_key"] = key_input

    question = st.text_input("Enter your research question:")
    run_clicked = st.button("Search & Summarize", type="primary")

    if run_clicked:
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        if not get_api_key():
            st.error(
                "No OpenAI API key found. Please add one in the sidebar, "
                "or in Streamlit secrets."
            )
            return

        with st.spinner("Searching the web and summarizing..."):
            try:
                graph = build_graph()
                result = graph.invoke(
                    {"question": question, "search_results": "", "answer": ""}
                )

                st.subheader("Answer")
                st.write(result["answer"])

                with st.expander("Show raw search results"):
                    st.write(result["search_results"])

            except Exception as e:
                st.error(f"Something went wrong while running the agent: {e}")


if __name__ == "__main__":
    main()
