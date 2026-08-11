import os
from typing import TypedDict

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="centered",
)


# ============================================================
# 2. GET GROQ API KEY
# ============================================================

def get_groq_api_key():
    """
    Get the Groq API key from:
    1. Streamlit Secrets (recommended for Streamlit Cloud)
    2. Environment variable (useful for local testing)
    """

    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.environ.get("GROQ_API_KEY")


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    search_results: str
    answer: str


# ============================================================
# 4. SEARCH NODE
# ============================================================

def search_node(state: AgentState) -> AgentState:
    """
    Search the web using DuckDuckGo.
    """

    tool = DuckDuckGoSearchRun()

    try:
        search_results = tool.run(state["question"])

        return {
            **state,
            "search_results": search_results,
        }

    except Exception as e:
        return {
            **state,
            "search_results": f"[Search failed: {e}]",
        }


# ============================================================
# 5. SUMMARIZATION NODE
# ============================================================

def summarize_node(state: AgentState) -> AgentState:
    """
    Send the question and web-search results to Groq
    and generate a concise research answer.
    """

    api_key = get_groq_api_key()

    if not api_key:
        return {
            **state,
            "answer": (
                "[Configuration error] GROQ_API_KEY was not found. "
                "Please add it to Streamlit Secrets or the environment."
            ),
        }

    # Make the key available to ChatGroq.
    os.environ["GROQ_API_KEY"] = api_key

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

        prompt = f"""
You are a simple AI research assistant.

Question:
{state["question"]}

Web search results:
{state["search_results"]}

Write a clear and concise answer using only the information
contained in the web search results.

Rules:
- Do not invent facts.
- If the search results do not contain enough information,
  say so honestly.
- Focus directly on the user's question.
- Keep the answer easy to understand.
"""

        response = llm.invoke(prompt)

        return {
            **state,
            "answer": response.content,
        }

    except Exception as e:
        return {
            **state,
            "answer": f"[Summarization failed: {e}]",
        }


# ============================================================
# 6. BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_research_agent():
    """
    Simple workflow:

    START
      ↓
    Search
      ↓
    Summarize
      ↓
    END
    """

    graph = StateGraph(AgentState)

    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


research_agent = build_research_agent()


# ============================================================
# 7. STREAMLIT USER INTERFACE
# ============================================================

st.title("🔎 AI Research Agent")

st.write(
    """
Ask a question and the AI Research Agent will search the web
using DuckDuckGo and then use a Groq LLM to create a concise answer.
"""
)

st.info(
    "Workflow: User Question → DuckDuckGo Search → "
    "Groq Summarization → Final Answer"
)


question = st.text_area(
    "Research Question",
    placeholder=(
        "Example: What are the main applications "
        "of artificial intelligence in education?"
    ),
    height=120,
)


# ============================================================
# 8. RUN RESEARCH
# ============================================================

if st.button("🔍 Research", type="primary"):

    if not question.strip():
        st.warning("Please enter a research question.")

    elif not get_groq_api_key():
        st.error(
            "GROQ_API_KEY is missing. "
            "Please add it to your Streamlit Secrets."
        )

    else:
        initial_state: AgentState = {
            "question": question.strip(),
            "search_results": "",
            "answer": "",
        }

        with st.spinner(
            "Searching the web and preparing your answer..."
        ):
            result = research_agent.invoke(initial_state)

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        st.subheader("Research Answer")

        if result["answer"].startswith("["):
            st.error(result["answer"])
        else:
            st.markdown(result["answer"])

        # ----------------------------------------------------
        # Search results
        # ----------------------------------------------------

        with st.expander("View Web Search Results"):
            st.write(result["search_results"])


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About")

    st.write(
        """
This is a beginner-friendly AI research agent built with:

- LangGraph
- LangChain
- DuckDuckGo
- Groq
- Streamlit
"""
    )

    st.caption(
        "Simple workflow: Search → Summarize"
    )
