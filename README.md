# 🔎 AI Research Agent

A simple beginner-friendly AI research agent built with **LangGraph**, **LangChain**, **DuckDuckGo**, **Groq**, and **Streamlit**.

## Architecture

```text
User Question
      ↓
Web Search
      ↓
Collect Search Results
      ↓
Groq LLM Summarization
      ↓
Final Answer
```

## Technologies

- **Python** — main programming language
- **LangGraph** — manages the research workflow
- **LangChain** — connects the application to the LLM and search tools
- **DuckDuckGo** — web search
- **Groq** — provides the LLM
- **Streamlit** — web interface
- **GitHub + Streamlit Community Cloud** — deployment

## Current LangGraph Workflow

The application uses a simple two-node workflow:

```text
START
  ↓
Search
  ↓
Summarize
  ↓
END
```

The search node uses `DuckDuckGoSearchRun` to search for the user's question.

The summarize node sends the question and search results to:

```text
llama-3.3-70b-versatile
```

through Groq.

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## API Key

The application requires a Groq API key.

### Google Colab

For the current Colab version, the key is loaded from Google Colab Secrets:

```python
from google.colab import userdata
import os

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

Create a Colab secret named:

```text
GROQ_API_KEY
```

Do not put the actual API key directly into the Python code.

### Streamlit Community Cloud

For Streamlit Community Cloud, add the secret in the application's **Secrets** settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not commit your API key to GitHub.

## Running the Streamlit Application

If `app.py` is a Streamlit application, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the application can be opened in a browser.

## Deployment

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud.
4. Select the GitHub repository.
5. Select `app.py` as the main file.
6. Add `GROQ_API_KEY` under the application's Secrets.
7. Deploy the application.

## Important Note About the Current Colab File

The Python file currently developed in Google Colab is a **Colab notebook export**. It contains:

```python
!pip install ...
```

and:

```python
from google.colab import userdata
```

Those are Colab-specific and should not remain in the final Streamlit `app.py`.

Before deploying to Streamlit Community Cloud, convert the Colab code into a normal Streamlit `app.py`:

- Remove `!pip install ...`
- Remove `from google.colab import userdata`
- Read `GROQ_API_KEY` from `st.secrets`
- Add the Streamlit interface using `st.title()`, `st.text_area()`, `st.button()`, etc.
- Keep the LangGraph search and summarization workflow

The `requirements.txt` in this repository contains both the packages used by the current Colab workflow and Streamlit for the final web application.

## Example Question

```text
What are the main applications of artificial intelligence in education?
```

The agent searches the web and sends the collected information to the Groq LLM to produce a concise answer.

## Security

Never upload API keys or other secrets to GitHub.

Use:

- Google Colab Secrets during Colab testing
- Streamlit Secrets during Streamlit Community Cloud deployment

## Project Structure

```text
ai-research-agent/
│
├── app.py
├── requirements.txt
└── README.md
```

## Future Improvements

Possible improvements after the basic version works:

- Display individual sources more clearly
- Add search-result links
- Add source citations to the final answer
- Add a loading/progress indicator
- Add configurable number of search results
- Add conversation history
- Add more research tools
- Add LangGraph conditional routing
