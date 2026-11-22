# Market Lens — AI Stock Market Assistant

Market Lens is a deployable Streamlit research prototype that combines historical stock data, financial-news retrieval, LangChain prompt pipelines, and a LangGraph multi-agent workflow. Its BUY / HOLD / SELL output is a **simulated, rule-based research signal**, not financial advice or a forecast.

## Deliverables

- `app.py` — Streamlit application
- `stock_assistant/` — market/news data utilities and LangChain/LangGraph workflow
- `notebooks/market_lens_walkthrough.ipynb` — runnable workflow walkthrough
- `presentation/Market_Lens_Overview.pptx` — project presentation
- `tools/build_presentation.py` — source used to regenerate the presentation
- `document_qa_app.py` — separate document-question-answering application
- `document_qa/` — document ingestion, LlamaIndex retrieval, and LangChain answer generation
- `notebooks/document_qa_walkthrough.ipynb` — document Q&A walkthrough
- `presentation/Document_QA_Overview.pptx` — document Q&A presentation

## Run locally

Python 3.10+ is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

### Document Q&A app

The document Q&A system is a separate app and leaves the stock assistant unchanged:

```powershell
python -m pip install -r requirements.txt
streamlit run document_qa_app.py
```

Set `OPENAI_API_KEY` in `.env` to enable LangChain answer generation. The app embeds chunks locally using LlamaIndex's Hugging Face integration; the first indexing run downloads the configured embedding model. Upload PDF/TXT/Markdown files or supply public web-page URLs. Retrieved passages are shown with source/page references; uploaded documents and indexes stay in Streamlit session memory and are not persisted by the application. **Retrieved passages are sent to OpenAI to generate answers.** Web ingestion only allows public HTTP(S) hosts and text/HTML responses.

The document Q&A walkthrough can be opened from `notebooks/document_qa_walkthrough.ipynb`. Run focused document-Q&A tests with:

```powershell
python -m unittest discover -s document_qa_tests -v
```

The document Q&A app requires the API key for answer generation. Its notebook demonstrates the same flow using a small in-memory sample document.

Set `OPENAI_API_KEY` in `.env` to enable LangChain `ChatOpenAI` narrative generation. Without a key, the app still runs the LangGraph workflow and gives transparent local explanations. An LLM API error is surfaced in the interface; affected narratives fall back to the disclosed deterministic summary.

For an offline demonstration, enable **Use demo data (offline)**. Synthetic prices and sample headlines are explicitly labeled and never represented as live data. With demo mode off, price history is fetched from Yahoo Finance and headlines from Google News RSS; either source may be unavailable.

## Multi-agent design

`stock_assistant/workflow.py` builds the graph and holds shared state:

```text
START → Market analyst ──conditional──→ Insufficient data → END
                         └────────────→ News researcher / RAG
                                         → Market insight agent
                                         → Risk analyst
                                         → Recommendation agent → END
```

- **Market analyst:** invokes a LangChain market-data tool, then computes price change, 20-session average, and historical volatility.
- **News researcher:** invokes a financial-news tool, wraps articles as LangChain `Document`s, and runs a keyword-based LangChain `BaseRetriever`. If an API key is configured, retrieved context is passed through a LangChain prompt/LLM/output-parser runnable.
- **Market insight agent:** explains the calculated metrics through a LangChain prompt chain when available.
- **Risk analyst:** highlights volatility and data/news limitations.
- **Recommendation agent:** computes a deterministic signal from price trend, moving average, and a disclosed keyword tone proxy. It does not delegate the BUY/HOLD/SELL decision to an LLM.

The conditional edge prevents a signal from being inferred when price history is missing or has fewer than five observations. If RSS is unavailable, the error is surfaced and the recommendation is marked as price-only.

## Validation

Run the offline workflow test after installing requirements:

```powershell
python -m unittest discover -s tests -v
```

The offline test does not require network access or an API key.

## Limitations

- Yahoo Finance and Google News RSS are external, potentially delayed sources with no uptime guarantee.
- The retrieved-news keyword tone is a basic, explainable proxy; it is not a validated financial sentiment model.
- The assistant does not account for personal circumstances, portfolio composition, transaction costs, or suitability.
- Do not use the output as a substitute for independent research or professional advice.
