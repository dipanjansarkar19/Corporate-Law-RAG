import os
import json
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

# CONFIGURATION

DOCUMENT_NAME = "Indian Corporate Law"
PERSIST_DIR = "vectorstore1000"
TOP_K = 8
MODEL_NAME = "gpt-5-nano"
MAX_QUERIES_PER_SESSION = 10
EMBEDDING_MODEL = "text-embedding-3-small"

# PAGE CONFIG

st.set_page_config(
    page_title="Indian Corporate Law Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Indian Corporate Law Assistant")
st.caption(
    f"Ask questions about {DOCUMENT_NAME}. "
    "Answers include the retrieved sources and are evaluated automatically."
)

# API KEY

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error(
        "No API key found. Add OPENAI_API_KEY to Streamlit Secrets "
        "or your .env file."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# VECTOR STORE CHECK

if not os.path.exists(PERSIST_DIR):
    st.error(
        f"No vector store found at '{PERSIST_DIR}'. "
        "Run ingest.py first."
    )
    st.stop()

# SESSION STATE

if "history" not in st.session_state:
    st.session_state.history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# LOAD RETRIEVER

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    return retriever

# LOAD RAG CHAIN

@st.cache_resource
def load_chain():
    llm = ChatOpenAI(model=MODEL_NAME, temperature = 0.5)

    prompt = ChatPromptTemplate.from_template(
        """
You are a careful assistant answering questions about Indian Corporate Law.

Answer ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the answer is not present in the context, say:
   "I don't have enough information in the provided documents."
4. Explain the answer in simple language.
5. Mention the relevant section number(s) whenever they are available.
6. Do not invent section numbers.

Context:
{context}

Question:
{input}

Answer:
"""
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    chain = create_retrieval_chain(
        retriever=load_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    return chain

chain = load_chain()

# EVALUATION LLM

@st.cache_resource
def load_evaluator():
    
    evaluator = ChatOpenAI(
        model=MODEL_NAME
    )

    return evaluator

evaluator = load_evaluator()

# EVALUATION FUNCTION

def evaluate_answer(question, answer, documents):
    """
    Evaluate one RAG interaction.

    Metrics:

    1. Faithfulness
       Is the answer supported by retrieved context?

    2. Answer Relevance
       Does the answer directly address the question?

    3. Context Relevance
       Is the retrieved context relevant to the question?

    4. Overall
       Average of the three scores.

    Scores are from 0 to 1.
    """

    if not documents:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
            "overall": 0.0,
            "reason": "No context was retrieved."
        }

    context_parts = []

    for i, doc in enumerate(documents, 1):
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"""
--- CONTEXT {i} ---
Page: {page}

{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    evaluation_prompt = """
You are an evaluator for a Retrieval-Augmented Generation (RAG)
system answering questions about Indian Corporate Law.

Evaluate the RAG answer using ONLY the information provided below.

QUESTION:
{question}

ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}


Evaluate three metrics.

1. FAITHFULNESS

Question:
Is every important claim in the answer supported by the retrieved context?

Score:
1.0 = completely supported
0.8 = mostly supported, only very minor issue
0.6 = partially supported
0.4 = significant unsupported claims
0.2 = mostly unsupported
0.0 = completely unsupported


2. ANSWER RELEVANCE

Question:
Does the answer directly address the user's question?

Score:
1.0 = directly and completely answers
0.8 = good answer with minor omissions
0.6 = partially answers
0.4 = substantially incomplete
0.2 = barely addresses the question
0.0 = does not answer the question


3. CONTEXT RELEVANCE

Question:
How relevant is the retrieved context to answering the question?

Score:
1.0 = highly relevant context
0.8 = mostly relevant
0.6 = moderately relevant
0.4 = considerable irrelevant material
0.2 = mostly irrelevant
0.0 = completely irrelevant


Return ONLY valid JSON.

Use exactly this format:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "context_relevance": 0.0,
    "reason": "Short explanation of the evaluation."
}}

The scores must be numbers between 0 and 1.
"""

    prompt = ChatPromptTemplate.from_template(
        evaluation_prompt
    )

    evaluation_chain = prompt | evaluator

    try:
        response = evaluation_chain.invoke(
            {
                "question": question,
                "answer": answer,
                "context": context
            }
        )

        raw = response.content

        # Remove accidental markdown code fences
        raw = raw.strip()

        raw = re.sub(
            r"^```json\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )

        data = json.loads(raw)

        faithfulness = float(
            data.get("faithfulness", 0)
        )

        answer_relevance = float(
            data.get("answer_relevance", 0)
        )

        context_relevance = float(
            data.get("context_relevance", 0)
        )

        # Clamp values between 0 and 1
        faithfulness = max(
            0.0,
            min(1.0, faithfulness)
        )

        answer_relevance = max(
            0.0,
            min(1.0, answer_relevance)
        )

        context_relevance = max(
            0.0,
            min(1.0, context_relevance)
        )

        overall = (
            faithfulness
            + answer_relevance
            + context_relevance
        ) / 3

        return {
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_relevance": context_relevance,
            "overall": overall,
            "reason": data.get(
                "reason",
                "No explanation provided."
            )
        }

    except Exception as e:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
            "overall": 0.0,
            "reason": f"Evaluation failed: {str(e)}"
        }

# SOURCE RENDERING

def render_sources(sources):
    with st.expander("📚 Show sources used for this answer"):
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get(
                "page",
                "?"
            )

            source_doc = doc.metadata.get(
                "source_doc",
                DOCUMENT_NAME
            )

            st.markdown(
                f"**Excerpt {i} — {source_doc}, page {page}:**"
            )

            snippet = doc.page_content[:500]

            st.text(
                snippet
                + ("..." if len(doc.page_content) > 500 else "")
            )

# EVALUATION DASHBOARD

def render_evaluation_dashboard():
    st.header("📊 Live RAG Evaluation")

    evaluations = st.session_state.evaluations

    if not evaluations:
        st.info(
            "No evaluations yet. Ask a question in the Chat tab "
            "and the answer will be evaluated automatically."
        )
        return

    df = pd.DataFrame(evaluations)

    # SUMMARY

    avg_faithfulness = df["faithfulness"].mean()
    avg_answer_relevance = df["answer_relevance"].mean()
    avg_context_relevance = df["context_relevance"].mean()
    avg_overall = df["overall"].mean()

    st.subheader("Session Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Faithfulness",
            f"{avg_faithfulness:.2f}"
        )

    with col2:
        st.metric(
            "Answer Relevance",
            f"{avg_answer_relevance:.2f}"
        )

    with col3:
        st.metric(
            "Context Relevance",
            f"{avg_context_relevance:.2f}"
        )

    with col4:
        st.metric(
            "Overall",
            f"{avg_overall:.2f}"
        )

    st.divider()


    # SCORE INTERPRETATION

    st.subheader("Score Interpretation")

    st.markdown(
        """
**0.90 – 1.00:** Excellent  
**0.75 – 0.89:** Good  
**0.60 – 0.74:** Needs improvement  
**Below 0.60:** Poor
"""
    )

    st.divider()

    # --------------------------------------------------------
    # SCORE CHART
    # --------------------------------------------------------
    st.subheader("📈 Evaluation Trend")

    chart_df = df[
        [
            "faithfulness",
            "answer_relevance",
            "context_relevance",
            "overall"
        ]
    ].copy()

    chart_df.index = range(
        1,
        len(chart_df) + 1
    )

    st.line_chart(chart_df)

    st.divider()


    # DETAILED RESULTS

    st.subheader("🔎 Detailed Evaluations")

    for i, row in df.iloc[::-1].iterrows():
        question = row["question"]

        with st.expander(
            f"#{i + 1} — {question}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Faithfulness",
                    f"{row['faithfulness']:.2f}"
                )

            with col2:
                st.metric(
                    "Answer Relevance",
                    f"{row['answer_relevance']:.2f}"
                )

            with col3:
                st.metric(
                    "Context Relevance",
                    f"{row['context_relevance']:.2f}"
                )

            with col4:
                st.metric(
                    "Overall",
                    f"{row['overall']:.2f}"
                )

            st.markdown("**Question**")
            st.write(row["question"])

            st.markdown("**Answer**")
            st.write(row["answer"])

            st.markdown("**Evaluator reasoning**")
            st.write(row["reason"])

    st.divider()


    # DOWNLOAD RESULTS

    st.subheader("⬇️ Export Evaluation")

    csv_df = df.copy()

    csv_data = csv_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download evaluation results as CSV",
        data=csv_data,
        file_name="rag_evaluation_results.csv",
        mime="text/csv"
    )


# SIDEBAR

with st.sidebar:
    st.header("⚙️ RAG Configuration")

    st.write(
        f"**Model:** `{MODEL_NAME}`"
    )

    st.write(
        f"**Embeddings:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Top K:** `{TOP_K}`"
    )

    st.write(
        f"**Queries:** "
        f"{st.session_state.query_count}/"
        f"{MAX_QUERIES_PER_SESSION}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat & Evaluation"
    ):
        st.session_state.history = []
        st.session_state.evaluations = []
        st.session_state.query_count = 0
        st.rerun()


# MAIN INTERFACE

chat_tab, evaluation_tab = st.tabs(
    [
        "💬 Chat",
        "📊 Live Evaluation"
    ]
)

# CHAT TAB

with chat_tab:
    # Render previous conversation
    for role, content, sources in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

            if sources:
                render_sources(sources)

    query = st.chat_input(
        f"Ask a question about {DOCUMENT_NAME}..."
    )

    if query:
        if (
            st.session_state.query_count
            >= MAX_QUERIES_PER_SESSION
        ):
            st.warning(
                f"You have reached the maximum number "
                f"of queries ({MAX_QUERIES_PER_SESSION}) "
                "for this session."
            )
            st.stop()

        st.session_state.query_count += 1


        # USER MESSAGE

        st.session_state.history.append(
            (
                "user",
                query,
                None
            )
        )

        with st.chat_message("user"):
            st.markdown(query)


        # RAG RESPONSE
        
        with st.chat_message("assistant"):
            with st.spinner(
                "Retrieving documents and generating answer..."
            ):
                try:
                    answer_start = time.perf_counter()

                    result = chain.invoke(
                        {
                            "input": query
                        }
                    )
                    answer_time = time.perf_counter() - answer_start

                    answer = result["answer"]

                    sources = result.get(
                        "context",
                        []
                    )

                    st.markdown(answer)

                    st.caption(
                        f"Answer generated in {answer_time:.2f} seconds"
                    )

                    if sources:
                        render_sources(sources)

                except Exception as e:
                    st.error(
                        f"RAG error: {str(e)}"
                    )
                    st.stop()

        # LIVE EVALUATION
        
        with st.spinner("Evaluating answer..."):

            evaluation_start = time.perf_counter()
            
            evaluation = evaluate_answer(
                question=query,
                answer=answer,
                documents=sources
            )

            evaluation_time = time.perf_counter() - evaluation_start
   
        # SAVE EVALUATION
        
        evaluation_record = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "question": query,
            "answer": answer,
            "answer_time": answer_time,
            "evaluation_time": evaluation_time,
            "faithfulness": evaluation[
                "faithfulness"
            ],
            "answer_relevance": evaluation[
                "answer_relevance"
            ],
            "context_relevance": evaluation[
                "context_relevance"
            ],
            "overall": evaluation[
                "overall"
            ],
            "reason": evaluation[
                "reason"
            ]
        }

        st.session_state.evaluations.append(
            evaluation_record
        )

       
        # SHOW LIVE SCORE
    
        st.success(
            "✅ Answer evaluated"
        )

        st.caption(
            f"Evaluation completed in {evaluation_time:.2f} seconds"
        )

        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)

        with eval_col1:
            st.metric(
                "Faithfulness",
                f"{evaluation['faithfulness']:.2f}"
            )

        with eval_col2:
            st.metric(
                "Answer Relevance",
                f"{evaluation['answer_relevance']:.2f}"
            )

        with eval_col3:
            st.metric(
                "Context Relevance",
                f"{evaluation['context_relevance']:.2f}"
            )

        with eval_col4:
            st.metric(
                "Overall",
                f"{evaluation['overall']:.2f}"
            )

        with st.expander(
            "🔍 Why did the evaluator give this score?"
        ):
            st.write(
                evaluation["reason"]
            )

        
        # SAVE ASSISTANT MESSAGE
        
        st.session_state.history.append(
            (
                "assistant",
                answer,
                sources
            )
        )


# EVALUATION TAB

with evaluation_tab:
    render_evaluation_dashboard()
