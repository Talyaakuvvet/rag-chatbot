#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- imports ----------
import os, time, warnings
import streamlit as st
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# HF SDK
from huggingface_hub import HfFolder, HfApi, InferenceClient

# ---------- bootstrap ----------
load_dotenv()
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("Chatbot")

# Optional: silence some LC deprecation noise
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

# ---------- HF TOKEN (robust & exported) ----------
def resolve_hf_token():
    # 1) streamlit secrets
    try:
        if hasattr(st, "secrets") and "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
            tok = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            if isinstance(tok, str) and tok.startswith("hf_"):
                return tok.strip()
    except Exception:
        pass
    # 2) env
    for k in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        v = os.getenv(k)
        if v and v.startswith("hf_"):
            return v.strip()
    # 3) hf cli cache
    v = HfFolder.get_token()
    if v and v.startswith("hf_"):
        return v.strip()
    return None

# clear rotten envs
for bad in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
    v = os.getenv(bad)
    if v and not v.startswith("hf_"):
        os.environ.pop(bad, None)

hf_token = resolve_hf_token()
if not (hf_token and hf_token.startswith("hf_") and len(hf_token) > 20):
    st.error("HF token yok/geçersiz. `.streamlit/secrets.toml` içine\n"
             'HUGGINGFACEHUB_API_TOKEN = "hf_********" yaz, ya da ENV/`hf auth login`.')
    st.stop()

# make sure ALL downstream libs see the same token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
os.environ["HUGGINGFACE_HUB_TOKEN"]   = hf_token
os.environ["HF_TOKEN"]                = hf_token

# quick whoami (fail-fast)
try:
    who = HfApi().whoami(token=hf_token)
    st.caption(f"HF user: {who.get('name','?')} | token head: {hf_token[:6]}… len={len(hf_token)}")
except Exception as e:
    st.error(f"Hugging Face whoami başarısız: {e}")
    st.stop()

# ---------- Pinecone ----------
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
except KeyError:
    st.error("PINECONE_API_KEY eksik. `.env` içinde ayarla.")
    st.stop()

try:
    INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
except KeyError:
    st.error("PINECONE_INDEX_NAME eksik. `.env` içine yaz.")
    st.stop()

CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
REGION     = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE  = os.getenv("PC_NAMESPACE", "")

# ---------- Embeddings ----------
emb_model = os.getenv("EMB_MODEL", "BAAI/bge-m3")
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    encode_kwargs={"normalize_embeddings": True},
)
try:
    embed_dim = len(embeddings.embed_query("ping"))
except Exception as e:
    st.error(f"Embedding modeli indirilemedi/çağrı hatası: {e}")
    st.stop()

# ---------- Index create-if-missing ----------
existing = [x["name"] for x in pc.list_indexes()]
if INDEX_NAME not in existing:
    st.info(f"Creating Pinecone index `{INDEX_NAME}` (dim={embed_dim}, {CLOUD}/{REGION})")
    pc.create_index(
        name=INDEX_NAME, dimension=embed_dim, metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    # wait ready
    for _ in range(120):
        if pc.describe_index(INDEX_NAME).status.get("ready"):
            break
        time.sleep(1)

index = pc.Index(INDEX_NAME)
st.caption(f"Using index: **{INDEX_NAME}**, namespace: **{NAMESPACE or '(empty)'}**")

# ---------- Vector store ----------
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)

# ---------- HF Chat LLM adapter ----------
class HFChatLLM:
    def __init__(self, model: str, token: str, temperature: float = 0.3,
                 max_new_tokens: int = 350, timeout: int = 120):
        self.client = InferenceClient(model=model, token=token, timeout=timeout)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def invoke(self, prompt: str) -> str:
        res = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are concise and use retrieved context when provided."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return res.choices[0].message.content

# ---------- LLM (InferenceClient, conversational) ----------
HF_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HFChatLLM(
    model=HF_REPO,
    token=hf_token,
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
    max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "350")),
    timeout=120,
)
st.caption(f"LLM: {HF_REPO} (conversational via InferenceClient)")

# ---------- Chat geçmişi ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            "You are an assistant for question-answering tasks. "
            "Use retrieved context. If unknown, say you don't know. Keep answers concise."
        )
    ]

# geçmişi göster
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(m.content)

# ---------- Girdi (DÖNGÜNÜN DIŞINDA!) ----------
prompt = st.chat_input("Ask me anything…", key="main_chat_input")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # 1) Retrieve
    retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},  # 3→5 ve 0.5→0.3
    )

    try:
        docs = retriever.invoke(prompt)
    except Exception as e:
        docs = []
        st.warning(f"Retriever error: {e}")

    docs_text = "\n\n".join(d.page_content for d in docs) if docs else ""

    # 2) Prompt helper
    def is_greeting(s: str) -> bool:
        s = s.strip().lower()
        return any(k in s for k in ["hi", "hello", "hey", "selam", "merhaba"])

    # 3) Compose full prompt
    if not docs_text.strip() or is_greeting(prompt):
        system_prompt = (
            "You are a friendly, concise assistant. "
            "If the user's message is a greeting, respond briefly and warmly. "
            "If it's a general question, answer normally using your own knowledge. "
            "Keep replies under 3 short sentences."
        )
        full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
    else:
        system_prompt = (
            "You are an assistant for question-answering tasks.\n"
            "Use the retrieved context when relevant, otherwise answer normally.\n"
            "Keep it concise (<=3 sentences).\n"
            f"Context:\n{docs_text}\n"
        )
        full_prompt = f"{system_prompt}\nQuestion: {prompt}\nAnswer:"

    # 4) LLM çağrısı
    try:
        answer_text = llm.invoke(full_prompt)
    except Exception as e:
        answer_text = f"LLM error: {e}"

    # 5) Göster + kaydet
    with st.chat_message("assistant"):
        st.markdown(answer_text)
    st.session_state.messages.append(AIMessage(answer_text))

    # 6) Kaynakları göster
    if docs:
        with st.expander("Sources"):
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", "?")
                st.markdown(f"**[{i}]** {src} — page {page}")

