# ingestion.py — solid version (free-friendly)

import os, time, hashlib, json
from dotenv import load_dotenv
from typing import List, Dict, Any

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------- 1) ENV --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME")
CLOUD            = os.getenv("PINECONE_CLOUD", "aws")
REGION           = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE        = os.getenv("PC_NAMESPACE", "").strip()  # "" = default

DOCS_DIR         = os.getenv("DOCS_DIR", "documents")      # klasör parametreli olsun
EMB_MODEL        = os.getenv("EMB_MODEL", "BAAI/bge-m3")   # local, ücretsiz
KEEP_PAGES       = int(os.getenv("KEEP_PAGES", "20"))      # ilk N sayfa

assert PINECONE_API_KEY and INDEX_NAME, "PINECONE_API_KEY ve PINECONE_INDEX_NAME zorunlu."

# HF token’ı varsa tek kaynaktan okut ve ENV’e zorla (bazı alt çağrılar ENV’i kullanıyor)
hf_tok = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_tok:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_tok
    os.environ["HF_TOKEN"] = hf_tok
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_tok

# -------------------- 2) Pinecone --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings — BAAI/bge-m3 ücretsiz ve local (HF’den indirir ama inference token gerekmez)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True}
)
try:
    embed_dim = len(embeddings.embed_query("ping"))
except Exception as e:
    raise RuntimeError(f"Embedding modeli alınamadı: {e}")

# index yoksa oluştur
existing = [x["name"] for x in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"[pc] creating index: {INDEX_NAME} (dim={embed_dim}, {CLOUD}/{REGION})")
    pc.create_index(
        name=INDEX_NAME,
        dimension=embed_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    # hazır olana kadar bekle
    for _ in range(120):
        if pc.describe_index(INDEX_NAME).status.get("ready"):
            break
        time.sleep(1)

index = pc.Index(INDEX_NAME)
print(f"[pc] Using index: {INDEX_NAME} | namespace: {NAMESPACE or '(empty)'}")

# -------------------- 3) Belgeleri yükle --------------------
if not os.path.isdir(DOCS_DIR):
    raise FileNotFoundError(f"Doküman klasörü bulunamadı: {DOCS_DIR}")

loader = PyPDFDirectoryLoader(DOCS_DIR)
raw_docs = loader.load()
if not raw_docs:
    raise RuntimeError(f"{DOCS_DIR} içinde PDF bulunamadı.")

# sayfa filtresi güvenli (metadata 'page' str/None olabilir)
def safe_page(meta: Dict[str, Any]) -> int:
    p = meta.get("page", 0)
    try:
        return int(p)
    except Exception:
        return 0

raw_docs = [d for d in raw_docs if safe_page(d.metadata) < KEEP_PAGES]

# dedup: source+page bazında tekilleştir (aynı dosyayı yeniden ingest edince çakışma olmasın)
seen = set()
deduped = []
for d in raw_docs:
    key = f"{d.metadata.get('source','unknown')}#p{safe_page(d.metadata)}"
    if key not in seen:
        seen.add(key)
        deduped.append(d)
raw_docs = deduped

print(f"[load] PDFs: {len(seen)} unique pages (<= page {KEEP_PAGES-1}) from {DOCS_DIR}")

# -------------------- 4) Split --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n### ","\n## ","\n# ","\n\n", "\n", " "],
)
docs = splitter.split_documents(raw_docs)
print(f"[split] chunks: {len(docs)}")

# -------------------- 5) Deterministic ID üretimi --------------------
# Aynı içerik/sayfa geldiğinde aynı ID oluşsun → duplicate upsert olmaz
def chunk_id(d) -> str:
    src = str(d.metadata.get("source", "unknown"))
    pg  = str(safe_page(d.metadata))
    # metin + kritik metadata’dan hash
    h = hashlib.sha1()
    h.update(d.page_content.encode("utf-8", errors="ignore"))
    h.update(f"|{src}|{pg}".encode())
    return h.hexdigest()

ids = [chunk_id(d) for d in docs]

# -------------------- 6) Vector store (namespace’li) --------------------
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# batch upsert (rate-limit friendly)
BATCH = int(os.getenv("BATCH", "64"))
print(f"[upsert] uploading in batches of {BATCH}…")
for i in range(0, len(docs), BATCH):
    batch_docs = docs[i:i+BATCH]
    batch_ids  = ids[i:i+BATCH]
    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    print(f"  upserted {i+len(batch_docs)}/{len(docs)}")

# -------------------- 7) Stats --------------------
stats = index.describe_index_stats()
stats = index.describe_index_stats()
try:
    # Bazı sürümlerde zaten dict gelir
    print("[stats]", json.dumps(stats, indent=2))
except TypeError:
    # Yeni SDK: response objesi -> dict'e çevir
    try:
        print("[stats]", json.dumps(stats.to_dict(), indent=2))
    except Exception:
        # En kötü ihtimal: stringe bas
        print("[stats]", str(stats))
print("[done] ingestion completed")

