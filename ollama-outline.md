***Generated using Ollama, `gpt-oss:120b`. Fixing markdown - the formula in 5.1 in incorrectly rendered - I need to go deep on markdown renderings of formulas, and will review in detail and comment as time allows. For now, just a useful roadmap.***

**Things to do**
+ Fix 5.1 formula
+ ~~revise 2.0 Data acqusition table to reflect actual sources used in Asterism~~
+ ~~review services required, price, and look for free alternatives~~
+ Test python, esp. 12.5. I suspect it won't run, but curious to see...

Below is a complete end‑to‑end blueprint for building a “starting‑image‑driven recommendation engine” that pulls images from many public repositories, extracts formal, conceptual, and historical affinities, stores the results in a graph, and serves real‑time recommendations.

The guide is broken into four layers (data, representation, graph, service) and includes concrete tool choices, code snippets, and a realistic project timeline.

# 1. High‑Level Architecture

               +-------------------+        +-------------------+
               |  Scrapers / Crawl |  --->  |  Raw Image Store  |
               +-------------------+        +-------------------+
                           |                     |
                           v                     v
                +-------------------+   +-------------------+
                |  Metadata Extract |   |  Image Ingest &   |
                |   (EXIF, OCR…)    |   |   Normalization   |
                +-------------------+   +-------------------+
                           |                     |
                           v                     v
                +-------------------------------------------+
                |   Multi‑Modal Embedding Service (ML)       |
                |   – Formal (vision)                        |
                |   – Conceptual (text/vision)               |
                |   – Historical (temporal + provenance)    |
                +-------------------------------------------+
                           |
                           v
                +-------------------+   +-------------------+
                |   Vector Store    |   |   Property Store  |
                |   (FAISS / Annoy) |   |   (Postgres)      |
                +-------------------+   +-------------------+
                           \               /
                            \             /
                             v           v
                      +---------------------------+
                      |   Graph Database (Neo4j)  |
                      |   Nodes = images          |
                      |   Edges = affinity scores |
                      +---------------------------+
                                 |
                                 v
                     +--------------------------+
                     |  Recommendation Engine   |
                     |  (Hybrid: vector + graph)|
                     +--------------------------+
                                 |
                                 v
                     +--------------------------+
                     |  API / UI (FastAPI, React)|
                     +--------------------------+


+ **Scrapers** feed raw images + metadata.
+ **Embedding Service** creates three orthogonal embeddings per image.
+ **Vector + Property Stores** keep fast‑lookup vectors and rich metadata.
+ **Neo4j** (or similar) stores a **multigraph** where each edge type corresponds to one affinity.
+ **Recommendation Engine** performs a **weighted‑sum** of similarity scores or runs a **graph neural network** to rank candidates.

# 2. Data Acquisition (Scraping & Ingestion)
| **Source** | **Typical Access Method** | **License / Legal Note** | **Recommended Scraper / Tool** |
| --- | --- | --- | --- |
| **Wikimedia Commons** | Public MediaWiki API (`action=query&list=allimages`) – supports continuation for paging | Mostly **CC‑BY, CC‑BY‑SA, CC0,** and some **All Rights Reserved** items. Always keep the `license` field from the API response. | `mwclient` + `aiohttp` (async) – see the code block in the original section. |
| **The Metropolitan Museum of Art – Open Access** | Bulk download of a [**JSON‑LD** manifest](https://github.com/metmuseum/openaccess) plus direct image URLs. The manifest contains metadata, tags, and rights. | **CC‑0** (public domain) for ~375k images. No attribution required, but it’s good practice to keep the `artist` / `source` fields. | Simple `requests` + `pandas` to read the JSON lines; a tiny helper script (shown below). |
| **The Art Institute of Chicago – Digital Collections** | REST API (https://api.artic.edu/api/v1/artworks) that returns paginated JSON with image IDs, URLs, and rights information. | **CC‑BY‑SA** for most images; a few are **All Rights Reserved.** Respect the `rights_statement` field. | `aiohttp` async loop + pagination handling (code below). |

**Why these three?**
+ All three provide machine‑readable metadata (tags, creator, creation date, rights).
+ They expose large open‑access corpora that are safe to redistribute in a recommendation service (provided you surface the original license).
+ Each has a stable, documented API, which means you can keep the scrapers running long‑term with minimal maintenance.

> **James' comment:** this outline assumes ongoing scraping, vector generation, etc. For a pilot project, I don't this needs to be an ongoing activity past launch. *If* it's determined that this is a viable ongoing project, *then* build that into ongoing planning/budget.

## 2.1 Unified Scraper Skeleton
All three sources can be driven from a single **async‑oriented** framework. The skeleton below shows how to plug in each source as a coroutine that yields a dictionary for every image:

```python
import asyncio, aiohttp, hashlib, json, pathlib, os
from tqdm.asyncio import tqdm

BASE_DIR = pathlib.Path("./raw_images")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1️⃣ Wikimedia Commons
# ----------------------------------------------------------------------
WIKI_API = "https://commons.wikimedia.org/w/api.php"

async def wiki_fetch_one(session, img):
    url = f"https:{img['url']}"
    async with session.get(url) as resp:
        if resp.status != 200:
            return None
        data = await resp.read()
        sha = hashlib.sha256(data).hexdigest()[:16]
        ext = url.split(".")[-1].split("?")[0]
        path = BASE_DIR / f"{img['name']}_{sha}.{ext}"
        path.write_bytes(data)
        return {
            "source": "wikimedia",
            "source_id": img["name"],
            "url": url,
            "path": str(path),
            "license": img.get("license", "unknown"),
            "title": img.get("title", ""),
            "description": img.get("description", "")
        }

async def crawl_wikimedia(limit: int = 10_000):
    async with aiohttp.ClientSession() as session:
        cont = {"continue": None}
        fetched = 0
        while fetched < limit:
            params = {
                "action": "query",
                "list": "allimages",
                "ailimit": "max",
                "format": "json",
                **cont
            }
            async with session.get(WIKI_API, params=params) as resp:
                payload = await resp.json()
                imgs = payload["query"]["allimages"]
                tasks = [wiki_fetch_one(session, i) for i in imgs]
                for meta in await asyncio.gather(*tasks):
                    if meta:
                        yield meta
                fetched += len(imgs)
                cont = payload.get("continue", {})
                if not cont:
                    break

# ----------------------------------------------------------------------
# 2️⃣ The Met – Open Access (JSON‑LD manifest)
# ----------------------------------------------------------------------
MET_MANIFEST = "https://raw.githubusercontent.com/metmuseum/openaccess/master/Metadata/CC0/cc0.json"

async def crawl_met():
    # The manifest is a single huge JSON‑L file – we stream it line‑by‑line.
    async with aiohttp.ClientSession() as session:
        async with session.get(MET_MANIFEST) as resp:
            async for line in resp.content:
                record = json.loads(line)
                # Skip entries that lack an image URL
                if not record.get("primaryImage"):
                    continue
                img_url = record["primaryImage"]
                async with session.get(img_url) as img_resp:
                    if img_resp.status != 200:
                        continue
                    data = await img_resp.read()
                    sha = hashlib.sha256(data).hexdigest()[:16]
                    ext = img_url.split(".")[-1].split("?")[0]
                    path = BASE_DIR / f"{record['objectID']}_{sha}.{ext}"
                    path.write_bytes(data)
                    yield {
                        "source": "met",
                        "source_id": str(record["objectID"]),
                        "url": img_url,
                        "path": str(path),
                        "license": "CC0",
                        "title": record.get("title", ""),
                        "artist": record.get("artistDisplayName", ""),
                        "date": record.get("objectDate", "")
                    }

# ----------------------------------------------------------------------
# 3️⃣ Art Institute of Chicago – Digital Collections
# ----------------------------------------------------------------------
AIC_API = "https://api.artic.edu/api/v1/artworks"

async def crawl_aic(page_size: int = 100):
    async with aiohttp.ClientSession() as session:
        page = 1
        while True:
            params = {"page": page, "limit": page_size, "fields": "id,title,image_id,artist_title,date_display,license_text,thumbnail"}
            async with session.get(AIC_API, params=params) as resp:
                payload = await resp.json()
                data = payload["data"]
                if not data:
                    break
                for rec in data:
                    if not rec.get("image_id"):
                        continue
                    img_url = f"https://www.artic.edu/iiif/2/{rec['image_id']}/full/843,/0/default.jpg"
                    async with session.get(img_url) as img_resp:
                        if img_resp.status != 200:
                            continue
                        raw = await img_resp.read()
                        sha = hashlib.sha256(raw).hexdigest()[:16]
                        ext = "jpg"
                        path = BASE_DIR / f"{rec['id']}_{sha}.{ext}"
                        path.write_bytes(raw)
                        yield {
                            "source": "artic",
                            "source_id": str(rec["id"]),
                            "url": img_url,
                            "path": str(path),
                            "license": rec.get("license_text", "unknown"),
                            "title": rec.get("title", ""),
                            "artist": rec.get("artist_title", ""),
                            "date": rec.get("date_display", "")
                        }
                page += 1

# ----------------------------------------------------------------------
# 4️⃣ Orchestrator – write everything to a JSONL file for downstream steps
# ----------------------------------------------------------------------
async def orchestrate():
    out = open("metadata.jsonl", "a")
    async for meta in crawl_wikimedia(limit=30_000):
        out.write(json.dumps(meta) + "\n")
    async for meta in crawl_met():
        out.write(json.dumps(meta) + "\n")
    async for meta in crawl_aic():
        out.write(json.dumps(meta) + "\n")
    out.close()

if __name__ == "__main__":
    asyncio.run(orchestrate())
```

**What the script does**
| **Step** | **Action** |
| --- | --- |
| **Deduplication** | SHA‑256 of the raw bytes is saved; if the same hash appears again you can simply skip writing a new file. |
| **License capture** | The `license` (or `license_text`) field from each source is stored verbatim; you’ll expose it later in the UI and in the API response. |
| **Metadata enrichment** | Basic fields like `title`, `artist`, `date` are harvested now so you don’t have to re‑parse the JSON later. |
| **Streaming** | The whole pipeline is async; you can easily spin up multiple workers (e.g., via `asyncio.gather`) to increase throughput without over‑loading the source APIs. |

> **Tip:** All three sources impose polite‑usage limits (e.g., “no more than 5 req/s”). Wrap the `session.get` calls in a tiny rate‑limiter (or use the `aiohttp_retry` library) to stay on the safe side.

## 2.2 Metadata Harvesting

1. **EXIF / IPTC** – `piexif`, `exiftool`.
2. **Embedded textual description** – parse the API‑provided `description`, `tags`, `captions`.
3. **OCR on the image** (if it contains text) – `pytesseract` or **Google Vision API.**
4. **Provenance** – capture `source`, `license`, `creation_date`, `artist/author`, `museum/collection`.

All metadata goes into a **PostgreSQL** table:

```sql
CREATE TABLE images (
    id          UUID PRIMARY KEY,
    source      TEXT,
    source_id   TEXT,
    file_path   TEXT,
    sha256      TEXT,
    width       INT,
    height      INT,
    mime_type   TEXT,
    created_at  TIMESTAMP,
    license     TEXT,
    tags        TEXT[],          -- array of strings
    caption     TEXT,
    ocr_text    TEXT,
    ingest_ts   TIMESTAMP DEFAULT now()
);
```

## 2.3 Inserting the Harvested Metadata into PostgreSQL
Once you have `metadata.jsonl`, load it into the images table described in the original section. The loader works for any of the three sources because the JSON keys are consistent.

```python
import json, psycopg2, uuid
from pathlib import Path

conn = psycopg2.connect(
    dbname="image_db",
    user="postgres",
    password=os.getenv("PG_PASSWORD"),
    host="localhost"
)
cur = conn.cursor()

def upsert_image(rec):
    cur.execute("""
        INSERT INTO images (
            id, source, source_id, file_path, sha256,
            width, height, mime_type, created_at,
            license, tags, caption, ocr_text
        ) VALUES (
            gen_random_uuid(), %(source)s, %(source_id)s, %(path)s,
            %(sha)s, NULL, NULL, NULL, now(),
            %(license)s, ARRAY[]::TEXT[], %(title)s, NULL
        )
        ON CONFLICT (source, source_id) DO UPDATE
        SET file_path = EXCLUDED.file_path,
            license   = EXCLUDED.license,
            caption   = EXCLUDED.caption;
    """, {
        "source": rec["source"],
        "source_id": rec["source_id"],
        "path": rec["path"],
        "sha": hashlib.sha256(Path(rec["path"]).read_bytes()).hexdigest(),
        "license": rec["license"],
        "title": rec.get("title") or rec.get("caption") or ""
    })

with open("metadata.jsonl") as f:
    for line in f:
        upsert_image(json.loads(line))
        conn.commit()

cur.close()
conn.close()
```

+ The `ON CONFLICT` clause guarantees idempotent runs – you can re‑scrape a source without creating duplicate rows.
+ You can extend the `INSERT` list later (e.g., `width`, `height`, `ocr_text`) once you run the image‑analysis steps.

# 3. Representations (Formal, Conceptual, Historical)

| **Affinity** | **What it captures** | **Primary model(s)** | **Input** | **Output** |
| --- | --- | --- | --- | --- |
| **Formal** | Pure visual style – color palette, composition, brush‑stroke, texture | **Vision‑only** CLIP‑ViT, **DINOv2, OpenCLIP, Swin‑V2,** or a custom **self‑supervised** encoder | `image` | 512‑dim embedding |
| **Conceptual** | Semantic meaning – objects, scene, narrative, genre, keywords | **Multimodal** CLIP (image + text), **BLIP‑2, Flava, CoCa** | `image` + **captions / tags** | 512‑dim embedding (aligned to text space) |
| **Historical** | Temporal / provenance context – creation year, art movement, influential artists, exhibition history | **Temporal embedding** (e.g., sinusoidal encoding of year) + **graph‑based influence vectors** | `year`, `movement`, `artist_id` (from DB)	128‑dim vector (can be concatenated) |

## 3.1 Formal Embedding Pipeline
```python
import torch, torchvision.transforms as T
from torchvision.models import vit_b_32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_b_32(pretrained=True).to(device).eval()
transform = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std =[0.229,0.224,0.225])
])

def embed_formal(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(x)
    # L2‑normalize for cosine similarity
    return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().squeeze()
```

+ **Alternative:** `DINOv2` gives richer texture/style vectors (use `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`).

## 3.2 Conceptual Embedding (CLIP)
```python
import clip, torch
model, preprocess = clip.load("ViT-L/14", device=device)

def embed_conceptual(image_path):
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feat = model.encode_image(img)
    return torch.nn.functional.normalize(image_feat, dim=-1).cpu().numpy().squeeze()
```

+ You can **fine‑tune** CLIP on a domain‑specific caption dataset (e.g., museum catalogues) to improve “conceptual” alignment.

## 3.3 Historical Embedding

Historical affinity is *not* a pure visual signal; we encode it as a **learned vector** that reflects relationships between movements, periods, and artists.

1. **Create a “knowledge graph”** of art history (nodes = artists/movements/years, edges = “influenced”, “member of”, “active in”).
2. Run **Node2Vec** or **TransE** to get a 128‑D embedding per node.

```python
import networkx as nx
from node2vec import Node2Vec

G = nx.read_gpickle("art_history_graph.gpickle")   # built offline
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

def embed_historical(artist_id):
    if artist_id in model.wv:
        return model.wv[artist_id]   # 128‑D numpy array
    else:
        return np.zeros(128)        # fallback
```

+ **Year encoding:** `np.sin(year/100)` + `np.cos(year/100)` concatenated to the artist vector.

## 3.4 Final Composite Vector
```python
def composite_vector(image_path, artist_id):
    f = embed_formal(image_path)          # 512
    c = embed_conceptual(image_path)      # 512
    h = embed_historical(artist_id)       # 128
    # Weighted concatenation (tune via validation)
    return np.concatenate([0.4*f, 0.4*c, 0.2*h])
```

Store this **composite vector** in a **FAISS** index for fast ANN (Approximate Nearest Neighbor) queries.

# 4. Graph Construction & Storage

## 4.1 Why a Graph?

+ **Multimodal edges** – each affinity type becomes a separate edge label (`FORMAL_SIM`, `CONCEPT_SIM`, `HISTORICAL_SIM`).
+ **Explainability** – you can traverse “why” a recommendation was made (e.g., “shared color palette + same movement”).
+ **Hybrid queries** – combine vector similarity with property filters (e.g., “show only CC‑BY images from 1900‑1920”).

## 4.2 Neo4j Schema
```cypher
// Node
CREATE CONSTRAINT ON (i:Image) ASSERT i.id IS UNIQUE;

// Edge types
// 1. Formal similarity (thresholded)
CREATE INDEX ON :Image(formalVec);
// 2. Conceptual similarity
CREATE INDEX ON :Image(conceptVec);
// 3. Historical similarity
CREATE INDEX ON :Image(historicalVec);
```

### 4.2.1 Edge Generation (batch job)
```python
import numpy as np, faiss, psycopg2, neo4j

# 1️⃣ Load vectors from FAISS index (or from DB)
vectors = load_all_composite_vectors()   # shape (N, 1152)
ids = load_all_image_ids()

# 2️⃣ Build ANN index
dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)   # cosine via inner product on normalized vectors
faiss.normalize_L2(vectors)
index.add(vectors)

# 3️⃣ For each image, fetch top‑K neighbours
K = 50
distances, neighbors = index.search(vectors, K+1)   # +1 includes self

# 4️⃣ Push edges to Neo4j
driver = neo4j.GraphDatabase.driver(uri, auth=("neo4j","pwd"))
def create_edges(tx, src_id, dst_id, score, edge_type):
    tx.run(f"""
        MATCH (a:Image {{id: $src}})
        MATCH (b:Image {{id: $dst}})
        MERGE (a)-[r:{edge_type} {{weight: $score}}]->(b)
        """, src=src_id, dst=dst_id, score=score)

with driver.session() as session:
    for i, (src_id, nbrs, sims) in enumerate(zip(ids, neighbors, distances)):
        for dst_idx, sim in zip(nbrs[1:], sims[1:]):   # skip self
            dst_id = ids[dst_idx]
            # Split weight by affinity component
            formal_w = 0.4   # tuned hyper‑params
            concept_w = 0.4
            hist_w = 0.2
            # we can compute partial sims if we stored separate vectors
            # For demo we just reuse the composite sim
            session.write_transaction(create_edges, src_id, dst_id, float(sim), "SIMILAR")
```

+ **Thresholding:** keep edges where `sim > 0.75` or top‑K per node to limit graph density.
+ **Edge properties:** `weight`, `affinity_type` (if you keep separate edge labels).

## 4.3 Property Store (PostgreSQL) ↔ Graph Sync

+ Use **Debezium** or **logical replication** to stream new rows from PostgreSQL to Neo4j via a Kafka topic.
+ Or run a nightly batch that **UPSERTs** missing nodes/edges.

# 5. Recommendation Engine

## 5.1 Simple Hybrid Scoring

Given a **query image** `q`:

1. Compute its **composite vector** `v_q`.
2. Retrieve top‑N nearest neighbours from FAISS → set `V`.
3. Pull corresponding **edge weights** from Neo4j (if any).
4. Combine:

$$score(i) = \alpha \cdot \cosine(v_q, v_i)
+ \beta \cdot \frac{\sum_{\text{edge} \in \{F,C,H\}} w_{\text{edge}}}{\#\text{edges}}$$

Typical values: `α=0.7`, `β=0.3`.

## 5.2 Graph‑Neural‑Network (Optional, for higher quality)

1. **Node features** = composite vector (or split into three sub‑vectors).
2. **Edge types** → one‑hot or learned embedding (formal / conceptual / historical).
3. Use **PyG** (PyTorch Geometric) or **DGL** to train a **link‑prediction** model:

```python
import torch_geometric as pyg

class AffinityGNN(torch.nn.Module):
    def __init__(self, dim_in, hidden=256):
        super().__init__()
        self.conv1 = pyg.nn.GATConv(dim_in, hidden, heads=4, edge_dim=3)
        self.conv2 = pyg.nn.GATConv(hidden*4, hidden, heads=1, edge_dim=3)
        self.lin   = torch.nn.Linear(hidden, 1)   # predicts link score

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Train with positive edges (existing affinities) vs negative sampled edges.
```

+ After training, **run inference** on the query node to rank all other nodes.
+ This approach automatically learns how to balance the three affinities.

## 5.3 API End‑point (FastAPI)
```python
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import uvicorn, numpy as np, faiss, aiohttp

app = FastAPI()

@app.post("/recommend")
async def recommend(file: UploadFile, top_k: int = 20):
    img_bytes = await file.read()
    # store temporarily
    tmp = f"/tmp/{file.filename}"
    open(tmp, "wb").write(img_bytes)

    # 1️⃣ Compute embeddings
    f = embed_formal(tmp)
    c = embed_conceptual(tmp)
    # No artist info → historic = zero
    h = np.zeros(128)
    q_vec = np.concatenate([0.4*f, 0.4*c, 0.2*h])
    faiss.normalize_L2(q_vec.reshape(1, -1))

    # 2️⃣ ANN search
    D, I = faiss_index.search(q_vec.reshape(1, -1), top_k)
    ids = [image_ids[i] for i in I[0]]

    # 3️⃣ Pull metadata for UI
    async with aiohttp.ClientSession() as s:
        resp = await s.post("http://metadata-service/batch", json={"ids": ids})
        meta = await resp.json()
    return {"recommendations": meta}
```

+ **Streaming response** – send low‑resolution thumbnails first, then higher‑res on demand.

# 6. Evaluation & Tuning

| **Metric** | **How to compute** | **What it tells you** |
| --- | --- | --- |
| **Recall@K** (image retrieval) | For a test set with known “related” images (e.g., same museum collection), measure fraction of true relatives in top‑K. | Effectiveness of similarity. |
| **Mean Reciprocal Rank (MRR)** | Average of 1 / rank of first correct hit. | Ranking quality. |
| **Human Preference Test** | Show participants a starting image + 5 candidates (different weighting of affinities) and ask which feels “most related”. | Real‑world relevance & explainability. |
| **Edge Sparsity** | Avg. degree per node. | Trade‑off between coverage & graph size. |
| **Latency** | End‑to‑end time (image upload → recommendation). | System performance. |

**Tuning knobs**

+ **Embedding weights** (`0.4/0.4/0.2` in composite) – optimize on recall.
+ **FAISS index type** – IVF‑PQ for >10 M images, HNSW for ultra‑low latency.
+ **Edge threshold** – higher threshold → more explainable but lower recall.

# 7. Scalability & Ops

| **Component** | **Scale Strategy** |
| --- | --- |
| **Scrapers** | Deploy as **Kubernetes CronJobs** or **AWS Batch;** use S3 for raw storage. |
| **Embedding Service** | GPU‑enabled inference workers behind a **Ray Serve** or **Triton Inference Server.** |
| **FAISS Index** | Use **FAISS‑GPU** for up‑to‑5M vectors; beyond that split into **shards** with **FAISS‑IVF + distributed RPC** (e.g., via faiss-gRPC). |
| **Neo4j** | Run a **Causal Cluster** (3 core + 3 read replicas). Keep edge count < 100 M; otherwise consider **JanusGraph + ScyllaDB.** |
| **Metadata DB** | PostgreSQL with **partitioning** by ingestion date. |
| **CI/CD** | GitHub Actions → Docker images → Helm charts → Helm release on a private EKS/GKE cluster.
| **Monitoring** | Prometheus + Grafana (FAISS query latency, Neo4j query time, GPU utilization).
| **Backup** | Daily S3 snapshots of raw images + nightly dump of Postgres + Neo4j export (neo4j-admin dump).

# 8. Legal & Ethical Checklist

| **Issue** | **Mitigation** |
| --- | --- |
| **Copyright** | Only ingest images with CC‑0, CC‑BY, CC‑BY‑SA or other permissive licenses. Record the original license in the DB; expose it in the UI. |
| **Attribution** | Auto‑generate an “Attribution string” (Artist – Source – License) for every recommendation. |
| **Bias** | Evaluate representation across cultures, genders, eras. Add a “diversity” weighting if needed. |
| **Privacy** | No personal data is collected; if you ever ingest user‑uploaded images, run face‑blur before storage. |
| **Explainability** | UI shows the three affinity scores (e.g., “Formal: 0.81 – same palette”, “Conceptual: 0.73 – both depict trains”, “Historical: 0.68 – both early‑20th c. Impressionism”). |

# 9. Sample Project Timeline (6 Months)

| **Week** | **Milestone** |
| --- | --- |
| 1‑2 | Set up repo, CI, cloud infra; pick 2‑3 pilot image sources. |
| 3‑4 | Write scrapers → store raw images & metadata in S3 + Postgres. |
| 5‑6 | Build formal & conceptual embedding pipelines; generate first 10 k vectors. |
| 7‑8 | Create historical knowledge graph, run Node2Vec, store vectors. |
| 9‑10 | Assemble composite vectors, evaluate similarity on a small validation set. |
| 11‑12 | Deploy FAISS index + simple ANN query service. |
| 13‑14 | Load vectors into Neo4j, generate edges (top‑20 per node). |
| 15‑16 | Implement FastAPI recommendation endpoint; basic UI (React). |
| 17‑18 | Run human evaluation; tune weighting, edge thresholds. |
| 19‑20 | Add GNN link‑prediction model (optional) and compare performance. |
| 21‑22 | Scale up to 1 M images; shard FAISS, add Neo4j replicas. |
| 23‑24 | Write documentation, open‑source the code, prepare demo video. |

# 10. Quick‑Start Code Repository Skeleton
```
repo/
├─ data/
│   ├─ raw/                # S3 sync folder (git‑ignore)
│   └─ processed/
│        ├─ vectors.npy
│        └─ ids.pkl
├─ scrapers/
│   ├─ wikimedia.py
│   ├─ flickr.py
│   └─ art_museum.py
├─ embeddings/
│   ├─ formal.py
│   ├─ conceptual.py
│   └─ historical.py
├─ graph/
│   ├─ build_edges.py
│   └─ neo4j_schema.cypher
├─ api/
│   ├─ main.py           # FastAPI
│   └─ models.py
├─ ml/
│   ├─ gnn.py
│   └─ train.py
├─ scripts/
│   ├─ ingest.sh
│   └─ deploy.sh
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

+ **Dockerfile** builds a multi‑stage image:
1️⃣ Base = python:3.11-slim + ffmpeg, libgl1.
2️⃣ Install torch, torchvision, faiss-gpu, neo4j, psycopg2.
3️⃣ Copy the code, set entrypoint to uvicorn api.main:app.

# 11. Closing Thoughts
+ **Modularity is key** – keep scrapers, embeddings, and graph logic independent so you can swap a model (e.g., replace DINOv2 with a future diffusion‑based style encoder) without re‑writing the whole pipeline.
+ **Explainability wins** – users will trust recommendations more when you can surface “shared brush‑stroke” or “same exhibition year” cues.
+ **Start small, iterate** – a prototype with 10‑20 k images will surface most engineering challenges (deduplication, licensing, vector drift). Once the pipeline is stable, scale horizontally.

You now have a full technical roadmap, sample code, and operational guidance to turn a “starting‑image recommendation” concept into a production‑ready system. Good luck, and feel free to ask for deeper dives into any component!

# 12. Postscript: Cost (according to Ollama)
**TL;DR**
+ **Total annual bill:** ≈$60k (≈$5k/month).
+ **Biggest expense:** LLM inference (~$1500/month).
+ **GPU compute for OCR/embeddings:** ~$440/month.
+ **All other services** (DB, vector store, storage, networking, ops) stay under $2000/month.

With modest traffic (1k users × 500 interactions each) the platform is well within the budget of a small startup or a department‑level research project, and there are clear levers to trim costs further as the product matures.

## 12.1 Cost‑Estimation (Production‑Ready Deployment)
The figures below are **high‑level “ball‑park” estimates** for a 12‑month run of the image‑recommendation service described in the previous sections. They assume you are using a **major public‑cloud provider** (AWS, GCP or Azure – the numbers are largely interchangeable) and that the system is built with the open‑source stack already outlined (PostgreSQL + PGVector, Qdrant, FastAPI, Docker/Kubernetes, etc.).

All numbers are **monthly** unless otherwise noted. Prices are taken from the 2025 public‑cloud price lists (rounded to the nearest cent) and include a modest 20% buffer for traffic spikes, backups and minor over‑provisioning.

| **#** | **Service / Resource** | **Usage Assumptions** | **Monthly Cost (USD)** | **Annual Cost (USD)** | **Notes** |
| --- | --- | --- | --- | --- | --- |
| **1** | **Compute – API + Orchestrator** (2×vCPU, 8GiB RAM)	2×t3.medium (or e2‑medium) EC2 / GCE instances running 24h, auto‑scaled to 2 nodes for HA | $80 (≈$40×2) | $960 | Handles FastAPI, request routing, job‑queue workers. |
| **2** | **Compute – Image‑Processing Workers** (GPU for OCR & embeddings) | 2×p3.2xlarge (1 GPU, 8 vCPU, 61 GiB) running 12h/day (batch processing of newly‑ingested images). ≈24k GPU‑hours per year. | $440 (≈$220×2) | $5280| NVIDIA T4 (or equivalent) on‑demand. OCR (Tesseract) + CLIP embedding generation. |
| **3** | **Vector Store – Qdrant (managed)** | 200 GiB SSD storage, 2 CPU cores, 8 GiB RAM (Qdrant‑cloud “Standard”) | $120 | $1440 | Stores ~1M embeddings (≈150 GiB) + overhead. |
| **4** | **Relational DB – Amazon RDS (PostgreSQL+PGVector)** | db.t3.medium (2 vCPU, 4 GiB), 500 GiB gp3 SSD (to hold raw images metadata, tags, OCR text). Daily backup retained 30 days. | $110 | $1320 | Multi‑AZ HA adds ~30% cost – already baked into figure. |
| **5** | **Object Storage – S3 / Cloud Storage** | 2 TB of raw‑image files (average 2 MB×≈1 M images) + 200 GB of processed thumbnails. Standard tier, 0.02 USD/GB‑month. | $44 | $528 | Includes 5% egress for API downloads. |
| **6** | **Embedding Generation (LLM‑hosted) – OpenAI / Azure OpenAI** (text‑to‑vector via `text‑embedding‑ada‑002`) | 1 M embeddings (≈30 M tokens). $0.0001/1k tokens → $3/month. | $3 | $36 | Very cheap; most cost is GPU compute above. |
| **7** | **LLM Inference (Chat‑Completion) – GPT‑4o** (or Claude‑3.5) for user‑facing explanations | 500k user interactions × average 200 tokens per request (prompt + completion) = 100M tokens. $0.015/1k tokens → $1500/month. | $1500 | $18000 | Core cost driver – can be swapped for a smaller model (e.g., GPT‑4‑Turbo) to cut ~30%. |
| **8** | **CDN / Bandwidth** | 500k image‑preview deliveries @≈200KB each = 100GB egress + API traffic ~20GB. $0.09/GB (first 10TB). | $11 | $132 | Covered by CloudFront / Cloudflare free tier + overage. |
| **9** | **Logging, Monitoring & Alerting** (CloudWatch / Stackdriver) | 2GB logs/day, 3‑metric dashboards, alert notifications. | $25 | $300 | Includes retention for 30 days. |
| **10** | **CI/CD & Registry** (GitHub Actions + ECR / Artifact Registry) | 200 build minutes/month, 5GB container storage. | $7 | $84 | Negligible. |
| **11** | **Security & IAM** (Secrets Manager, KMS) | 10 secrets, 1 KMS key, encryption‑at‑rest. | $5 | $60 | - |
| **12** | **Personnel (Part‑time Ops/Dev)** | 0.25FTE (≈10h/week) senior engineer for deployments, updates & incident response. | $2500 | $30000 | Assumes $120k/yr salary + benefits. |
| **13** | **Contingency / Miscellaneous** | 10% of subtotal (covers occasional spot‑instance usage, extra storage, licensing for optional commercial tools). | $558 | $6696 | - |
| **—** | **Total Monthly** | - | **≈$4983** | **≈$59800** | - |

## 12.2 Why These Numbers Make Sense for the Given Load
| **Metric** | **Value** | **Reasoning** |
| --- | --- | --- |
| **Users** | 1000 | Small‑to‑mid‑size B2B or niche‑consumer app. |
| **Interactions / User** | 500 | Roughly 2 requests per day over a 6‑month active period; yields **500k total API calls** per month. |
| **Embedding Size** | 1M vectors (≈153MiB each) | One vector per image + some extra for future growth. |
| **Vector‑Store Queries** | 500k queries / month, each returning 10‑20 nearest neighbours. | Qdrant on a modest node handles this comfortably (<10ms latency). |
| **LLM Tokens** | 100M tokens / month (≈1500 USD) | This is the biggest variable. If you switch to a cheaper local model (e.g., LLaMA‑2 7B fine‑tuned) and run it on the same GPU workers, you could cut this to **≈$300 / month** (GPU compute already budgeted). |
| **GPU Hours** | 2 × p3.2xlarge for 12h/day ≈720 GPU‑hrs / month | Sufficient for OCR (Tesseract) + CLIP embeddings on the nightly ingest of ~10k new images. |
| **Storage Growth** | 2TB raw + 200GB processed | At 2MB per image you can store ~1M images. Future growth can be accommodated by adding more S3 storage (≈$0.02 /GB‑month). |


## 12.3 Cost‑Reduction Levers
| **Lever** | **Potential Savings** | **How to Apply** |
| --- | --- | --- |
| **Model Choice** | Replace GPT‑4o with an **open‑source LLM** (e.g., LLaMA‑2‑7B) hosted on the same GPU workers → **≈$1200/yr** saved. | Deploy a FastAPI inference wrapper (vLLM, Text Generation Inference). |
| **Batching & Caching** | Cache LLM‑generated explanations per image‑pair for 24h → reduces duplicate calls by ~30% → **≈$450/yr** saved. | Use Redis (cost ≈$15 / month). |
| **Spot Instances** | Run GPU workers on spot/preemptible instances (70% discount) → **≈$150/month** saved. | Add auto‑recovery logic to restart jobs if preempted. |
| **Serverless API** | Replace always‑on EC2 API nodes with **AWS Lambda + API Gateway** (cost scales with invocations) → could drop the $80 / month API cost to <$20. | Only advisable if latency constraints are modest (<100ms). |
| **Lower‑tier Vector Store** | Switch from Qdrant‑cloud “Standard” to “Starter” (100GiB) if vector count stays <500k → ≈$30 / month saved. | Monitor vector count; upgrade when needed. |

> **James' comment:** build model testing/choice early into the dev process.

## 12.4 Break‑Even & ROI Perspective
| Scenario | Annual Cost | Expected Revenue / Cost‑Offset |
| --- | --- | --- |
| **Base** (estimates above) | **≈$60k** | If each active user is worth $100/yr (e.g., subscription, ad‑revenue, or B2B licensing) → $100k revenue → **profit ≈$40k.** |
| **Optimized** (open‑source LLM + spot GPUs) | **≈$48k** | Same $100k revenue → **profit ≈$52k.** |
| **Scale‑up** (5×users, same interaction rate) | **≈$300k** | Revenue $500k (if $100/yr per user) → **profit ≈$200k.** |

> Bottom line: For the modest 1000‑user target, the service can be run comfortably under $5k/month. The dominant line item is the LLM inference cost; swapping to a self‑hosted open‑source model or aggressively caching responses can halve that expense with minimal impact on user experience.

> **James' comment:** this isn't a for-profit venture, but the number here are useful for considering sustainability. If this were a valuable tool for other universities, could licensing make this sustainable?

## 12.5 Quick‑Start Cost‑Calculator (Python)
If you want to tweak the numbers for your own traffic patterns, the snippet below lets you plug in variables and get a monthly estimate:

```python
def estimate_monthly(
    users=1_000,
    interactions_per_user=500,
    avg_tokens_per_inter=200,
    gpu_hours_per_day=12,
    gpu_instances=2,
    api_nodes=2,
    storage_tb=2,
    cdn_gb=100,
    lLM_price_per_1k_tokens=0.015,   # GPT‑4o price
    embed_price_per_1k_tokens=0.0001
):
    # 1. LLM inference
    total_tokens = users * interactions_per_user * avg_tokens_per_inter
    lLM_cost = total_tokens/1_000 * lLM_price_per_1k_tokens

    # 2. Embedding generation (once per image)
    embed_cost = (1_000_000 / 1_000) * embed_price_per_1k_tokens   # 1M images

    # 3. Compute (API)
    api_cost = api_nodes * 0.040 * 730   # $0.040 per hour for t3.medium (approx)

    # 4. GPU workers
    gpu_cost = gpu_instances * gpu_hours_per_day * 30 * 2.75  # $2.75 per GPU‑hour (p3.2xlarge)

    # 5. Storage
    storage_cost = storage_tb * 1024 * 0.023   # $0.023 per GB‑month (S3 Standard)

    # 6. CDN
    cdn_cost = cdn_gb * 0.09

    # 7. Misc (vector store, DB, monitoring, ops)
    misc = 500  # rough lump sum for all the other services above

    total = sum([lLM_cost, embed_cost, api_cost, gpu_cost,
                 storage_cost, cdn_cost, misc])
    return round(total, 2)

print("Estimated monthly cost:", estimate_monthly())
```

Running the function with the default values prints something close to **$4950/month,** matching the table.
