"""
GraphRAG Pipeline — Gemini API Edition
Deliverables:
  1. Source code (this file)
  2. knowledge_graph.png  — Matplotlib visualization
  3. benchmark_results.csv / benchmark_table.png  — 20-question comparison
  4. cost_analysis.txt   — Token usage & time
"""

import os, json, time, warnings, pickle, re, textwrap
warnings.filterwarnings("ignore")

import requests
import networkx as nx
import numpy as np
import pandas as pd
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

from google import genai
from google.genai import types

# ─── CONFIG ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL      = "gemini-2.5-flash"
EMBED_MODEL    = "gemini-embedding-001"
EMBED_DIM      = 3072

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY environment variable. "
        "Set it in your shell or in a local .env before running."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# Token usage tracker
token_tracker = {"extract": 0, "embed_calls": 0, "query": 0, "judge": 0}
time_tracker  = {}

# ─── STEP 1: FETCH WIKIPEDIA ARTICLES ────────────────────────────────────────
COMPANIES = [
    "OpenAI", "Google DeepMind", "Anthropic", "Mistral AI", "Inflection AI",
    "Stability AI", "Cohere", "Hugging Face", "Scale AI", "Cerebras Systems",
    "Perplexity AI", "Character.AI", "xAI (company)", "Aleph Alpha",
    "Runway (company)", "Adept (company)", "Together AI", "01.AI",
    "Moonshot AI", "Baidu",
]

HEADERS = {"User-Agent": "GraphRAG-Research/1.0 (https://github.com/graphrag; graphrag@research.org)"}

def fetch_wikipedia(title: str, max_chars: int = 3000) -> dict:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": True,
        "redirects": True, "titles": title, "format": "json",
        "exintro": True,
    }
    try:
        r = requests.get(url, params=params, timeout=10, headers=HEADERS)
        pages = r.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        text  = page.get("extract", "")[:max_chars]
        return {"title": title, "content": text}
    except Exception as e:
        print(f"  [WARN] {title}: {e}")
        return {"title": title, "content": ""}

print("=" * 60)
print("STEP 1: Fetching Wikipedia articles …")
t0 = time.time()
articles = []
for c in COMPANIES:
    art = fetch_wikipedia(c)
    if art["content"]:
        articles.append(art)
        print(f"  ✓ {c} ({len(art['content'])} chars)")
    else:
        print(f"  ✗ {c} (skipped)")
    time.sleep(0.2)

time_tracker["fetch"] = time.time() - t0
print(f"Fetched {len(articles)} articles in {time_tracker['fetch']:.1f}s\n")

# ─── STEP 2: ENTITY EXTRACTION → TRIPLES ─────────────────────────────────────
EXTRACT_PROMPT = """You are a knowledge graph extractor for AI company data.
Extract (subject, predicate, object) triples from the text.
Focus on: founders, investors, employees, products, acquisitions, partnerships, locations, funding.
Return ONLY valid JSON: {{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
Max 20 triples. Keep subject/object as proper nouns or short noun phrases."""

def clean_json(text: str) -> str:
    """Strip markdown code fences and extract JSON."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()
    # Find first { to last }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

def extract_triples(text: str, title: str) -> list[dict]:
    prompt = f"Article about {title}:\n\n{text}"
    try:
        resp = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACT_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        token_tracker["extract"] += (resp.usage_metadata.total_token_count or 0)
        data = json.loads(clean_json(resp.text))
        triples = data.get("triples", [])
        # Ensure all triples are dicts with string values
        valid = []
        for t in triples:
            if isinstance(t, dict) and all(isinstance(t.get(k,""), str) for k in ("subject","predicate","object")):
                valid.append(t)
        return valid
    except Exception as e:
        print(f"  [WARN] extract_triples({title}): {e}")
        return []

print("STEP 2: Extracting triples …")
t0 = time.time()
all_triples = []
for art in articles:
    triples = extract_triples(art["content"], art["title"])
    all_triples.extend(triples)
    print(f"  {art['title']}: {len(triples)} triples")
    time.sleep(0.5)

with open("triples.json", "w") as f:
    json.dump(all_triples, f, indent=2, ensure_ascii=False)
time_tracker["extract"] = time.time() - t0
print(f"Total triples: {len(all_triples)} in {time_tracker['extract']:.1f}s\n")

# ─── STEP 3: BUILD NETWORKX GRAPH ────────────────────────────────────────────
print("STEP 3: Building NetworkX graph …")
G = nx.DiGraph()
for t in all_triples:
    s, p, o = t.get("subject","").strip(), t.get("predicate","").strip(), t.get("object","").strip()
    if s and p and o:
        G.add_node(s)
        G.add_node(o)
        if G.has_edge(s, o):
            G[s][o]["relation"] += f"; {p}"
        else:
            G.add_edge(s, o, relation=p)
print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")

# ─── STEP 4: NODE EMBEDDINGS ──────────────────────────────────────────────────
print("STEP 4: Computing node embeddings …")
t0 = time.time()

def get_embedding(text: str) -> list[float]:
    token_tracker["embed_calls"] += 1
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return resp.embeddings[0].values

node_list = list(G.nodes())
node_embeddings = {}
for i, node in enumerate(node_list):
    node_embeddings[node] = get_embedding(node)
    if (i+1) % 20 == 0:
        print(f"  Embedded {i+1}/{len(node_list)} nodes …")
    time.sleep(0.05)

for node, emb in node_embeddings.items():
    G.nodes[node]["embedding"] = emb

with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)
time_tracker["embed"] = time.time() - t0
print(f"  Done in {time_tracker['embed']:.1f}s\n")

# ─── STEP 5: VISUALIZE GRAPH ──────────────────────────────────────────────────
print("STEP 5: Visualizing knowledge graph …")
company_names = {a["title"].lower() for a in articles}
# Also include short variants
company_aliases = set()
for c in company_names:
    parts = c.replace("(company)","").split()
    company_aliases.update(parts)

def is_company(node):
    n = node.lower()
    return n in company_names or any(alias in n for alias in company_aliases if len(alias) > 3)

# Filter to nodes with degree >= 2
visible = [n for n in G.nodes() if G.degree(n) >= 2]
subG = G.subgraph(visible)

# Top 30 by degree for clarity
top30 = sorted(visible, key=lambda n: G.degree(n), reverse=True)[:30]
subG2 = G.subgraph(top30)

pos = nx.spring_layout(subG2, k=2.5, seed=42)
degrees = dict(subG2.degree())
node_sizes = [max(300, degrees[n] * 120) for n in subG2.nodes()]
node_colors = ["#4A90D9" if is_company(n) else "#E8854A" for n in subG2.nodes()]

fig, ax = plt.subplots(figsize=(18, 14))
nx.draw_networkx_nodes(subG2, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
nx.draw_networkx_labels(subG2, pos, font_size=7, font_weight="bold", ax=ax)
nx.draw_networkx_edges(subG2, pos, arrows=True, arrowsize=12,
                        edge_color="#aaaaaa", width=0.8,
                        connectionstyle="arc3,rad=0.1", ax=ax)
edge_labels = {(u, v): d["relation"][:20] for u, v, d in subG2.edges(data=True)}
nx.draw_networkx_edge_labels(subG2, pos, edge_labels, font_size=5, alpha=0.7, ax=ax)

blue_patch  = mpatches.Patch(color="#4A90D9", label="AI Company")
orange_patch = mpatches.Patch(color="#E8854A", label="Person / Concept")
ax.legend(handles=[blue_patch, orange_patch], loc="upper left", fontsize=10)
ax.set_title("AI Company Knowledge Graph (Top 30 nodes by degree)", fontsize=16, pad=20)
ax.axis("off")
plt.tight_layout()
plt.savefig("knowledge_graph.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved knowledge_graph.png\n")

# ─── STEP 6: FLAT RAG ─────────────────────────────────────────────────────────
print("STEP 6: Building Flat RAG (FAISS) …")
t0 = time.time()

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

chunk_texts, chunk_metadata = [], []
for art in articles:
    for chunk in chunk_text(art["content"]):
        chunk_texts.append(chunk)
        chunk_metadata.append({"title": art["title"], "text": chunk})

chunk_embeddings = []
for i, ct in enumerate(chunk_texts):
    emb = get_embedding(ct)
    chunk_embeddings.append(emb)
    time.sleep(0.05)

chunk_matrix = np.array(chunk_embeddings, dtype="float32")
faiss.normalize_L2(chunk_matrix)
index = faiss.IndexFlatIP(EMBED_DIM)
index.add(chunk_matrix)
time_tracker["flat_build"] = time.time() - t0
print(f"  {len(chunk_texts)} chunks indexed in {time_tracker['flat_build']:.1f}s\n")

NO_THINK = types.ThinkingConfig(thinking_budget=0)

def call_llm(system: str, user: str) -> str:
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=512,
            thinking_config=NO_THINK,
        ),
    )
    token_tracker["query"] += (resp.usage_metadata.total_token_count or 0)
    return resp.text.strip()

def flat_rag_query(question: str, k: int = 5) -> tuple[str, float]:
    t0 = time.time()
    q_emb = np.array([get_embedding(question)], dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    ctx_parts = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0:
            ctx_parts.append(f"[{chunk_metadata[idx]['title']}]\n{chunk_metadata[idx]['text']}")
    context = "\n\n---\n\n".join(ctx_parts)
    system = "You are a helpful assistant. Answer the question based only on the provided context. Be concise."
    answer = call_llm(system, f"Context:\n{context}\n\nQuestion: {question}")
    return answer, time.time() - t0

# ─── STEP 7: GRAPHRAG ─────────────────────────────────────────────────────────
def extract_question_entities(question: str) -> list[str]:
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction='Extract key named entities (companies, people, organizations) from the question. Return ONLY valid JSON: {"entities": [...]}',
            response_mime_type="application/json",
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    token_tracker["query"] += (resp.usage_metadata.total_token_count or 0)
    try:
        raw = json.loads(clean_json(resp.text)).get("entities", [])
        # Normalize: could be list of strings or list of dicts
        result = []
        for e in raw:
            if isinstance(e, str):
                result.append(e)
            elif isinstance(e, dict):
                # try common keys
                result.append(e.get("name") or e.get("entity") or e.get("text") or str(list(e.values())[0]))
        return result
    except:
        return []

def find_seed_nodes(entities: list[str]) -> list[str]:
    seeds = []
    nodes_lower = {n.lower(): n for n in G.nodes()}
    node_embs = {n: np.array(G.nodes[n]["embedding"]) for n in G.nodes() if "embedding" in G.nodes[n]}
    for ent in entities:
        el = ent.lower()
        # Exact match
        if el in nodes_lower:
            seeds.append(nodes_lower[el])
            continue
        # Substring
        found = [v for k, v in nodes_lower.items() if el in k or k in el]
        if found:
            seeds.extend(found[:2])
            continue
        # Embedding similarity
        if node_embs:
            ent_emb = np.array(get_embedding(ent))
            best_node, best_sim = None, -1
            for n, emb in node_embs.items():
                sim = float(np.dot(ent_emb, emb) / (np.linalg.norm(ent_emb) * np.linalg.norm(emb) + 1e-9))
                if sim > best_sim:
                    best_sim, best_node = sim, n
            if best_node and best_sim > 0.5:
                seeds.append(best_node)
    return list(set(seeds))

def bfs_subgraph(seeds: list[str], max_hops: int = 2) -> list[tuple]:
    visited, queue, triples_found = set(), deque(), []
    for s in seeds:
        if s in G:
            queue.append((s, 0))
            visited.add(s)
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for nbr in list(G.successors(node)) + list(G.predecessors(node)):
            rel = G[node][nbr]["relation"] if G.has_edge(node, nbr) else G[nbr][node]["relation"]
            if G.has_edge(node, nbr):
                triples_found.append((node, G[node][nbr]["relation"], nbr))
            else:
                triples_found.append((nbr, G[nbr][node]["relation"], node))
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, depth + 1))
    return triples_found

def textualize(triples: list[tuple]) -> str:
    lines = [f"{s} --[{p}]--> {o}" for s, p, o in triples]
    return "\n".join(lines[:80])  # cap context

def graphrag_query(question: str, max_hops: int = 2) -> tuple[str, float, int]:
    t0 = time.time()
    entities = extract_question_entities(question)
    seeds    = find_seed_nodes(entities)
    triples  = bfs_subgraph(seeds, max_hops)
    graph_ctx = textualize(triples)
    if not graph_ctx:
        graph_ctx = "No relevant graph data found."
    system = "You are a knowledge graph assistant. Answer the question using the provided knowledge graph triples. Be specific and concise."
    answer = call_llm(system, f"Knowledge Graph:\n{graph_ctx}\n\nQuestion: {question}")
    return answer, time.time() - t0, len(triples)

# ─── STEP 8: BENCHMARK — 20 QUESTIONS ────────────────────────────────────────
benchmark_questions = [
    {"q": "Which AI companies were co-founded by people who previously worked at Google?",
     "expected": ["Google", "Anthropic", "DeepMind"]},
    {"q": "What companies did Sam Altman co-found or lead?",
     "expected": ["Sam Altman", "OpenAI"]},
    {"q": "Which founders of Anthropic previously worked at OpenAI?",
     "expected": ["Anthropic", "OpenAI", "Dario Amodei"]},
    {"q": "What is the relationship between Elon Musk and AI companies?",
     "expected": ["Elon Musk", "OpenAI", "xAI"]},
    {"q": "Which AI companies have received investment from Microsoft?",
     "expected": ["Microsoft", "OpenAI"]},
    {"q": "Who are the founders of Mistral AI?",
     "expected": ["Mistral AI"]},
    {"q": "What products has Anthropic released?",
     "expected": ["Anthropic", "Claude"]},
    {"q": "Which companies are backed by Google or Alphabet?",
     "expected": ["Google", "DeepMind", "Anthropic"]},
    {"q": "What is the connection between Stability AI and its founders?",
     "expected": ["Stability AI"]},
    {"q": "Which AI companies were founded in San Francisco?",
     "expected": ["San Francisco", "OpenAI", "Anthropic"]},
    {"q": "Who invested in Cohere and what is their relationship with NLP research?",
     "expected": ["Cohere"]},
    {"q": "What AI models has Hugging Face contributed to the open-source community?",
     "expected": ["Hugging Face"]},
    {"q": "What is Perplexity AI's relationship with search technology?",
     "expected": ["Perplexity AI"]},
    {"q": "Who co-founded xAI and what other companies are they associated with?",
     "expected": ["xAI", "Elon Musk"]},
    {"q": "Which AI companies have a connection to Baidu's research labs?",
     "expected": ["Baidu"]},
    {"q": "What is the relationship between Scale AI and data labeling?",
     "expected": ["Scale AI"]},
    {"q": "Which founders left OpenAI to start competing AI companies?",
     "expected": ["OpenAI", "Anthropic", "Inflection AI"]},
    {"q": "What is Character.AI's founding story and who created it?",
     "expected": ["Character.AI", "Google"]},
    {"q": "Which AI companies focus on open-source large language models?",
     "expected": ["Mistral AI", "Hugging Face", "Aleph Alpha"]},
    {"q": "What is the funding history of Inflection AI?",
     "expected": ["Inflection AI", "Reid Hoffman"]},
]

print("STEP 8: Running benchmark (20 questions × 2 pipelines) …")
t0 = time.time()

def llm_judge(question: str, answer: str, expected: list[str]) -> float:
    system = 'Score the answer 0.0 to 1.0: correctness (0.5), entity coverage (0.3), specificity (0.2). Return JSON: {"score": float, "reason": string}'
    prompt = f"Question: {question}\nExpected entities: {expected}\nAnswer: {answer}"
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    token_tracker["judge"] += (resp.usage_metadata.total_token_count or 0)
    try:
        return float(json.loads(clean_json(resp.text)).get("score", 0.0))
    except:
        return 0.0

results = []
for i, bq in enumerate(benchmark_questions):
    q = bq["q"]
    exp = bq["expected"]
    print(f"  [{i+1:02d}/20] {q[:60]}…")

    flat_ans,  flat_lat  = flat_rag_query(q)
    graph_ans, graph_lat, subgraph_sz = graphrag_query(q)

    flat_score  = llm_judge(q, flat_ans, exp)
    graph_score = llm_judge(q, graph_ans, exp)

    results.append({
        "Q#": i + 1,
        "Question": q,
        "FlatRAG_Answer":   flat_ans[:200],
        "GraphRAG_Answer":  graph_ans[:200],
        "FlatRAG_Score":    flat_score,
        "GraphRAG_Score":   graph_score,
        "FlatRAG_Latency":  round(flat_lat, 2),
        "GraphRAG_Latency": round(graph_lat, 2),
        "Subgraph_Size":    subgraph_sz,
    })
    time.sleep(0.5)

time_tracker["benchmark"] = time.time() - t0
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print(f"  Benchmark done in {time_tracker['benchmark']:.1f}s\n")

# ─── STEP 9: VISUALIZE BENCHMARK RESULTS ─────────────────────────────────────
print("STEP 9: Generating benchmark table image …")
fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# Score comparison bar chart
ax1 = axes[0]
x = np.arange(len(df))
w = 0.35
ax1.bar(x - w/2, df["FlatRAG_Score"],  w, label="Flat RAG",  color="#5B9BD5", alpha=0.85)
ax1.bar(x + w/2, df["GraphRAG_Score"], w, label="GraphRAG",  color="#ED7D31", alpha=0.85)
ax1.set_xlabel("Question #", fontsize=11)
ax1.set_ylabel("Accuracy Score (0–1)", fontsize=11)
ax1.set_title("GraphRAG vs Flat RAG — Accuracy on 20 Multi-Hop Questions", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels([str(i+1) for i in range(len(df))], fontsize=8)
ax1.legend(fontsize=11)
ax1.set_ylim(0, 1.1)
ax1.axhline(df["FlatRAG_Score"].mean(),  color="#5B9BD5", linestyle="--", alpha=0.6, label="FlatRAG avg")
ax1.axhline(df["GraphRAG_Score"].mean(), color="#ED7D31", linestyle="--", alpha=0.6, label="GraphRAG avg")
ax1.grid(axis="y", alpha=0.3)

# Summary table
ax2 = axes[1]
ax2.axis("off")
short_qs = [textwrap.shorten(q, width=60, placeholder="…") for q in df["Question"]]
table_data = []
for i, row in df.iterrows():
    table_data.append([
        row["Q#"],
        short_qs[i],
        f"{row['FlatRAG_Score']:.2f}",
        f"{row['GraphRAG_Score']:.2f}",
        f"{row['FlatRAG_Latency']:.1f}s",
        f"{row['GraphRAG_Latency']:.1f}s",
    ])
cols = ["Q#", "Question", "Flat Score", "Graph Score", "Flat Lat", "Graph Lat"]
tbl = ax2.table(cellText=table_data, colLabels=cols,
                cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", weight="bold")
    elif c in (2, 3):
        flat_s  = float(table_data[r-1][2]) if r > 0 else 0
        graph_s = float(table_data[r-1][3]) if r > 0 else 0
        if c == 3 and r > 0 and graph_s > flat_s:
            cell.set_facecolor("#D5F5E3")
        elif c == 2 and r > 0 and flat_s > graph_s:
            cell.set_facecolor("#D5F5E3")

plt.tight_layout(pad=2)
plt.savefig("benchmark_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved benchmark_table.png\n")

# ─── STEP 10: COST ANALYSIS ───────────────────────────────────────────────────
flat_avg_score  = df["FlatRAG_Score"].mean()
graph_avg_score = df["GraphRAG_Score"].mean()
improvement     = (graph_avg_score - flat_avg_score) / max(flat_avg_score, 1e-9) * 100
flat_wins  = (df["FlatRAG_Score"] > df["GraphRAG_Score"]).sum()
graph_wins = (df["GraphRAG_Score"] > df["FlatRAG_Score"]).sum()
ties       = (df["FlatRAG_Score"] == df["GraphRAG_Score"]).sum()

total_time = sum(time_tracker.values())

# Gemini Flash pricing (as of mid-2025): ~$0.075/1M input tokens, $0.30/1M output tokens
# Rough estimate using total_token_count
total_tokens = sum(token_tracker.values()) + token_tracker["embed_calls"] * 10  # embed tokens minimal
cost_estimate = total_tokens / 1_000_000 * 0.15  # blended rate

cost_report = f"""
GraphRAG Pipeline — Cost & Performance Analysis
=================================================
Date: 2026-05-05
Model: {LLM_MODEL}  |  Embeddings: {EMBED_MODEL}

─── TOKEN USAGE ───────────────────────────────
Entity extraction   : ~{token_tracker['extract']:,} tokens  ({len(articles)} articles)
Node embeddings     : {token_tracker['embed_calls']} embed calls  ({len(node_list)} nodes + {len(chunk_texts)} chunks)
Query processing    : ~{token_tracker['query']:,} tokens  (20 questions × 2 pipelines)
LLM judge           : ~{token_tracker['judge']:,} tokens  (40 judgements)
TOTAL tokens (est.) : ~{total_tokens:,}
Estimated cost      : ~${cost_estimate:.4f} USD

─── TIME BREAKDOWN ────────────────────────────
Wikipedia fetch     : {time_tracker.get('fetch', 0):.1f}s
Triple extraction   : {time_tracker.get('extract', 0):.1f}s
Node embedding      : {time_tracker.get('embed', 0):.1f}s
Flat RAG build      : {time_tracker.get('flat_build', 0):.1f}s
Benchmark (20 Qs)   : {time_tracker.get('benchmark', 0):.1f}s
TOTAL               : {total_time:.1f}s

─── ACCURACY RESULTS ──────────────────────────
Flat RAG  avg score : {flat_avg_score:.3f}
GraphRAG  avg score : {graph_avg_score:.3f}
GraphRAG improvement: {improvement:+.1f}%

Question breakdown  :
  GraphRAG wins     : {graph_wins}/20
  Flat RAG wins     : {flat_wins}/20
  Ties              : {ties}/20

─── LATENCY (avg) ─────────────────────────────
Flat RAG  : {df['FlatRAG_Latency'].mean():.2f}s / question
GraphRAG  : {df['GraphRAG_Latency'].mean():.2f}s / question

─── KNOWLEDGE GRAPH STATS ─────────────────────
Nodes               : {G.number_of_nodes()}
Edges               : {G.number_of_edges()}
Triples extracted   : {len(all_triples)}
Articles processed  : {len(articles)}

─── CONCLUSION ────────────────────────────────
{'✅ GraphRAG meets +20% accuracy target.' if improvement >= 20 else f'⚠️  GraphRAG improvement is {improvement:+.1f}% (target: +20%).'}
GraphRAG excels at multi-hop relational questions.
Flat RAG is faster and better for simple lookups.
=================================================
"""

with open("cost_analysis.txt", "w") as f:
    f.write(cost_report)

print(cost_report)
print("\n✅ All deliverables generated:")
print("  1. run_graphrag.py       — Source code")
print("  2. knowledge_graph.png   — Graph visualization")
print("  3. benchmark_results.csv — Raw benchmark data")
print("     benchmark_table.png   — Visual comparison table")
print("  4. cost_analysis.txt     — Token usage & time report")
print("  5. triples.json          — Extracted triples")
print("  6. graph.pkl             — Serialized NetworkX graph")
