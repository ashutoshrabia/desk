
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import logging
from pydantic import BaseModel
from typing import List, Dict

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set cache for HF models
os.environ["HF_HOME"] = "/app/cache"

# FastAPI app
app = FastAPI(
    title="News Article Similarity Search API",
    description="API for finding similar news articles",
    version="1.0.3"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for API
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

# Load prebuilt FAISS index
INDEX_PATH = "news.index"
META_PATH = "meta.json"

try:
    INDEX = faiss.read_index(INDEX_PATH)
    logger.info(f"Loaded FAISS index from {INDEX_PATH}")
except Exception as e:
    logger.error(f"Failed loading FAISS index: {e}")
    raise

# Load metadata
try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)
    logger.info(f"Loaded metadata from {META_PATH}, total articles: {len(META)}")
except Exception as e:
    logger.error(f"Failed loading metadata: {e}")
    raise

# SentenceTransformer for queries only
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("SentenceTransformer model initialized for query embeddings.")


def get_head(title: str) -> str:
    """
    Generate the HTML <head> section, injecting the page title.
    """
    template = """
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
  <style>
    html, body { height: 100%; margin: 0; }
    body { background: #e9ecef; }
    .hero { position: relative; width: 100%; min-height: 100vh; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #343a40; }
    .hero img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.3; z-index: 0; }
    .hero .content { position: relative; z-index: 1; color: #fff; text-align: center; }
    .hero h1 { font-weight: 800; font-size: 4rem; }
    .hero p { font-size: 1.5rem; }
    .form-range { width: 150px; }
    .accordion-button { font-weight: bold; }
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js\"></script>
</head>
"""
    return template.format(title=title)


@app.get("/", response_class=HTMLResponse)
async def root():
    head = get_head("News API Home")
    return f"""
<html>{head}
<body>
  <div class=\"hero\">
    <img src=\"/static/free-old-newspaper-vector.jpg\" alt=\"News Background\" />
    <div class=\"content\">
      <h1 class=\"display-3\">News Similarity Explorer</h1>
      <p class=\"lead\">Instantly find related news articles and explore insights!</p>
      <a href=\"/search\" class=\"btn btn-lg btn-primary\">Start Searching</a>
    </div>
  </div>
</body>
</html>
"""


@app.get("/search", response_class=HTMLResponse)
async def search_form(request: Request, query: str = "", top_k: int = 5):
    head = get_head("Search News")

    # Empty form
    if not query:
        return f"""
<html>{head}
<body>
  <div class=\"container py-5\">
    <h2 class=\"text-center mb-4\">🔍 Search News Articles</h2>
    <form method=\"get\" class=\"d-flex justify-content-center gap-2\">
      <input type=\"text\" name=\"query\" class=\"form-control w-50\" placeholder=\"Your search term...\" required>
      <input type=\"number\" name=\"top_k\" class=\"form-control w-auto\" value=\"5\" min=\"1\" max=\"20\" title=\"Number of results\">
      <button type=\"submit\" class=\"btn btn-success\">Go</button>
    </form>
  </div>
</body>
</html>
"""

    # Perform search
    emb = MODEL.encode([query])
    dists, idxs = INDEX.search(np.array(emb, dtype="float32"), top_k)
    results = []
    for dist, i in zip(dists[0], idxs[0]):
        sim = 1 - (dist / 2)
        meta = META[i]
        results.append({**meta, "similarity": float(sim)})

    # Build accordion items
    items = "\n".join([
        f"""
        <div class=\"accordion-item\">
          <h2 class=\"accordion-header\" id=\"heading{idx}\">
            <button class=\"accordion-button collapsed\" type=\"button\" data-bs-toggle=\"collapse\" data-bs-target=\"#collapse{idx}\" aria-expanded=\"false\" aria-controls=\"collapse{idx}\">
              {r['heading']} — <small class=\"text-muted\">{r['date']}, {r['news_type']}</small> <span class=\"badge bg-info ms-2\">{r['similarity']:.2f}</span>
            </button>
          </h2>
          <div id=\"collapse{idx}\" class=\"accordion-collapse collapse\" aria-labelledby=\"heading{idx}\" data-bs-parent=\"#resultsAccordion\">
            <div class=\"accordion-body\">{r['article']}</div>
          </div>
        </div>
        """ for idx, r in enumerate(results)
    ])

    return f"""
<html>{head}
<body>
  <div class=\"container py-5\">
    <h2 class=\"mb-4\">Results for: <em>{query}</em></h2>
    <div class=\"accordion\" id=\"resultsAccordion\">{items or '<p class=\"text-muted\">No matches found.</p>'}</div>
    <div class=\"text-center mt-4\">
      <a href=\"/search\" class=\"btn btn-outline-primary\">🔄 New Search</a>
    </div>
  </div>
</body>
</html>
"""


@app.post("/api/search")
async def api_search_post(query: SearchQuery):
    emb = MODEL.encode([query.query])
    dists, idxs = INDEX.search(np.array(emb, dtype="float32"), query.top_k)
    results = []
    for dist, i in zip(dists[0], idxs[0]):
        sim = 1 - (dist / 2)
        meta = META[i]
        results.append({**meta, "similarity": float(sim)})
    return {"results": results}


@app.get("/api/search")
async def api_search_get(q: str, top_k: int = 5):
    emb = MODEL.encode([q])
    dists, idxs = INDEX.search(np.array(emb, dtype="float32"), top_k)
    results = []
    for dist, i in zip(dists[0], idxs[0]):
        sim = 1 - (dist / 2)
        meta = META[i]
        results.append({**meta, "similarity": float(sim)})
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
