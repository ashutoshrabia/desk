from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory for Hugging Face
os.environ["HF_HOME"] = "/app/cache"

app = FastAPI(
    title="News Article Similarity Search API",
    description="API for finding similar news articles",
    version="1.0.3"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class DocumentStore:
    def __init__(self, csv_path: str = "Articles.csv"):
        self.model = None
        self.dimension = 384
        self.index = None
        self.documents = []
        self.csv_path = csv_path
        self.load_csv()

    def load_csv(self):
        logger.info(f"Loading CSV from: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding='latin1')
        for col in ['Article', 'Date', 'Heading', 'NewsType']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        articles = df['Article'].astype(str).tolist()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self.model.encode(articles, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
        for idx, row in df.iterrows():
            self.documents.append({
                'id': idx,
                'article': row['Article'],
                'date': row['Date'],
                'heading': row['Heading'],
                'news_type': row['NewsType']
            })
        logger.info(f"Indexed {len(self.documents)} articles")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        embed = self.model.encode([query])[0]
        dists, idxs = self.index.search(np.array([embed]), top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            sim = 1 - (dist / 2)
            results.append({**doc, 'similarity': float(sim)})
        return results

doc_store = DocumentStore()

def get_head(title: str) -> str:
    """
    Generates the HTML <head> section with CSS and JS links.
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

@app.on_event("startup")
async def startup_event():
    if not doc_store.documents:
        raise RuntimeError("No articles loaded.")

@app.get("/", response_class=HTMLResponse)
async def root():
    head = get_head("News API Home")
    return f"""
    <html>{head}<body>
      <div class=\"hero\">
        <img src=\"/static/free-old-newspaper-vector.jpg\" alt=\"Newspaper Background\"> 
        <div class=\"content\">
          <h1 class=\"display-3\">News Similarity Explorer</h1>
          <p class=\"lead\">Instantly find related news articles and explore insights!</p>
          <a href=\"/search\" class=\"btn btn-lg btn-primary\">Start Searching</a>
        </div>
      </div>
    </body></html>
    """

@app.get("/search", response_class=HTMLResponse)
async def search_form(request: Request, query: str = "", top_k: int = 5):
    head = get_head("Search News")
    if not query:
        return f"""
        <html>{head}<body>
          <div class=\"container py-5\">... (form HTML unchanged) ...
        </body></html>
        """
    results = doc_store.search(query, top_k)
    items = "\n".join([...])  # build accordion items as before
    return f"""
    <html>{head}<body>... (results HTML unchanged) ...</body></html>
    """

@app.post("/api/search")
async def api_search(query: SearchQuery):
    return {"results": doc_store.search(query.query, query.top_k)}

@app.get("/api/search")
async def api_search_get(q: str, top_k: int = 5):
    return {"results": doc_store.search(q, top_k)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
