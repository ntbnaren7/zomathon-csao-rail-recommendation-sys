"""
CSAO FastAPI Serving Endpoint — Production-ready API with latency tracking.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from serving.inference_pipeline import InferencePipeline

app = FastAPI(title="CSAO Add-On Recommendation API", version="1.0.0")

# Initialize pipeline (loaded once at startup)
pipeline: Optional[InferencePipeline] = None


@app.on_event("startup")
async def startup():
    global pipeline
    print("Loading inference pipeline...")
    pipeline = InferencePipeline()
    print("Pipeline ready!")


# ─── Request/Response Models ───

class RecommendRequest(BaseModel):
    user_id: str = "U00001"
    restaurant_id: str
    cart_item_ids: List[str]
    hour: int = 19
    day_of_week: int = 3
    meal_period: str = "dinner"
    season: str = "winter"
    festival: str = "none"
    is_weekend: bool = False
    top_k: int = 8


class ItemResponse(BaseModel):
    item_id: str
    name: str
    price: float
    meal_role: str
    category: str
    is_veg: bool
    score: float
    explanation: str


class RecommendResponse(BaseModel):
    recommendations: List[dict]
    meal_dna: dict
    meal_completion: float
    missing_roles: List[str]
    latency_ms: float
    n_candidates_scored: int


# ─── API Endpoints ───

@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Get add-on recommendations for a cart."""
    if pipeline is None:
        raise HTTPException(500, "Pipeline not initialized")

    result = pipeline.recommend(
        user_id=req.user_id,
        restaurant_id=req.restaurant_id,
        cart_item_ids=req.cart_item_ids,
        hour=req.hour,
        day_of_week=req.day_of_week,
        meal_period=req.meal_period,
        season=req.season,
        festival=req.festival,
        is_weekend=req.is_weekend,
        top_k=req.top_k,
    )
    return result


@app.get("/api/restaurants")
async def get_restaurants(city: str = None):
    """Get list of restaurants."""
    if pipeline is None:
        raise HTTPException(500, "Pipeline not initialized")
    return pipeline.get_restaurants(city)


@app.get("/api/menu/{restaurant_id}")
async def get_menu(restaurant_id: str):
    """Get menu for a restaurant."""
    if pipeline is None:
        raise HTTPException(500, "Pipeline not initialized")
    menu = pipeline.get_restaurant_menu(restaurant_id)
    if not menu:
        raise HTTPException(404, f"Restaurant {restaurant_id} not found")
    return menu


@app.get("/api/cities")
async def get_cities():
    """Get available cities."""
    return ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Lucknow"]


# ─── Serve static demo ───
app.mount("/static", StaticFiles(directory="demo"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive demo."""
    with open("demo/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=False)
