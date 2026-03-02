<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png" alt="Zomato" width="200"/>
</p>

<h1 align="center">🍽️ CSAO Rail — Cart Super Add-On Recommendation System</h1>

<p align="center">
  <strong>An intelligent ML system that suggests the perfect add-on items to complement your Zomato cart</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0+-006600?logo=xgboost" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/CUDA-GPU_Accelerated-76b900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Latency-~5ms-brightgreen" alt="Latency"/>
</p>

---

## 🎯 What is CSAO Rail?

When a customer builds their cart on Zomato, the **CSAO (Cart Super Add-On) Rail** recommends complementary items to complete their meal — a missing drink with biryani, a dessert after dinner, or raita to go with a kebab platter. Our system uses **Meal DNA**, **Item2Vec embeddings**, and a **XGBoost + DCN-V2 ensemble** to intelligently suggest add-ons that feel natural, not pushy.

### The Numbers

| Metric | Value |
|:---|:---|
| 🎯 Model AUC | **0.611** |
| 📊 Score Separation | **0.035** (3× improvement over baseline) |
| 🏷️ HitRate@8 | **99.7%** |
| ⚡ Inference Latency | **~5ms** (target <300ms) |
| 📈 Projected AOV Lift | **+34.9%** |
| 🚀 Acceptance Lift | **4.27×** over random baseline |

---

## 🏗️ Architecture

```
┌──────────┐     ┌───────────────────────────────────────────┐     ┌──────────┐
│  User    │     │           Feature Engine (72-dim)         │     │  CSAO    │
│  Cart    │────▶│  Meal DNA · Item2Vec · User · Temporal    │────▶│  Rail    │
│  Context │     │  Cart Completion · Co-occurrence · Price   │     │  (Top 8) │
└──────────┘     └───────────────┬───────────────────────────┘     └──────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             ┌─────────────┐          ┌─────────────┐
             │   XGBoost   │          │   DCN-V2    │
             │  117 trees  │          │  Multi-task  │
             │  depth=8    │          │  Dual heads  │
             │  AUC: 0.611 │          │  AUC: 0.583 │
             └──────┬──────┘          └──────┬──────┘
                    │     Ensemble (α=0.15)  │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Post-Ranker (MMR)    │
                    │  Diversity · Explain    │
                    │  Thompson Sampling      │
                    └─────────────────────────┘
```

---

## ✨ Key Innovations

<table>
<tr>
<td width="50%">

### 🧬 Meal DNA
6-dimensional, **cuisine-aware meal completion** encoding. Knows that a South Indian breakfast needs sambar, while a Mughlai dinner needs naan. Provides explainable gap scores: *"Your cart is missing a drink."*

</td>
<td width="50%">

### 🔗 Item2Vec Embeddings
32-dim embeddings trained on **1.7M item co-occurrence pairs** via Skip-gram. Generalizes beyond direct co-occurrence — if Biryani→Raita and Pulao≈Biryani, then Pulao→Raita is inferred automatically.

</td>
</tr>
<tr>
<td width="50%">

### ⚔️ Hard Negative Mining
3-strategy sampling: **same-role** (another protein when user chose this one), **popularity-weighted** (popular items user skipped), **random**. Teaches fine-grained discrimination.

</td>
<td width="50%">

### 🎲 Thompson Sampling
Bayesian exploration bonus for **cold-start items**. New menu items get higher variance in scoring, ensuring fair exposure without sacrificing quality for established items.

</td>
</tr>
</table>

---

## 📁 Project Structure

```
csao-recommendation-zomato/
│
├── 📊 data/
│   ├── menu_catalog.py           # 660 items · 52 restaurants · 15 cuisines
│   ├── data_generator.py         # 10K users · 116K orders · hard negative mining
│   └── generated/                # Output CSVs
│
├── 🔧 features/
│   ├── meal_dna.py               # 6-dim cuisine × period meal completion encoder
│   ├── item2vec.py               # 32-dim item embeddings (Skip-gram, GPU)
│   └── feature_engineering.py    # 72-dim feature assembly + StandardScaler
│
├── 🧠 models/
│   ├── reranker.py               # DCN-V2 · cross network + deep MLP · dual heads
│   ├── ensemble.py               # XGBoost + DCN-V2 ensemble trainer
│   ├── post_ranker.py            # MMR diversification + Thompson Sampling
│   ├── train.py                  # GPU training · AMP · cosine warmup · early stop
│   └── checkpoints/              # Saved models, scaler, embeddings
│
├── 📈 evaluation/
│   ├── evaluator.py              # NDCG@K · MRR · HitRate · segment analysis
│   └── ab_testing.py             # A/B framework · guardrails · phased rollout
│
├── 🌐 serving/
│   ├── api.py                    # FastAPI · 4 endpoints · ~5ms latency
│   └── inference_pipeline.py     # End-to-end candidate → score → rank
│
├── 🎨 demo/
│   └── index.html                # Interactive 3-panel web demo
│
├── SUBMISSION.md                 # Detailed submission document
└── README.md                     # ← You are here
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended for training)
- 8GB+ VRAM (tested on RTX 4060)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/csao-recommendation-zomato.git
cd csao-recommendation-zomato

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn fastapi uvicorn jinja2 xgboost scipy
```

### Run the Full Pipeline

```bash
# Step 1: Generate synthetic data (10K users, 116K orders)
python -m data.data_generator

# Step 2: Train Item2Vec embeddings (660 items → 32-dim vectors)
python -m features.item2vec

# Step 3: Train DCN-V2 model (GPU-accelerated, ~2.5 min)
python -m models.train

# Step 4: Train XGBoost ensemble
python -m models.ensemble

# Step 5: Run evaluation
python -m evaluation.evaluator

# Step 6: Launch interactive demo
python -m serving.api
```

Then open **http://localhost:8000** to see the demo.

---

## 🎮 Interactive Demo

The demo features a three-panel layout:

| Panel | What it Shows |
|:---|:---|
| **🏪 Controls** | Select city, restaurant, browse menu, build your cart |
| **💡 Recommendations** | Top-8 CSAO suggestions with scores, prices, and explanations |
| **🧬 Meal DNA** | Real-time visualization of cart completeness and meal gaps |

---

## 📊 Model Evolution

We iterated through **3 versions**, each building on the last:

```
v1 (Baseline)              v2 (Improved)              v3 (Final)
─────────────              ─────────────              ──────────
DCN-V2 only                + XGBoost ensemble         + Item2Vec embeddings
51 features                + 17 enriched features     + 4 embedding features
Random negatives           + Hard negative mining      72 total features
LR=1e-3, batch=1024       + StandardScaler            α increased 0.05→0.15
                           + Cosine warmup
                           + Label smoothing

AUC: 0.596                 AUC: 0.612                 AUC: 0.611
Score Sep: 0.018            Score Sep: 0.035            Score Sep: 0.035
```

---

## 📋 Evaluation Results

### Overall Metrics

| Metric | Value | What it Means |
|:---|:---|:---|
| AUC | 0.611 | Model ranks relevant items above irrelevant 61% of the time |
| NDCG@8 | 0.597 | Relevant items appear near the top of recommendations |
| HitRate@8 | 99.7% | Almost every rail contains ≥1 relevant item |
| MRR | 0.622 | First relevant item appears at position ~1.6 on average |
| Score Separation | 0.035 | Clear gap between positive/negative predictions |

### Segment Performance

| Best Segments | AUC | Worst Segments | AUC |
|:---|:---|:---|:---|
| Rolls cuisine | 0.652 | Solo orders | 0.594 |
| Lucknow city | 0.629 | Snack period | 0.595 |
| Late-night period | 0.620 | Delhi city | 0.603 |
| Party orders | 0.619 | Breakfast period | 0.598 |

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| **Language** | Python 3.10 |
| **ML Framework** | PyTorch 2.x (CUDA), XGBoost 2.x (GPU) |
| **Feature Engineering** | NumPy, Pandas, scikit-learn |
| **Serving** | FastAPI + Uvicorn |
| **Embeddings** | Custom Item2Vec (Skip-gram + NS) |
| **Training** | Mixed Precision (AMP), Cosine Warmup, Early Stopping |
| **Hardware** | NVIDIA RTX 4060 (8GB VRAM) |

---

## 📖 Documentation

| Document | Contents |
|:---|:---|
| [`SUBMISSION.md`](SUBMISSION.md) | Full submission mapped to 6 evaluation criteria |
| [`evaluation/evaluator.py`](evaluation/evaluator.py) | Ranking metrics, segment analysis, business impact |
| [`evaluation/ab_testing.py`](evaluation/ab_testing.py) | A/B test framework, guardrails, rollout strategy |

---

<p align="center">
  <strong>Built for the Zomato CSAO Hackathon</strong><br/>
  <em>Increasing Average Order Value through intelligent, context-aware add-on recommendations</em>
</p>
