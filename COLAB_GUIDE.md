# Google Colab Deployment Guide

## Quick Start

Your Mac doesn't have GPU (MPS), so backtests are **very slow** on CPU. Use Google Colab's free GPU instead!

---

## Step 1: Train Model Locally

1. Open your Streamlit dashboard (already running)
2. Go to **Model** tab
3. Configure and train your model
4. Once training completes, click **"💾 Download Model for Colab"**
5. Save `model_export.pt` to your Downloads folder

---

## Step 2: Prepare Data

Export your market data CSV:
1. Go to **Data** tab in dashboard
2. Note which CSV file you loaded
3. Copy it from `data/` folder to your Downloads

---

## Step 3: Upload to Google Colab

1. **Open Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Upload notebook**: 
   - Click "Upload" tab
   - Upload [`colab_backtest.ipynb`](file:///Users/manupareek/diffusionmodel/colab_backtest.ipynb)
3. **Enable GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU (T4)**
   - Click Save

---

## Step 4: Upload Project Files

**Option A - GitHub (Recommended)**:
```bash
# On your Mac, create git repo
cd /Users/manupareek/diffusionmodel
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
# Then in Colab, clone it
```

**Option B - Manual Upload**:
1. Zip your project:
   ```bash
   cd /Users/manupareek
   zip -r diffusion-trading.zip diffusionmodel/ -x "*.pyc" "**/__pycache__/*" ".git/*"
   ```
2. In Colab notebook, upload the zip file when prompted

---

## Step 5: Run Backtest

1. Upload `model_export.pt` when prompted
2. Upload your market data CSV
3. Run all cells sequentially
4. **Progress will show**: "Initialized on CUDA" (GPU confirmed!)
5. Backtest should complete in **2-5 minutes** (vs 40+ minutes on CPU)

---

## Expected GPU Performance

| Bars | Samples | CPU Time | GPU Time  |
|------|---------|----------|-----------|
| 50   | 10      | ~4 min   | ~20 sec   |
| 50   | 100     | ~40 min  | ~2 min    |
| 500  | 100     | ~6 hours | ~15 min   |

---

## Downloading Results

The notebook will generate:
- `backtest_equity.csv` - Portfolio value over time
- `backtest_trades.csv` - All trades executed
- Interactive Plotly charts (viewable in Colab)

Click the download buttons to save results to your Mac.

---

## Troubleshooting

**"No GPU available"**:
- Check Runtime → Change runtime type → GPU
- Restart runtime

**"Module not found"**:
- Make sure you uploaded the full project
- Run the `!pip install` cell

**"Model load error"**:
- Verify `model_export.pt` was uploaded
- Check PyTorch versions match (should auto-handle)
