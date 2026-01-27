# Google Colab Quick Start Guide

## What You Need to Upload

You need **3 things** in Google Colab:

### 1. Your Project Code (as a ZIP file)

**Create the ZIP:**
```bash
cd /Users/manupareek
zip -r diffusion-model.zip diffusionmodel/ \
  -x "diffusionmodel/.git/*" \
  -x "diffusionmodel/**/__pycache__/*" \
  -x "diffusionmodel/**/*.pyc" \
  -x "diffusionmodel/.streamlit/*"
```

This creates `diffusion-model.zip` in `/Users/manupareek/`

---

### 2. Your Trained Model (optional if training in Colab)

**If you already trained a model locally:**
1. Go to Model tab in Streamlit
2. Click "💾 Download Model for Colab" 
3. Save `model_export.pt`

**If you haven't trained yet:**
- Skip this, you'll train in Colab

---

### 3. Your Market Data CSV

The CSV file you loaded in the Data tab. It's in:
```
/Users/manupareek/diffusionmodel/data/
```

For example:
- `SPY_5min.csv`
- `AAPL_5min.csv`
- Or whatever you uploaded

---

## Steps in Google Colab

### Step 1: Open Colab
Go to: [colab.research.google.com](https://colab.research.google.com)

### Step 2: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to: **T4 GPU**
3. Click **Save**

### Step 3: Upload Your Notebook
1. Click **File** → **Upload notebook**
2. Upload: `/Users/manupareek/diffusionmodel/colab_backtest.ipynb`

### Step 4: Run Cells
Execute cells **in order**:

**Cell 1 - Upload Project:**
```python
from google.colab import files
uploaded = files.upload()  # Upload diffusion-model.zip
!unzip -q diffusion-model.zip
%cd diffusionmodel
```

**Cell 2 - Install Dependencies:**
```python
!pip install -q streamlit plotly pandas numpy torch
```

**Cell 3 - Verify GPU:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
```

**Cell 4 - Upload Data:**
```python
uploaded_data = files.upload()  # Upload your CSV
data_file = list(uploaded_data.keys())[0]
```

**Cell 5 - (Optional) Upload Trained Model OR Train New:**
```python
# Option A: Upload pre-trained model
uploaded_model = files.upload()  # Upload model_export.pt

# Option B: Train new model (will take ~5-10 min on GPU)
# ... (notebook has training code)
```

**Cell 6 - Run Backtest:**
```python
# This is in the notebook - just click Run!
```

---

## What You'll See

1. **Upload prompts** for each file
2. **Progress bars** showing backtest progress
3. **Results**:
   - Total Return: 15.2%
   - Sharpe Ratio: 1.8
   - Max Drawdown: -8.3%
   - Win Rate: 62%
4. **Charts** (equity curve, drawdown)
5. **Download buttons** for results CSVs

---

## Expected Timeline

- **Upload files**: 1-2 minutes
- **Install packages**: 1-2 minutes  
- **Training** (if needed): 5-10 minutes on GPU
- **Backtest** (100 bars, 100 samples): **2-3 minutes on GPU** ⚡

Compare to: 40+ minutes on your Mac CPU!

---

## If You Get Errors

**"No module named 'src'":**
- Make sure you ran the `unzip` and `%cd diffusionmodel` commands

**"GPU not available":**
- Runtime → Change runtime type → GPU → Save
- Runtime → Restart runtime

**"File not found":**
- Re-upload the missing file when prompted

---

## Quick Command Reference

**Package your project:**
```bash
cd /Users/manupareek
zip -r diffusion-model.zip diffusionmodel/ -x "*.pyc" "**/__pycache__/*"
```

**Files to upload in Colab:**
1. `diffusion-model.zip` (your code)
2. `model_export.pt` (optional, your trained model)
3. `SPY_5min.csv` (or your data file)

That's it! 🚀
