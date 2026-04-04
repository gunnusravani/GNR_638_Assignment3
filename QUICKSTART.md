# SegNet Implementation - Quick Start Guide

## 1️⃣ Initial Setup (Do This Once)

```bash
# Navigate to project directory
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2️⃣ Prepare Dataset

```bash
# Download CamVid dataset (~500MB)
# Option A: Automatic download (if available)
python src/setup_dataset.py

# Option B: Manual download
# - Visit: http://mi.eng.cam.ac.uk/research/projects/VideoSegmentation/data.html
# - Download and extract to: data/CamVid/
```

## 3️⃣ Train Custom SegNet Implementation

```bash
# Basic training (CPU - slower, but no CUDA required)
python src/train.py \
    --model custom \
    --epochs 50 \
    --batch_size 2 \
    --dataset_path data/CamVid/ \
    --checkpoint_dir models/custom_segnet/ \
    --device cpu

# Fast training (GPU - requires CUDA)
python src/train.py \
    --model custom \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 0.01 \
    --dataset_path data/CamVid/ \
    --checkpoint_dir models/custom_segnet/ \
    --device cuda
```

**Expected training time:**
- GPU (CUDA): ~30-60 min for 100 epochs
- CPU: ~2-3 hours for 100 epochs

## 4️⃣ Evaluate Custom Model

```bash
# Evaluate on test set
python src/evaluate.py \
    --model_path models/custom_segnet/best_model.pth \
    --dataset_path data/CamVid/ \
    --split test \
    --output_dir results/predictions/ \
    --metrics_output results/metrics/custom_metrics.json \
    --visualize \
    --device cuda
```

## 5️⃣ Train Official SegNet (for comparison)

```bash
python src/train.py \
    --model official \
    --epochs 100 \
    --batch_size 4 \
    --dataset_path data/CamVid/ \
    --checkpoint_dir models/official_segnet/ \
    --device cuda
```

## 6️⃣ Compare Custom vs Official

```bash
python src/compare.py \
    --custom_model_path models/custom_segnet/best_model.pth \
    --official_model_path models/official_segnet/best_model.pth \
    --dataset_path data/CamVid/ \
    --output_dir results/comparison/ \
    --metrics_output results/comparison_metrics.json \
    --device cuda
```

## 7️⃣ View Analysis in Jupyter

```bash
# Start Jupyter
jupyter notebook

# Open: notebook/comparison_analysis.ipynb
# Run all cells to see:
# - Training curves
# - Comparison metrics
# - Visualizations
```

## 8️⃣ Check Results

All results will be saved to:
- **Metrics**: `results/metrics/` (JSON files)
- **Predictions**: `results/predictions/` (PNG images)
- **Comparisons**: `results/comparison/` (comparison plots)
- **Models**: `models/custom_segnet/` and `models/official_segnet/`

---

## File Structure After Running

```
results/
├── metrics/
│   ├── custom_metrics.json          ← Custom model performance
│   ├── official_metrics.json        ← Official model performance
│   └── comparison_metrics.json      ← Side-by-side comparison
├── predictions/
│   ├── custom/
│   │   ├── image_1_prediction.png
│   │   ├── image_1_ground_truth.png
│   │   └── ...
│   └── official/
│       ├── image_1_prediction.png
│       ├── image_1_ground_truth.png
│       └── ...
└── comparison/
    ├── comparison_metrics.png    ← Bar chart of metrics
    ├── side_by_side_samples.png  ← Visual comparison
    └── inference_time.png        ← Speed comparison

models/
├── custom_segnet/
│   ├── best_model.pth          ← Best checkpoint
│   └── latest_model.pth        ← Latest checkpoint
└── official_segnet/
    ├── best_model.pth
    └── latest_model.pth
```

---

## Expected Metrics (CamVid 11-class)

**Custom SegNet:**
- Global Accuracy: ~89-91%
- Class Average Accuracy: ~65-72%
- mIoU: ~57-62%
- Boundary F1: ~40-45%
- Inference Time: 50-150ms/image

**Official SegNet:**
- Should be similar or slightly better
- Any differences due to framework/batch norm variants

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce batch_size to 2 or use CPU |
| CUDA not found | Use `--device cpu` or reinstall CUDA |
| Dataset not found | Run `python src/setup_dataset.py` |
| ModuleNotFoundError | Run `pip install -r requirements.txt` again |
| Slow training | Use GPU instead of CPU |

---

## Timeline to Completion

- Setup: 5-10 min
- Dataset prep: 10-20 min
- Train custom model: 30-120 min (depending on hardware)
- Evaluate: 5-10 min
- Train official: 30-120 min
- Comparison: 5 min
- Analysis: 10-15 min
- **Total: 1-4 hours** (depending on device)

---

## Next Steps After Completion

1. Review all metrics and visualizations in `results/`
2. Run Jupyter notebook for detailed analysis
3. Write assignment report based on findings
4. Push code to GitHub with TA as collaborator
5. Submit report and metrics via Google Form

