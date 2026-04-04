# Command Reference - Copy & Paste Ready

## Environment Setup

```bash
# Activate virtual environment (ALWAYS do this first)
source /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3/.venv/bin/activate

# Verify Python is correct
which python
python --version
```

## One-Line Setup (Run Once)

```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt
```

## Download Dataset

```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3
python src/setup_dataset.py
```

---

## Training Commands

### Train Custom SegNet (CPU - Safe but slow)
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python src/train.py \
  --model custom \
  --epochs 50 \
  --batch_size 2 \
  --learning_rate 0.01 \
  --momentum 0.9 \
  --dataset_path data/CamVid/ \
  --checkpoint_dir models/custom_segnet/ \
  --results_dir results/ \
  --device cpu \
  --num_classes 11 \
  --early_stopping_patience 10
```

### Train Custom SegNet (GPU - Faster)
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python src/train.py \
  --model custom \
  --epochs 100 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --momentum 0.9 \
  --dataset_path data/CamVid/ \
  --checkpoint_dir models/custom_segnet/ \
  --results_dir results/ \
  --device cuda \
  --num_classes 11 \
  --early_stopping_patience 10
```

### Train Official SegNet (for comparison)
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python src/train.py \
  --model official \
  --epochs 100 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --momentum 0.9 \
  --dataset_path data/CamVid/ \
  --checkpoint_dir models/official_segnet/ \
  --results_dir results/ \
  --device cuda \
  --num_classes 11 \
  --early_stopping_patience 10
```

---

## Evaluation Commands

### Evaluate Custom Model on Test Set
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python src/evaluate.py \
  --model_path models/custom_segnet/best_model.pth \
  --dataset_path data/CamVid/ \
  --split test \
  --output_dir results/predictions/custom/ \
  --metrics_output results/metrics/custom_metrics.json \
  --visualize true \
  --device cuda
```

### Evaluate Official Model on Test Set
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python src/evaluate.py \
  --model_path models/official_segnet/best_model.pth \
  --dataset_path data/CamVid/ \
  --split test \
  --output_dir results/predictions/official/ \
  --metrics_output results/metrics/official_metrics.json \
  --visualize true \
  --device cuda
```

---

## Comparison Commands

### Compare Models
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python src/compare.py \
  --custom_model_path models/custom_segnet/best_model.pth \
  --official_model_path models/official_segnet/best_model.pth \
  --dataset_path data/CamVid/ \
  --output_dir results/comparison/ \
  --metrics_output results/comparison_metrics.json \
  --device cuda
```

---

## Jupyter Notebook

```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
jupyter notebook notebook/comparison_analysis.ipynb
```

---

## Check Results

### List all results
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
find results/ -type f | head -20
```

### View metrics
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
cat results/metrics/custom_metrics.json | json_pp
```

### Count predictions generated
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
ls -la results/predictions/custom/ | wc -l
```

---

## Monitoring (While Training)

### View real-time loss (in another terminal)
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
tail -f training.log
```

### Monitor GPU usage (in another terminal)
```bash
watch -n 1 nvidia-smi
```

### Check disk space
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
du -sh *
```

---

## Cleanup Commands

```bash
# Remove old results (be careful!)
rm -rf results/predictions/*
rm -rf results/metrics/*

# Remove old models (be careful!)
rm -rf models/custom_segnet/*

# Clean cache
rm -rf data/processed/*
```

---

## Debugging Commands

### Test imports
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

### Test dataset loading
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python -c "from src.dataset import CamVidDataset; ds = CamVidDataset('data/CamVid/', split='train'); print(f'Dataset size: {len(ds)}')"
```

### Test model creation
```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3 && \
python -c "from src.segnet_model import SegNet; model = SegNet(num_classes=11); print(model)"
```

---

## Running the Complete Pipeline (Copy & Paste One Go)

```bash
#!/bin/bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3

# Setup
source .venv/bin/activate

# Download dataset (if needed)
# python src/setup_dataset.py

# Train custom
echo "Training custom SegNet..."
python src/train.py --model custom --epochs 100 --batch_size 4 --device cuda --dataset_path data/CamVid/ --checkpoint_dir models/custom_segnet/

# Train official
echo "Training official SegNet..."
python src/train.py --model official --epochs 100 --batch_size 4 --device cuda --dataset_path data/CamVid/ --checkpoint_dir models/official_segnet/

# Evaluate both
echo "Evaluating custom model..."
python src/evaluate.py --model_path models/custom_segnet/best_model.pth --dataset_path data/CamVid/ --split test --device cuda --output_dir results/predictions/custom/ --metrics_output results/metrics/custom_metrics.json

echo "Evaluating official model..."
python src/evaluate.py --model_path models/official_segnet/best_model.pth --dataset_path data/CamVid/ --split test --device cuda --output_dir results/predictions/official/ --metrics_output results/metrics/official_metrics.json

# Compare
echo "Comparing models..."
python src/compare.py --custom_model_path models/custom_segnet/best_model.pth --official_model_path models/official_segnet/best_model.pth --dataset_path data/CamVid/ --device cuda --output_dir results/comparison/ --metrics_output results/comparison_metrics.json

echo "Complete! Check results/ folder"
```

---

## Check Your Environment

```bash
# Run this to verify everything is set up correctly
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3

echo "Python version:"
python --version

echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "CUDA available:"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo "Project structure:"
ls -la src/
ls -la data/
ls -la models/
ls -la results/
```

