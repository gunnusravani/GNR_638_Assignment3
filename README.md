# SegNet: Semantic Segmentation Implementation

This project implements SegNet (Badrinarayanan et al., 2016) from scratch and compares it against official implementations for GNR_638 Assignment 3.

## Project Structure

```
GNR_638_Assignment3/
├── data/                          # Dataset directory
│   ├── CamVid/                    # CamVid dataset (to be downloaded)
│   │   ├── train/                 # Training images and labels
│   │   ├── val/                   # Validation images and labels
│   │   └── test/                  # Test images and labels
│   └── processed/                 # Preprocessed data cache
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── segnet_model.py            # Custom SegNet implementation
│   ├── dataset.py                 # Dataset loading and preprocessing
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation and metrics
│   └── utils.py                   # Utility functions
│
├── models/                        # Saved model checkpoints
│   ├── custom_segnet/             # Custom implementation checkpoints
│   └── official_segnet/           # Official implementation checkpoints
│
├── results/                       # Results output
│   ├── predictions/               # Segmentation predictions (images)
│   ├── metrics/                   # Metrics (JSON/CSV)
│   └── visualizations/            # Comparison visualizations
│
├── notebook/                      # Jupyter notebooks
│   └── comparison_analysis.ipynb  # Analysis and comparison
│
├── report/                        # Assignment report
│   └── assignment_report.md       # Detailed report
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- CUDA 11.0+ (recommended for GPU training)
- 8GB+ RAM
- ~2GB disk space for dataset

### 2. Clone and Navigate to Repository

```bash
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3
```

### 3. Create Virtual Environment

```bash
# Using Python venv
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download and Prepare Dataset

```bash
# From the repository root, run:
python src/setup_dataset.py

# This will:
# - Download CamVid dataset (if not exists)
# - Extract images and labels
# - Create train/val/test splits
# - Store paths in a manifest file
```

Alternative: Download manually from [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoSegmentation/data.html)

---

## Running the Project

### 1. Train Custom SegNet Implementation

```bash
python src/train.py \
    --model custom \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 0.01 \
    --dataset_path data/CamVid/ \
    --checkpoint_dir models/custom_segnet/ \
    --results_dir results/ \
    --device cuda  # or 'cpu'
```

**Arguments:**
- `--model`: 'custom' or 'official'
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Initial learning rate (default: 0.01)
- `--momentum`: SGD momentum (default: 0.9)
- `--dataset_path`: Path to CamVid dataset
- `--checkpoint_dir`: Where to save model checkpoints
- `--results_dir`: Where to save results
- `--device`: 'cuda' or 'cpu'
- `--num_classes`: Number of segmentation classes (default: 11 for CamVid)
- `--early_stopping_patience`: Epochs without improvement before stopping (default: 10)

**Output:**
- Checkpoint saved to `models/custom_segnet/best_model.pth`
- Training logs to terminal/tensorboard
- Validation metrics every epoch

### 2. Evaluate on Test Set

```bash
python src/evaluate.py \
    --model_path models/custom_segnet/best_model.pth \
    --dataset_path data/CamVid/ \
    --split test \
    --output_dir results/predictions/ \
    --metrics_output results/metrics/custom_metrics.json \
    --visualize \
    --device cuda
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--dataset_path`: Path to test dataset
- `--split`: 'train', 'val', or 'test'
- `--output_dir`: Where to save predictions
- `--metrics_output`: JSON file for metrics
- `--visualize`: Save visualizations (True/False)
- `--device`: 'cuda' or 'cpu'

**Output:**
- Segmentation masks in `results/predictions/`
- Metrics (mIoU, accuracy, boundary F1) in JSON format
- Visualized predictions (overlay with ground truth)

### 3. Train Official SegNet Implementation

```bash
python src/train.py \
    --model official \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 0.01 \
    --dataset_path data/CamVid/ \
    --checkpoint_dir models/official_segnet/ \
    --results_dir results/ \
    --device cuda
```

### 4. Compare Custom vs Official

```bash
python src/compare.py \
    --custom_model_path models/custom_segnet/best_model.pth \
    --official_model_path models/official_segnet/best_model.pth \
    --dataset_path data/CamVid/ \
    --output_dir results/comparison/ \
    --metrics_output results/comparison_metrics.json \
    --device cuda
```

**Output:**
- Side-by-side predictions comparison
- Performance metrics table (accuracy, mIoU, BF score, inference time)
- Memory usage comparison
- Visualization plots

### 5. Run Jupyter Notebook for Analysis

```bash
jupyter notebook notebook/comparison_analysis.ipynb
```

This notebook includes:
- Data exploration
- Model architecture visualization
- Training curves and loss plots
- Detailed results comparison
- Error analysis

---

## Expected Results

### Custom SegNet on CamVid (11 classes):
- **mIoU**: ~57-60%
- **Global Accuracy**: ~89-90%
- **Inference Time**: ~50-100ms per 512×512 image
- **GPU Memory**: ~1GB

### Comparison Metrics:
Both implementations should achieve similar performance when:
- Using identical hyperparameters
- Training on same dataset
- Using same preprocessing

Differences may arise from:
- PyTorch vs original Caffe implementation
- Batch normalization variants
- Random seed initialization

---

## File Descriptions

### Core Implementation Files

**`src/segnet_model.py`**
- `SegNetEncoder`: VGG16-based encoder with pooling indices
- `SegNetDecoder`: Decoder with index-based upsampling
- `SegNet`: Full architecture combining encoder/decoder

**`src/dataset.py`**
- `CamVidDataset`: PyTorch dataset class for CamVid
- Data augmentation: rotations, flips, elastic deformations
- Preprocessing: normalization, resizing

**`src/train.py`**
- Training loop with validation
- Loss function: Cross-entropy with class balancing
- Optimizer: SGD with momentum
- Checkpointing and early stopping

**`src/evaluate.py`**
- Evaluation on test set
- Metrics calculation: mIoU, accuracy, boundary F1
- Visualization of predictions
- JSON output for comparison

**`src/utils.py`**
- Metric computation functions
- Visualization helpers
- Class frequency computation
- Weight map generation

**`src/compare.py`**
- Side-by-side comparison
- Performance benchmarking
- Statistical analysis

---

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python src/train.py --batch_size 2

# Or use CPU (slower)
python src/train.py --device cpu
```

### CUDA Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Fall back to CPU
python src/train.py --device cpu
```

### Dataset Not Found
```bash
# Manually download from CamVid website and extract to data/CamVid/
# Ensure folder structure:
# data/CamVid/train_images, train_labels, test_images, test_labels, val_images, val_labels
```

### Model Not Converging
- Reduce learning rate: `--learning_rate 0.001`
- Increase batch size: `--batch_size 8`
- Add more data augmentation
- Check class weights are balanced

---

## Performance Optimization

### For Faster Training:
```bash
python src/train.py --batch_size 8 --num_workers 4 --device cuda
```

### For Better Accuracy:
```bash
python src/train.py \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --weight_decay 1e-4
```

### For Memory Efficiency:
```bash
python src/train.py --batch_size 2 --device cuda
```

---

## Assignment Submission

1. **Code**: Push to GitHub repository (ensure TA is added as collaborator)
2. **Report**: PDF in `report/assignment_report.md`
3. **Models**: Save best checkpoints in `models/`
4. **Results**: Include metrics JSON and visualizations
5. **Notebook**: Complete Jupyter notebook with analysis
6. **Form**: Submit via Google Form with blog and report links

---

## References

- SegNet Paper: [arXiv:1511.00561](https://arxiv.org/abs/1511.00561)
- CamVid Dataset: [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoSegmentation/data.html)
- PyTorch Documentation: [Official Docs](https://pytorch.org/docs/)

---

## Author & Date

- Assignment: GNR_638 (Deep Learning)
- Deadline: April 4, 2026 05:30 IST
- Implementation Date: April 2026