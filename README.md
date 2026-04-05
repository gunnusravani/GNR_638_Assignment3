# GNR 638 Assignment 3: SegNet Semantic Segmentation

Custom PyTorch implementation of **SegNet** (Badrinarayanan et al., 2015) for semantic segmentation on the **CamVid** dataset.

**Key Features:**
- Encoder-decoder architecture with 14.6M parameters
- Pooling-indices-based unpooling (no learned deconvolution)
- 12-class CamVid dataset support
- Test accuracy: **82.72%** | Mean IoU: **43.38%**

---

## Project Structure

```
GNR_638_Assignment3/
├── src/
│   ├── segnet_model.py         # SegNet architecture (encoder-decoder)
│   ├── dataset.py              # CamVid dataset loader (12 classes)
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation and metrics
│   └── utils.py                # Loss functions and utilities
│
├── models/
│   ├── custom_segnet/          # Trained model checkpoints
│   └── custom_segnet_v2/       # Retrained with 12-class fix
│
├── results/
│   ├── plots/                  # Training and evaluation visualizations
│   └── metrics/                # JSON evaluation results
│
├── report/
│   └── assignment_report_v2.tex # LaTeX report (7500+ words, 8 chapters)
│
├── data/
│   └── CamVid/                 # Dataset directory (downloaded separately)
│
├── requirements.txt            # Python dependencies
├── generate_plots.py           # Visualization script
└── README.md                   # This file
```

---

## Prerequisites

- **Python:** 3.10+
- **GPU:** CUDA 11.0+ (optional, CPU supported)
- **RAM:** 8GB+ recommended
- **Disk:** 2GB+ for dataset and models
- **OS:** macOS, Linux, or Windows

---

## Step 1: Clone and Setup Virtual Environment

```bash
# Navigate to project directory
cd /Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

---

## Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch 2.0+ (torch, torchvision, torchaudio)
- NumPy, Pillow, OpenCV
- Matplotlib for visualization

---

## Step 3: Download CamVid Dataset

The CamVid dataset must be downloaded manually. Follow these steps:

### Option A: Manual Download
1. Visit: [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoSegmentation/data.html)
2. Download the following files:
   - `701_StillsRaw_full.zip` (training images)
   - `LabeledApproved_full.zip` (training labels)
3. Place in `data/CamVid/` directory with this structure:

```
data/CamVid/
├── train/
│   ├── [image files] *.png
│   └── [label files] labelids/*.png
├── val/
│   ├── [image files] *.png
│   └── [label files] labelids/*.png
└── test/
    ├── [image files] *.png
    └── [label files] labelids/*.png
```

### Option B: Using Script (if available)
```bash
# Note: Manual download and extraction is recommended for CamVid
# due to licensing and data availability policies
```

**Verify the dataset:**
```bash
# Check if dataset exists
ls -la data/CamVid/train/ | head -5
```

---

## Step 4: Training the Model

Run the training script with the following command:

```bash
python src/train.py \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 0.01 \
    --checkpoint_dir models/custom_segnet_v2/ \
    --device cuda
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Initial learning rate (default: 0.01)
- `--checkpoint_dir`: Directory to save model checkpoints
- `--device`: `cuda` for GPU or `cpu` for CPU training
- `--num_classes`: Number of segmentation classes (default: 12 for CamVid)

**Training Output:**
- Best model saved to: `models/custom_segnet_v2/best_model.pth`
- Training history saved to: `training_history.json`
- Console logs showing epoch progress, loss, and validation metrics

**Expected Training Time:**
- GPU (CUDA): ~30-45 minutes for 100 epochs
- CPU: ~3-4 hours for 100 epochs

---

## Step 5: Evaluate on Test Set

After training completes, evaluate the model on the test set:

```bash
python src/evaluate.py \
    --model_path models/custom_segnet_v2/best_model.pth \
    --split test \
    --device cuda
```

**Evaluation Parameters:**
- `--model_path`: Path to trained model checkpoint
- `--split`: Dataset split (`test`, `val`, or `train`)
- `--device`: `cuda` or `cpu`
- `--num_classes`: Number of classes (default: 12)

**Evaluation Output:**
- Metrics saved to: `evaluation_results.json`
- Console output showing:
  - Global Accuracy
  - Mean Intersection over Union (mIoU)
  - Per-class accuracy and IoU
  - Frequency-weighted IoU

**Expected Test Metrics:**
- Global Accuracy: ~82% - 85%
- Mean IoU: ~40% - 45%
- Class Avg Accuracy: ~50% - 55%

---

## Step 6: Generate Visualizations

Create training and evaluation plots:

```bash
python generate_plots.py
```

**Output plots (saved to `results/plots/`):**
1. `training_curves.png` - Loss, accuracy, and mIoU progression
2. `per_class_metrics.png` - Per-class IoU and accuracy
3. `test_results_comparison.png` - Custom implementation metrics
4. `class_analysis.png` - Class frequency vs performance analysis

---

## Quick Start (Complete Pipeline)

Run the entire pipeline in sequence:

```bash
# 1. Activate environment (if not already active)
source .venv/bin/activate

# 2. Train model (skip if already trained)
python src/train.py --epochs 100 --batch_size 4 --learning_rate 0.01 --checkpoint_dir models/custom_segnet_v2/ --device cuda

# 3. Evaluate on test set
python src/evaluate.py --model_path models/custom_segnet_v2/best_model.pth --split test --device cuda

# 4. Generate visualizations
python generate_plots.py

# 5. View the report
# Open report/assignment_report_v2.tex in LaTeX editor (or PDF if compiled)
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size: `--batch_size 2`
- Use CPU: `--device cpu`
- Reduce image resolution in `src/dataset.py`

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue: "CamVid dataset not found"
**Solution:**
- Ensure dataset is in `data/CamVid/` directory
- Check directory structure matches the format above
- Run: `ls -la data/CamVid/train/ | head` to verify

### Issue: "Model checkpoint not found"
**Solution:**
- Ensure training completed successfully
- Check checkpoint path: `ls -la models/custom_segnet_v2/`

### Issue: "Class index 11 exceeding num_classes=11"
**Solution:**
- This fix is already applied in current code
- Ensure you're using the latest version from this repository

---

## Model Architecture

**SegNet** is an encoder-decoder semantic segmentation architecture:

### Encoder (VGG-16 based)
- 5 blocks with 2 convolutional layers each
- Pooling operations store **max-pool indices**
- Progressively downsamples from 360×480 to 11×15 spatial dimensions

### Decoder
- 5 blocks mirroring encoder structure
- Uses stored **pooling indices** for upsampling (no learned parameters)
- Preserves fine object boundaries

### Key Innovation
**Pooling-indices-based unpooling:**
- Eliminates learned deconvolution parameters
- Maintains 14.6M total model parameters (vs 136M for FCN)
- Empirically achieves boundary F-measure: 81.4%

**Model Parameters:** 14,615,553 (≈14.6M)

---

## Dataset: CamVid

**Cambridge-driving Labeled Video Database**

### Statistics
- **Dataset Size:** 701 training + 101 validation + 233 test images
- **Image Resolution:** 360 × 480 pixels each
- **Number of Classes:** 12 semantic classes (indices 0-11)
- **Class Distribution:** Highly imbalanced (buildings/sky dominate; bicyclists are rare)

### 12 Semantic Classes
| Index | Class | Notes |
|-------|-------|-------|
| 0 | Road | Most frequent |
| 1 | Sidewalk | - |
| 2 | Tree | - |
| 3 | Car | Vehicle class |
| 4 | Fence | - |
| 5 | Pedestrian | Minority class |
| 6 | Building | Dominant class |
| 7 | Pole | - |
| 8 | Sky | Dominant class |
| 9 | Bicycle | Minority class |
| 10 | Sign | - |
| 11 | Void | Unlabeled/invalid pixels |

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | SGD | With momentum 0.9 |
| Learning Rate | 0.01 | Fixed (or adapt with schedule) |
| Batch Size | 4 | Limited by GPU memory |
| Max Epochs | 100 | With early stopping (~70 epochs typically) |
| Loss Function | Weighted Cross-Entropy | Median frequency balancing |
| Image Normalization | ImageNet stats | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Weight Decay | 0.0 | Not used |
| Momentum | 0.9 | SGD momentum |

---

## Evaluation Metrics

### Global Accuracy
Percentage of correctly classified pixels across entire test set.
```
Global Accuracy = (Correct Pixels) / (Total Pixels) × 100%
```

### Mean Intersection over Union (mIoU)
Average IoU across all 12 classes. Primary metric for segmentation.
```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = (1/K) × Σ IoU_c
```

### Per-Class Metrics
- **Class Accuracy:** Recall for each individual class
- **Class IoU:** Precision and recall balance per class
- **Frequency-Weighted IoU:** Weighted by class prevalence in test set

---

## Critical Bug Fixes Applied

### Issue 1: Class Index Mismatch (Train/Test)
**Problem:** CamVid labels contain 12 classes (indices 0-11), but model was configured for 11 classes (indices 0-10).
- Caused CUDA runtime error: "device-side assert triggered"
- Resulted in 27.59% test accuracy vs 91.11% validation accuracy

**Solution Applied:**
- ✅ Updated `src/dataset.py`: CLASS_NAMES list expanded to 12 items
- ✅ Updated `src/train.py`: `num_classes` default changed from 11 to 12
- ✅ Updated `src/evaluate.py`: Added `--num_classes` parameter (default: 12)

### Issue 2: Missing Image Normalization in Evaluation
**Problem:** Training applied ImageNet normalization, but evaluation didn't.
- Caused artificially low test metrics
- Explanation: Model learned features based on normalized images

**Solution Applied:**
- ✅ Added ImageNet normalization to evaluation pipeline in `src/evaluate.py`
- ✅ Applied same preprocessing as training: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- ✅ Test metrics normalized after fix

---

## Test Results (Final)

After both bug fixes applied:

| Metric | Value | Notes |
|--------|-------|-------|
| **Global Accuracy** | 82.72% | Percentage of correctly classified pixels |
| **Mean IoU (mIoU)** | 43.38% | Primary segmentation metric |
| **Class Avg Accuracy** | 52.65% | Average recall across 12 classes |
| **Inference Time** | ~60ms | Per image (360×480) on GPU |
| **Model Size** | 14.6M params | Lightweight architecture |

### Per-Class Breakdown
- **High Performance:** Road, sky, building (>80% IoU)
- **Medium Performance:** Sidewalk, fence, pole, sign (30-70% IoU)
- **Low Performance:** Bicycle, pedestrian (<20% IoU) - due to dataset imbalance

---

## File Reference

### Source Code Files

**`src/segnet_model.py`**
- `SegNetEncoder` class: 5-block VGG-based encoder with pooling indices
- `SegNetDecoder` class: Mirrored decoder using stored pooling indices
- `SegNet` class: Complete encoder-decoder model
- **Key Method:** `SegNetDecoder.forward()` - uses `F.max_unpool2d()` with stored indices

**`src/dataset.py`**
- `toyDataset` class: PyTorch Dataset for CamVid
- **Updated:** CLASS_NAMES now list 12 items (added "void")
- Converts RGB label images to class index tensors
- Applies ImageNet normalization and resizing

**`src/train.py`**
- `SegNetTrainer` class: Encapsulates training loop
- **Fixed:** `--num_classes` default changed from 11 to 12
- Weighted cross-entropy loss with median frequency balancing
- SGD optimizer with momentum and optional learning rate decay
- Checkpointing and early stopping logic

**`src/evaluate.py`**
- Inference script for test set evaluation
- **Fixed:** Applied ImageNet normalization to input images
- **Added:** `--num_classes` argument for flexibility
- Computes: accuracy, IoU, per-class metrics
- Outputs: JSON metrics file

**`generate_plots.py`**
- Reads `training_history.json` and `evaluation_results.json`
- Generates 4 visualization plots
- Saves to `results/plots/` directory

### Report

**`report/assignment_report_v2.tex`**
- LaTeX report suitable for academic submission
- 8 chapters + bibliography
- Includes code snippets, architecture tables, 4 integrated figures
- Test results: 82.72% accuracy, 43.38% mIoU
- Professional formatting with color highlights

---

## Citation

If you use this code in your research, please cite the original SegNet paper:

```bibtex
@article{badrinarayanan2015segnet,
  title={SegNet: A deep convolutional encoder-decoder architecture for image segmentation},
  author={Badrinarayanan, Vijay and Kendall, Alex and Cipolla, Roberto},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2015},
  volume={39},
  number={12},
  pages={2481--2495}
}
```

---

## FAQ

**Q: Can I train on CPU?**  
A: Yes, use `--device cpu`. Training will be ~5-10x slower.

**Q: How long does training take?**  
A: ~40 minutes on GPU (NVIDIA RTX), ~4 hours on CPU.

**Q: Why is test accuracy lower than validation accuracy?**  
A: Test set may have different distribution or use different labels. This is expected behavior.

**Q: What batch size should I use?**  
A: Batch size 4 is recommended. Use smaller (2, 1) on GPUs with limited memory.

**Q: How do I use a pre-trained model?**  
A: Load checkpoint: `torch.load('models/custom_segnet_v2/best_model.pth')`

---

## License & Disclaimer

This is an educational implementation for GNR 638 Assignment 3. The CamVid dataset has its own licensing terms.

**Disclaimer:** Results may vary depending on:
- GPU/CPU hardware differences
- Random seed initialization
- PyTorch version differences
- Preprocessing variations

---

## Support

For questions or issues:
1. Check the README sections above
2. Review code comments in `src/` directory
3. Consult `report/assignment_report_v2.pdf` for detailed technical discussion