# SegNet Implementation - Testing & Verification Guide

This guide helps you verify that the SegNet implementation works correctly before full training.

## ✅ Quick Verification (5 minutes)

### 1. Test Model Creation and Forward Pass

```bash
# Run the test suite
python test_segnet.py
```

**Expected output:**
```
✅ All imports successful
✅ Model created successfully
   Total parameters: 29,462,027
✅ Forward pass successful
✅ Dataset and DataLoader working
✅ Metrics computed
✅ Loss computed
✅ Training step successful

All tests passed! SegNet is ready for training.
```

### 2. Run Tutorial

```bash
# Run the complete tutorial (shows how everything works)
python tutorial.py
```

**What it covers:**
- Model architecture overview
- Forward pass and outputs
- Dataset and DataLoader
- Loss function and metrics
- Single training step
- Full training loop example

---

## 🚀 First Training Run (Quick Test)

### Option 1: Use Toy Dataset (5-10 minutes)

```bash
# Train for just 5 epochs with toy dataset (no real data needed)
python src/train.py \
    --use_toy_dataset \
    --epochs 5 \
    --batch_size 2 \
    --device cpu
```

**Output:**
- Checkpoints: `models/custom_segnet/`
- History: `results/training_history.json`
- Logs: `logs/segnet_training/`

### Option 2: Full Training (30 minutes - 2 hours)

```bash
# Train with more epochs
python src/train.py \
    --use_toy_dataset \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.01 \
    --device cuda
```

---

## 📊 Training with Real CamVid Dataset

### Step 1: Prepare Dataset

```bash
# Create dummy CamVid dataset for testing (no download needed)
python src/setup_dataset.py --dummy --num_train 10 --num_val 5 --num_test 5
```

This creates synthetic images and labels in `data/CamVid/` for quick testing.

### Step 2: Train on CamVid

```bash
python src/train.py \
    --dataset_path data/CamVid/ \
    --epochs 50 \
    --batch_size 4 \
    --device cuda
```

---

## 🎯 Evaluate Trained Model

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

---

## 📈 Training Progress Monitoring

### Monitor in Real-time

**Terminal 1** - Start training:
```bash
python src/train.py --use_toy_dataset --epochs 50 --device cuda
```

**Terminal 2** - Monitor GPU (if using CUDA):
```bash
watch -n 1 nvidia-smi
```

**Terminal 3** - Monitor TensorBoard (optional):
```bash
tensorboard --logdir logs/segnet_training/
# Open http://localhost:6006 in browser
```

---

## 📁 Checking Results

```bash
# List all training checkpoints
ls -lh models/custom_segnet/

# View training metrics
cat results/training_history.json

# List evaluation results
ls -lh results/predictions/

# View metric summary
cat results/metrics/custom_metrics.json
```

---

## ⚙️ Customization

### Change Learning Rate

```bash
python src/train.py \
    --use_toy_dataset \
    --epochs 50 \
    --learning_rate 0.001    # Smaller LR for stability
```

### Change Batch Size

```bash
python src/train.py \
    --use_toy_dataset \
    --epochs 50 \
    --batch_size 8          # Larger batch = faster but needs more memory
```

### Use Different Device

```bash
# CPU (slower, but works everywhere)
python src/train.py --use_toy_dataset --device cpu

# GPU (much faster)
python src/train.py --use_toy_dataset --device cuda

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🔍 Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
python src/train.py --use_toy_dataset --batch_size 2 --device cuda

# Or use CPU
python src/train.py --use_toy_dataset --device cpu
```

### CUDA Issues

```bash
# Check PyTorch CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# If CUDA not available, use CPU
python src/train.py --use_toy_dataset --device cpu
```

### Dataset Issues

```bash
# Create dummy dataset
python src/setup_dataset.py --dummy

# Verify dataset
python -c "from src.dataset import CamVidDataset; ds = CamVidDataset('data/CamVid/', split='train'); print(f'Dataset: {len(ds)} images')"
```

---

## 📝 Understanding the Code

### Model Architecture (src/segnet_model.py)
- `SegNetEncoder`: VGG16-style encoder with 13 conv layers
- `SegNetDecoder`: Symmetric decoder using pooling indices
- `SegNet`: Complete architecture

### Dataset (src/dataset.py)
- `ToySegmentationDataset`: For quick testing
- `CamVidDataset`: Real CamVid dataset loader
- `SegmentationTransform`: Data augmentation

### Training (src/train.py)
- `SegNetTrainer`: Handles training loop
- Learning rate scheduling
- Checkpoint saving
- Early stopping

### Evaluation (src/evaluate.py)
- `SegNetEvaluator`: Evaluation on test set
- Metrics computation
- Visualization of predictions

### Comparison (src/compare.py)
- `ModelComparator`: Compare two models
- Side-by-side metrics
- Inference time benchmark

---

## 🎓 Learning Path

1. **Start here**: `python test_segnet.py` - Verify everything works
2. **Understand**: `python tutorial.py` - Learn the pipeline
3. **Train**: `python src/train.py --use_toy_dataset --epochs 5` - Quick training
4. **Evaluate**: `python src/evaluate.py --model_path models/custom_segnet/best_model.pth`
5. **Production**: Download real CamVid data and train fully

---

## 🚀 Full Pipeline (One Command)

```bash
#!/bin/bash

# Create dummy dataset
python src/setup_dataset.py --dummy

# Train model
python src/train.py --use_toy_dataset --epochs 50 --device cuda

# Evaluate
python src/evaluate.py \
    --model_path models/custom_segnet/best_model.pth \
    --dataset_path data/CamVid/ \
    --split test \
    --visualize \
    --device cuda

echo "Pipeline complete! Check results/"
```

---

## 📊 Expected Results

After training for 50 epochs with toy dataset:
- **Loss**: Should decrease from ~2.4 to ~0.5
- **Accuracy**: Should increase towards 95%+
- **Training time**: 5-30 minutes (depending on device)

With real CamVid dataset after 100 epochs:
- **mIoU**: 55-60%
- **Global Accuracy**: 88-91%
- **Inference time**: 50-150ms/image

---

## 📞 Support

If you encounter issues:

1. Check test output: `python test_segnet.py`
2. Review error message carefully
3. Check troubleshooting section above
4. Verify all dependencies: `pip list | grep torch`
5. Check file structure: `find src/ -name "*.py"`

---

**Ready to start? Run:** `python test_segnet.py`
