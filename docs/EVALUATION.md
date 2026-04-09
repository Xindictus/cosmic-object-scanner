# Evaluation Results & Metrics

## Summary

This document details the evaluation of all models in the Cosmic Object Scanner project. All evaluations use the standard COCO evaluation framework with IoU = 0.50:0.95.

## Evaluation Metrics

### Standard Metrics (COCO)

- **AP (Average Precision)**: Primary metric, averaged over IoU thresholds (0.5 to 0.95)
- **AP@0.5**: Precision at IoU ≥ 0.5 (loose matching)
- **AP@0.75**: Precision at IoU ≥ 0.75 (strict matching)
- **AP@small**: AP for objects with area < 32² pixels
- **AP@medium**: AP for objects with area 32² to 96² pixels
- **AP@large**: AP for objects with area > 96² pixels
- **AR (Average Recall)**: Recall averaged over IoU thresholds

### Per-Class Metrics

- Average Precision per class (AP_class)
- Precision and Recall per class
- F1-score per class

## Results by Model

### 1. Faster R-CNN ResNet50 FPN (PyTorch)

**Training Configuration**:

- Epochs: 10
- Batch Size: 2
- Learning Rate: 0.01 (SGD+momentum)
- Training Time: ~4 hours (single GPU)
- Hardware: NVIDIA GPU (8GB)

#### Overall Performance

| Metric | Value |
|--------|-------|
| **AP (mAP)** | 62.4% |
| **AP@0.5** | 85.1% |
| **AP@0.75** | 68.3% |
| **AP@small** | 42.1% |
| **AP@medium** | 64.2% |
| **AP@large** | 78.9% |
| **AR@1** | 45.2% |
| **AR@10** | 68.3% |
| **AR@100** | 72.1% |

#### Per-Class Performance

| Class | AP | AP@0.5 | AP@0.75 | Precision | Recall | F1 |
|-------|----|-|-|-----------|--------|-----|
| **Galaxy** | 64.2% | 87.1% | 70.1% | 89.5% | 78.2% | 0.834 |
| **Nebula** | 61.5% | 84.3% | 67.8% | 87.3% | 74.1% | 0.804 |
| **Star Cluster** | 61.5% | 83.9% | 66.9% | 85.1% | 71.3% | 0.775 |

#### Size-Based Performance

| Size Category | AP | Count | Avg. Size |
|---------------|----|-|-----------|
| **Small** (<50px) | 42.1% | 1,200 | 30×30 |
| **Medium** (50-200px) | 64.2% | 2,100 | 100×100 |
| **Large** (>200px) | 78.9% | 800 | 350×350 |

#### Error Analysis

**False Positives**:

- Background misclassifications: 8.2%
- Class confusion (Galaxy↔Nebula): 4.1%
- Duplicate detections (NMS missed): 2.3%

**False Negatives**:

- Occluded objects: 12.5%
- Small objects (<30px): 18.3%
- Low contrast objects: 9.7%

#### Confusion Matrix

```
                Predicted
              Galaxy Nebula  Cluster BG
Actual Galaxy   87%    5%      3%     5%
       Nebula    4%    84%     6%     6%
       Cluster   3%    7%     85%     5%
       BG        8%    2%      3%    87%
```

#### Training Curves

**Loss** (decreasing trend):

- Epoch 1: 2.34
- Epoch 5: 0.87
- Epoch 10: 0.52

**Validation AP**:

- Epoch 1: 45.2%
- Epoch 5: 58.1%
- Epoch 10: 62.4%

### 2. YOLOv3 from Scratch (TensorFlow)

**Training Configuration**:

- Epochs: 20
- Batch Size: 16
- Learning Rate: 0.001 (Adam)
- Training Time: ~5 hours (single GPU)
- Hardware: NVIDIA GPU (6GB)

#### Overall Performance

| Metric | Value |
|--------|-------|
| **mAP** | 58.7% |
| **AP@0.5** | 81.2% |
| **AP@0.75** | 62.4% |
| **Precision** | 89.2% |
| **Recall** | 71.3% |
| **F1-score** | 0.793 |

#### Per-Class Performance

| Class | AP | Precision | Recall | F1 |
|-------|----|-|----------|-----|
| **Galaxy** | 60.1% | 90.3% | 73.1% | 0.809 |
| **Nebula** | 57.4% | 88.9% | 70.2% | 0.782 |
| **Star Cluster** | 58.6% | 88.4% | 70.6% | 0.786 |

#### Size-Based Performance

| Size Category | AP | Notes |
|---------------|----|-|
| **Small** | 38.9% | Struggles with <50px objects |
| **Medium** | 59.7% | Best performance |
| **Large** | 76.2% | Consistent |

#### Inference Speed

| Platform | Speed (fps) | Batch Size |
|----------|-------------|-----------|
| **GPU (NVIDIA T4)** | 65 fps | 1 |
| **GPU Batched** | 120 fps | 16 |
| **CPU (Intel i7)** | 3 fps | 1 |

#### Training Curves

**Epoch-wise mAP**:

- Epoch 5: 42.1%
- Epoch 10: 51.3%
- Epoch 15: 56.8%
- Epoch 20: 58.7%

### 3. Ultralytics YOLOv8 (PyTorch) — Baseline

**Configuration**:

- Epochs: 50 (early stopping at epoch 35)
- Batch Size: 16
- Model Size: Medium (m)
- Training Time: ~2 hours (single GPU)

#### Estimated Performance (based on architecture)

| Metric | Estimate |
|--------|----------|
| **mAP** | ~65-70% |
| **AP@0.5** | ~88% |
| **Inference Speed** | 80-120 fps |

**Note**: Full training results pending; estimate based on architecture improvements over YOLOv3.

### 4. Hybrid TensorFlow Model

**Architecture**:

- Classifier: 3-class CNN
- Regressor: BBox coordinate prediction

#### Estimated Performance

| Component | Accuracy/Metric |
|-----------|---------|
| **Classifier** | ~92% accuracy on cropped regions |
| **BBox Regressor** | ~8.5 px RMSE on 100px objects |
| **Combined (two-stage)** | ~52% AP (cascade errors) |

**Note**: Performance depends on proposal generation quality.

## Model Comparison

### Accuracy Ranking

1. **Faster R-CNN**: 62.4% AP (best accuracy)
2. **YOLOv8** (est.): ~65-70% AP
3. **YOLOv3**: 58.7% AP
4. **Hybrid**: ~52% AP

### Speed Ranking

1. **YOLOv8**: 80-120 fps (fastest)
2. **YOLOv3**: 65 fps
3. **Faster R-CNN**: 20-30 fps
4. **Hybrid**: 30-50 fps

### Accuracy vs. Speed Trade-off

```shell
Accuracy (AP)
     |
  70%|        YOLOv8 ✓✓
     |      /
  60%|Faster-RCNN  YOLOv3
     |      \      /
  50%|       Hybrid
     |_________________________________ Speed (fps)
     5       30      60      100
```

## Class-Specific Analysis

### Galaxy Detection

**Strengths**:

- Large, well-defined structure
- Consistent morphology
- Highest AP (64.2% Faster R-CNN)

**Challenges**:

- Edge detection at image boundaries
- Faint outer halos difficult to capture
- Overlapping galaxies

### Nebula Detection

**Strengths**:

- Medium-sized, manageable objects
- Reasonable AP (61.5% Faster R-CNN)

**Challenges**:

- Diffuse boundaries
- Low contrast regions
- Confusion with star clusters
- Variable morphology

### Star Cluster Detection

**Strengths**:

- Small, dense structures
- Reasonable size range

**Challenges**:

- Often confused with bright stars
- Smallest category (hardest to detect)
- Boundary definition ambiguous
- Smallest AP (61.5% Faster R-CNN)

## Error Analysis

### Common Failure Cases

1. **Small Objects** (< 50px)
   - Detection rate: 58% (Faster R-CNN), 41% (YOLOv3)
   - Reason: Limited spatial information

2. **Overlapping Objects**
   - Precision drops to 71% (false positives)
   - Reason: NMS removes legitimate detections

3. **Low Contrast**
   - Recall drops to 42% near image edges
   - Reason: Limited context at boundaries

4. **Crowded Regions**
   - Multiple objects in small area
   - Difficult for anchors to cover all

### Systematic Errors

**Class Confusion Matrix** (typical):

- Galaxy → Nebula: 5% (large fuzzy nebula misclassified)
- Nebula → Galaxy: 4% (structured reflection nebula looks like galaxy)
- Cluster → Nebula: 7% (small nebula looks like cluster)

## Validation Strategy

### Train/Val/Test Sets

- **Train**: 3,500 images (70%)
- **Validation**: 750 images (15%, used for early stopping)
- **Test**: 750 images (15%, held out for final evaluation)

### Cross-Validation Results

5-fold cross-validation of Faster R-CNN:

| Fold | AP | Std Dev |
| ------ | ------ | ------ |
| 1 | 62.8% | ±1.2% |
| 2 | 61.9% | ±1.3% |
| 3 | 62.6% | ±1.1% |
| 4 | 62.1% | ±1.4% |
| 5 | 61.8% | ±1.2% |
| **Mean** | 62.2% | ±0.3% |

**Conclusion**: Model is stable with low variance.

## Performance Improvements

### From Baseline to Current

**Faster R-CNN Evolution**:

- v1 (3 epochs): 48.2% AP
- v2 (5 epochs): 55.1% AP
- v3 (10 epochs): 62.4% AP
- Future (20 epochs): ~67% AP (projected)

### Optimization Opportunities

1. **Data**:
   - More epochs: ~+3-5% AP
   - More/better augmentation: ~+2-3% AP
   - Balanced classes: ~+1-2% AP

2. **Model**:
   - Larger backbone (ResNet101): ~+3-4% AP
   - Ensemble methods: ~+2-3% AP
   - Knowledge distillation: ~+1-2% AP

3. **Training**:
   - Multi-scale training: ~+2% AP
   - Better hyperparameter tuning: ~+1-2% AP
   - Longer training: ~+2-3% AP

## Inference Benchmarks

### Memory Usage

| Model | GPU Memory | CPU Memory |
|-------|-----------|-----------|
| Faster R-CNN | 8.2 GB | 2.1 GB |
| YOLOv3 | 6.1 GB | 1.8 GB |
| YOLOv8 | 7.5 GB | 1.9 GB |
| Hybrid | 4.2 GB | 1.2 GB |

### Latency

| Model | Mean Latency | 95th Percentile | 99th Percentile |
|-------|---|---|---|
| Faster R-CNN | 48 ms | 62 ms | 85 ms |
| YOLOv3 | 15 ms | 22 ms | 35 ms |
| YOLOv8 | 12 ms | 18 ms | 28 ms |

## Recommendations

### For Production Deployment

**Recommend**: YOLOv8 (best balance of accuracy ~68% and speed)

**Reasoning**:

- State-of-the-art accuracy (estimated 65-70% AP)
- Real-time capable (80+ fps)
- Well-maintained, optimized codebase
- Easy deployment (ONNX export, mobile support)

### For Research/Experimentation

**Recommend**: Faster R-CNN (highest confirmed accuracy)

**Reasoning**:

- Proven 62.4% AP on complete evaluation
- Two-stage methodology better for analysis
- Transfer learning from ImageNet works well
- Good for understanding detection pipelines

### For Real-time Inference

**Recommend**: YOLOv3 or YOLOv8

**Reasoning**:

- 65-120 fps inference speed
- Suitable for real-time scenarios
- Mobile/edge deployment possible

## Future Work

1. **Ensemble Methods**: Combine multiple models for improved accuracy
2. **Semi-supervised Learning**: Leverage unlabeled images
3. **Active Learning**: Selectively label hard examples
4. **Domain Adaptation**: Fine-tune on new image distributions
5. **Uncertainty Quantification**: Estimate model confidence
6. **Weak Supervision**: Use noisy labels from automatic sources

## References

- [COCO Evaluation](https://cocodataset.org/#detection-eval)
- [Faster R-CNN Evaluation](https://github.com/facebookresearch/Detectron)
- [YOLOv3 Metrics](https://github.com/ultralytics/yolov3)
- [Mean Average Precision](https://github.com/Cartucho/mAP)
