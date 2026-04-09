# Experimental Workflow Guide for `6970-project`

## 1. Overview of Relevant Files

### 1.1 Core Code Files for the Experimental Workflow

- Data processing: `main_model/dataset.py`
- Main model: `main_model/model.py`
- Training entry point: `main_model/train.py`
- Evaluation entry point: `main_model/evaluate.py`
- Pretrained backbone: `main_model/pretrained/w600k_r50.onnx`

### 1.2 Landmark

Training results:
- With landmark: best validation accuracy = `0.8809`, at epoch `56`, test accuracy = `0.8856`
- Without landmark: best validation accuracy = `0.8760`, at epoch `36`, test accuracy = `0.8889`

Testing:
- Landmark was used during testing.

We finally chose **not to use landmark during training**.

### 1.3 Training Hyperparameters

```yaml
data_root: ../RAF-DBdataset
save_dir: checkpoints_v7b_cosine80_more_seed55_landmark
image_size: 112
batch_size: 32
num_workers: 4
epochs: 120
patience: 25
cosine_t_max: 80
backbone_lr: 2e-6
head_lr: 5e-4
weight_decay: 1e-4
dropout: 0.4
focal_gamma: 1.0
label_smoothing: 0.1
lam_center: 0.01
center_lr: 0.5
lam_sce: 0.5
sce_beta: 0.5
no_pretrained: false
landmarks_dir: .
stage1_ckpt: null
disable_strong_fd_aug: true
mixup_prob: 0.0
cutmix_prob: 0.0
final_tune_epochs: 0
seeds: "55"
```

### 1.4 Multi-Seed Background

This experiment was **not** a single random run with one seed. Instead, multiple seeds were tested under the same configuration:

- seed 2
- seed 7
- seed 13
- seed 55
- seed 88

Finally, **seed = 55** was selected.

## 2. Data Pipeline: From RAF-DB to Training / Validation / Test Samples

### 2.1 Data Source

The dataset used in this experiment is **RAF-DB**, located under `../RAF-DBdataset`. The core files include:

- `train_labels.csv`
- `test_labels.csv`
- `DATASET/train/`
- `DATASET/test/`

The emotion category order is defined in `EMOTION_NAMES` in `main_model/dataset.py` as:

- Surprise
- Fear
- Disgust
- Happiness
- Sadness
- Anger
- Neutral

### 2.2 Source Relationship of Train / Validation / Test Sets

This experiment did **not** repartition the entire RAF-DB dataset. Instead, the following strategy was used:

1. The official `train_labels.csv` was used as the full training pool.
2. A stratified `9:1` split was performed within this training pool to obtain the training set and validation set.
3. The official `test_labels.csv` was kept unchanged as the test set.

This logic is implemented in `main_model/dataset.py` through two functions:

- `stratified_split_indices`
- `create_dataloaders`

The procedure of `stratified_split_indices` is:

- Take sample indices separately for each class
- Shuffle within each class
- Split out the validation set with `val_ratio = 0.1`
- Put the remaining samples into the training set

### 2.3 Input Resolution

The input resolution in this experiment was fixed at:

- `112 x 112`

This was not chosen arbitrarily. It is aligned with the backbone design. The current main model uses an **IR-50 style face recognition backbone**, which is naturally designed for `112x112` inputs.

### 2.4 Training Augmentation and Validation / Test Preprocessing

The training augmentations are defined in `get_transforms(split='train', image_size=112)` in `main_model/dataset.py`.

The basic augmentations for the training set include:

1. `RandomResizedCrop(112, scale=(0.82, 1.0), ratio=(0.95, 1.05))`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomRotation(12)`
4. `ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05)`
5. `ToTensor()`
6. `RandomErasing(p=0.25, scale=(0.02, 0.18))`
7. `Normalize(mean, std)`, using ImageNet-style mean and standard deviation

The validation set and test set do not use random augmentation. They only apply:

1. `Resize((112, 112))`
2. `ToTensor()`
3. `Normalize(mean, std)`

### 2.5 Class Imbalance Handling

The original RAF-DB training data has clear class imbalance, especially for:

- Fear
- Disgust
- Anger

Their sample counts are significantly smaller than majority classes such as Happiness and Neutral.

At the data level, this experiment adopted **moderate oversampling**, corresponding to `moderate_oversample` in `dataset.py`:

- For each class in the training set, if the sample count is less than `2000`, samples are duplicated with replacement until the class reaches `2000`
- If the class already has more than `2000` samples, it is kept unchanged

Therefore, after expansion, the approximate target sample counts for the training set are:

- Surprise: 2000
- Fear: 2000
- Disgust: 2000
- Happiness: unchanged (already above 2000)
- Sadness: 2000
- Anger: 2000
- Neutral: unchanged (already above 2000)

This step does not change the validation set or the test set. It only affects the training sample indices.

### 2.6 Class Weights and Additional Emphasis on Fear / Disgust

In addition to oversampling, this experiment also used class weights in the loss function. The basic idea of `get_class_weights` in `dataset.py` is inverse frequency:

- Fewer samples in a class → larger weight
- More samples in a class → smaller weight

On top of this, two difficult classes were further emphasized:

- `Fear x1.5`
- `Disgust x1.8`

The actual meaning is:

- Misclassifying Fear incurs a larger loss
- Misclassifying Disgust incurs a larger loss

This is **not** data augmentation. It is stronger emphasis on minority classes at the loss level.

## 3. Model Pipeline

### 3.1 Source and Role of the Pretrained Backbone

The pretrained face recognition backbone file is:

- `main_model/pretrained/w600k_r50.onnx`

It is **not** a weight file trained on RAF-DB. Instead, it contains a pretrained **IR-50** model trained on a large-scale face dataset. The current code uses this file to initialize the model backbone as a strong feature extractor that already knows **how to represent faces**, rather than learning everything from scratch.

### 3.2 Main Architecture of the Model

1. **IR-50 backbone for feature extraction**
2. **Mid-level and high-level features pass through CBAM**
3. **Multi-scale fusion**
4. **Self-Attention models regional relationships**
5. **Global pooling + BN + Dropout + FC(7)** complete the classification

More specifically, the main forward path in the current model is:

1. `self.backbone(x)` returns two feature maps:
   - `stage3`: `256 x 14 x 14`
   - `stage4`: `512 x 7 x 7`

2. `stage3` and `stage4` each pass through `CBAM`

3. `stage3` is projected to `512` channels through `proj3`, then aligned to `7 x 7` by `adaptive_avg_pool2d`

4. A learnable parameter `fuse_weight` is used for weighted fusion:
   - One branch comes from the fine-grained semantic features of `stage3`
   - The other branch comes from the high-level semantic features of `stage4`

5. The fused `7 x 7` feature map is then fed into `SelfAttentionBlock`

6. Finally, the output passes through:
   - `AdaptiveAvgPool2d(1)`
   - `BatchNorm1d(512)`
   - `Dropout(0.4)`
   - `Linear(512, 7)`

to produce 7-class emotion logits.

Therefore, this experiment can be summarized as:

> Pretrained IR-50 backbone + CBAM + multi-scale fusion + Self-Attention + linear classification head

### 3.3 Parameter Count Explanation

The current main-model script reports the approximate parameter counts as:

- Total parameters: `45,851,405`
- Backbone: `43,572,288`
- Added modules: `2,279,117`

This indicates:

- The vast majority of parameters come from the pretrained IR-50 backbone
- The attention modules and classification head you added contribute only a relatively small number of parameters

This is also why pretraining has a major influence on the results: for a small dataset like RAF-DB, it is very difficult to learn face representations of comparable quality from scratch.

## 4. Training Pipeline: From Dataloader to Best Checkpoint

### 4.1 Training Entry Point

The training entry point is:

- `main_model/train.py`

This script completes the following tasks:

1. Parse training arguments
2. Build train / val / test dataloaders
3. Build the model
4. Build the loss functions and optimizers
5. Run the epoch loop
6. Save the best checkpoint according to validation results
7. Evaluate on the test set using the best checkpoint, and write the result into `best_seed55.pth`

### 4.2 How the Main Model Is Connected to the Training Script

The current `train.py` does not hardcode:

```python
from model import build_model
```

at the top of the file. Instead, it uses dynamic import:

- When `model_type == 'main'`, it imports `model.py`
- When `model_type == 'baseline'`, it imports `baseline_model.py`

This experiment clearly belongs to the main-model route, so the actual function used is:

- `build_model` from `main_model/model.py`

The model is then instantiated by:

```python
build_model(num_classes=7, pretrained=True, dropout=0.4)
```

The checkpoint parameter:

- `no_pretrained = False`

also indicates that the pretrained backbone was indeed enabled in this experiment.

### 4.3 Optimizer and Differential Learning Rates

The main optimizer in this experiment is:

- `AdamW`

and differential learning rates are used:

- backbone: `2e-6`
- head: `5e-4`

This reflects a typical fine-tuning strategy:

- The pretrained backbone is only slightly adjusted
- The newly added attention modules and classification head learn much faster

At the same time, the center vectors in Center Loss are updated with a separate optimizer:

- `SGD`
- `center_lr = 0.5`

Therefore, two optimizers are actually working in parallel during training:

1. `AdamW` for model parameters
2. `SGD` for `CenterLoss.centers`

### 4.4 Three-Part Loss Function

This experiment did not use standard cross-entropy alone as the only classification loss. Instead, it used three cooperating components:

#### 1. Focal Loss

Parameters:

- `gamma = 1.0`
- `label_smoothing = 0.1`

Role:

- It is still centered around cross-entropy
- But it reduces the influence of easy samples and increases the contribution of hard samples
- Combined with class weights, it is more friendly to minority classes

#### 2. Center Loss

Parameter:

- `lam_center = 0.01`

Role:

- It makes feature vectors of the same class more compact in the embedding space
- It helps alleviate overly scattered feature distributions among visually similar expression classes

#### 3. SCE Loss

Parameters:

- `lam_sce = 0.5`
- `sce_beta = 0.5`

Role:

- It is more robust to potential noise introduced by crowd-sourced labels
- It adds a reverse CE term on top of ordinary CE

The total loss can be written as:

```text
Total Loss = Focal + 0.01 * Center + 0.5 * SCE
```

The three parts can be understood as:

- Focal: responsible for classification
- Center: responsible for tightening intra-class feature distributions
- SCE: responsible for improving robustness to noisy labels

### 4.5 AMP, Gradient Clipping, and Learning Rate Scheduling

To improve training stability and efficiency, this experiment also used:

- AMP mixed-precision training
- Gradient clipping: `max_norm = 5.0`
- `CosineAnnealingLR`

Learning-rate scheduling parameters:

- `epochs = 120`
- `cosine_t_max = 80`
- `eta_min = 1e-7`

The meaning of this setup is:

- A complete cosine annealing cycle is finished within the first 80 epochs
- After the learning rate decays to a very low value, training is still allowed to continue for a while
- Together with early stopping, this avoids forcing the model to train through all epochs unnecessarily

### 4.6 Early Stopping Was Enabled

Monitoring metric:
- validation accuracy

Patience:
- 25 epochs

Rule:
- stop if there is no validation improvement for 25 consecutive epochs

## 5. Evaluation Metrics

- Accuracy
- Macro F1
- Weighted F1
- MAE

In addition, precision / recall / f1-score for each class are also reported.

## 6. Complete Workflow Summary of This Experiment

Putting everything together, the complete pipeline of `v7b_cosine80_more_seed55` can be summarized as follows:

1. Read the official `train_labels.csv` and `test_labels.csv` of RAF-DB
2. Perform a stratified `9:1` split inside the official training set to obtain train / val
3. Apply moderate oversampling to the training portion, bringing minority classes up to `2000`
4. Build training augmentations and validation / test preprocessing
5. Build the main model: `IR-50 pretrained backbone + CBAM + multi-scale fusion + Self-Attention + FC(7)`
6. Train with the three-part loss: `Focal + 0.01*Center + 0.5*SCE`
7. Train using AdamW + differential learning rates + AMP + gradient clipping + CosineAnnealingLR
8. Evaluate on the validation set after each epoch and save the best weights according to validation performance
