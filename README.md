# Synthetic Data Augmentation for Medical Audio Classification
## Reproduction & Extension of McShannon et al. (2025)

> **Coswara COVID-19 Cough Detection** — Pipeline toàn diện từ tiền xử lý, sinh dữ liệu, đánh giá chất lượng, đến phân loại và ensemble.

---

## 📂 Kiến Trúc Hệ Thống

```text
├── 01_data_preprocessing.py   # Tiền xử lý & phân tách dữ liệu
├── 02_generative_models.py    # Train VAE / WGAN-GP / Diffusion
├── 03_evaluate_fad.py         # Đánh giá chất lượng sinh (FAD)
├── 04_classifier.py           # CNN classifier — 4 scenarios
└── 05_ensemble.py             # Ensemble 4 models → kết quả cuối
```

---

## 📝 Chi Tiết Từng Phần

### 1. `01_data_preprocessing.py` — Tiền Xử Lý Dữ Liệu

**Chức năng:** Quét cây thư mục Coswara, đọc `metadata.json`, trích xuất cough recordings và nhãn tương ứng.

**Luồng xử lý:**
- Lọc giữ lại `healthy` (label=0) và `positive` (label=1), loại bỏ `exposed` / `recovered`
- Resample về **16,000 Hz mono**; center-crop hoặc zero-pad về đúng **3 giây** ($48{,}000$ samples)
- Trích xuất **Log-Mel Spectrogram**: $N_{fft}=2048$, $Hop=512$, $N_{mels}=128$, $f_{min}=0$, $f_{max}=8000$ Hz → tensor $(128 \times 94 \times 1)$
- Loại bỏ các file silent (all-zero)

**Phân tách dữ liệu — cải tiến so với bài báo gốc:**

| | Bài báo gốc | Code này |
|---|---|---|
| Phương pháp | `StratifiedShuffleSplit` | **`GroupShuffleSplit` theo `user_id`** |
| Rủi ro | Data leakage: cùng user xuất hiện ở nhiều splits | Không: một user chỉ nằm trong đúng 1 split |

Tỷ lệ: **70% Train / 15% Val / 15% Test** (stratified theo label).

**Chuẩn hóa:** Z-score ($\mu$, $\sigma$) tính **chỉ trên tập Train**, áp dụng cho Val và Test — tránh information leakage.

**Output:** `dataset.pt` chứa `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, `norm_mean`, `norm_std`.

---

### 2. `02_generative_models.py` — Mô Hình Sinh Dữ Liệu

**Chức năng:** Train 3 mô hình sinh **chỉ trên COVID+ training samples**, mỗi model sinh thêm **50%** (≈558 samples) để augment lớp thiểu số.

| Model | Kiến trúc | Epochs | Optimizer | Đặc điểm |
|---|---|---|---|---|
| **VAE** | 4 ConvBlocks Encoder + Decoder đối xứng, latent dim=128 | 200 | Adam lr=1e-4 | Loss = MSE + β·KL, β=0.1 |
| **WGAN-GP** | Generator (dense→deconv) + Discriminator (spectral norm) | 300 | RMSprop lr=5e-5 | λ_gp=10, n_critic=5, ổn định hội tụ |
| **Diffusion** | U-Net 2D + Self-Attention (bottleneck 64×64) | 400 | AdamW lr=1e-4 | T=1000 steps, linear schedule β₁=1e-4→β_T=0.02 |

**Resume training** (nếu bị ngắt giữa chừng):
```bash
python 02_generative_models.py --model diffusion --resume diffusion_epoch_200.pth
```

**Generate only** (không train lại):
```bash
python 02_generative_models.py --model vae --generate_only --resume vae_final.pth
```

**Output:** `vae_samples.pt`, `gan_samples.pt`, `diffusion_samples.pt`

---

### 3. `03_evaluate_fad.py` — Đánh Giá Chất Lượng (FAD)

**Chức năng:** Định lượng chất lượng dữ liệu sinh bằng **Fréchet Audio Distance** — chỉ số càng thấp, phân phối sinh càng gần thật.

**Cơ chế:**
- Feature extractor: **ResNet18 pretrained** (ImageNet), conv1 được adapt 3-channel → 1-channel bằng cách **average weights** (thay vì random init) để giữ lại spatial patterns đã học
- Trích xuất embedding 512-dim cho cả real COVID+ và synthetic samples
- Tính FAD theo công thức Fréchet:

$$d^2 = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

**Cải tiến so với bài báo:** Bài báo không có bước đánh giá FAD. Code này bổ sung thêm **FAD Real-vs-Real baseline** (chia đôi tập real COVID+ → tính FAD giữa 2 nửa) làm **lower bound tham chiếu** — cho biết mức FAD tốt nhất lý thuyết có thể đạt.

**Output in ra:**
```
FAD (Real vs Real split):   XX.XX  ← lower bound
FAD (Real vs VAE):          XX.XX  [+XX.XX vs baseline]
FAD (Real vs WGAN-GP):      XX.XX  [+XX.XX vs baseline]
FAD (Real vs Diffusion):    XX.XX  [+XX.XX vs baseline]
```

---

### 4. `04_classifier.py` — CNN Classifier (4 Scenarios)

**Chức năng:** Kiểm chứng giá trị của dữ liệu tăng cường bằng cách train CNN phân loại nhị phân trên 4 kịch bản.

> ⚠️ **Lưu ý quan trọng:** Code dùng **CNN train from scratch** (không phải ResNet18 fine-tuning), đúng với thiết kế của bài báo gốc (Section 2.3) nhằm isolate contribution của synthetic data, tránh nhiễu từ pretrained knowledge.

**Kiến trúc CNN:**
```
Input (1×128×94)
→ ConvBlock(1→32)   [Conv2D 3×3 + BN + ReLU + MaxPool2×2]
→ ConvBlock(32→64)
→ ConvBlock(64→128)
→ ConvBlock(128→256)
→ AdaptiveAvgPool → Dropout(0.5) → Linear(256→2)
```

**Training config** (theo bài báo Section 2.3):
- Optimizer: Adam, lr=0.001
- Scheduler: CosineAnnealingLR, T_max=100
- Early stopping: patience=15 epoch, theo **val macro F1**
- Max epochs: 100

**Cải tiến so với bài báo:** Sử dụng **`WeightedRandomSampler`** đảm bảo mỗi batch cân bằng class — đặc biệt quan trọng sau khi thêm synthetic COVID+ samples làm lệch phân phối batch.

**4 Scenarios:**

| Scenario | Training data |
|---|---|
| Baseline | Real data only |
| Real + VAE | Real + 558 VAE synthetic COVID+ |
| Real + WGAN-GP | Real + 558 GAN synthetic COVID+ |
| Real + Diffusion | Real + 558 Diffusion synthetic COVID+ |

**Metrics** (primary: **macro F1**, secondary: AUROC):

| Scenario | Paper F1 | Paper AUROC |
|---|---|---|
| Baseline | 0.645 | 0.745 |
| Real + VAE | 0.646 | 0.748 |
| Real + WGAN-GP | 0.609 | 0.726 |
| Real + Diffusion | 0.644 | 0.746 |

**Output:** `results.json`, `model_Baseline.pth`, `model_Real+VAE.pth`, `model_Real+WGAN-GP.pth`, `model_Real+Diffusion.pth`

---

### 5. `05_ensemble.py` — Ensemble & Final Evaluation

**Chức năng:** Kết hợp 4 models bằng probability averaging (Equation 1, bài báo Section 2.5).

$$P_{\text{ensemble}}(\text{COVID+}) = \frac{1}{4} \sum_{i=1}^{4} p_i(\text{COVID+})$$

**Kết quả bài báo gốc:**

| Scenario | Paper F1 | Paper AUROC |
|---|---|---|
| Ensemble | **0.664** | **0.761** |

**Output:**
- `ensemble_results.json` — metrics tất cả scenarios + ensemble
- `ensemble_results.png` — 4-panel figure: F1 bar chart, AUROC bar chart, confusion matrix, ROC curves

**Extensions (không có trong bài báo):**
- **t-SNE visualization**: embedding của real vs synthetic trong feature space của CNN
- **Per-class F1**: breakdown Healthy vs COVID+ cho từng scenario

---

## 🚀 Hướng Dẫn Chạy

```bash
# Bước 1: Tiền xử lý
python 01_data_preprocessing.py

# Bước 2: Train generative models (chạy từng model hoặc tất cả)
python 02_generative_models.py --model all
# Hoặc từng model:
python 02_generative_models.py --model vae
python 02_generative_models.py --model wgan
python 02_generative_models.py --model diffusion

# Bước 3: Đánh giá FAD
python 03_evaluate_fad.py

# Bước 4: Train classifier
python 04_classifier.py

# Bước 5: Ensemble & kết quả cuối
python 05_ensemble.py
```

> **Môi trường khuyến nghị:** Google Colab T4 GPU. Thời gian ước tính: ~3–5 giờ cho toàn bộ pipeline (phần lớn ở Diffusion 400 epochs).

---

## 📊 Tóm Tắt Kết Quả Kỳ Vọng

| Scenario | F1 (paper) | AUROC (paper) | Ghi chú |
|---|---|---|---|
| Baseline | 0.645 | 0.745 | CNN from scratch, no augmentation |
| Real + VAE | 0.646 | 0.748 | Neutral (+0.001) |
| Real + WGAN-GP | 0.609 | 0.726 | Degraded (−0.036) |
| Real + Diffusion | 0.644 | 0.746 | Neutral (−0.001) |
| **Ensemble** | **0.664** | **0.761** | Best result (+0.019 vs baseline) |

**Kết luận chính:** Synthetic augmentation không cải thiện individual models trong bài toán này. Lợi ích duy nhất đến từ ensemble — các models tạo ra error patterns khác nhau, kết hợp giúp cải thiện nhẹ F1.

---

## 🔧 Cải Tiến So Với Bài Báo Gốc

| Điểm | Bài báo gốc | Code này |
|---|---|---|
| Data split | StratifiedShuffleSplit | **GroupShuffleSplit** theo user_id (không leakage) |
| Batch sampling | Shuffle thông thường | **WeightedRandomSampler** (cân bằng class) |
| Đánh giá generative | Không có | **FAD + Real-vs-Real lower bound** |
| Ensemble | Mô tả kết quả | **File riêng** với full metrics + visualization |
| Resume training | Không đề cập | **Checkpoint mỗi 50 epoch**, resume-aware |
| Analysis | Không có | **t-SNE embedding** + **per-class F1** breakdown |

---

## 📦 Dependencies

```bash
pip install torch librosa scikit-learn seaborn matplotlib numpy pandas scipy
pip install torchvision  # cho ResNet18 feature extractor (FAD)
```

---

## 📁 Output Files

| File | Mô tả |
|---|---|
| `dataset.pt` | Mel-spectrograms + labels, tất cả splits |
| `vae_samples.pt` | 558 VAE synthetic COVID+ spectrograms |
| `gan_samples.pt` | 558 WGAN-GP synthetic COVID+ spectrograms |
| `diffusion_samples.pt` | 558 Diffusion synthetic COVID+ spectrograms |
| `results.json` | F1 + AUROC của 4 classifier scenarios |
| `ensemble_results.json` | F1 + AUROC của tất cả models + ensemble |
| `ensemble_results.png` | Figure tổng hợp 4 panels |
| `model_*.pth` | Checkpoints của 4 CNN models |

---

## 📖 Tài Liệu Tham Khảo

McShannon, D., Mella, A., & Dietrich, N. (2025). *Synthetic Data Augmentation for Medical Audio Classification: A Preliminary Evaluation*. Independent Researcher / University of Toronto.

Bhattacharya, D., et al. (2023). Coswara — A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis. *Interspeech 2023*.
