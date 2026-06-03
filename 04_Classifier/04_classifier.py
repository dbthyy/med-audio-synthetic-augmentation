"""
04_classifier.py
================
Huấn luyện và đánh giá CNN classifier trên 4 kịch bản dữ liệu:
  1. Baseline — chỉ dữ liệu thật
  2. Real + VAE augmented
  3. Real + WGAN-GP augmented
  4. Real + Diffusion augmented

"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3          # Adam lr=0.001 theo bài báo Section 2.3
PATIENCE   = 15            # Early stopping patience theo bài báo
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH        = "dataset.pt"
VAE_REPRODUCE_PATH  = "reproduce_vae_samples.pt"
VAE_IMPROVE_PATH    = "improve_vae_samples.pt"
GAN_REPRODUCE_PATH  = "reproduce_gan_samples.pt" 
GAN_IMPROVE_PATH    = "improve_gan_samples.pt"
DIFF_REPRODUCE_PATH = "reproduce_diffusion_samples.pt" 
DIFF_IMPROVE_PATH   = "improve_diffusion_samples.pt"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True


# ── 1. DATASETS ───────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y.long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.shape[-1] == 1:
            x = x.permute(2, 0, 1)    # (H, W, 1) → (1, H, W)
        return x, self.y[idx]


class SyntheticDataset(Dataset):
    """Dữ liệu sinh — tự động gán nhãn COVID+ (1)."""
    def __init__(self, X):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = torch.ones(len(self.X), dtype=torch.long)

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]


def make_standard_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    labels = []
    for _, y in dataset:
        labels.append(int(y))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )



# ── 2. CLASSIFICATION MODELS ───────────────────────────────────────────────────────
# CNN FROM SCRATCH
class ConvBlock(nn.Sequential):
    """Conv2D → BN → ReLU → MaxPool (đúng theo bài báo Section 2.3)."""
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


class AudioCNN(nn.Module):
    """
    [FIX] CNN train from scratch — đúng bài báo Section 2.3.
    4 ConvBlocks: 32 → 64 → 128 → 256 filters, 3×3 kernel.
    Trained from random initialization (không dùng pretrained weights).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
        
    def embed(self, x):
        x = self.features(x)
        return self.pool(x).flatten(1)

# EFFICIENTBET-B0 
class AudioEfficientNetB0(nn.Module):
    """
    Modifications:
    - Input channels: 3 -> 1
    - Output classes: configurable (default = 2)
    - Pretrained ImageNet weights are reused
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        original_conv = self.effnet.features[0][0]
        
        # Replace the first convolution layer to accept single-channel spectrograms instead of RGB images.
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        with torch.no_grad():
            self.effnet.features[0][0].weight.copy_(torch.sum(original_conv.weight, dim=1, keepdim=True))

        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 1, 128, 94]
        return self.effnet(x)

    def embed(self, x):
        # t-SNE visualization
        # [B, H, W, 1] -> [B, 1, H, W]
        if x.dim() == 4 and x.size(-1) == 1:
            x = x.permute(0, 3, 1, 2)

        x = self.effnet.features(x)
        x = self.effnet.avgpool(x)
        x = torch.flatten(x, 1) # [Batch, 1280]
        return x


# ── 3. TRAIN & EVALUATE ───────────────────────────────────────────────────────
def plot_misclassified_samples(false_positives: list, false_negatives: list, scenario_name: str):
    """
    Visualize misclassified samples (up to 10 samples per error category).
    """
    count_fp = min(10, len(false_positives))
    count_fn = min(10, len(false_negatives))
    total_errors = len(false_positives) + len(false_negatives)

    print(f"\n  [Error Analysis] Scenario '{scenario_name}': {total_errors} misclassified samples.")
    print(f"  - Healthy misclassified as COVID+: {len(false_positives)} samples (Displaying {count_fp})")
    print(f"  - COVID+ misclassified as Healthy: {len(false_negatives)} samples (Displaying {count_fn})")

    if count_fp == 0 and count_fn == 0:
        print("  -> No misclassified samples found.")
        return

    ncols = 10
    nrows = (1 if count_fp > 0 else 0) + (1 if count_fn > 0 else 0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.5 * nrows))
    fig.suptitle(f"Misclassified Analysis (Max 10 per class) - Scenario: {scenario_name}", fontsize=14, fontweight='bold')

    # Ensure axes is always a 2D array with shape [nrows, ncols]
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    current_row = 0

    # Row 1: True Healthy -> Predicted COVID+
    if count_fp > 0:
        for idx in range(ncols):
            ax = axes[current_row, idx]
            if idx < count_fp:
                item = false_positives[idx]
                ax.imshow(item["image"], origin="lower", cmap="viridis")
                ax.set_title(
                    f"Idx: {item['index']}\nTrue: Healthy\nPred: COVID+ ({item['prob_covid']:.2f})",
                    fontsize=8,
                    color="red"
                )
            ax.axis("off")
        current_row += 1

    # Row 2: True COVID+ -> Predicted Healthy
    if count_fn > 0:
        for idx in range(ncols):
            ax = axes[current_row, idx]
            if idx < count_fn:
                item = false_negatives[idx]
                ax.imshow(item["image"], origin="lower", cmap="viridis")
                ax.set_title(
                    f"Idx: {item['index']}\nTrue: COVID+\nPred: Healthy ({item['prob_covid']:.2f})",
                    fontsize=8,
                    color="darkorange"
                )
            ax.axis("off")

    plt.tight_layout()
    plt.show()



def train_and_evaluate(model, model_name, classweight, train_loader, val_loader, test_loader,
                       scenario_name: str, verbose: bool = True) -> dict:
    """
    Train the CNN model and collect misclassified samples
    for later analysis in the main function.
    """
    if verbose:
        print(f"\nTraining: {scenario_name}")

    criterion = nn.CrossEntropyLoss(weight=classweight)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1  = -1.0
    best_wts     = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        # ===== Training Phase =====
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs).float(), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ===== Validation Phase =====
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                _, preds = torch.max(model(inputs), 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_wts    = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if verbose and (epoch % 10 == 0):
            print(f"  Epoch {epoch:03d}/{EPOCHS} | val_macro_F1={val_f1:.4f} "
                  f"| best={best_val_f1:.4f} | patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch}.")
            break

    # ===== Test Evaluation & Misclassification Collection =====
    model.load_state_dict(best_wts)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    false_positives = []
    false_negatives = []
    sample_idx = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            inputs_np = inputs.cpu().numpy()
            preds_np  = preds.cpu().numpy()
            labels_np = labels.numpy()
            probs_np  = probs.cpu().numpy()

            for i in range(len(labels_np)):
                true_label = labels_np[i]
                pred_label = preds_np[i]
                prob_covid = probs_np[i]

                if pred_label != true_label:
                    img_data = inputs_np[i]
                    if img_data.shape[0] == 1:
                        img_data = img_data[0]

                    sample_info = {
                        "index": sample_idx,
                        "true_class": "COVID+" if true_label == 1 else "Healthy",
                        "pred_class": "COVID+" if pred_label == 1 else "Healthy",
                        "prob_covid": float(prob_covid),
                        "image": img_data
                    }

                    if true_label == 0:
                        false_positives.append(sample_info)
                    else:
                        false_negatives.append(sample_info)

                sample_idx += 1

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            all_probs.extend(probs_np)

    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auroc     = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)

    # Save model checkpoint (overwrite existing file, independent of run count)
    model_path = f"{model_name}_{scenario_name.replace(' ', '_')}.pth"

    if verbose:
        print(f"  Result ==== precision: {precision:.4f} | recall: {recall:.4f} | macro F1: {macro_f1:.4f} | AUROC: {auroc:.4f}")
        torch.save(model.state_dict(), model_path)

    return {
        "macro_F1":        round(macro_f1, 4),
        "AUROC":           round(auroc, 4),
        "probs":           all_probs,
        "labels":          all_labels,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# ── 4. RUN MODEL ──────────────────────────────────────────────────────────────────────
def run_ensemble_from_models(models_dict, test_loader, model_name, verbose=True):
    """
    Ensemble multiple trained models using Soft Voting
    (average predicted probabilities).
    """
    all_probs_list = []
    all_labels_list = []

    # Flag to ensure labels are collected only once from the first model
    collected_labels = False

    for scenario_name, model in models_dict.items():
        model.eval()
        run_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                run_probs.extend(probs.cpu().numpy())

                # Collect labels only during the first model evaluation
                if not collected_labels:
                    all_labels_list.extend(labels.numpy())

        all_probs_list.append(run_probs)
        collected_labels = True  # Prevent repeated label collection for efficiency

    # Convert the complete label list into a NumPy array
    all_labels = np.array(all_labels_list)

    # Soft voting: average probabilities across all models
    ensemble_probs = np.mean(all_probs_list, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    # Compute evaluation metrics
    macro_f1  = f1_score(all_labels, ensemble_preds, average="macro", zero_division=0)
    auroc     = roc_auc_score(all_labels, ensemble_probs)
    precision = precision_score(all_labels, ensemble_preds, zero_division=0)
    recall    = recall_score(all_labels, ensemble_preds, zero_division=0)

    if verbose:
        print(f"ENSEMBLE RESULTS ==== precision: {precision:.4f} | recall: {recall:.4f} | macro F1: {macro_f1:.4f} | AUROC: {auroc:.4f}")
        print(f"recall    = {round(recall, 4)}")
        print(f"macro_f1  = {round(macro_f1, 4)}")
        print(f"auroc     = {round(auroc, 4)}")
        print("=" * 50)

    return {
        "macro_F1": round(macro_f1, 4),
        "AUROC": round(auroc, 4)
    }

def run_model(model_class, model_name, num_runs, result_path):
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Dataset file not found.")

    data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    real_train_ds = AudioDataset(data["X_train"], data["y_train"])
    val_ds        = AudioDataset(data["X_val"],   data["y_val"])
    test_ds       = AudioDataset(data["X_test"],  data["y_test"])

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_counts = np.bincount(data["y_train"])
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # Load synthetic datasets
    def load_pt(path):
        return torch.load(path, map_location="cpu") if os.path.exists(path) else None

    vae_X_1 = load_pt(VAE_REPRODUCE_PATH)
    vae_X_2 = load_pt(VAE_IMPROVE_PATH)
    gan_X_1 = load_pt(GAN_REPRODUCE_PATH)
    gan_X_2 = load_pt(GAN_IMPROVE_PATH)
    diff_X_1 = load_pt(DIFF_REPRODUCE_PATH)
    diff_X_2 = load_pt(DIFF_IMPROVE_PATH)

    # Define training scenarios
    scenarios = {
        "Baseline": make_standard_loader(real_train_ds),
    }
    if vae_X_1 is not None:
        scenarios["Real+VAE_reproduce"] = make_standard_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(vae_X_1)])
        )
    if vae_X_2 is not None:
        scenarios["Real+VAE_improve"] = make_standard_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(vae_X_2)])
        )
    if gan_X_1 is not None:
        scenarios["Real+WGAN-GP_reproduce"] = make_standard_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(gan_X_1)])
        )
    if gan_X_2 is not None:
        scenarios["Real+WGAN-GP_improve"] = make_standard_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(gan_X_2)])
        )
    if diff_X_1 is not None:
        scenarios["Real+DIFF_reproduce"] = make_standard_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(diff_X_1)])
        )
    if diff_X_2 is not None:
        scenarios["Real+DIFF_improve"] = make_standard_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(diff_X_2)])
        )

    # Store results across all runs
    all_results = {}

    for run_idx in range(1, num_runs + 1):
        is_first_run = (run_idx == 1)
        print(f"\n" + "=" * 20 + f" RUN {run_idx}/{num_runs} " + "=" * 20)

        # Store trained models for ensemble evaluation
        trained_models = {}
        results = {}

        for name, train_loader in scenarios.items():
            current_model = model_class(num_classes=2).to(DEVICE)

            eval_res = train_and_evaluate(
                model=current_model,
                model_name=model_name,
                classweight=class_weights,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                scenario_name=f"{name}",
                verbose=is_first_run
            )

            results[name] = {
                "macro_F1": eval_res["macro_F1"],
                "AUROC": eval_res["AUROC"]
            }

            # Save trained model for ensemble construction
            trained_models[name] = current_model

            # Display misclassified samples only during the first run
            if is_first_run:
                plot_misclassified_samples(
                    false_positives=eval_res["false_positives"],
                    false_negatives=eval_res["false_negatives"],
                    scenario_name=name
                )

        # Display first-run performance summary
        if is_first_run:
            print("\n" + "=" * 55)
            print(f"{'RUN 1 RESULTS':^55}")
            print("=" * 55)
            print(f"{'Scenario':<28} | {'macro F1':>8} | {'AUROC':>7}")
            print("-" * 55)

            baseline_f1 = results.get("Baseline", {}).get("macro_F1", 0)
            for name, m in results.items():
                delta = m["macro_F1"] - baseline_f1
                sign = "+" if delta >= 0 else ""
                print(f"{name:<28} | {m['macro_F1']:>8.4f} | {m['AUROC']:>7.4f} ({sign}{delta:.3f})")
            print("=" * 55)

        # ==== Select the best model from each generation method ====
        # Compare F1 scores between reproduce and improve variants
        def get_best_scenario(results_dict, model_type):
            rep_key = f"Real+{model_type}_reproduce"
            imp_key = f"Real+{model_type}_improve"

            rep_f1 = results_dict.get(rep_key, {}).get("macro_F1", 0)
            imp_f1 = results_dict.get(imp_key, {}).get("macro_F1", 0)

            return rep_key if rep_f1 >= imp_f1 else imp_key

        # Build the ensemble scenario list
        ensemble_scenarios = ["Baseline"]

        for model_type in ["VAE", "WGAN-GP", "DIFF"]:
            best = get_best_scenario(results, model_type)
            if best in trained_models:
                ensemble_scenarios.append(best)

        # Keep only available trained models
        ensemble_models = {
            name: trained_models[name]
            for name in ensemble_scenarios
            if name in trained_models
        }

        print(f"\nEnsemble models: {list(ensemble_models.keys())}")

        # Run ensemble evaluation
        ensemble_result = run_ensemble_from_models(
            models_dict=ensemble_models,
            test_loader=test_loader,
            model_name=model_name,
            verbose=is_first_run
        )

        # Save results for the current run (including ensemble)
        all_results[f"run_{run_idx}"] = {
            **results,
            "ensemble": ensemble_result
        }

        # Save after each run to avoid losing progress due to crashes
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"Saved run {run_idx} results to {result_path}")


# ──  5. T-SNE VISUALIZATION ────────────────────────────────────────────────────────────────
name_map = {
    1: 'WGAN-GP_reproduce', 
    2: 'WGAN-GP_improve',
    3: 'VAE_reproduce',
    4: 'VAE_improve',
    5: 'DIFF_reproduce',
    6: 'DIFF_improve'
}

path_data = [
    GAN_REPRODUCE_PATH,
    GAN_IMPROVE_PATH,
    VAE_REPRODUCE_PATH,
    VAE_IMPROVE_PATH,
    DIFF_REPRODUCE_PATH,
    DIFF_IMPROVE_PATH
]

def get_embeddings(model, tensors, label_val, max_n=200):
    tensors = tensors[:max_n]
    if tensors.ndim == 4 and tensors.shape[-1] == 1:
        tensors = tensors.permute(0,3,1,2)
    elif tensors.ndim == 4 and tensors.shape[1] != 1:
        tensors = tensors
    with torch.no_grad():
        emb = model.embed(tensors.to(DEVICE)).cpu().numpy()
    return emb, np.full(len(emb), label_val)

def visualize_tsne(baseline_model, model_name):
    # Collect real samples from the training dataset
    real_covid_X = data['X_train'][y_train == 1]
    healthy_X = data['X_train'][y_train == 0]

    # Extract embeddings for REAL data
    emb_covid, lbl_covid = get_embeddings(baseline_model, real_covid_X, 0)      # Original COVID+ samples labeled as 0
    emb_healthy, lbl_healthy = get_embeddings(baseline_model, healthy_X, 1)      # Original Healthy samples labeled as 1

    # Initialize lists with the two real-data groups
    emb_list = [emb_covid, emb_healthy]
    lbl_list = [lbl_covid, lbl_healthy]

    # Extract embeddings for SYNTHETIC data
    for i, path in enumerate(path_data, 2):
        if os.path.exists(path):
            s = torch.load(path, map_location='cpu')
            emb_s, lbl_s = get_embeddings(baseline_model, s, i)
            emb_list.append(emb_s)
            lbl_list.append(lbl_s)

    # Concatenate all feature matrices and label arrays
    all_emb = np.concatenate(emb_list)
    all_lbl = np.concatenate(lbl_list)

    # Apply t-SNE dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(all_emb)

    # Define display labels corresponding to class indices
    labels_tsne = [
        'Real COVID+',
        'Real Healthy',
        'WGAN-GP_reproduce',
        'WGAN-GP_improve',
        'VAE_reproduce',
        'VAE_improve',
        'DIFF_reproduce',
        'DIFF_improve'
    ]

    # Define color palette corresponding to class indices
    colors_tsne = [
        '#1565C0',  # Real COVID+ - Dark blue
        '#8E24AA',  # Real Healthy - Magenta purple
        '#E53935',  # WGAN-GP_reproduce - Dark red
        '#FF8A80',  # WGAN-GP_improve - Light red
        '#2E7D32',  # VAE_reproduce - Dark green
        '#81C784',  # VAE_improve - Light green
        '#F9A825',  # DIFF_reproduce - Amber
        '#FFD54F'   # DIFF_improve - Light amber
    ]

    # 5. Generate the visualization
    fig, ax = plt.subplots(figsize=(10, 8)) 

    for lv in np.unique(all_lbl):
        mask = all_lbl == lv
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=colors_tsne[int(lv)],
            label=labels_tsne[int(lv)],
            alpha=0.6,
            s=25,
            edgecolors='none'
        )

    ax.legend(fontsize=11, markerscale=2, loc='upper right')

    # Title describing the full comparison scenario
    ax.set_title(
        't-SNE: Real (COVID+ vs Healthy) vs Synthetic Embeddings\n'
        '(Feature Space Visualization)',
        fontsize=13,
        fontweight='bold'
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'plot_03_tsne_{model_name}.png', dpi=120, bbox_inches='tight')
    plt.show()


# ──── 6. HYPOTHESIS TESTING ─────────────────────────────────────────
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group results by scenario
    scenarios_data = {}

    for run_id, run_data in data.items():
        for scenario, metrics in run_data.items():
            if scenario not in scenarios_data:
                scenarios_data[scenario] = {
                    "macro_F1": [],
                    "AUROC": []
                }

            scenarios_data[scenario]["macro_F1"].append(
                metrics.get("macro_F1", 0)
            )
            scenarios_data[scenario]["AUROC"].append(
                metrics.get("AUROC", 0)
            )

    return scenarios_data

def run_statistical_test(json_file_path):
    # Check whether the file exists
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return

    # Load data from the JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Define evaluation metrics and retrieve model names from the first run
    metrics = ["macro_F1", "AUROC"]
    first_run = list(data.keys())[0]
    models = list(data[first_run].keys())

    # Initialize a structure to aggregate results across runs
    # Example: extracted_data['Baseline']['macro_F1'] = [0.7268, 0.7182, ...]
    extracted_data = {m: {met: [] for met in metrics} for m in models}

    for run_id, model_dict in data.items():
        for model_name, metric_dict in model_dict.items():
            if model_name not in extracted_data:
                continue

            for met in metrics:
                if met in metric_dict:
                    extracted_data[model_name][met].append(metric_dict[met])

    # Perform paired t-tests
    results = []
    baseline_f1 = extracted_data["Baseline"]["macro_F1"]
    baseline_auc = extracted_data["Baseline"]["AUROC"]

    print("=" * 85)
    print(f"STATISTICAL TEST RESULTS FOR FILE: {json_file_path}")
    print("=" * 85)
    print(
        f"Baseline mean values -> macro_F1: {np.mean(baseline_f1):.4f} | "
        f"AUROC: {np.mean(baseline_auc):.4f}\n"
    )

    for model in models:
        if model == "Baseline":
            continue

        row = {"Model": model}

        for met in metrics:
            baseline_vals = extracted_data["Baseline"][met]
            model_vals = extracted_data[model][met]

            # Compute the mean score of the current model
            mean_base = np.mean(baseline_vals)
            mean_model = np.mean(model_vals)

            # Perform a two-tailed paired t-test
            t_stat, p_val = stats.ttest_rel(model_vals, baseline_vals)

            row[f"{met}_Mean"] = round(mean_model, 4)
            row[f"{met}_p-value"] = round(p_val, 4)

            # Determine statistical significance (alpha = 0.05)
            if p_val < 0.05:
                if mean_model > mean_base:
                    row[f"{met}_Result"] = "Significantly better (p < 0.05)"
                else:
                    row[f"{met}_Result"] = "Significantly worse (p < 0.05)"
            else:
                row[f"{met}_Result"] = "No significant difference (p >= 0.05)"

        results.append(row)

    # Format results as a Pandas table for cleaner output
    df_results = pd.DataFrame(results)

    print("--- macro_F1 Metric ---")
    print(
        df_results[
            ["Model", "macro_F1_Mean", "macro_F1_p-value", "macro_F1_Result"]
        ].to_string(index=False)
    )

    print("\n" + "-" * 85 + "\n")

    print("--- AUROC Metric ---")
    print(
        df_results[
            ["Model", "AUROC_Mean", "AUROC_p-value", "AUROC_Result"]
        ].to_string(index=False)
    )

    print("=" * 85)

def run_statistical_test_with_fixed_baseline(baseline_json_path, compare_json_paths):
    # Load baseline data
    baseline_data = load_data(baseline_json_path)

    # Load comparison models
    compare_models = {}
    for model_name, json_path in compare_json_paths.items():
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue
        compare_models[model_name] = load_data(json_path)

    # Identify scenarios for comparison
    scenarios = list(baseline_data.keys())

    # Perform statistical tests for each scenario
    for scenario in scenarios:
        if scenario not in baseline_data:
            continue

        print(f"\n{'=' * 100}")
        print(f"SCENARIO: {scenario}")
        print(f"{'=' * 100}")

        # Baseline statistics
        base_f1 = baseline_data[scenario]["macro_F1"]
        base_auroc = baseline_data[scenario]["AUROC"]

        print(f"\nBaseline:")
        print(f"  macro_F1:  {np.mean(base_f1):.4f} +- {np.std(base_f1):.4f}")
        print(f"  AUROC:     {np.mean(base_auroc):.4f} +- {np.std(base_auroc):.4f}")

        # Compare against each model
        results = []

        for model_name, model_data in compare_models.items():
            if scenario not in model_data:
                print(f"\nWarning: {model_name}: Scenario '{scenario}' not found")
                continue

            model_f1 = model_data[scenario]["macro_F1"]
            model_auroc = model_data[scenario]["AUROC"]

            # Check whether the number of runs matches
            if len(model_f1) != len(base_f1):
                print(
                    f"Warning: {model_name}: Number of runs does not match "
                    f"({len(model_f1)} vs {len(base_f1)})"
                )

                # Use the smaller number of runs
                min_len = min(len(model_f1), len(base_f1))
                model_f1 = model_f1[:min_len]
                model_auroc = model_auroc[:min_len]
                base_f1_trimmed = base_f1[:min_len]
                base_auroc_trimmed = base_auroc[:min_len]
            else:
                base_f1_trimmed = base_f1
                base_auroc_trimmed = base_auroc

            # Paired t-test
            t_stat_f1, p_val_f1 = stats.ttest_rel(
                model_f1,
                base_f1_trimmed
            )
            t_stat_auroc, p_val_auroc = stats.ttest_rel(
                model_auroc,
                base_auroc_trimmed
            )

            # Compute performance differences
            diff_f1 = np.mean(model_f1) - np.mean(base_f1_trimmed)
            diff_auroc = np.mean(model_auroc) - np.mean(base_auroc_trimmed)

            # Interpret F1 results
            if p_val_f1 < 0.05:
                if diff_f1 > 0:
                    f1_result = "Better"
                else:
                    f1_result = "Worse"
            else:
                f1_result = "No significant difference"

            # Interpret AUROC results
            if p_val_auroc < 0.05:
                if diff_auroc > 0:
                    auroc_result = "Better"
                else:
                    auroc_result = "Worse"
            else:
                auroc_result = "No significant difference"

            results.append({
                "Model": model_name,
                "F1_Mean": np.mean(model_f1),
                "F1_Std": np.std(model_f1),
                "F1_Diff": diff_f1,
                "F1_p-value": p_val_f1,
                "F1_Result": f1_result,
                "AUROC_Mean": np.mean(model_auroc),
                "AUROC_Std": np.std(model_auroc),
                "AUROC_Diff": diff_auroc,
                "AUROC_p-value": p_val_auroc,
                "AUROC_Result": auroc_result
            })

        # Display results as tables
        df = pd.DataFrame(results)

        print("\n--- macro_F1 METRIC ---")
        print(
            df[
                ["Model", "F1_Mean", "F1_Std", "F1_Diff", "F1_p-value", "F1_Result"]
            ].to_string(index=False)
        )

        print("\n--- AUROC METRIC ---")
        print(
            df[
                [
                    "Model",
                    "AUROC_Mean",
                    "AUROC_Std",
                    "AUROC_Diff",
                    "AUROC_p-value",
                    "AUROC_Result"
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 100)

# ──── ENTRY POINT ─────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  choices=["CNN", "EfficientNet", "all"],
                   default="all", help="Which model to train")
    p.add_argument("--num_runs", type=int, default=5,
                   help="Number of independent training runs (default: 5)")
    return p.parse_args()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_cnn = args.model in ("CNN",  "all")
    run_eff = args.model in ("EfficientNet",  "all")
    num_runs = args.num_runs
    
    if run_cnn:
        print("\n=== CNN ===")
        run_model(AudioCNN, "CNN", num_runs, "results_1.json")

        baseline_model = AudioCNN().to(DEVICE)
        baseline_model.load_state_dict(torch.load('CNN_Baseline.pth', map_location=DEVICE))
        baseline_model.eval()
        visualize_tsne(baseline_model, "CNN")

        run_statistical_test("results_1.json")
    if run_eff:
        print("\n=== EfficientNet-B0 ===")
        run_model(AudioEfficientNetB0, "EfficientNetB0", num_runs, "results_2.json")

        baseline_model = AudioEfficientNetB0().to(DEVICE)
        baseline_model.load_state_dict(torch.load('EfficientNetB0_Baseline.pth', map_location=DEVICE))
        baseline_model.eval()

        visualize_tsne(baseline_model, "EfficientNetB0")

        run_statistical_test("results_2.json")

    if args.model == "all":
        run_statistical_test_with_fixed_baseline('results_1.json', {'EfficientNetB0': 'results_2.json'})