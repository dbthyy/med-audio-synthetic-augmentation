# 2. GENERATING MODELS STRUCTURE

generative_audio/
├── __init__.py
├── config.py                     ← Tất cả hyperparameter & device
├── main.py                       ← CLI entry point
│
├── data/
│   ├── __init__.py
│   └── datasets.py               ← AudioDataset, SyntheticDataset, make_augmented_loader
│
├── models/
│   ├── __init__.py
│   ├── vae.py                    ← VAEEncoder, VAEDecoder, VAE
│   ├── wgan.py                   ← Generator, Discriminator, gradient_penalty
│   └── diffusion.py              ← SinusoidalEmbedding, UNet, Diffusion
│
├── training/
│   ├── __init__.py
│   ├── train_vae.py              ← train_vae()
│   ├── train_wgan.py             ← train_wgan_gp()
│   └── train_diffusion.py        ← train_diffusion()
│
├── evaluation/
│   ├── __init__.py
│   └── fad.py                    ← ResNetEmbeddingExtractor, compute_fad
│
└── utils/
    ├── __init__.py
    ├── tracking.py               ← MetricsLogger, FADAdaptiveTracker, adapt callbacks
    └── plotting.py               ← plot_metrics()