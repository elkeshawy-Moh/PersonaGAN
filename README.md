<div align="center">
  <img src="assets/banner.png" alt="PersonaGAN Banner" width="100%" />
</div>

# 🎭 PersonaGAN: High-Resolution Human Face Synthesis

[![Open In Kaggle](https://kaggle.com)](https://kaggle.com)
![Python](https://shields.io)
![TensorFlow](https://shields.io)
![Gradio](https://shields.io)
![License](https://shields.io)

**PersonaGAN** is a deep generative adversarial network (GAN) designed to synthesize realistic `128x128` human faces from random latent vectors. Built with TensorFlow and Keras, the model incorporates modern GAN stabilization techniques—including **Spectral Normalization**, **Minibatch Standard Deviation**, **DiffAugment**, **Mode Seeking Regularization**, and **Exponential Moving Average (EMA)**—to achieve stable convergence and diverse facial generation without mode collapse.

---

## ✨ Key Features & Architecture

* **Generator (`128x128x3`):** Uses bilinear `UpSampling2D` followed by dual `3x3` convolutions, `BatchNormalization`, and `LeakyReLU` activations to prevent checkerboard artifacts.
* **Discriminator:** Employs **Spectral Normalization** across all convolutional and dense layers to enforce the 1-Lipschitz constraint, paired with a custom **Minibatch Standard Deviation** layer to encourage sample diversity.
* **Training Stabilization:**
  * **Differentiable Augmentation (`DiffAugment`):** Applies random horizontal flips, translations (pad & crop), and brightness jittering to both real and generated images.
  * **Two-Time-Scale Update Rule (TTUR):** Uses separate learning rates for Generator (`1.5e-4`) and Discriminator (`2.0e-4`) with `N_CRITIC = 2`.
  * **Mode Seeking Loss:** Penalizes the generator if distinct latent vectors map to overly similar facial features.
  * **EMA Weights:** Maintains a shadow moving average (`decay = 0.999`) of generator weights for crisp inference results.
  * **Mixed Precision (`mixed_float16`):** Accelerates training throughput on NVIDIA T4 GPUs.
* **Interactive Web Demo:** Includes a **Gradio** interface allowing users to generate faces on demand using a custom random seed and truncation/diversity slider.

---

## 📊 Training Results & Progression

The model was trained on a subset of **15,000 aligned face images** from the **CelebA Dataset** (`128x128` resolution, normalized to `[-1, 1]`).

### Sample Progression

| Early Stage (`Step 500`) | Mid Training (`Step 10,000`) | Final EMA Output (`Step 30,000`) |
| :---: | :---: | :---: |
| ![Step 500](assets/faces_step_00500.png) | ![Step 10000](assets/faces_step_10000.png) | ![Step 30000](assets/faces_step_30000.png) |

### Loss Convergence
![Training Loss Curve](assets/training_losses.png)

---

## 🚀 Getting Started

### 1. Run Directly on Kaggle (Recommended)
Click the **"Open In Kaggle"** badge at the top of this page to launch the notebook with free GPU acceleration and pre-attached CelebA data.

### 2. Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd PersonaGAN
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow numpy pillow matplotlib gradio
   ```

3. **Launch the Gradio Demo (using pre-trained weights):**
   Place `generator_human_faces.weights.h5` in the root directory and run:
   ```python
   import gradio as gr
   import numpy as np
   import tensorflow as tf
   # Load create_generator() from the notebook
   generator = create_generator(latent_dim=128, channels=3)
   generator.load_weights("generator_human_faces.weights.h5")
   ```

---

## 📁 Repository Structure

```text
PersonaGAN/
├── PersonaGAN.ipynb                     # Main training, evaluation, and Gradio notebook
├── generator_human_faces.weights.h5     # Final trained EMA Generator weights (or via Release)
├── assets/
│   ├── banner.png                       # Project header banner
│   ├── faces_step_00500.png             # Early generation sample grid
│   ├── faces_step_10000.png           # Mid-training sample grid
│   ├── faces_step_30000.png           # Final high-quality sample grid
│   └── training_losses.png              # Generator vs. Discriminator loss plot
└── README.md                            # Project documentation
```

---

## 🙏 Acknowledgments
* Dataset: [CelebFaces Attributes (CelebA) Dataset](https://kaggle.com)
* Techniques inspired by *ProGAN* (MinibatchStdDev), *SNGAN* (Spectral Normalization), and *DiffAugment*.
