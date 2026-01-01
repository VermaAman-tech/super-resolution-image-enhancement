# Super-Resolution Image Enhancement

This project implements an advanced deep learning model for **4x Image Super-Resolution (ISR)**, upscaling low-resolution (LR) images to high-quality high-resolution (HR) versions. The architecture is inspired by the state-of-the-art **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)**, specifically utilizing **Residual-in-Residual Dense Blocks (RRDB)**.

## 🚀 Key Features

- **4x Upscaling**: Efficiently increases image resolution by a factor of 4.
- **InsaneSRNet Architecture**: A custom deep neural network featuring:
  - **Residual Dense Blocks (RDB)**: Extracts local features through dense connections.
  - **Residual-in-Residual Dense Blocks (RRDB)**: Enhances feature representation without increasing computational overhead significantly.
  - **Global Skip Connections**: Combines high-level features with upscaled versions of the original image to preserve structural details.
  - **PixelShuffle**: Used for efficient upsampling at the end of the generator.
- **Mixed Precision Training**: Utilizes `torch.cuda.amp` for faster training and reduced memory footprint on modern GPUs.
- **Evaluation Metrics**: Tracks performance using:
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **SSIM (Structural Similarity Index)**
  - **Joint Metric**: A weighted combination ($40 \times SSIM + PSNR$) for better correlation with human perception.

## 🏗️ Model Architecture: InsaneSRNet

The model consists of four main parts:
1.  **Feature Extraction (Head)**: Initial convolutional layer.
2.  **Mapping (Trunk)**: A sequence of 5 RRDBs to extract deep semantic features.
3.  **Upsampling**: Two `UpscaleBlock` stages, each performing 2x upscaling using PixelShuffle and LeakyReLU.
4.  **Reconstruction (Tail)**: Final convolution to produce the 3-channel RGB image.

A global bilinear interpolation skip connection ensures that the model focuses on learning the high-frequency residuals rather than the entire image content.

## 📊 Dataset & Training

- **Data**: Handled via custom PyTorch `Dataset` and `DataLoader` classes.
- **Optimization**: Adam optimizer with a learning rate of `1e-4`.
- **Loss Function**: Mean Squared Error (MSE) loss.
- **Hardware**: Configured to run on CUDA-enabled GPUs with fallback to CPU.

## 🛠️ Setup and Usage

1.  **Requirements**:
    - Python 3.x
    - PyTorch & Torchvision
    - OpenCV (`cv2`)
    - Scikit-Image
    - NumPy, Matplotlib, Pandas, tqdm

2.  **Training**:
    The notebook `notebook3b74e19863 (2).ipynb` contains the complete pipeline from data loading to model training and evaluation.
    ```bash
    # Ensure your LR/HR image paths are correctly set in the notebook
    TRAIN_LR_DIR = "path/to/lr"
    TRAIN_HR_DIR = "path/to/hr"
    ```

3.  **Inference**:
    The model saves weights (`best.pth`, `latest.pth`) which can be reloaded for upscaling new images.

## 📈 Evaluation Results

The project includes an evaluation function that computes average metrics over a validation set. Typical outputs include:
- `PSNR: ~25.00+`
- `SSIM: ~0.80+`
- `Joint Metric: ~60.00+`

---
*Created as part of the GNR 638 project.*