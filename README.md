# Liquid Impulse Neural Network (LNN-SNN) for Multimodal Information Fusion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A hybrid neural network model integrating Liquid Neural Networks (LNN) and Spiking Neural Networks (SNN) for robust fault diagnosis through heterogeneous fusion of multimodal signals (e.g., vibration, SP image).

## 📦 Repository Structure
LNN_SNN/
├── LNN_SNN.py # Core model architecture
├── TFDA_Visual.py # Time-Frequency Domain Attention visualization tools
├── Train_LNNS.py # Training script for LNN-SNN fusion
├── Train_SNN.py # Training script for standalone SNN
├── best_lnn_snn_model.pth # Pre-trained weights
└── README.md # This document


## ✨ Key Features

- **Multimodal Fusion**: Combines temporal (LNN) and event-driven (SNN) processing for heterogeneous sensor data
- **Interpretable Analysis**: 
  - Time-Frequency Domain Attention (TFDA) for signal periodicity localization
  - Grad-CAM++ activation maps for fault feature visualization
- **Noise Robustness**: Validated under SNR=-6dB to 0dB conditions (see [performance analysis](#-performance))

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your_username/LNN_SNN.git
cd LNN_SNN
pip install -r requirements.txt

📚 Citation

@article{yourpaper2023,
  title={A liquid-impulse neural network model based on heterogeneous fusion of multimodal information for interpretable rotating machinery fault diagnosis},
  author={Keshun You, Yingkui Gu, Haidong Shao and Yajun Wange},
  year={2025}
}
