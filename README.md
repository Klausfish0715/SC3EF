# SC³EF

Official implementation of our paper:
**SC³EF: A Joint Self-Correlation and Cross-Correspondence Estimation Framework for Visible and Thermal Image Registration**  
*Accepted by IEEE Transactions on Intelligent Transportation Systems (T-ITS)*

[https://arxiv.org/abs/2504.12869] Paper Link

[https://github.com/Klausfish0715/SC3EF] Project Page

## 🧩 Introduction
This repository contains the source code for SC³EF, a novel joint self-correlation and cross-correspondence estimation framework to improve RGB-T image registration by leveraging both local representative features and global contextual cues. A convolution-transformer-based pipeline is developed to extract intra-modality self-correlation and estimate inter-modality correspondences. Considering that human observers use both local and global cues to establish correspondences, a convolution-based local feature extractor and a transformer-based global self-correlation encoder are introduced. The extracted features and encoded correlations are then utilized to estimate inter-modality correspondences, which are merged and progressively refined using a hierarchical optical flow decoder. Experimental results on benchmark RGB-T datasets show that SC³EF outperforms state-of-the-art methods. Furthermore, it demonstrates competitive generalization capabilities across challenging scenarios, including large parallax, severe occlusions, adverse weather, and other cross-modal datasets (e.g., RGB-N and RGB-D).

## 📦 Requirements
- Python 3.8+
- PyTorch 1.10.1+
- timm 0.4.12
- OpenCV, NumPy, etc.

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Run Demo

## 📁 Project Structure

## 📜 Citation
If you find this work helpful in your research, please cite our paper:
```bibtex
@article{tong2025sc,
  title={SC3EF: A Joint Self-Correlation and Cross-Correspondence Estimation Framework for Visible and Thermal Image Registration},
  author={Tong, Xi and Luo, Xing and Yang, Jiangxin and Li, Xin and Cao, Yanpeng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```
