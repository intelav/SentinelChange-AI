# 🛰️ SentinelChange-AI  
**Copernicus Sentinel-2 Change Detection with GPU-Accelerated Deep Learning (ChangeStar + SAM + ESRGAN)**

![LEVIR-CD Example](results/levir_cd_1.png)
*High-resolution LEVIR-CD benchmark — ChangeStar fine-tuned baseline*

![Copernicus Example](results/copernicus_cd_1.png)
*Copernicus Sentinel-2 (10 m) experiment — work in progress, Ground Truth Refinement , Model Re-Training , GPU optimization ongoing*

---

## 🌍 Overview

**SentinelChange-AI** is an applied research pipeline for **multi-temporal satellite change detection** using **Copernicus Sentinel-2 imagery**.  
It integrates **ChangeStar** for coarse semantic change prediction, **Segment Anything Model (SAM)** for precise boundary refinement, and **ESRGAN/RRDBNet** for super-resolving 10 m scenes to near-aerial visual quality.

Beyond accuracy, this repository emphasizes **GPU acceleration, profiling, and workload optimization** — ideal for researchers, practitioners, and AI engineers exploring **geospatial deep learning on high-volume remote-sensing data**.

---

## 🔑 Key Features

- 🛰️ **Copernicus Sentinel-2 Change Detection Pipeline** (10 m resolution)  
- ⚙️ **ChangeStar Deep Change-Detection Backbone** (Zheng et al., CVPR 2022)  
- ✂️ **SAM-Based Label Refinement** for high-quality masks  
- 🔍 **Automatic Sentinel-2 Pairing** by tile ID & year (MGRS)  
- 🧠 **Super-Resolution (ESRGAN / RRDBNet)** for visual enhancement  
- 🚀 **GPU-Optimized Training & Inference** workflows (CUDA / TorchRun)  
- 📊 **Benchmark-grade workloads** for Nsight profiling and GPU efficiency studies  

---

## 🧭 Attribution

This repository builds upon the open-source [**ChangeStar** framework by Zheng et al. (CVPR 2022)](https://github.com/Z-Zheng/ChangeStar).  
Model definitions, configuration templates, and baseline training logic are reused under their MIT license.  
All **Copernicus Sentinel-2 processing, SAM integration, ESRGAN upscaling**, and **GPU acceleration workflows** were independently developed by **Atul Vaish**.

---

## 📂 Repository Structure

```
SentinelChange-AI/
├── ChangeStar/                       # Forked base framework (Zheng et al.)
│   ├── configs/                      # Model configs for training/fine-tuning
│   ├── module/                       # Core model architectures
│   ├── generate_labels_sam.py        # SAM label generation (1024×1024 tiles)
│   ├── generate_labels_sam_upscaled.py
│   ├── inference_sentinel.py         # Tile-wise inference visualization
│   ├── train_sup_change.py           # Supervised change-detection training
│   ├── runtraining.sh                # TorchRun GPU training launcher
│   └── ...
│
├── pair_image_files.py               # Pair Sentinel-2 SAFE scenes by year/tile
├── process_images.py                 # Convert .SAFE → RGB tiles (10 m)
├── upscale_images.py                 # ESRGAN upscaling to ×4 resolution
├── RRDBNet_arch.py                   # RRDBNet architecture for ESRGAN
├── unzip_dataset.py                  # Extract all .SAFE.zip archives
├── results/
│   ├── levir_cd_1.png                # Benchmark reference result
│   └── copernicus_cd_1.png           # Copernicus 10 m sample result
└── datasets/                         # Input SAFE files & processed tiles
```

---

## 🧠 End-to-End Workflow

### 1️⃣ Unzip Copernicus `.SAFE` archives  
```bash
python unzip_dataset.py
```

### 2️⃣ Pair multi-temporal Sentinel-2 scenes by tile ID and year  
```bash
python pair_image_files.py
```

### 3️⃣ Convert spectral bands to RGB tiles (10 m)  
```bash
python process_images.py
```

### 4️⃣ (Optional) Enhance tiles using ESRGAN ×4 super-resolution  
```bash
python upscale_images.py
```

### 5️⃣ Generate SAM-refined labels from ChangeStar predictions  
```bash
python ChangeStar/generate_labels_sam.py
# or for upscaled data
python ChangeStar/generate_labels_sam_upscaled.py
```

### 6️⃣ Train or fine-tune ChangeStar on your dataset  
```bash
bash ChangeStar/runtraining.sh
```

### 7️⃣ Visualize predictions and ground truth  
```bash
python ChangeStar/inference_sentinel.py
```

---

## ⚙️ Environment Setup

```bash
conda create -n change_detection python=3.9
conda activate change_detection

pip install torch torchvision torchaudio
pip install ever-alpha tqdm opencv-python pillow rasterio matplotlib segment-anything
pip install numpy urllib3
```

> For ESRGAN upscaling, ensure GPU support and install **PyTorch ≥ 2.0** with CUDA 11.x or higher.  
> For ChangeStar training, recommended GPUs: RTX 3060+, A100, or Jetson Orin for edge experiments.

---

## 🛰️ Datasets

- **LEVIR-CD** — 30 cm high-resolution aerial imagery dataset (benchmark for validation).  
- **Copernicus Sentinel-2** — 10 m multispectral imagery (ESA Copernicus Open Access Hub).  
  - Automatically paired by **MGRS Tile ID** and **acquisition year**.  
  - RGB tiles are created using bands B04, B03, B02.  
  - SAM-refined masks used as weak supervision for fine-tuning.

Experiments on Copernicus data are **in progress**; visual quality and segmentation precision improve significantly when super-resolved with ESRGAN and retrained with ChangeStar.

---

## 🚀 GPU Acceleration & Profiling

Training and inference workloads are ideal for **GPU utilization analysis** and **Nsight profiling**.

Example profiling command:
```bash
nsys profile -t cuda,nvtx -o runs/profile bash ChangeStar/runtraining.sh
```

You can track:
- **SM efficiency & occupancy**
- **Tensor Core utilization**
- **DRAM throughput**
- **Kernel launch latency**
- **GPU power draw & timeline traces**

This repository forms part of a broader **Agentic GPU Optimization** research effort — profiling model workloads to enable autonomous sense–think–act–learn optimization loops for AI acceleration.

---

## 🧩 Research & Applications

- 🌾 **Land-use and Land-cover change analysis**  
- 🏙️ **Urban expansion and infrastructure mapping**  
- 🌊 **Flood and disaster impact assessment**  
- 🌋 **Environmental monitoring (deforestation, mining, glaciers)**  
- 🛰️ **GPU benchmarking for Earth Observation AI workloads**

---

## 🧱 Roadmap

| Milestone | Description | Status |
|------------|-------------|--------|
| ✅ Baseline Copernicus Change Detection | End-to-end pipeline completed | Done |
| ✅ ESRGAN Super-Resolution | Working upscaler integrated | In Progress |
| 🧩 SAM Integration | Precise label refinement | In Progress |
| 🧠 DINOv2 + SAM Fusion | Improved embedding quality | In Progress |
| ⚙️ GPU Profiling Integration | Nsight metrics + agentic optimization | In Progress |
| 🌐 Multi-sensor Fusion | Extend to Sentinel-1 & Landsat-8 | Planned |

---

## 🏗️ Work in Progress

- 🌐 Fusion of **DINOv2 + SAM** for adaptive feature extraction  
- 🔬 Nsight-driven **GPU workload profiling module**  
- 🧩 Integration with **agentic GPU optimization** (Sense → Think → Act → Learn)  
- 🛰️ Expansion to **Sentinel-1/3 multimodal change detection**

---

## 📄 License

This project adopts the **MIT License**, consistent with the original [ChangeStar](https://github.com/Z-Zheng/ChangeStar).  
All additional modules authored by **Atul Vaish** are released under the same terms.

---

## ✨ Citation

If you use or reference this work, please cite both the original ChangeStar paper and this applied project:

```text
@inproceedings{zheng2022changestar,
  title={ChangeStar: A Universal Change Detection Framework},
  author={Zheng, Zilong and Zhong, Yanfei and Wang, Zheng and Li, Zhen and Zhang, Liangpei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@software{vaish2025_sentinelchangeai,
  author = {Atul Vaish},
  title  = {SentinelChange-AI: Copernicus Sentinel-2 Change Detection with GPU-Accelerated Deep Learning},
  year   = {2025},
  url    = {https://github.com/avaish/SentinelChange-AI}
}
```

---

## 👨‍💻 Author

**Atul Vaish**  
Independent Applied AI & Geospatial Research  
GPU Optimization | Satellite AI | Edge Intelligence  
🌐 [https://aifusion.in](https://aifusion.in)  
📧 atul7911@gmail.com  
📍 India | EU Collaboration Ready
