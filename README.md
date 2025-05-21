<p align="center">
  <!-- Generic Project Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic"><img src="https://img.shields.io/badge/PupilTrackingGPUPublic-v1.0.0-blueviolet"/></a>
  <!-- Code Repository Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic"><img src="https://img.shields.io/badge/code-Source-yellowgreen"/></a>
  <!-- License Badge - Custom for Academic Use -->
  <a href="#license"><img src="https://img.shields.io/badge/license-Academic%20%26%20Research%20Only-red"/></a>
  <!-- Last Commit Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic/commits/main"><img src="https://img.shields.io/github/last-commit/artuppp/PupilTrackingGPUPublic"/></a>
  <br>
  <!-- Stars Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic/stargazers"><img src="https://img.shields.io/github/stars/artuppp/PupilTrackingGPUPublic?style=social"/></a>
  <!-- Forks Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic/network/members"><img src="https://img.shields.io/github/forks/artuppp/PupilTrackingGPUPublic?style=social"/></a>
  <!-- Watchers Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic/watchers"><img src="https://img.shields.io/github/watchers/artuppp/PupilTrackingGPUPublic?style=social"/></a>
  <!-- Open Issues Badge -->
  <a href="https://github.com/artuppp/PupilTrackingGPUPublic/issues"><img src="https://img.shields.io/github/issues/artuppp/PupilTrackingGPUPublic"/></a>
</p>

<p align="center">
  <!-- Optional: Add a logo here if you have one -->
  <!-- <img src="path/to/your/logo.png" alt="PupilTrackingGPU Logo" width="200"/> -->
</p>

<h1 align="center">Pupil Tracking GPU <sub>v1.0.0</sub></h1>

<p align="center">
  <i>Harnessing GPU acceleration for efficient and accurate pupil tracking in real-time applications.</i>
</p>
<hr>

## 📢 Disclaimer

> This software is developed by the **University of Murcia** and is intended solely for **academic and research purposes**. It **must not** be used for commercial or business applications.

<p align="center">
  <!-- TODO: Add a cool GIF or screenshot of the tracker in action! -->
  <img src='https://univmurcia-my.sharepoint.com/:i:/g/personal/arturo_vicentej_um_es/ER33LoS62rNLlU2OGk7VTYUBnuzMtSwkXQRi-Nn0wDNabQ?e=BfDwxp' width='750'>
  <!-- <i>A placeholder for a demonstration GIF or image.</i> -->
</p>

## 🌟 Introduction

Welcome to the **Pupil Tracking GPU** repository! This project is designed to leverage the power of modern GPUs for high-performance, real-time pupil tracking. Whether you're involved in eye-tracking research, developing advanced human-computer interaction (HCI) systems, or exploring applications in medical diagnostics, this repository offers a robust and adaptable foundation. Our goal is to provide a tool that is both powerful and accessible to the research community.

## ✨ Features

*   🚀 **GPU Acceleration**: Utilizes CUDA for significant performance gains, enabling complex computations at high frame rates.
*   ⏱️ **Real-Time Processing**: Achieves low-latency tracking suitable for interactive and time-sensitive applications.
*   🔧 **Customizable Algorithms**: Implements multiple pupil detection algorithms (ELSE, EXCUSE, and their greedy variants) allowing users to choose based on their specific needs for accuracy and speed.
*   🛠️ **Flexible Platform**: Supports execution on both GPU and CPU for broader compatibility and testing.
*   📊 **Performance Measurement**: Integrated option to measure execution time for benchmarking and optimization.
*   📖 **Open Source**: Developed for the academic and research community, encouraging contributions and collaborative improvement.

## ⚙️ Installation

Follow these steps to get the Pupil Tracking GPU software up and running:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/artuppp/PupilTrackingGPU.git
    cd PupilTrackingGPU
    ```

2.  **Install Dependencies**:
    This project relies on OpenCV.
    ```bash
    sudo apt-get update
    sudo apt-get install libopencv-dev
    ```

3.  **GPU Setup**:
    *   Ensure you have a CUDA-compatible NVIDIA GPU.
    *   Install the latest NVIDIA drivers for your GPU.
    *   Install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) compatible with your drivers and project requirements.

4.  **Build the Project**:
    A `Makefile` is provided for easy compilation.
    ```bash
    make
    ```
    This will create the executable in the `build/` directory.

## 🚀 Usage

The main executable `pupil_tracking` can be run from the command line with several arguments to control its behavior.

**Synopsis**:
```bash
./build/pupil_tracking <image_path> <measure_time> <platform> <num_repetitions> <algorithm>
