# Pupil Tracking GPU

Welcome to the **Pupil Tracking GPU** repository! This project is designed to leverage GPU acceleration for efficient and accurate pupil tracking in real-time applications. Whether you're working on eye-tracking research, human-computer interaction, or medical diagnostics, this repository provides a robust foundation for your needs.
> **Disclaimer**: This software is developed by the University of Murcia and is intended solely for academic and research purposes. It must not be used for commercial or business applications.

## Features

- **GPU Acceleration**: Harness the power of modern GPUs for high-performance pupil tracking.
- **Real-Time Processing**: Achieve low-latency tracking suitable for real-time applications.
- **Customizable**: Easily adapt the codebase to your specific requirements.
- **Open Source**: Contribute and collaborate with the community to improve the project.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/PupilTrackingGPU.git
    cd PupilTrackingGPU
    ```

2. Install dependencies:
    ```bash
    sudo apt-get install libopencv-dev
    ```

3. Ensure you have the necessary GPU drivers and CUDA toolkit installed.

## Usage

Run the main script to test tracking:
```bash
./build/pupil_tracking resources/testimage1.png <(measure_time)1|0> <(platform)gpu|cpu> <num_repetitions> <(algorithm)else|else_greedy_i|else_greedy_ii|excuse|excuse_greedy_i|excuse_greedy_ii>
```