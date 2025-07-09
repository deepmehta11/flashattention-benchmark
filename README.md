# FlashAttention Benchmark

A micro-benchmark comparing **vanilla attention** vs **FlashAttention-2.8** performance and memory usage on the publicly available GPT-J 6B model.

---

## 📊 Results

| Configuration          | Throughput (tokens/sec) | Peak GPU Memory (GB) |
| ---------------------- | ----------------------- | -------------------- |
| **Vanilla Attention**  | 8,729                   | 34.71                |
| **FlashAttention-2.8** | 14,289                  | 24.21                |

* **Speed‑up:** \~1.64× faster forward pass
* **Memory Savings:** \~30% reduction in peak GPU memory

---

## 🚀 Quickstart

### Prerequisites

* A machine with an NVIDIA GPU (A100/H100 recommended)
* [Docker](https://docs.docker.com/get-docker/) installed for containerized runs
* Or Python 3.11+ and `pip`

### Clone the Repo

```bash
git clone https://github.com/<USERNAME>/flashattention-benchmark.git
cd flashattention-benchmark
```

### Run with Docker

Build the image and run benchmark:

```bash
docker build -t flashattention-bench .

docker run --gpus all --rm flashattention-bench
```

Results will be written to `results.csv` and displayed in the container logs.

### Run Locally (without Docker)

1. **Create a Python environment**:

   ```bash
   conda create -n flash3 python=3.11 -y
   conda activate flash3
   ```

2. **Install dependencies**:

   ```bash
   pip install torch==2.7.1+cu126 torchvision --extra-index-url https://download.pytorch.org/whl/cu126
   pip install flash-attn==2.8.0.post2 transformers accelerate bitsandbytes
   ```

3. **Run the benchmark**:

   ```bash
   python benchmark.py
   cat results.csv
   ```

---

## 🔧 Benchmark Script

* **File:** `benchmark.py`
* **Model:** `EleutherAI/gpt-j-6B` (FP16)
* **Batch:** 1, **Sequence length:** 2048
* **Steps:** 200 forward passes each for vanilla vs FlashAttention-2.8

The script measures average **seconds per step** and **peak GPU memory**, outputting a CSV for easy plotting.

---

## 📁 Repo Structure

```text
flashattention-benchmark/
├── README.md
├── .gitignore
├── benchmark.py       # Timing script
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container for reproducibility
├── results.csv        # Sample output
├── assets/            # Generated plots
│   ├── throughput_comparison.png
│   └── memory_usage_comparison.png
```

---

## 🤝 Contributing

Feel free to open issues or PRs to:

* Add support for custom models or attention implementations
* Integrate profiling traces (Nsight, PyTorch Profiler)
* Extend to distributed benchmarks (multi-GPU via FSDP/ZeRO)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
