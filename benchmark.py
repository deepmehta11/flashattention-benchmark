# benchmark.py — vanilla vs FlashAttention-2.8 micro-benchmark (using attn_implementation="flash_attention_2")

import time
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID  = "EleutherAI/gpt-j-6B"    # open 6B-param model
BATCH     = 1
SEQ_LEN   = 2048
STEPS     = 200                       # ~3–6s on A100
DEVICE    = "cuda"
DTYPE     = torch.float16
# ──────────────────────────────────────────────────────────────────────────────

def load_model(use_flash: bool):
    """
    Load the model with either:
     - eager attention (vanilla)
     - flash_attention_2 (FlashAttention v2.8)
    """
    impl = "flash_attention_2" if use_flash else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation=impl,
    )
    model.eval()

    # create dummy input
    inp = torch.randint(
        0,
        model.config.vocab_size,
        (BATCH, SEQ_LEN),
        device=DEVICE,
    )
    return model, inp

def run_benchmark(model, inp):
    """Warm up, then measure STEPS forward passes."""
    with torch.no_grad():
        for _ in range(10):
            _ = model(inp)

    times = []
    for _ in range(STEPS):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(inp)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg_sec = sum(times) / len(times)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return avg_sec, round(peak_gb, 2)

def main():
    results = []
    for use_flash in (False, True):
        # reset memory stats
        torch.cuda.reset_peak_memory_stats()

        model, inp = load_model(use_flash=use_flash)
        sec, gb = run_benchmark(model, inp)

        results.append({
            "flash2.8": use_flash,
            "sec_per_step": sec,
            "gb_peak": gb,
        })

        # cleanup
        del model, inp
        torch.cuda.empty_cache()

    # write CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("✅ Done! See results.csv")

if __name__ == "__main__":
    main()