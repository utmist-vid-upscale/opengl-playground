# Benchmark Analysis

All benchmarks on NVIDIA RTX A4000 (16 GB VRAM), Intel Xeon W-2265, Linux 5.15.0-139-generic, NVIDIA driver 535.230.02. Vsync off. 60-frame warmup, averages reported every 60 frames.

---

## Raw Runs

### 800x600 — C benchmark (`bench.cu`)

**CPU compute → `glTexSubImage2D` upload:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120           22.188       0.231       0.625      23.044        43
180           22.138       0.230       0.628      22.996        43
240           22.171       0.225       0.632      23.029        43
300           22.114       0.229       0.626      22.969        44
```

**CUDA kernel → CUDA-GL interop:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120            0.020       0.133       0.190       0.344      2910
180            0.020       0.119       0.152       0.290      3447
240            0.020       0.171       0.229       0.419      2385
300            0.020       0.130       0.174       0.324      3082
360            0.016       0.044       0.057       0.118      8505
```

### 1920x1080 — C benchmark (`bench.cu`)

**CPU compute → `glTexSubImage2D` upload:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120           95.617       2.715       0.034      98.366        10
```

**CUDA kernel → CUDA-GL interop:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120            0.068       0.052       0.012       0.133      7530
180            0.068       0.045       0.001       0.115      8693
```

### 1920x1080 — PyTorch + `cv2.imshow` (`bench_cv2.py`)

```
frame       generate    download     display       total       FPS
                (ms)        (ms)        (ms)        (ms)
----------------------------------------------------------------------
120            2.166       2.068      13.513      17.747        56
180            2.178       2.167      13.817      18.162        55
240            2.218       2.239      13.909      18.366        54
300            2.191       2.254      13.567      18.011        56
360            2.170       2.427      13.007      17.605        57
420            2.144       2.376      11.772      16.292        61
480            2.158       2.052      12.812      17.022        59
540            2.168       1.742      12.575      16.485        61
600            2.181       2.276      13.321      17.777        56
660            2.135       2.509      11.921      16.566        60
720            2.152       2.177      13.137      17.466        57
```

### 1920x1080 — PyTorch + CUDA-GL interop (`bench_gl.py`)

```
frame       generate     display       total       FPS
                (ms)        (ms)        (ms)
-------------------------------------------------------
120            2.085       0.197       2.282       438
180            2.108       0.306       2.413       414
240            2.082       0.179       2.260       442
300            2.093       0.170       2.263       442
360            2.082       0.177       2.259       443
420            2.085       0.176       2.261       442
```

---

## Summary — all runs at 1920x1080

| Method | Generate (ms) | Download (ms) | Upload/Display (ms) | Total (ms) | FPS |
|--------|:---:|:---:|:---:|:---:|:---:|
| **CPU compute → `glTexSubImage2D`** (`bench cpu`) | 95.62 | — | 2.75 | 98.37 | 10 |
| **CUDA kernel → CUDA-GL interop** (`bench gpu`) | 0.07 | — | 0.06 | 0.12 | 7530 |
| **PyTorch GPU → `tensor.cpu()` → `cv2.imshow`** (`bench_cv2.py`) | 2.20 | 2.35 | 13.20 | 17.75 | 56 |
| **PyTorch GPU → `data_ptr()` → CUDA-GL interop** (`bench_gl.py`) | 2.09 | — | 0.20 | 2.29 | 440 |

For a real ML display pipeline (bottom two rows), the display overhead comparison:

| | cv2.imshow | CUDA-GL interop | Speedup |
|---|:---:|:---:|:---:|
| **Display overhead** | 15.55 ms | 0.20 ms | **~78x** |
| **% of 33ms budget (30 FPS)** | 47% | 0.6% | |
| **Time left for inference** | 17.4 ms | 32.8 ms | |

---

## Why is `cv2.imshow` so slow?

```
GPU VRAM → PCIe → CPU RAM (tensor.cpu)
  → OpenCV BGR conversion (CPU memcpy)
    → Qt event loop + widget paint
      → X11 shared memory
        → compositor uploads to GPU texture (PCIe again)
          → GPU composites → display
```

Two PCIe bus crossings, three memory copies, two event loops. The CUDA-GL interop path:

```
GPU VRAM (tensor) → GPU VRAM (GL texture) → display
```

One VRAM-to-VRAM copy (~0.05ms), one GL draw call (~0.01ms), buffer swap.
