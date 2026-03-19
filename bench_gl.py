"""
Benchmark: GPU tensor → CUDA-GL interop display path.

Uses a small C/CUDA shared library (libgl_display.so) to display
a PyTorch CUDA tensor directly via OpenGL, without ever touching the CPU.

Compare results with bench_cv2.py (tensor.cpu().numpy() + cv2.imshow).
"""

import ctypes
import os
import torch
import time
import math

WIDTH = 1920
HEIGHT = 1080
WARMUP_FRAMES = 60
REPORT_INTERVAL = 60

# Load the shared library
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgl_display.so")
lib = ctypes.CDLL(lib_path)

# Define function signatures
lib.gl_display_init.argtypes = [ctypes.c_int, ctypes.c_int]
lib.gl_display_init.restype = ctypes.c_int

lib.gl_display_show_frame.argtypes = [ctypes.c_ulonglong]
lib.gl_display_show_frame.restype = ctypes.c_int

lib.gl_display_should_close.restype = ctypes.c_int

lib.gl_display_cleanup.restype = None


def generate_plasma_gpu(width, height, t, device):
    """Generate plasma pattern on GPU using torch ops (stands in for model inference)."""
    y_coords = torch.linspace(0, 1, height, device=device).unsqueeze(1)
    x_coords = torch.linspace(0, 1, width, device=device).unsqueeze(0)

    v1 = torch.sin(x_coords * 10.0 + t)
    v2 = torch.sin(y_coords * 10.0 + t * 0.7)
    v3 = torch.sin((x_coords + y_coords) * 10.0 + t * 1.3)
    v4 = torch.sin(torch.sqrt(x_coords ** 2 + y_coords ** 2) * 20.0 - t * 2.0)
    v = (v1 + v2 + v3 + v4) / 4.0

    r = ((torch.sin(v * math.pi) * 0.5 + 0.5) * 255)
    g = ((torch.sin(v * math.pi + 2.094) * 0.5 + 0.5) * 255)
    b = ((torch.sin(v * math.pi + 4.188) * 0.5 + 0.5) * 255)
    a = torch.full_like(r, 255)

    img = torch.stack([r, g, b, a], dim=-1).to(torch.uint8)  # RGBA for OpenGL
    return img


def main():
    device = torch.device("cuda")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Mode: GPU tensor → .data_ptr() → CUDA-GL interop (stays in VRAM)")

    # Init GL display
    ret = lib.gl_display_init(WIDTH, HEIGHT)
    if ret != 0:
        print("Failed to init GL display")
        return

    # CUDA events for timing
    ev_frame_start = torch.cuda.Event(enable_timing=True)
    ev_generate_done = torch.cuda.Event(enable_timing=True)
    ev_display_done = torch.cuda.Event(enable_timing=True)

    print(f"\nWarming up ({WARMUP_FRAMES} frames)...")
    print(f"{'frame':<8}  {'generate':>10}  {'display':>10}  {'total':>10}  {'FPS':>8}")
    print(f"{'':8}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}  {'':>8}")
    print("-" * 55)

    accum_generate = 0.0
    accum_display = 0.0
    accum_total = 0.0
    frame_count = 0
    total_frames = 0
    t_start = time.monotonic()

    while not lib.gl_display_should_close():
        t = time.monotonic() - t_start

        # -- Generate on GPU --
        ev_frame_start.record()
        img_gpu = generate_plasma_gpu(WIDTH, HEIGHT, t, device)
        torch.cuda.synchronize()
        ev_generate_done.record()

        # -- Display via CUDA-GL interop (stays on GPU) --
        ret = lib.gl_display_show_frame(img_gpu.data_ptr())
        ev_display_done.record()

        ev_display_done.synchronize()

        if ret == 1:  # window closed
            break

        total_frames += 1
        if total_frames <= WARMUP_FRAMES:
            continue

        ms_generate = ev_frame_start.elapsed_time(ev_generate_done)
        ms_display = ev_generate_done.elapsed_time(ev_display_done)
        ms_total = ev_frame_start.elapsed_time(ev_display_done)

        accum_generate += ms_generate
        accum_display += ms_display
        accum_total += ms_total
        frame_count += 1

        if frame_count % REPORT_INTERVAL == 0:
            n = REPORT_INTERVAL
            print(f"{total_frames:<8}  {accum_generate/n:>10.3f}  {accum_display/n:>10.3f}  "
                  f"{accum_total/n:>10.3f}  {1000.0/(accum_total/n):>8.0f}")
            accum_generate = 0.0
            accum_display = 0.0
            accum_total = 0.0

    leftover = frame_count % REPORT_INTERVAL
    if leftover > 0:
        n = leftover
        print(f"{total_frames:<8}  {accum_generate/n:>10.3f}  {accum_display/n:>10.3f}  "
              f"{accum_total/n:>10.3f}  {1000.0/(accum_total/n):>8.0f}")

    lib.gl_display_cleanup()


if __name__ == "__main__":
    main()
