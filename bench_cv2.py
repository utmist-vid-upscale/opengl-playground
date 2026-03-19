"""
Benchmark: GPU tensor → cv2.imshow display path.

Measures the real cost of tensor.cpu().numpy() + cv2.imshow,
which is what most people do to display neural net output.

Compare results with ./bench gpu (CUDA-GL interop path).
"""

import torch
import cv2
import time
import math

WIDTH = 1920
HEIGHT = 1080
WARMUP_FRAMES = 60
REPORT_INTERVAL = 60


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

    img = torch.stack([b, g, r], dim=-1).to(torch.uint8)  # BGR for OpenCV
    return img


def main():
    device = torch.device("cuda")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Mode: GPU tensor → .cpu().numpy() → cv2.imshow")

    # Create CUDA events for timing
    ev_frame_start = torch.cuda.Event(enable_timing=True)
    ev_generate_done = torch.cuda.Event(enable_timing=True)
    ev_download_done = torch.cuda.Event(enable_timing=True)
    ev_display_done = torch.cuda.Event(enable_timing=True)

    cv2.namedWindow("bench_cv2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("bench_cv2", WIDTH, HEIGHT)

    print(f"\nWarming up ({WARMUP_FRAMES} frames)...")
    print(f"{'frame':<8}  {'generate':>10}  {'download':>10}  {'display':>10}  {'total':>10}  {'FPS':>8}")
    print(f"{'':8}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}  {'(ms)':>10}  {'':>8}")
    print("-" * 70)

    accum_generate = 0.0
    accum_download = 0.0
    accum_display = 0.0
    accum_total = 0.0
    frame_count = 0
    total_frames = 0
    t_start = time.monotonic()

    while True:
        t = time.monotonic() - t_start

        # -- Generate on GPU --
        ev_frame_start.record()
        img_gpu = generate_plasma_gpu(WIDTH, HEIGHT, t, device)
        ev_generate_done.record()

        # -- Download to CPU (this is tensor.cpu().numpy()) --
        img_cpu = img_gpu.cpu().numpy()
        ev_download_done.record()

        # -- Display with OpenCV --
        cv2.imshow("bench_cv2", img_cpu)
        key = cv2.waitKey(1)
        ev_display_done.record()

        # Sync and collect timings
        ev_display_done.synchronize()

        total_frames += 1
        if total_frames <= WARMUP_FRAMES:
            continue
        if key == 27:  # ESC
            break

        ms_generate = ev_frame_start.elapsed_time(ev_generate_done)
        ms_download = ev_generate_done.elapsed_time(ev_download_done)
        ms_display = ev_download_done.elapsed_time(ev_display_done)
        ms_total = ev_frame_start.elapsed_time(ev_display_done)

        accum_generate += ms_generate
        accum_download += ms_download
        accum_display += ms_display
        accum_total += ms_total
        frame_count += 1

        if frame_count % REPORT_INTERVAL == 0:
            n = REPORT_INTERVAL
            print(f"{total_frames:<8}  {accum_generate/n:>10.3f}  {accum_download/n:>10.3f}  "
                  f"{accum_display/n:>10.3f}  {accum_total/n:>10.3f}  {1000.0/(accum_total/n):>8.0f}")
            accum_generate = 0.0
            accum_download = 0.0
            accum_display = 0.0
            accum_total = 0.0

    # Final partial batch
    leftover = frame_count % REPORT_INTERVAL
    if leftover > 0:
        n = leftover
        print(f"{total_frames:<8}  {accum_generate/n:>10.3f}  {accum_download/n:>10.3f}  "
              f"{accum_display/n:>10.3f}  {accum_total/n:>10.3f}  {1000.0/(accum_total/n):>8.0f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
