# OpenGL Display Pipeline: CPU vs GPU

This project demonstrates two ways to get pixel data onto a screen using OpenGL, and benchmarks them against each other:

1. **CPU pipeline** (`main.c`) — generate pixels on the CPU, upload them to the GPU, display
2. **GPU pipeline** (`main_cuda.cu`) — generate pixels on the GPU, copy directly to an OpenGL texture without ever touching the CPU
3. **Benchmark** (`bench.cu`) — same binary runs either pipeline with detailed per-stage timing

Both render an animated plasma pattern. The pattern itself doesn't matter — it's a stand-in for any image source (neural network output, video frame, simulation, etc.).

## Prerequisites

- NVIDIA GPU with driver installed
- CUDA toolkit (`nvcc`)
- GLFW3 (`libglfw3-dev` on Ubuntu/Debian)
- A running X11 display (set `DISPLAY=:1` or whichever your display is)

Install GLFW:
```bash
sudo apt-get install -y libglfw3-dev
```

GLAD (the OpenGL function loader) is already generated in `glad/`. If you need to regenerate it:
```bash
pip install glad
python3 -m glad --generator c --out-path glad --api gl=3.3 --profile core
```

## Build

```bash
make          # builds all three: demo, demo_cuda, bench
make demo     # CPU-only version
make demo_cuda # GPU version
make bench    # benchmark (supports both modes)
```

## Run

You need a display. If you're on a remote server via VNC, find your display number first:
```bash
ls /tmp/.X11-unix/   # shows X0, X1, etc.
```

Then prefix every command with `DISPLAY=:1` (or whichever number).

```bash
# CPU demo
DISPLAY=:1 ./demo

# GPU demo (CUDA → OpenGL interop)
DISPLAY=:1 ./demo_cuda

# Benchmark — CPU pipeline
DISPLAY=:1 ./bench cpu

# Benchmark — GPU pipeline
DISPLAY=:1 ./bench gpu
```

Press **ESC** to close the window.

---

## How it works

### Background: what is OpenGL?

OpenGL is an API for talking to the GPU's graphics hardware. It lets you create textures (rectangular grids of pixels stored in GPU memory), write small programs called "shaders" that run on the GPU, and draw geometry to the screen.

To display an image, you:
1. Create a **texture** on the GPU
2. Fill it with pixel data
3. Draw a triangle (or quad) that covers the whole screen, with the texture mapped onto it
4. The GPU's fragment shader reads from the texture and outputs the color for each screen pixel

The key question is: **how does the pixel data get into the texture?**

### The libraries

| Library | What it does |
|---------|-------------|
| **GLFW** | Creates a window and an OpenGL context. Handles keyboard/mouse input and the event loop. Without it, you'd need to write hundreds of lines of platform-specific window code. |
| **GLAD** | Loads OpenGL function pointers at runtime. OpenGL functions aren't linked like normal libraries — they're provided by the GPU driver and must be looked up by name. GLAD does this for you. |
| **CUDA runtime** | NVIDIA's GPU compute API. Used in the GPU pipeline to run kernels and manage GPU memory. |
| **cuda_gl_interop** | The bridge between CUDA and OpenGL. Lets CUDA read/write OpenGL textures directly in GPU memory. |

### Shaders

Both pipelines use the same two shaders. Shaders are small programs written in GLSL that run on the GPU.

**Vertex shader** — runs once per vertex. We draw a single triangle with 3 vertices that covers the entire screen. The shader generates vertex positions and texture coordinates purely from the vertex ID (0, 1, 2), so no vertex buffer is needed:

```glsl
#version 330 core
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
```

This generates an oversized triangle that covers the screen:
```
vertex 0: position (-1,-1), uv (0,0)  — bottom-left
vertex 1: position ( 3,-1), uv (2,0)  — far right
vertex 2: position (-1, 3), uv (0,2)  — far top
```
The parts outside the screen get clipped. The UV coordinates let us sample the texture.

**Fragment shader** — runs once per pixel on screen. It reads the color from the texture at the interpolated UV coordinate:

```glsl
#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D tex;
void main() {
    color = texture(tex, uv);
}
```

That's it. The entire rendering side is just "sample a texture and output the color."

---

## CPU pipeline (`main.c`)

### Data flow
```
CPU (generate_image)  →  glTexSubImage2D  →  GPU texture  →  fragment shader  →  screen
         RAM                PCIe bus              VRAM
```

### Step by step

#### 1. Create a window and OpenGL context (lines 83–106)

```c
glfwInit();
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
GLFWwindow *win = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Texture Display", NULL, NULL);
glfwMakeContextCurrent(win);
gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
```

- `glfwInit` initializes the library
- `glfwWindowHint` requests OpenGL 3.3 core profile (no legacy/deprecated features)
- `glfwCreateWindow` creates an 800x600 window with an OpenGL context
- `glfwMakeContextCurrent` binds that context to the current thread — all subsequent GL calls operate on this context
- `gladLoadGLLoader` looks up all the OpenGL function pointers from the driver

#### 2. Compile shaders and create the program (lines 108–109)

```c
GLuint program = create_program();
```

This compiles the vertex and fragment shader source strings into GPU machine code, links them into a "program," and returns a handle. The `compile_shader` helper checks for compilation errors.

#### 3. Create a VAO (lines 111–113)

```c
GLuint vao;
glGenVertexArrays(1, &vao);
```

A VAO (Vertex Array Object) stores vertex attribute configuration. OpenGL 3.3 core profile *requires* one to be bound before drawing, even though we don't use any vertex buffers (our vertex shader generates positions from `gl_VertexID`).

#### 4. Create the texture (lines 115–122)

```c
GLuint texture;
glGenTextures(1, &texture);
glBindTexture(GL_TEXTURE_2D, texture);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
```

- `glGenTextures` allocates a texture handle
- `glBindTexture` makes it the "current" texture for subsequent calls
- `glTexParameteri` sets filtering to linear interpolation (smooth scaling)
- `glTexImage2D` allocates GPU memory for an 800x600 RGB texture. The `NULL` data pointer means "allocate but don't fill yet"

#### 5. Allocate a CPU pixel buffer (line 125)

```c
unsigned char *pixels = malloc(WIDTH * HEIGHT * 3);
```

800 * 600 * 3 bytes (RGB) = 1.44 MB of CPU RAM.

#### 6. Render loop (lines 128–151)

Each frame:

**a. Generate the image on the CPU (line 136):**
```c
generate_image(pixels, WIDTH, HEIGHT, time);
```
Loops over every pixel (480,000 of them) sequentially, computes `sinf` values for the plasma pattern, writes RGB bytes into the `pixels` buffer. This is the slow part — ~22ms on a single CPU core.

**b. Upload to the GPU (lines 140–141):**
```c
glBindTexture(GL_TEXTURE_2D, texture);
glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
```
`glTexSubImage2D` copies the CPU `pixels` buffer to the GPU texture over the PCIe bus. For 1.44 MB this takes ~0.23ms.

**c. Draw and present (lines 143–150):**
```c
glClear(GL_COLOR_BUFFER_BIT);
glUseProgram(program);
glBindVertexArray(vao);
glDrawArrays(GL_TRIANGLES, 0, 3);
glfwSwapBuffers(win);
glfwPollEvents();
```
- `glClear` clears the screen
- `glUseProgram` activates our shader program
- `glDrawArrays(GL_TRIANGLES, 0, 3)` tells the GPU to run the vertex shader 3 times (producing a triangle), then the fragment shader for every pixel inside it
- `glfwSwapBuffers` swaps the front and back buffers (double buffering prevents tearing)
- `glfwPollEvents` processes window events (keyboard, mouse, close button)

---

## GPU pipeline (`main_cuda.cu`)

### Data flow
```
CUDA kernel  →  cudaMemcpy2DToArray  →  GL texture  →  fragment shader  →  screen
   VRAM             VRAM → VRAM            VRAM
```

The CPU never touches pixel data. Everything stays in GPU memory (VRAM).

### What's different from the CPU version

#### Texture is RGBA, not RGB (line 131)

```c
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
```

CUDA works best with 4-byte aligned data. `uchar4` (RGBA) is 4 bytes per pixel, neatly aligned. RGB would be 3 bytes, causing misaligned memory accesses and reduced throughput.

#### Register the GL texture with CUDA (lines 134–138)

```c
cudaGraphicsResource *cudaTexResource;
cudaGraphicsGLRegisterImage(
    &cudaTexResource, texture, GL_TEXTURE_2D,
    cudaGraphicsRegisterFlagsWriteDiscard
);
```

This is the key interop step. It tells CUDA: "this OpenGL texture exists, and I want to write to it." The `WriteDiscard` flag means CUDA will overwrite the entire texture every frame (no need to preserve old contents).

This is done **once** at setup, not every frame.

#### The CUDA kernel (lines 74–93)

```c
__global__ void plasma_kernel(uchar4 *surface, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    // ... same sinf math as the CPU version ...
    surface[y * width + x] = make_uchar4(r, g, b, 255);
}
```

- `__global__` marks this as a CUDA kernel — a function that runs on the GPU across thousands of threads simultaneously
- Each thread handles **one pixel**. `blockIdx`/`threadIdx` identify which pixel this thread is responsible for
- The math is identical to the CPU version, but instead of looping over 480,000 pixels sequentially, the GPU runs ~480,000 threads in parallel
- `make_uchar4(r, g, b, 255)` packs the color into a 4-byte struct

The kernel is launched with a 2D grid of 16x16 thread blocks:
```c
dim3 block(16, 16);   // 256 threads per block
dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);  // enough blocks to cover the image
plasma_kernel<<<grid, block>>>(d_buffer, WIDTH, HEIGHT, time);
```

#### Per-frame interop: map, write, unmap (lines 154–181)

Each frame follows a three-step protocol:

**a. Map the GL texture for CUDA access:**
```c
cudaGraphicsMapResources(1, &cudaTexResource, 0);
cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaTexResource, 0, 0);
```
This "locks" the texture so CUDA can write to it. OpenGL must not touch it while mapped.

The `cudaArray` you get back is an opaque handle to the texture's memory. You can't write to a `cudaArray` directly from a kernel (it's in a tiled memory layout optimized for texture sampling). So the kernel writes to a flat staging buffer instead.

**b. Kernel writes to staging buffer, then copy to the GL texture:**
```c
plasma_kernel<<<grid, block>>>(d_buffer, WIDTH, HEIGHT, time);

cudaMemcpy2DToArray(cuArray, 0, 0,
    d_buffer, WIDTH * sizeof(uchar4),
    WIDTH * sizeof(uchar4), HEIGHT,
    cudaMemcpyDeviceToDevice);
```
`cudaMemcpyDeviceToDevice` means GPU → GPU. The data never leaves VRAM. In a real application, `d_buffer` would be your PyTorch tensor's data pointer — no staging buffer needed.

**c. Unmap so OpenGL can render:**
```c
cudaGraphicsUnmapResources(1, &cudaTexResource, 0);
```
Now OpenGL owns the texture again and can draw it.

---

## Benchmark (`bench.cu`)

A single binary that runs either pipeline, controlled by a command-line argument. Uses `cudaEvent` timing to measure each stage with GPU-clock precision.

### Timing methodology

CUDA events are timestamps recorded on the GPU's command queue:

```c
cudaEvent_t ev_start, ev_end;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_end);

cudaEventRecord(ev_start, 0);    // timestamp before
// ... work ...
cudaEventRecord(ev_end, 0);      // timestamp after
cudaEventSynchronize(ev_end);    // wait for GPU to reach this point

float ms;
cudaEventElapsedTime(&ms, ev_start, ev_end);  // difference in milliseconds
```

The benchmark measures three stages per frame:
- **generate** — CPU sinf loop (CPU mode) or CUDA kernel (GPU mode)
- **upload** — `glTexSubImage2D` over PCIe (CPU mode) or map + `cudaMemcpy2DToArray` within VRAM + unmap (GPU mode)
- **render** — OpenGL draw call + buffer swap (same for both)

It skips the first 60 frames (warmup) and prints averaged results every 60 frames.

Vsync is disabled (`glfwSwapInterval(0)`) so the benchmark measures raw throughput, not monitor-limited frame rate.

---

## Test System

| Component | Spec |
|-----------|------|
| **CPU** | Intel Xeon W-2265 @ 3.50GHz (12 cores / 24 threads) |
| **GPU** | NVIDIA RTX A4000 (16 GB VRAM) |
| **RAM** | 32 GB |
| **Driver** | NVIDIA 535.230.02 |
| **CUDA** | Toolkit 10.1 (driver supports up to 12.2) |
| **OS** | Linux 5.15.0-139-generic |

## Benchmark Results

Vsync off. 60-frame warmup, averages reported every 60 frames.

### 800x600 (1.92 MB RGBA per frame)

**CPU pipeline:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120           22.188       0.231       0.625      23.044        43
180           22.138       0.230       0.628      22.996        43
240           22.171       0.225       0.632      23.029        43
300           22.114       0.229       0.626      22.969        44
```

**GPU pipeline:**
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

| Stage | CPU | GPU | Speedup |
|-------|-----|-----|---------|
| **Generate** | 22.15 ms | 0.02 ms | ~1100x |
| **Upload** | 0.23 ms | 0.13 ms | ~1.8x |
| **Render** | 0.63 ms | 0.17 ms | ~3.7x |
| **Total** | 23.01 ms | 0.32 ms | ~72x |

### 1920x1080 (8.29 MB RGBA per frame)

**CPU pipeline:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120           95.617       2.715       0.034      98.366        10
```

**GPU pipeline:**
```
frame       generate      upload      render       total       FPS
                (ms)        (ms)        (ms)        (ms)
---------------------------------------------------------------
120            0.068       0.052       0.012       0.133      7530
180            0.068       0.045       0.001       0.115      8693
```

| Stage | CPU | GPU | Speedup |
|-------|-----|-----|---------|
| **Generate** | 95.62 ms | 0.07 ms | ~1366x |
| **Upload** | 2.72 ms | 0.05 ms | ~54x |
| **Render** | 0.03 ms | 0.01 ms | ~3x |
| **Total** | 98.37 ms | 0.12 ms | ~820x |

### Analysis

**Generate** is irrelevant to a real ML use case — your model runs on GPU regardless. The plasma kernel is just a stand-in.

**Upload is the metric that matters.** At 1080p:
- CPU path: **2.72ms** — `glTexSubImage2D` sends 8.3 MB over the PCIe bus
- GPU path: **0.05ms** — `cudaMemcpy2DToArray` copies within VRAM

That's a **2.67ms difference** per frame. Against a 33ms budget (30 FPS target), the CPU path spends **8%** of your frame time just moving data. The GPU path spends **0.15%**.

Whether this matters depends on your model's inference time:
- Model takes 15ms → 33 - 15 = 18ms headroom. 2.7ms is fine, use the simpler CPU path.
- Model takes 28ms → 33 - 28 = 5ms headroom. 2.7ms eats over half your remaining budget. Use GPU path.
- Model takes 32ms → you're already borderline. 2.7ms pushes you past the deadline. GPU path is mandatory.
