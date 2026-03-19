# OpenGL Display Pipeline: CPU vs GPU

This project demonstrates two ways to get pixel data onto a screen using OpenGL, and benchmarks them against the standard `cv2.imshow` path.

### Files

| File | Language | What it does |
|------|----------|-------------|
| `main.c` | C | CPU pipeline — generate pixels on CPU, upload to GPU via `glTexSubImage2D`, display |
| `main_cuda.cu` | CUDA/C | GPU pipeline — generate pixels with CUDA kernel, copy to GL texture via `cudaMemcpy2DToArray`, display. Data never leaves VRAM. |
| `bench.cu` | CUDA/C | Benchmark binary — runs either CPU or GPU pipeline (`./bench cpu` or `./bench gpu`) with per-stage `cudaEvent` timing |
| `gl_display.cu` | CUDA/C | Shared library (`libgl_display.so`) — exposes a C API (`init`, `show_frame`, `cleanup`) so Python can display a CUDA tensor via CUDA-GL interop without touching the CPU |
| `bench_cv2.py` | Python | Benchmark — generates image on GPU with PyTorch, displays via `tensor.cpu().numpy()` + `cv2.imshow` |
| `bench_gl.py` | Python | Benchmark — generates image on GPU with PyTorch, displays via `tensor.data_ptr()` + `libgl_display.so` (CUDA-GL interop) |

All demos/benchmarks render an animated plasma pattern. The pattern itself doesn't matter — it's a stand-in for any image source (neural network output, video frame, simulation, etc.).

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
make                  # builds everything: demo, demo_cuda, bench, libgl_display.so
make demo             # CPU-only demo
make demo_cuda        # GPU demo (CUDA-GL interop)
make bench            # C benchmark (supports cpu/gpu modes)
make libgl_display.so # shared library for Python benchmarks
```

For the Python benchmarks:
```bash
python3 -m venv .venv
.venv/bin/pip install torch opencv-python
```

## Run

You need a display. If you're on a remote server via VNC, find your display number first:
```bash
ls /tmp/.X11-unix/   # shows X0, X1, etc.
```

Then prefix every command with `DISPLAY=:1` (or whichever number).

```bash
# CPU demo — generate on CPU, upload to GPU, display
DISPLAY=:1 ./demo

# GPU demo — generate on GPU, CUDA-GL interop, display
DISPLAY=:1 ./demo_cuda

# C benchmark — CPU or GPU pipeline
DISPLAY=:1 ./bench cpu
DISPLAY=:1 ./bench gpu

# Python benchmark — PyTorch + cv2.imshow
DISPLAY=:1 .venv/bin/python bench_cv2.py

# Python benchmark — PyTorch + CUDA-GL interop
DISPLAY=:1 .venv/bin/python bench_gl.py
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

## Python display library (`gl_display.cu` → `libgl_display.so`)

This is the bridge that lets Python/PyTorch use CUDA-GL interop. It compiles to a shared library with a simple C API, called from Python via `ctypes`.

### API

```c
int  gl_display_init(int width, int height);        // create window + GL context + CUDA interop
int  gl_display_show_frame(unsigned long long ptr);  // display a CUDA tensor (pass tensor.data_ptr())
int  gl_display_should_close(void);                  // check if window was closed
void gl_display_cleanup(void);                       // tear down everything
```

### How `show_frame` works

The function takes a raw CUDA device pointer (the `unsigned long long` from `tensor.data_ptr()`) and does the same map/copy/unmap/draw cycle as `main_cuda.cu`:

```c
int gl_display_show_frame(unsigned long long data_ptr) {
    // 1. Map GL texture for CUDA access
    cudaGraphicsMapResources(1, &cuda_tex_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cu_array, cuda_tex_resource, 0, 0);

    // 2. Copy tensor data → GL texture (GPU → GPU, no PCIe)
    cudaMemcpy2DToArray(cu_array, 0, 0,
        (void *)data_ptr, width * 4,
        width * 4, height,
        cudaMemcpyDeviceToDevice);

    // 3. Unmap, draw, swap
    cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glfwSwapBuffers(win);
}
```

The tensor must be RGBA `uint8`, shape `(H, W, 4)`, contiguous, on CUDA. In Python:

```python
import ctypes
lib = ctypes.CDLL("./libgl_display.so")
lib.gl_display_init(1920, 1080)

# After model inference:
img = model(input)  # returns (H, W, 4) uint8 CUDA tensor
lib.gl_display_show_frame(img.data_ptr())
```

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

## Benchmark Results

See [analysis.md](analysis.md) for raw runs and detailed breakdown. Summary at 1920x1080:

| Method | Generate (ms) | Download (ms) | Upload/Display (ms) | Total (ms) | FPS |
|--------|:---:|:---:|:---:|:---:|:---:|
| **CPU compute → `glTexSubImage2D`** | 95.62 | — | 2.75 | 98.37 | 10 |
| **CUDA kernel → CUDA-GL interop** | 0.07 | — | 0.06 | 0.12 | 7530 |
| **PyTorch → `tensor.cpu()` → `cv2.imshow`** | 2.20 | 2.35 | 13.20 | 17.75 | 56 |
| **PyTorch → `data_ptr()` → CUDA-GL interop** | 2.09 | — | 0.20 | 2.29 | 440 |

## Test System

| Component | Spec |
|-----------|------|
| **CPU** | Intel Xeon W-2265 @ 3.50GHz (12 cores / 24 threads) |
| **GPU** | NVIDIA RTX A4000 (16 GB VRAM) |
| **RAM** | 32 GB |
| **Driver** | NVIDIA 535.230.02 |
| **CUDA** | Toolkit 10.1 (driver supports up to 12.2) |
| **OS** | Linux 5.15.0-139-generic |

