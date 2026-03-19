// Benchmark: CPU→GPU upload vs CUDA→GL GPU-only pipeline
// Usage:
//   ./bench cpu    — generate on CPU, upload via glTexSubImage2D
//   ./bench gpu    — generate on GPU, copy via cudaMemcpy2DToArray (stays in VRAM)
//
// Prints per-frame timing breakdown and running averages.

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH  800
#define HEIGHT 600
#define WARMUP_FRAMES 60
#define REPORT_INTERVAL 60  // print averages every N frames

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- Shaders ----

static const char *vert_src =
    "#version 330 core\n"
    "out vec2 uv;\n"
    "void main() {\n"
    "    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static const char *frag_src =
    "#version 330 core\n"
    "in vec2 uv;\n"
    "out vec4 color;\n"
    "uniform sampler2D tex;\n"
    "void main() {\n"
    "    color = texture(tex, uv);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    int ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, NULL, log);
        fprintf(stderr, "Shader error: %s\n", log);
        exit(1);
    }
    return s;
}

static GLuint create_program(void) {
    GLuint v = compile_shader(GL_VERTEX_SHADER, vert_src);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ---- CPU image generation (same plasma, RGBA this time) ----

static void generate_image_cpu(unsigned char *pixels, int w, int h, float time) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float fx = (float)x / w;
            float fy = (float)y / h;
            float v1 = sinf(fx * 10.0f + time);
            float v2 = sinf(fy * 10.0f + time * 0.7f);
            float v3 = sinf((fx + fy) * 10.0f + time * 1.3f);
            float v4 = sinf(sqrtf(fx * fx + fy * fy) * 20.0f - time * 2.0f);
            float v = (v1 + v2 + v3 + v4) / 4.0f;

            int i = (y * w + x) * 4;
            pixels[i + 0] = (unsigned char)((sinf(v * 3.14159f) * 0.5f + 0.5f) * 255);
            pixels[i + 1] = (unsigned char)((sinf(v * 3.14159f + 2.094f) * 0.5f + 0.5f) * 255);
            pixels[i + 2] = (unsigned char)((sinf(v * 3.14159f + 4.188f) * 0.5f + 0.5f) * 255);
            pixels[i + 3] = 255;
        }
    }
}

// ---- CUDA kernel ----

__global__ void plasma_kernel(uchar4 *surface, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float fx = (float)x / width;
    float fy = (float)y / height;
    float v1 = sinf(fx * 10.0f + time);
    float v2 = sinf(fy * 10.0f + time * 0.7f);
    float v3 = sinf((fx + fy) * 10.0f + time * 1.3f);
    float v4 = sinf(sqrtf(fx * fx + fy * fy) * 20.0f - time * 2.0f);
    float v = (v1 + v2 + v3 + v4) / 4.0f;

    unsigned char r = (unsigned char)((sinf(v * 3.14159f) * 0.5f + 0.5f) * 255);
    unsigned char g = (unsigned char)((sinf(v * 3.14159f + 2.094f) * 0.5f + 0.5f) * 255);
    unsigned char b = (unsigned char)((sinf(v * 3.14159f + 4.188f) * 0.5f + 0.5f) * 255);

    surface[y * width + x] = make_uchar4(r, g, b, 255);
}

// ---- Timing accumulators ----

typedef struct {
    double generate;   // CPU: sinf loop / GPU: kernel execution
    double upload;     // CPU: glTexSubImage2D / GPU: map + memcpy + unmap
    double render;     // GL draw + swap
    double total;      // full frame
} FrameTiming;

int main(int argc, char **argv) {
    if (argc < 2 || (strcmp(argv[1], "cpu") != 0 && strcmp(argv[1], "gpu") != 0)) {
        fprintf(stderr, "Usage: %s <cpu|gpu>\n", argv[0]);
        return 1;
    }
    int use_gpu = strcmp(argv[1], "gpu") == 0;

    // ---- Init GLFW ----
    if (!glfwInit()) { fprintf(stderr, "Failed to init GLFW\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const char *title = use_gpu ? "BENCH: GPU pipeline" : "BENCH: CPU pipeline";
    GLFWwindow *win = glfwCreateWindow(WIDTH, HEIGHT, title, NULL, NULL);
    if (!win) { fprintf(stderr, "Failed to create window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);  // disable vsync for raw throughput measurement

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n"); return 1;
    }
    printf("OpenGL %s\n", (const char *)glGetString(GL_VERSION));
    printf("Mode: %s\n", use_gpu ? "GPU (CUDA kernel → GL texture, stays in VRAM)" :
                                    "CPU (sinf loop → glTexSubImage2D upload)");

    GLuint program = create_program();
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // ---- Texture ----
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // ---- CUDA setup (GPU mode) ----
    cudaGraphicsResource *cudaTexResource = NULL;
    uchar4 *d_buffer = NULL;
    if (use_gpu) {
        CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &cudaTexResource, texture, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard));
        CUDA_CHECK(cudaMalloc(&d_buffer, WIDTH * HEIGHT * sizeof(uchar4)));
        printf("CUDA-GL interop registered\n");
    }

    // ---- CPU buffer (CPU mode) ----
    unsigned char *pixels = NULL;
    if (!use_gpu) {
        pixels = (unsigned char *)malloc(WIDTH * HEIGHT * 4);
    }

    // ---- CUDA events for timing ----
    cudaEvent_t ev_frame_start, ev_generate_done, ev_upload_done, ev_render_done;
    CUDA_CHECK(cudaEventCreate(&ev_frame_start));
    CUDA_CHECK(cudaEventCreate(&ev_generate_done));
    CUDA_CHECK(cudaEventCreate(&ev_upload_done));
    CUDA_CHECK(cudaEventCreate(&ev_render_done));

    // ---- Accumulators ----
    FrameTiming accum = {0, 0, 0, 0};
    int frame_count = 0;
    int total_frames = 0;

    printf("\nWarming up (%d frames)...\n", WARMUP_FRAMES);
    printf("%-8s  %10s  %10s  %10s  %10s  %8s\n",
           "frame", "generate", "upload", "render", "total", "FPS");
    printf("%-8s  %10s  %10s  %10s  %10s  %8s\n",
           "", "(ms)", "(ms)", "(ms)", "(ms)", "");
    printf("---------------------------------------------------------------\n");

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    while (!glfwWindowShouldClose(win)) {
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, 1);

        float time = (float)glfwGetTime();

        // ---- TIMING: frame start ----
        CUDA_CHECK(cudaEventRecord(ev_frame_start, 0));

        if (use_gpu) {
            // GPU path: CUDA kernel
            plasma_kernel<<<grid, block>>>(d_buffer, WIDTH, HEIGHT, time);
            CUDA_CHECK(cudaEventRecord(ev_generate_done, 0));

            // GPU path: map + copy + unmap
            CUDA_CHECK(cudaGraphicsMapResources(1, &cudaTexResource, 0));
            cudaArray_t cuArray;
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaTexResource, 0, 0));
            CUDA_CHECK(cudaMemcpy2DToArray(
                cuArray, 0, 0,
                d_buffer, WIDTH * sizeof(uchar4),
                WIDTH * sizeof(uchar4), HEIGHT,
                cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaTexResource, 0));
            CUDA_CHECK(cudaEventRecord(ev_upload_done, 0));
        } else {
            // CPU path: generate pixels
            generate_image_cpu(pixels, WIDTH, HEIGHT, time);
            CUDA_CHECK(cudaEventRecord(ev_generate_done, 0));

            // CPU path: upload to GL texture
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT,
                            GL_RGBA, GL_UNSIGNED_BYTE, pixels);
            CUDA_CHECK(cudaEventRecord(ev_upload_done, 0));
        }

        // ---- GL render ----
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);
        glfwPollEvents();

        CUDA_CHECK(cudaEventRecord(ev_render_done, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_render_done));

        // ---- Collect timings ----
        float ms_generate, ms_upload, ms_render, ms_total;
        CUDA_CHECK(cudaEventElapsedTime(&ms_generate, ev_frame_start, ev_generate_done));
        CUDA_CHECK(cudaEventElapsedTime(&ms_upload, ev_generate_done, ev_upload_done));
        CUDA_CHECK(cudaEventElapsedTime(&ms_render, ev_upload_done, ev_render_done));
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_frame_start, ev_render_done));

        total_frames++;
        if (total_frames <= WARMUP_FRAMES) continue;  // skip warmup

        accum.generate += ms_generate;
        accum.upload   += ms_upload;
        accum.render   += ms_render;
        accum.total    += ms_total;
        frame_count++;

        if (frame_count % REPORT_INTERVAL == 0) {
            double n = REPORT_INTERVAL;
            printf("%-8d  %10.3f  %10.3f  %10.3f  %10.3f  %8.0f\n",
                   total_frames,
                   accum.generate / n,
                   accum.upload / n,
                   accum.render / n,
                   accum.total / n,
                   1000.0 / (accum.total / n));
            accum = (FrameTiming){0, 0, 0, 0};
        }
    }

    // ---- Final summary ----
    int leftover = frame_count % REPORT_INTERVAL;
    if (leftover > 0) {
        double n = leftover;
        printf("%-8d  %10.3f  %10.3f  %10.3f  %10.3f  %8.0f\n",
               total_frames,
               accum.generate / n,
               accum.upload / n,
               accum.render / n,
               accum.total / n,
               1000.0 / (accum.total / n));
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_frame_start));
    CUDA_CHECK(cudaEventDestroy(ev_generate_done));
    CUDA_CHECK(cudaEventDestroy(ev_upload_done));
    CUDA_CHECK(cudaEventDestroy(ev_render_done));
    if (cudaTexResource) CUDA_CHECK(cudaGraphicsUnregisterResource(cudaTexResource));
    if (d_buffer) CUDA_CHECK(cudaFree(d_buffer));
    free(pixels);
    glDeleteTextures(1, &texture);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
