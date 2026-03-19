// CUDA-OpenGL interop: generate an image on the GPU and display it WITHOUT
// ever touching the CPU. The plasma kernel writes directly to a GL texture.

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH  800
#define HEIGHT 600

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---- Shaders (same as CPU version) ----

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

// ---- CUDA kernel: generates plasma directly on GPU ----
// This is the part that would be your neural network output.
// Instead of sinf on pixels, imagine this is a tensor from PyTorch.

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

int main(void) {
    // 1. Init GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *win = glfwCreateWindow(WIDTH, HEIGHT, "CUDA -> OpenGL (zero-copy)", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);

    // 2. Load GL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n");
        return 1;
    }
    printf("OpenGL %s\n", glGetString(GL_VERSION));

    // 3. Shader + VAO
    GLuint program = create_program();
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // 4. Create GL texture (RGBA this time — CUDA likes 4-byte aligned pixels)
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // 5. Register the GL texture with CUDA — this is the key interop step
    cudaGraphicsResource *cudaTexResource;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cudaTexResource, texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard  // CUDA will only write, never read
    ));
    printf("CUDA-GL interop registered\n");

    // FPS tracking
    double lastTime = glfwGetTime();
    int frames = 0;

    // 6. Render loop — no CPU pixel buffer anywhere!
    while (!glfwWindowShouldClose(win)) {
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, 1);

        float time = (float)glfwGetTime();

        // --- CUDA writes directly to the GL texture ---

        // Map the GL texture so CUDA can access it
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaTexResource, 0));

        // Get a CUDA array that points to the texture's memory
        cudaArray_t cuArray;
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaTexResource, 0, 0));

        // Launch kernel into a temporary surface, then copy to the GL texture
        // (cudaArray doesn't support direct kernel writes, so we use a staging buffer)
        static uchar4 *d_buffer = NULL;
        if (!d_buffer) {
            CUDA_CHECK(cudaMalloc(&d_buffer, WIDTH * HEIGHT * sizeof(uchar4)));
        }

        dim3 block(16, 16);
        dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
        plasma_kernel<<<grid, block>>>(d_buffer, WIDTH, HEIGHT, time);

        // Copy from CUDA buffer to GL texture (GPU -> GPU, same VRAM)
        CUDA_CHECK(cudaMemcpy2DToArray(
            cuArray, 0, 0,
            d_buffer, WIDTH * sizeof(uchar4),
            WIDTH * sizeof(uchar4), HEIGHT,
            cudaMemcpyDeviceToDevice  // <-- stays on GPU!
        ));

        // Unmap so OpenGL can use the texture again
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaTexResource, 0));

        // --- Now draw it ---
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(win);
        glfwPollEvents();

        // Print FPS every second
        frames++;
        if (glfwGetTime() - lastTime >= 1.0) {
            printf("FPS: %d\n", frames);
            frames = 0;
            lastTime = glfwGetTime();
        }
    }

    // Cleanup
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaTexResource));
    glDeleteTextures(1, &texture);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
