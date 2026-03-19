// Shared library for GPU-direct display from Python.
// Python calls: init(w, h) → show_frame(data_ptr) → cleanup()
// data_ptr is a CUDA device pointer (e.g. from tensor.data_ptr())

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

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

static GLFWwindow *win = NULL;
static GLuint program = 0, vao = 0, gl_tex = 0;
static cudaGraphicsResource *cuda_tex_resource = NULL;
static int tex_width = 0, tex_height = 0;

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
        return 0;
    }
    return s;
}

extern "C" {

// Initialize the GL window and CUDA interop. Call once.
// Returns 0 on success, -1 on failure.
int gl_display_init(int width, int height) {
    tex_width = width;
    tex_height = height;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    win = glfwCreateWindow(width, height, "GPU Direct Display", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n");
        return -1;
    }

    // Shaders
    GLuint v = compile_shader(GL_VERTEX_SHADER, vert_src);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    if (!v || !f) return -1;
    program = glCreateProgram();
    glAttachShader(program, v);
    glAttachShader(program, f);
    glLinkProgram(program);
    glDeleteShader(v);
    glDeleteShader(f);

    glGenVertexArrays(1, &vao);

    // Texture
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Register with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_tex_resource, gl_tex, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));

    printf("gl_display: initialized %dx%d, CUDA-GL interop ready\n", width, height);
    return 0;
}

// Display a frame from a CUDA device pointer.
// data_ptr: device pointer to RGBA uint8 data, row-major, width*height*4 bytes.
// Returns: 0 on success, -1 on error, 1 if window was closed.
int gl_display_show_frame(unsigned long long data_ptr) {
    if (!win) return -1;

    if (glfwWindowShouldClose(win)) return 1;

    glfwPollEvents();
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) return 1;

    // Map GL texture for CUDA
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));

    cudaArray_t cu_array;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cu_array, cuda_tex_resource, 0, 0));

    // Copy from tensor's GPU memory to GL texture (GPU → GPU)
    CUDA_CHECK(cudaMemcpy2DToArray(
        cu_array, 0, 0,
        (void *)data_ptr, tex_width * 4,
        tex_width * 4, tex_height,
        cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

    // Render
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glfwSwapBuffers(win);

    return 0;
}

// Returns 1 if window should close, 0 otherwise.
int gl_display_should_close(void) {
    if (!win) return 1;
    return glfwWindowShouldClose(win);
}

// Cleanup. Call once when done.
void gl_display_cleanup(void) {
    if (cuda_tex_resource) {
        cudaGraphicsUnregisterResource(cuda_tex_resource);
        cuda_tex_resource = NULL;
    }
    if (gl_tex) { glDeleteTextures(1, &gl_tex); gl_tex = 0; }
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (program) { glDeleteProgram(program); program = 0; }
    if (win) { glfwDestroyWindow(win); win = NULL; }
    glfwTerminate();
}

}  // extern "C"
