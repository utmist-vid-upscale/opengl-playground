// Minimal OpenGL example: generate a pattern on the CPU and display it as a texture.
// This is the foundation for CUDA interop — you'd replace the CPU texture update
// with a CUDA kernel writing directly to the GL texture.

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH  800
#define HEIGHT 600

// Vertex shader: fullscreen quad via vertex ID (no vertex buffer needed)
static const char *vert_src =
    "#version 330 core\n"
    "out vec2 uv;\n"
    "void main() {\n"
    "    // Generate fullscreen triangle that covers the screen\n"
    "    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

// Fragment shader: sample the texture
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

// Generate a colorful animated pattern (this is what you'd replace with your neural net output)
static void generate_image(unsigned char *pixels, int w, int h, float time) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float fx = (float)x / w;
            float fy = (float)y / h;

            // Animated plasma effect
            float v1 = sinf(fx * 10.0f + time);
            float v2 = sinf(fy * 10.0f + time * 0.7f);
            float v3 = sinf((fx + fy) * 10.0f + time * 1.3f);
            float v4 = sinf(sqrtf(fx * fx + fy * fy) * 20.0f - time * 2.0f);
            float v = (v1 + v2 + v3 + v4) / 4.0f; // [-1, 1]

            int i = (y * w + x) * 3;
            pixels[i + 0] = (unsigned char)((sinf(v * 3.14159f) * 0.5f + 0.5f) * 255);
            pixels[i + 1] = (unsigned char)((sinf(v * 3.14159f + 2.094f) * 0.5f + 0.5f) * 255);
            pixels[i + 2] = (unsigned char)((sinf(v * 3.14159f + 4.188f) * 0.5f + 0.5f) * 255);
        }
    }
}

int main(void) {
    // 1. Init GLFW and create window
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *win = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Texture Display", NULL, NULL);
    if (!win) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);

    // 2. Load OpenGL functions via GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n");
        return 1;
    }
    printf("OpenGL %s\n", glGetString(GL_VERSION));

    // 3. Create shader program
    GLuint program = create_program();

    // 4. Create a VAO (required in core profile even if we don't use vertex buffers)
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // 5. Create texture — this is the surface we write our image to
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Allocate texture storage (NULL data for now)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    // CPU pixel buffer (in CUDA interop, this goes away — CUDA writes directly to the texture)
    unsigned char *pixels = malloc(WIDTH * HEIGHT * 3);

    // 6. Render loop
    while (!glfwWindowShouldClose(win)) {
        // ESC to close
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, 1);

        float time = (float)glfwGetTime();

        // Generate image on CPU (replace this with your neural net tensor!)
        generate_image(pixels, WIDTH, HEIGHT, time);

        // Upload to GPU texture
        // With CUDA interop, you skip this — the data is already on the GPU
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);

        // Draw fullscreen quad with the texture
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3); // 3 vertices = one big triangle covering screen

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // Cleanup
    free(pixels);
    glDeleteTextures(1, &texture);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
