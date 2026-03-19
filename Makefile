CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O2 -I glad/include
NVCCFLAGS = -O2 -I glad/include
LDFLAGS = -lglfw -lGL -lm -ldl
CUDA_LDFLAGS = -lglfw -lGL -lm -ldl -lcuda

all: demo demo_cuda bench libgl_display.so

demo: main.c glad/src/glad.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

demo_cuda: main_cuda.cu glad/src/glad.c
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

bench: bench.cu glad/src/glad.c
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

libgl_display.so: gl_display.cu glad/src/glad.c
	$(NVCC) $(NVCCFLAGS) --shared --compiler-options '-fPIC' -o $@ $^ $(CUDA_LDFLAGS)

clean:
	rm -f demo demo_cuda bench libgl_display.so

.PHONY: all clean
