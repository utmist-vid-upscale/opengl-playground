CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O2 -I glad/include
NVCCFLAGS = -O2 -I glad/include
LDFLAGS = -lglfw -lGL -lm -ldl
CUDA_LDFLAGS = -lglfw -lGL -lm -ldl -lcuda

all: demo demo_cuda bench

demo: main.c glad/src/glad.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

demo_cuda: main_cuda.cu glad/src/glad.c
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

bench: bench.cu glad/src/glad.c
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

clean:
	rm -f demo demo_cuda bench

.PHONY: all clean
