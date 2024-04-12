NVCC=nvcc

CUDA_SM := sm_86

CXXSTD := -std=c++17
CXXFLAGS = -Wall -Wextra -Werror -g $(CXXSTD)

CUDA_SRCS = $(shell find src -name "*.cu")
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
CUDA_HDRS = $(shell find src -name "*.h" -or -iname "*.hpp")

MAIN = main

.PHONY: all clean
.SUFFIXES: .cu

all: $(MAIN)

test: test_generate_input test_uniform_random

# Main source files
%.o: %.cu $(CUDA_HDRS)
	$(NVCC) -c $(CXXSTD) -arch $(CUDA_SM) -Xcompiler "$(CXXFLAGS)" $< -I src -o $@

$(MAIN): $(CUDA_OBJS)
	$(NVCC) $(CXXSTD) -arch $(CUDA_SM) -Xcompiler "$(CXXFLAGS)" $^ -o $@

# Testing files
test_generate_input: test/test_generate_input.cpp
	$(CXX) $(CXXFLAGS) $< -I src -o test_generate_input

test_uniform_random: test/test_uniform_random.cpp
	$(CXX) $(CXXFLAGS) $< -I src -o test_uniform_random

# Remove compiled objects and the main executable
clean:
	$(RM) -rf $(CUDA_OBJS) $(MAIN)
