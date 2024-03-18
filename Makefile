CXX=g++
NVCC=nvcc

CUDA_SM := sm_86

CXXSTD := -std=c++17
CXXFLAGS=-Wall -Wextra -Werror -g $(CXXSTD)

CUDA_SRCS = $(shell find src -name "*.cu")
CUDA_HDRS = $(shell find src -name "*.h" -or -iname "*.hpp")

.PHONY: all clean

all: main

test: test_generate_input test_uniform_random

# Main source files
main: $(CUDA_SRCS) $(CUDA_HDRS)
	$(NVCC) $(CXXSTD) -arch $(CUDA_SM) -Xcompiler "$(CXXFLAGS)" $(filter-out %.hpp,$^) -I src -o $@

# Testing files
test_generate_input: test/test_generate_input.cpp
	$(CXX) $(CXXFLAGS) test/test_generate_input.cpp -I src -o test_generate_input

test_uniform_random: test/test_uniform_random.cpp
	$(CXX) $(CXXFLAGS) test/test_uniform_random.cpp -I src -o test_uniform_random

# This removes all object (*.o) files and any files at the top-level with the
# executable permission set.
clean:
	rm -rf *.o && find . -type f -executable -delete
