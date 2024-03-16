CXX=g++
NVCC=nvcc
CXXFLAGS=-Wall -Wextra -Werror -g

.PHONY: all clean

all: clean main

test: test_generate_input test_uniform_random

# Main source files
main: src/main.cu
	$(NVCC) -Xcompiler "$(CXXFLAGS)" src/main.cu -I src -o main

# Testing files
test_generate_input: test/test_generate_input.cpp
	$(CXX) $(CXXFLAGS) test/test_generate_input.cpp -I src -o test_generate_input

test_uniform_random: test/test_uniform_random.cpp
	$(CXX) $(CXXFLAGS) test/test_uniform_random.cpp -I src -o test_uniform_random

# This removes all object (*.o) files and any files at the top-level with the
# executable permission set.
clean:
	rm -rf *.o && find . -type f -executable -delete
