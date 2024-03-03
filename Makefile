CXX=g++
CXXFLAGS=-Wall -Wextra -Werror -g

.PHONY: all clean

all: clean test_uniform_random

test_uniform_random: test/test_uniform_random.cpp
	$(CXX) $(CXXFLAGS) test/test_uniform_random.cpp -I src -o test_uniform_random

# This removes all object (*.o) files and any files at the top-level with the
# executable permission set.
clean:
	rm -rf *.o && find . -type f -executable -delete
