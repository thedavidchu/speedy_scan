CXX=g++
CXXFLAGS=-Wall -Wextra -Werror -g

.PHONY: all clean

all: clean test_uniform_random

test_uniform_random: test/test_uniform_random.cpp
	$(CXX) $(CXXFLAGS) test/test_uniform_random.cpp -I src

clean:
	rm -rf *.o a.out
