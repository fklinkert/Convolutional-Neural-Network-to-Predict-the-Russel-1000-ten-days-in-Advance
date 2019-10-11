CC  = clang++
CXX = clang++

CFLAGS   = -g -Wall 
CXXFLAGS = -g -Wall -std=c++11

LDFALGs = -g

executables = hw3
objects = hw3.o multi_init.o

$(executables):

$(objects):

.PHONY: clean
clean:
	rm -f *~ a.out core $(objects) $(executables)

.PHONY: default
default: clean $(executables)
