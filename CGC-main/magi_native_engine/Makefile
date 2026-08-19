CC = clang
CXX = clang++
CFLAGS = -O3 -std=c17 -Wall
CXXFLAGS = -O3 -std=c++17 -Wall

LLAMA_PATH = /Users/alexchuang/Documents/cgcjitload/llama.cpp
LLAMA_BUILD = $(LLAMA_PATH)/build/bin
LLAMA_INCLUDE = $(LLAMA_PATH)/ggml/include
LLAMA_SRC = $(LLAMA_PATH)/ggml/src

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -D__APPLE__
    CXXFLAGS += -D__APPLE__ -I$(LLAMA_INCLUDE) -I$(LLAMA_SRC)
endif

all: libmetal_runtime.a magi_infer

libmetal_runtime.a: metal_runtime.o
	ar rcs $@ $^

magi_infer: main.cpp libmetal_runtime.a
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm -framework Metal -framework Foundation -framework MetalKit \
		"$(LLAMA_BUILD)/libllama.dylib" \
		"$(LLAMA_BUILD)/libggml-metal.dylib" \
		"$(LLAMA_BUILD)/libggml-cpu.dylib" \
		"$(LLAMA_BUILD)/libggml-blas.dylib" \
		"$(LLAMA_BUILD)/libggml-base.dylib" \
		-Wl,-rpath,$(LLAMA_BUILD)

%.o: %.mm
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.a magi_infer

.PHONY: clean all