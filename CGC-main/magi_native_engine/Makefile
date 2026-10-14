CC = clang
CXX = clang++
CFLAGS = -O3 -std=c17 -Wall
CXXFLAGS = -O3 -std=c++17 -Wall

# 2026-10-14 整合：LLAMA_PATH 从已删除的 ~/Documents/cgcjitload/llama.cpp
# 改指 devserver 仓库内 llama.cpp 源码树（同一 repo，build/bin 已有全套 dylib）
LLAMA_PATH = $(abspath $(CURDIR)/../../src/llama.cpp)
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
		"$(LLAMA_BUILD)/libggml-base.dylib" \
		-Wl,-rpath,$(LLAMA_BUILD)

%.o: %.mm
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ===== P2 · MoE 四接口桩（不依赖 Metal/llama，纯 C 可编译） =====
moe_profile: moe_profile_main.c moe_inject.c
	$(CC) $(CFLAGS) $^ -o $@ -lpthread

moe_profile_metal: moe_profile_main.c moe_inject.c libmetal_runtime.a
	$(CXX) $(CXXFLAGS) $^ -o moe_profile_metal -lpthread -framework Metal -framework Foundation \
		"$(LLAMA_BUILD)/libllama.dylib" \
		"$(LLAMA_BUILD)/libggml-metal.dylib" \
		-Wl,-rpath,$(LLAMA_BUILD)

# ===== P2 · Metal 真实算量 GEMV/Router bench（compute 段真实化，仅 Metal 依赖） =====
moe_gemv_bench: moe_gemv_bench.mm moe_gemv_metal.mm moe_gemv_metal.h
	$(CXX) $(CXXFLAGS) -ObjC++ $^ -o $@ -framework Metal -framework Foundation

clean:
	rm -f *.o *.a magi_infer moe_profile moe_profile_metal moe_gemv_bench

.PHONY: clean all