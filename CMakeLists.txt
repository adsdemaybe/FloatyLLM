cmake_minimum_required(VERSION 3.20)
project(FloatyLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Core library: everything in src/ except the entry point.
file(GLOB CORE_SRC CONFIGURE_DEPENDS src/*.cu src/*.cpp)
list(REMOVE_ITEM CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
add_library(semillm_core STATIC ${CORE_SRC})
target_include_directories(semillm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(semillm_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt Threads::Threads)

# Optional: use llama.cpp's vocab (libllama) for tokenization. Point this at a built
# llama.cpp checkout (has include/llama.h and build/bin/libllama.so). Without it, the
# project still builds but the tokenizer/chat commands are disabled.
set(SEMILLM_LLAMA_DIR "" CACHE PATH "Path to a built llama.cpp (include/ + build/bin/libllama.so)")
if(SEMILLM_LLAMA_DIR)
  find_library(LLAMA_LIB llama PATHS ${SEMILLM_LLAMA_DIR}/build/bin REQUIRED)
  target_include_directories(semillm_core PUBLIC ${SEMILLM_LLAMA_DIR}/include ${SEMILLM_LLAMA_DIR}/ggml/include)
  target_link_libraries(semillm_core PUBLIC ${LLAMA_LIB})
  target_compile_definitions(semillm_core PUBLIC SEMILLM_HAVE_LLAMA=1)
  set(SEMILLM_LLAMA_RPATH ${SEMILLM_LLAMA_DIR}/build/bin)
  message(STATUS "FloatyLLM: tokenizer via llama.cpp at ${SEMILLM_LLAMA_DIR}")
endif()

add_executable(floatyllm src/main.cu)
target_link_libraries(floatyllm PRIVATE semillm_core)
if(SEMILLM_LLAMA_DIR)
  set_target_properties(floatyllm PROPERTIES BUILD_RPATH ${SEMILLM_LLAMA_RPATH} INSTALL_RPATH ${SEMILLM_LLAMA_RPATH})
endif()

enable_testing()
add_executable(test_dequant tests/test_dequant.cu)
target_link_libraries(test_dequant PRIVATE semillm_core)
add_test(NAME dequant COMMAND test_dequant)

add_executable(test_rmsnorm tests/test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE semillm_core)
add_test(NAME rmsnorm COMMAND test_rmsnorm)

add_executable(test_rope tests/test_rope.cu)
target_link_libraries(test_rope PRIVATE semillm_core)
add_test(NAME rope COMMAND test_rope)

add_executable(test_elementwise tests/test_elementwise.cu)
target_link_libraries(test_elementwise PRIVATE semillm_core)
add_test(NAME elementwise COMMAND test_elementwise)

add_executable(test_softmax tests/test_softmax.cu)
target_link_libraries(test_softmax PRIVATE semillm_core)
add_test(NAME softmax COMMAND test_softmax)

add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention PRIVATE semillm_core)
add_test(NAME attention COMMAND test_attention)

add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm PRIVATE semillm_core)
add_test(NAME gemm COMMAND test_gemm)

add_executable(test_layer tests/test_layer.cu)
target_link_libraries(test_layer PRIVATE semillm_core)
add_test(NAME layer COMMAND test_layer)

add_executable(test_stream tests/test_stream.cu)
target_link_libraries(test_stream PRIVATE semillm_core)
add_test(NAME stream COMMAND test_stream)

add_executable(test_runner tests/test_runner.cu)
target_link_libraries(test_runner PRIVATE semillm_core)
add_test(NAME runner COMMAND test_runner)

add_executable(test_moe tests/test_moe.cu)
target_link_libraries(test_moe PRIVATE semillm_core)
add_test(NAME moe COMMAND test_moe)

add_executable(test_moe_layer tests/test_moe_layer.cu)
target_link_libraries(test_moe_layer PRIVATE semillm_core)
add_test(NAME moe_layer COMMAND test_moe_layer)

add_executable(test_kquant_gpu tests/test_kquant_gpu.cu)
target_link_libraries(test_kquant_gpu PRIVATE semillm_core)
add_test(NAME kquant_gpu COMMAND test_kquant_gpu)

add_executable(test_fused_gemm tests/test_fused_gemm.cu)
target_link_libraries(test_fused_gemm PRIVATE semillm_core)
add_test(NAME fused_gemm COMMAND test_fused_gemm)
