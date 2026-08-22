cmake_minimum_required(VERSION 3.20)
project(SemiLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)

# Core library: everything in src/ except the entry point.
file(GLOB CORE_SRC CONFIGURE_DEPENDS src/*.cu src/*.cpp)
list(REMOVE_ITEM CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
add_library(semillm_core STATIC ${CORE_SRC})
target_include_directories(semillm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(semillm_core PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)

add_executable(semillm src/main.cu)
target_link_libraries(semillm PRIVATE semillm_core)

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
