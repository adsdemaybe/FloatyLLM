cmake_minimum_required(VERSION 3.20)
project(SemiLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB SEMILLM_SRC CONFIGURE_DEPENDS src/*.cu src/*.cpp)
add_executable(semillm ${SEMILLM_SRC})
target_link_libraries(semillm PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
