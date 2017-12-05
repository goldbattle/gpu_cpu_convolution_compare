cmake_minimum_required (VERSION 2.8.11)
project(conv_compare)


# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3  -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2 -mfpmath=sse")


add_executable(conv_single cpu/conv_single.cpp)
#target_link_libraries(conv_single LINK_PUBLIC Hello)


add_executable(conv_sse2 cpu/conv_sse2.cpp)
#target_link_libraries(conv_sse2 LINK_PUBLIC Hello)


find_package(CUDA REQUIRED)


# Find the compute capability
# https://stackoverflow.com/a/40665580
set(ENV{CUDA_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}")
set(ENV{CUDA_CUDART_LIBRARY} "${CUDA_CUDART_LIBRARY}")
message("Probing the (first) GPU for its compute capability...")
execute_process(COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/get_cuda_sm.sh"  OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY)
execute_process(COMMAND bash "-c" "echo \"${CUDA_TARGET_COMPUTE_CAPABILITY}\" | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n" OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY)
message("You Have Compute Capability ${FORMATTED_COMPUTE_CAPABILITY}.")
message("Formated Compute Capability ${CUDA_TARGET_COMPUTE_CAPABILITY}")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52)



cuda_add_executable(conv_cuda cuda/conv_cuda.cu cuda/conv_kernel.cu)
#target_link_libraries(conv_cuda LINK_PUBLIC Hello)


