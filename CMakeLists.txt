cmake_minimum_required(VERSION 3.16)
project(sr)

# Cấu hình đường dẫn CMake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Thiết lập cờ biên dịch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Thêm thư mục chứa header file của dự án
include_directories(${PROJECT_SOURCE_DIR}/src/include)

# Tìm và liên kết CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda-11.8/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-11.8/lib64)

# Tìm và liên kết OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Liên kết TensorRT
find_package(TensorRT REQUIRED)
include_directories(${TENSORRT_INCLUDE_DIRS})
link_directories(${TENSORRT_LIB_DIR})

# Thêm NVTX
include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda-11.8/include/nvtx3)

# Tạo tệp thực thi và liên kết các thư viện cần thiết
add_executable(${PROJECT_NAME} main.cpp)

# Liên kết các thư viện với tệp thực thi
target_link_libraries(${PROJECT_NAME} PRIVATE 
    nvinfer 
    cudart
    nvToolsExt 
    ${OpenCV_LIBS} 
    dl)  # Thêm libdl vào đây
