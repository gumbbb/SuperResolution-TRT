# SuperResolution-TRT

## Steps to Run the Code

1. **Clone the Repository**  
   First, clone the repository from GitHub:
   ```bash
   git clone SuperResolution-TRT
   cd SuperResolution-TRT
2. **Install Dependencies**  
   - CUDA 11.8  
   - TensorRT  
   - OpenCV 4.x  
   - CMake 3.16 or higher  
   - A C++17-compatible compiler  

3. **Modify `CMakeLists.txt`**  
   - Update the following paths in `CMakeLists.txt` if necessary:  
   ```cmake
   include_directories(/usr/local/cuda-11.8/targets/x86_64-linux/include)
   link_directories(/usr/local/cuda-11.8/lib64)
   find_package(TensorRT REQUIRED)
   include_directories(${TENSORRT_INCLUDE_DIRS})
   link_directories(${TENSORRT_LIB_DIR})

4. **Build the project**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    
5. **Run the Program**

   Use this command to execute the program (Replace <video_input_path> with the input video file (e.g., VID.mp4), <video_output_path> with the output file path (e.g., output/sr.mp4), and <engine_path> with the TensorRT engine file (e.g., sr.trt).):
   
   ```
    ./sr <video_input_path> <video_output_path> <engine_path> 
   ```

   
6. **Example Command**

   ```
   ./sr VID.mp4 output/sr.mp4 sr.trt
    ```
   
## Project Structure
├── cmake \
│   └── FindTensorRT.cmake \
├── CMakeLists.txt \
├── main.cpp \
├── src \
│   └── include \
│       ├── config \
│       │   └── config.hpp \
│       ├── cuda_utils.h \
│       ├── logging \
│       │   └── logging.h \
│       └── preprocess \
│           └── preprocess.hpp \
└── sr.trt

