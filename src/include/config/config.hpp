#ifndef REAL_ESRGAN_TRT_CONFIG_HPP
#define REAL_ESRGAN_TRT_CONFIG_HPP

#include <string>

// Tên blob input và output
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

// Sử dụng định dạng FP16 (nếu phần cứng hỗ trợ)
const bool USE_FP16 = false;

// Cấu hình batch size
static const int BATCH_SIZE = 1;

// Kích thước đầu vào (640 x 512)
static const int INPUT_C = 3;  // 3 channels (RGB)
static const int INPUT_H = 512;  // Chiều cao (height) 512
static const int INPUT_W = 640;  // Chiều rộng (width) 640

// Tỉ lệ upscale (4x)
static const int OUT_SCALE = 4;

// Kích thước đầu ra sau khi upscale
static const int OUTPUT_SIZE = BATCH_SIZE * 48 * INPUT_H * INPUT_W ;

#endif  //REAL_ESRGAN_TRT_CONFIG_HPP