#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <chrono>

#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>

#include "config/config.hpp"
#include "cuda_utils.h"
#include "logging/logging.h"
#include "preprocess/preprocess.hpp"

static Logger gLogger;

using namespace nvinfer1;

// Hàm inference
void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output) {
    nvtx3::scoped_range r{"doInference"};
    context.setBindingDimensions(0, Dims4(BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W));
    context.enqueueV2(buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <video_input_path> <video_output_path> <engine_path>" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];
    std::string output_path = argv[2]; 
    std::string engine_name = argv[3]; 

    // Deserialize engine
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read " << engine_name << std::endl;
        return -1;
    }

    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;

    assert(engine->getNbBindings() == 2);

    // Allocate GPU buffers
    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    std::vector<float> data(BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W);
    std::vector<float> output(BATCH_SIZE * OUTPUT_SIZE);

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << video_path << std::endl;
        return -1;
    }

    // Create VideoWriter to save output video
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer(output_path, fourcc, cap.get(cv::CAP_PROP_FPS), cv::Size(INPUT_W * OUT_SCALE, INPUT_H * OUT_SCALE));

    // Process video frames
    int frame_count = 0;
    auto global_start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        auto frame_start_time = std::chrono::high_resolution_clock::now(); // Bắt đầu đo thời gian cho frame

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Preprocess frame
        {
            nvtx3::scoped_range r{"Frame Preprocessing"};
            cv::resize(frame, frame, cv::Size(INPUT_W, INPUT_H));
            for (int b = 0; b < BATCH_SIZE; b++) {
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = frame.data + row * frame.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }
        }

        // Copy input data to GPU
        CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], data.data(), BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Run inference
        {
            nvtx3::scoped_range r{"Inference"};
            doInference(*context, stream, buffers, output.data());
        }

        // Postprocess and display
        {
            nvtx3::scoped_range r{"Postprocessing"};
            cv::Mat result(INPUT_H * OUT_SCALE, INPUT_W * OUT_SCALE, CV_8UC3);
            int OUTPUT_C = 3;
            int OUTPUT_H = INPUT_H * OUT_SCALE;
            int OUTPUT_W = INPUT_W * OUT_SCALE;

            for (int row = 0; row < OUTPUT_H; ++row) {
                uchar* uc_pixel = result.data + row * result.step;
                for (int col = 0; col < OUTPUT_W; ++col) {
                    auto r2 = std::round(output[row * OUTPUT_W + col] * 255.0);
                    auto g2 = std::round(output[OUTPUT_H * OUTPUT_W + row * OUTPUT_W + col] * 255.0);
                    auto b2 = std::round(output[2 * OUTPUT_H * OUTPUT_W + row * OUTPUT_W + col] * 255.0);
                    uc_pixel[0] = static_cast<uchar>(std::clamp(b2, 0.0, 255.0));
                    uc_pixel[1] = static_cast<uchar>(std::clamp(g2, 0.0, 255.0));
                    uc_pixel[2] = static_cast<uchar>(std::clamp(r2, 0.0, 255.0));
                    uc_pixel += 3;
                }
            }

            cv::namedWindow("Output", cv::WINDOW_NORMAL);
            cv::resizeWindow("Output", 1024, 1280);
            cv::imshow("Output", result);

            // Ghi khung hình vào video
            writer.write(result);

            if (cv::waitKey(1) == 27) {
                break;
            }
        }

        frame_count++;

        // Tính toán FPS
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - frame_start_time).count();
        double fps = 1000.0 / frame_time;  // Tính FPS từ thời gian xử lý frame
        std::cout << "Frame " << frame_count << ": " << fps << " FPS" << std::endl;
    }

    // Tính tổng FPS sau khi xử lý toàn bộ video
    auto global_end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(global_end_time - global_start_time).count();
    std::cout << "Processed " << frame_count << " frames in " << elapsed_seconds << " seconds (" << frame_count / elapsed_seconds << " FPS overall)." << std::endl;

    // Cleanup
    cap.release();
    writer.release();  // Giải phóng VideoWriter
    cv::destroyAllWindows();
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}