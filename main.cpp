#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "include/Else.h"
#include "include/Excuse.h"
#include "include/ElseGreedyI.h"
#include "include/ElseGreedyII.h"
#include "include/ExcuseGreedyI.h"
#include "include/ExcuseGreedyII.h"
#include <chrono>

using namespace cv;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <time_count> <gpu_or_cpu> <num_executions> <algorithm>" << std::endl;
        return -1;
    }

    // Parse command-line arguments
    std::string imagePath = argv[1];
    bool timeCount = std::stoi(argv[2]);
    std::string mode = argv[3];
    int numExecutions = std::stoi(argv[4]);
    std::string algorithm = argv[5];

    if (numExecutions <= 0) {
        std::cerr << "Error: num_executions must be greater than 0." << std::endl;
        return -1;
    }

    // Load the image
    Mat frame = imread(imagePath, IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Start timing if requested
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the function multiple times
    Pupil result;
    for (int i = 0; i < numExecutions; ++i) {
        if (mode == "gpu") {
            if (algorithm == "excuse") {
                result = EXCUSE_run(frame, i, 1);
            } else if (algorithm == "else") {
                result = ELSE_run(frame, i, 1);
            } else if (algorithm == "excuse_greedy_i") {
                result = EXCUSEGREEDYI_run(frame, i, 1);
            } else if (algorithm == "excuse_greedy_ii") {
                result = EXCUSEGREEDYII_run(frame, i, 1);
            } else if (algorithm == "else_greedy_i") {
                result = ELSEGREEDYI_run(frame, i, 1);
            } else if (algorithm == "else_greedy_ii") {
                result = ELSEGREEDYII_run(frame, i, 1);
            } else {
                std::cerr << "Error: Invalid algorithm. Use 'else', 'excuse', 'excuse_greedy_i', 'excuse_greedy_ii', 'else_greedy_i', or 'else_greedy_ii'." << std::endl;
                return -1;
            }
        } else if (mode == "cpu") {
            if (algorithm == "excuse") {
                result = EXCUSE_run(frame, i, 0);
            } else if (algorithm == "else") {
                result = ELSE_run(frame, i, 0);
            } else if (algorithm == "excuse_greedy_i") {
                result = EXCUSEGREEDYI_run(frame, i, 0);
            } else if (algorithm == "excuse_greedy_ii") {
                result = EXCUSEGREEDYII_run(frame, i, 0);
            } else if (algorithm == "else_greedy_i") {
                result = ELSEGREEDYI_run(frame, i, 0);
            } else if (algorithm == "else_greedy_ii") {
                result = ELSEGREEDYII_run(frame, i, 0);
            } else {
                std::cerr << "Error: Invalid algorithm. Use 'else', 'excuse', 'excuse_greedy_i', 'excuse_greedy_ii', 'else_greedy_i', or 'else_greedy_ii'." << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Error: Invalid mode. Use 'gpu' or 'cpu'." << std::endl;
            return -1;
        }
    }

    // Stop timing if requested
    if (timeCount) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Processing time for " << numExecutions << " executions: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Average time per execution: " << (elapsed.count() / numExecutions) << " seconds" << std::endl;
    }

    // Display the result from the last execution
    std::cout << "Pupil center: " << result.center << std::endl;
    std::cout << "Pupil size: " << result.size << std::endl;

    return 0;
}