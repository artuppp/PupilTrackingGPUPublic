#ifndef ELSEGREEDYII_H
#define ELSEGREEDYII_H
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Pupil.h"

using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif

Pupil ELSEGREEDYII_run(const Mat &frame, int iteration, int gpu);

#ifdef __cplusplus
}
#endif

#endif // ELSEGREEDYII_H