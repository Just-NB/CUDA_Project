#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <opencv2/core/cuda.hpp>

#define EYEWIDTH 80

cv::Mat KernelCall(cv::Mat origin, uchar *cuRc, uchar *cuBc, uchar *cuGc, uchar *cuDestR, uchar *cuDestB, uchar *cuDestG);
cv::Mat KernelCall2(cv::Mat origin, uint *ch, uint *Dest);
uint* KernelCall3(cv::Mat origin, uint *map_);
cv::Mat mapping(cv::Mat origin, uint *map_);
cv::Mat mappingInGpu(cv::Mat origin, uint *gpuMap, cv::cuda::GpuMat cvOrigin, cv::cuda::GpuMat cvRes);