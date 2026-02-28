#pragma once

#include "rknn_api.h"
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct Keypoint {
  float x;
  float y;
  float conf;
};

struct Object {
  cv::Rect box;
  float score;
  std::vector<Keypoint> kpts;
};

class FootPoseDetector {
public:
  explicit FootPoseDetector(const std::string &modelPath);
  ~FootPoseDetector();

  std::vector<Object> detect(const cv::Mat &orig_img);

private:
  rknn_context ctx = 0;
  rknn_input_output_num io_num{};
  rknn_tensor_attr *input_attrs = nullptr;
  rknn_tensor_attr *output_attrs = nullptr;

  const int INPUT_W = 512;
  const int INPUT_H = 384;
  const float CONF_THRES = 0.5f;
  const float IOU_THRES = 0.45f;
  const int NUM_KPTS = 6;

  void preprocess(const cv::Mat &img, cv::Mat &rgb_img, float &scale_x,
                  float &scale_y);
  std::vector<Object> foot_postprocess(float **output_bufs, float scale_x,
                                       float scale_y);

  float sigmoid(float x);
  float dfl_integral(const float *data, int area);
  float iou(const cv::Rect &a, const cv::Rect &b) const;
  std::vector<Object> nms(std::vector<Object> objects) const;
};
