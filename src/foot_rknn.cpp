#include "foot_rknn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
std::vector<unsigned char> read_binary_file(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open model file: " + path);
  }

  std::streamsize size = ifs.tellg();
  if (size <= 0) {
    throw std::runtime_error("Invalid model size: " + path);
  }

  ifs.seekg(0, std::ios::beg);
  std::vector<unsigned char> buffer(static_cast<size_t>(size));
  if (!ifs.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read model file: " + path);
  }
  return buffer;
}

int tensor_elements(const rknn_tensor_attr &attr) {
  int total = 1;
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    total *= static_cast<int>(attr.dims[i]);
  }
  return total;
}
} // namespace

FootPoseDetector::FootPoseDetector(const std::string &modelPath) {
  auto model_data = read_binary_file(modelPath);
  int ret = rknn_init(&ctx, model_data.data(), model_data.size(), 0, nullptr);
  if (ret != RKNN_SUCC) {
    throw std::runtime_error("rknn_init failed, ret=" + std::to_string(ret));
  }

  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    throw std::runtime_error("RKNN_QUERY_IN_OUT_NUM failed, ret=" +
                             std::to_string(ret));
  }

  input_attrs = new rknn_tensor_attr[io_num.n_input];
  output_attrs = new rknn_tensor_attr[io_num.n_output];

  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    std::memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
    input_attrs[i].index = i;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
  }

  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    std::memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
    output_attrs[i].index = i;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
  }
}

FootPoseDetector::~FootPoseDetector() {
  delete[] input_attrs;
  delete[] output_attrs;
  if (ctx != 0) {
    rknn_destroy(ctx);
    ctx = 0;
  }
}

void FootPoseDetector::preprocess(const cv::Mat &img, cv::Mat &rgb_img,
                                  float &scale_x, float &scale_y) {
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
  cv::cvtColor(resized, rgb_img, cv::COLOR_BGR2RGB);

  scale_x = static_cast<float>(img.cols) / static_cast<float>(INPUT_W);
  scale_y = static_cast<float>(img.rows) / static_cast<float>(INPUT_H);
}

std::vector<Object> FootPoseDetector::detect(const cv::Mat &orig_img) {
  if (orig_img.empty()) {
    return {};
  }

  cv::Mat rgb_img;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  preprocess(orig_img, rgb_img, scale_x, scale_y);

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = rgb_img.total() * rgb_img.elemSize();
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].buf = rgb_img.data;

  int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
    return {};
  }

  ret = rknn_run(ctx, nullptr);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_run failed: " << ret << std::endl;
    return {};
  }

  std::vector<rknn_output> outputs(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    outputs[i].index = i;
    outputs[i].is_prealloc = 0;
    outputs[i].want_float = 1;
  }

  ret = rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
    return {};
  }

  std::vector<float *> output_ptrs(io_num.n_output, nullptr);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    output_ptrs[i] = static_cast<float *>(outputs[i].buf);
  }

  auto result = foot_postprocess(output_ptrs.data(), scale_x, scale_y);
  result = nms(std::move(result));

  rknn_outputs_release(ctx, io_num.n_output, outputs.data());
  return result;
}

std::vector<Object> FootPoseDetector::foot_postprocess(float **output_bufs,
                                                       float scale_x,
                                                       float scale_y) {
  std::vector<Object> objects;
  if (io_num.n_output == 0 || output_bufs[0] == nullptr) {
    return objects;
  }

  const rknn_tensor_attr &attr = output_attrs[0];
  int elems = tensor_elements(attr);
  int stride = 5 + NUM_KPTS * 3;
  if (elems < stride) {
    return objects;
  }

  int num_det = elems / stride;
  const float *data = output_bufs[0];

  for (int i = 0; i < num_det; ++i) {
    const float *row = data + i * stride;
    float score = row[4];
    if (score < CONF_THRES) {
      continue;
    }

    float x1 = row[0] * scale_x;
    float y1 = row[1] * scale_y;
    float x2 = row[2] * scale_x;
    float y2 = row[3] * scale_y;
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    Object obj;
    obj.score = score;
    obj.box = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                       cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
    obj.kpts.reserve(NUM_KPTS);

    for (int k = 0; k < NUM_KPTS; ++k) {
      int base = 5 + 3 * k;
      Keypoint kp;
      kp.x = row[base + 0] * scale_x;
      kp.y = row[base + 1] * scale_y;
      kp.conf = row[base + 2];
      obj.kpts.push_back(kp);
    }

    objects.push_back(obj);
  }

  return objects;
}

float FootPoseDetector::sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float FootPoseDetector::dfl_integral(const float *data, int area) {
  float sum = 0.0f;
  float score_sum = 0.0f;
  for (int i = 0; i < area; ++i) {
    float p = sigmoid(data[i]);
    sum += p * static_cast<float>(i);
    score_sum += p;
  }
  return score_sum > 0.0f ? (sum / score_sum) : 0.0f;
}

float FootPoseDetector::iou(const cv::Rect &a, const cv::Rect &b) const {
  const int xx1 = std::max(a.x, b.x);
  const int yy1 = std::max(a.y, b.y);
  const int xx2 = std::min(a.x + a.width, b.x + b.width);
  const int yy2 = std::min(a.y + a.height, b.y + b.height);

  const int w = std::max(0, xx2 - xx1);
  const int h = std::max(0, yy2 - yy1);
  const float inter = static_cast<float>(w * h);
  const float uni = static_cast<float>(a.area() + b.area()) - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Object> FootPoseDetector::nms(std::vector<Object> objects) const {
  std::sort(objects.begin(), objects.end(), [](const Object &lhs, const Object &rhs) {
    return lhs.score > rhs.score;
  });

  std::vector<Object> kept;
  std::vector<bool> removed(objects.size(), false);
  for (size_t i = 0; i < objects.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    kept.push_back(objects[i]);
    for (size_t j = i + 1; j < objects.size(); ++j) {
      if (!removed[j] && iou(objects[i].box, objects[j].box) > IOU_THRES) {
        removed[j] = true;
      }
    }
  }
  return kept;
}
