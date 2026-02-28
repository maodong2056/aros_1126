#include "foot_rknn.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {1, 2}, {3, 4}, {4, 5}};

const std::vector<cv::Scalar> COLORS = {
    {0, 255, 0}, {0, 255, 0}, {255, 0, 0}, {255, 0, 0}};

void create_dir_if_not_exists(const std::string &path) {
  struct stat st = {0};
  if (stat(path.c_str(), &st) == -1) {
    mkdir(path.c_str(), 0777);
  }
}

bool has_suffix(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string resolve_model_path_from_config(const std::string &config_path,
                                           int model_id) {
  std::ifstream ifs(config_path);
  if (!ifs.is_open()) {
    return std::string();
  }

  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document doc;
  doc.ParseStream(isw);
  if (doc.HasParseError() || !doc.HasMember("models") || !doc["models"].IsArray()) {
    return std::string();
  }

  const auto &models = doc["models"];
  for (rapidjson::SizeType i = 0; i < models.Size(); ++i) {
    if (!models[i].IsObject() || !models[i].HasMember("name") ||
        !models[i]["name"].IsString()) {
      continue;
    }
    if (models[i].HasMember("id") && models[i]["id"].IsInt() &&
        models[i]["id"].GetInt() == model_id) {
      return models[i]["name"].GetString();
    }
  }

  if (model_id >= 0 && model_id < static_cast<int>(models.Size()) &&
      models[model_id].IsObject() && models[model_id].HasMember("name") &&
      models[model_id]["name"].IsString()) {
    return models[model_id]["name"].GetString();
  }

  return std::string();
}

int main(int argc, char **argv) {
  std::string modelPath = "foot_pose_rknn.rknn";
  std::string imagePattern = "images/*";
  std::string outputFolder = "inference_results";

  // 支持两种调用方式：
  // 1) rknn_foot_pose_demo <model.rknn> <image_pattern> <output_folder>
  // 2) 类 main_task 形式:
  //    rknn_foot_pose_demo <lib_so> <imgdir> <savedir> <config_or_model> <frame_id> <model_id>
  if (argc >= 7) {
    imagePattern = argv[2];
    outputFolder = argv[3];
    std::string config_or_model = argv[4];
    int model_id = std::atoi(argv[6]);

    if (has_suffix(config_or_model, ".json")) {
      std::string resolved = resolve_model_path_from_config(config_or_model, model_id);
      if (resolved.empty()) {
        std::cerr << "Failed to resolve model path from config: " << config_or_model
                  << ", model_id=" << model_id << std::endl;
        return -1;
      }
      modelPath = resolved;
    } else {
      modelPath = config_or_model;
    }
  } else {
    if (argc >= 2)
      modelPath = argv[1];
    if (argc >= 3)
      imagePattern = argv[2];
    if (argc >= 4)
      outputFolder = argv[3];
  }

  create_dir_if_not_exists(outputFolder);

  std::cout << "Loading RKNN model: " << modelPath << " ..." << std::endl;
  FootPoseDetector detector(modelPath);

  std::vector<cv::String> imageFiles;
  if (imagePattern.find("*") == std::string::npos &&
      imagePattern.find(".jpg") == std::string::npos &&
      imagePattern.find(".jpeg") == std::string::npos &&
      imagePattern.find(".png") == std::string::npos &&
      imagePattern.find(".bmp") == std::string::npos) {
    imagePattern += "/*";
  }
  cv::glob(imagePattern, imageFiles, false);
  std::cout << "Found " << imageFiles.size() << " images. Start inferencing..."
            << std::endl;

  for (const auto &imgPath : imageFiles) {
    cv::Mat img = cv::imread(imgPath.c_str());
    if (img.empty())
      continue;

    auto results = detector.detect(img);

    for (const auto &res : results) {
      cv::rectangle(img, res.box, cv::Scalar(0, 255, 255), 2);

      for (size_t i = 0; i < SKELETON.size(); ++i) {
        int idx1 = SKELETON[i].first;
        int idx2 = SKELETON[i].second;

        if (idx1 >= static_cast<int>(res.kpts.size()) ||
            idx2 >= static_cast<int>(res.kpts.size())) {
          continue;
        }

        const auto &kp1 = res.kpts[idx1];
        const auto &kp2 = res.kpts[idx2];

        if (kp1.conf > 0.5f && kp2.conf > 0.5f) {
          cv::Scalar color =
              (i < COLORS.size()) ? COLORS[i] : cv::Scalar(255, 0, 0);
          cv::line(img, cv::Point(static_cast<int>(kp1.x), static_cast<int>(kp1.y)),
                   cv::Point(static_cast<int>(kp2.x), static_cast<int>(kp2.y)),
                   color, 2);
        }
      }

      for (size_t i = 0; i < res.kpts.size(); ++i) {
        if (res.kpts[i].conf > 0.5f) {
          cv::Scalar c = (i < 3) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
          cv::circle(img, cv::Point(static_cast<int>(res.kpts[i].x),
                                    static_cast<int>(res.kpts[i].y)),
                     5, c, -1);
        }
      }
    }

    std::string filename = imgPath.substr(imgPath.find_last_of("/") + 1);
    std::string savePath = outputFolder + "/" + filename;
    cv::imwrite(savePath, img);
    std::cout << "Processed: " << filename << " -> " << results.size()
              << " objects" << std::endl;
  }

  std::cout << "All done!" << std::endl;
  return 0;
}
