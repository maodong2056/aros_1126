#include "foot_rknn.h"

#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>
#include <vector>

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

int main(int argc, char **argv) {
  std::string modelPath = "foot_pose_rknn.rknn";
  std::string inputFolder = "images";
  std::string outputFolder = "inference_results";

  if (argc >= 2)
    modelPath = argv[1];
  if (argc >= 3)
    inputFolder = argv[2];

  create_dir_if_not_exists(outputFolder);

  std::cout << "Loading RKNN model: " << modelPath << " ..." << std::endl;
  FootPoseDetector detector(modelPath);

  std::vector<std::string> imageFiles;
  DIR *dir = opendir(inputFolder.c_str());
  if (dir != nullptr) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string filename = entry->d_name;
      if (filename == "." || filename == "..")
        continue;

      std::string ext = filename.substr(filename.find_last_of(".") + 1);
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == "jpg" || ext == "png" || ext == "jpeg" || ext == "bmp") {
        imageFiles.push_back(inputFolder + "/" + filename);
      }
    }
    closedir(dir);
  } else {
    std::cerr << "Error: Input folder not found: " << inputFolder << std::endl;
    return -1;
  }

  std::cout << "Found " << imageFiles.size() << " images. Start inferencing..."
            << std::endl;

  for (const auto &imgPath : imageFiles) {
    cv::Mat img = cv::imread(imgPath);
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
