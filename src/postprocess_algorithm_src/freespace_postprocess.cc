#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include "postprocess_algorithm_h/freespace_postprocess.h"

namespace sweeper_ai
{

void nms(std::vector<BBox> &input_boxes, float nms_thresh) {
    float xx1   = -1;
    float yy1   = -1;
    float xx2   = -1;
    float yy2   = -1;
    float w     = -1;
    float h     = -1;
    float inter = -1;
    float ovr   = -1;

    std::sort(input_boxes.begin(), input_boxes.end(), [](BBox a, BBox b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).xmax - input_boxes.at(i).xmin + 1) * (input_boxes.at(i).ymax - input_boxes.at(i).ymin + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int jj = i + 1; jj < int(input_boxes.size());) {
            if(input_boxes[i].label == input_boxes[jj].label) {
                xx1   = (std::max)(input_boxes[i].xmin, input_boxes[jj].xmin);
                yy1   = (std::max)(input_boxes[i].ymin, input_boxes[jj].ymin);
                xx2   = (std::min)(input_boxes[i].xmax, input_boxes[jj].xmax);
                yy2   = (std::min)(input_boxes[i].ymax, input_boxes[jj].ymax);
                w     = (std::max)(float(0), xx2 - xx1 + 1);
                h     = (std::max)(float(0), yy2 - yy1 + 1);
                inter = w * h;
                ovr   = inter / (vArea[i] + vArea[jj] - inter);
                 // IOU 阈值判断
                if (ovr >= nms_thresh) {
                    input_boxes.erase(input_boxes.begin() + jj);
                    vArea.erase(vArea.begin() + jj);
                }
                else {
                    jj++;
                }
            }
            else {
                jj++;
            }
        }
    }
}

float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float sigmoid_freespace(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    return 0;
}

BBox disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int imgsrc_h, int imgsrc_w, int stride, int reg_max) {
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;

    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float dis_after_sm[reg_max + 1];
        // 对 reg_max + 1 长度的数据做softmax 得到dfl的概率
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);

        // 对dfl进行加权求和得到中心点到边的距离
        for (int jj = 0; jj < reg_max + 1; jj++) {
            dis += jj * dis_after_sm[jj];
        }
        // 缩放到输入分辨率大小
        dis *= stride;
        dis_pred[i] = dis;
    }

    // 中心点和四边距离转换成bbox
    float xmin = ct_x - dis_pred[0];
    float ymin = ct_y - dis_pred[1];
    float xmax = ct_x + dis_pred[2];
    float ymax = ct_y + dis_pred[3];
    xmin = xmin / BASE_INPUT_SIZE_WEIGH  * imgsrc_w;
    xmax = xmax / BASE_INPUT_SIZE_WEIGH  * imgsrc_w;
    ymin = ymin / BASE_INPUT_SIZE_HEIGHT * imgsrc_h;
    ymax = ymax / BASE_INPUT_SIZE_HEIGHT * imgsrc_h;
    return BBox{xmin, ymin, xmax, ymax, float(label), score};
}

void decode_infer(rknn_output &cls_pred, rknn_output &dis_pred, int nn_size, int imgsrc_h, int imgsrc_w, float threshold, std::vector<BBox> &results, int num_class, int reg_max, int feature_w, int feature_h) {
    // 得到每层的feature map大小
    int stride = nn_size / feature_h;
    float *cls_pred_data = (float *)(cls_pred.buf);
    float *dis_pred_data = (float *)(dis_pred.buf);

    threshold = log(threshold / (1 - threshold));

    for (int row = 0; row < feature_h; row++) {
        for (int col = 0; col < feature_w; col++) {
            float score     = -999.999f;
            int   cur_label = 0;
            for (int label = 0; label < num_class; label++) {
                int index = label * feature_w * feature_h + feature_w * row + col;
                float cur_score = cls_pred_data[index];
                if (cur_score > score) {
                    score = cur_score;
                    cur_label = label;
                }
            }
            if (score > threshold) {
                score = sigmoid_freespace(score);
                // 取dis_pred的最后一维数据（dis_pred 转置为了batch height width channel）

                int index = (row * feature_w + col) * 4 * (reg_max + 1);
                const float *bbox_pred = dis_pred_data + index;

                // dis_pred转换为bbox
                results.push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, imgsrc_h, imgsrc_w, stride, reg_max));
            }
        }
    }
}

void gfl_post_process(std::vector<BBox>& results, rknn_output *outputdata, int num_class, int outputdata_size, int nn_in_h, int nn_in_w, int imgsrc_h, int imgsrc_w, float score_threshold, float nms_thres, int **outputsize_weith, int **outputsize_height) {
    int num_levels = outputdata_size / 2;

    // 三层输出分层decode
    for (int i = 0; i < num_levels; i++) {
#ifdef USE_NEON
        decode_infer_neon(outputdata[i], outputdata[i + num_levels], nn_in_h, score_threshold, results, num_class, BASE_REG_MAX);
#else
        decode_infer(outputdata[i], outputdata[i + num_levels], nn_in_h, imgsrc_h, imgsrc_w, score_threshold, results, num_class, BASE_REG_MAX, (*outputsize_weith)[i], (*outputsize_height)[i]);
#endif
    }
    // 最后做NMS
    nms(results, nms_thres);
}


/**
 * @Autor: yangxiaodong
 * @description: 滑窗过滤边界点
 * @param {*}
 * @return {*}
 */
std::vector<cv::Point> maxInWindows(std::vector<cv::Point> &p, int window_size) {
    std::vector<cv::Point> res;
    if (!p.size() > 1)
        return res;
    if (p.size() == 2)
        return p;
    res.push_back(p.front());
    int st = 0;
    for (size_t i = 1; i < p.size();) {
        if (abs(p[st].x - p[i].x) > window_size || abs(p[st].y - p[i].y) > window_size) {
            st = i;
            i += 1;
            res.push_back(p[st]);
        }
        else {
            i += 1;
        }
    }
    res.push_back(p.back());
    return res;
}

/**
 * @Autor: zhangdongsheng
 * @description: 获取轮廓点
 * @param {*}
 * @return {*}
 */
void find_contours(cv::Mat &mask_ditan, std::vector<Pointf>& ps,cv::Mat& mask, cv::Mat& prob_mask, int h, int w, int org_h, int org_w, float label, int area_threshold, int win_size, bool bottom_flag,float& con_id) {
    float ymax_th = 0;
    float x       = -1;
    float y       = -1;
    float flag    = -1;
    int min_width  = 7,  max_width  = 247;
    int min_height = 96, max_height = 184;

    std::vector<cv::Mat> contours_mat;
    if (bottom_flag == true) {
        cv::findContours(mask, contours_mat, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    }
    else {
        // cv::findContours(mask, contours_mat, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
        cv::findContours(mask, contours_mat, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    }

    for (auto tmp1 : contours_mat) {
        //  // 获取分割区域下边沿
        if (bottom_flag == true) {
            int minIndex = 0;
            int maxIndex = 0;
            // 获取轮廓下边界起始位置和终点位置索引
            for (size_t i = 0; i < int(tmp1.rows); i++){
                if (tmp1.at<cv::Point>(i).x <= tmp1.at<cv::Point>(minIndex).x) {
                    minIndex = i;
                }
            }
            // 获取轮廓下边界起始位置和终点位置索引
            for (size_t i = int(tmp1.rows) - 1; i  > 0; i--){
                if (tmp1.at<cv::Point>(i).x >= tmp1.at<cv::Point>(maxIndex).x) {
                    maxIndex = i;
                }
            }

            if (maxIndex >= minIndex) {
                for (size_t i = minIndex; i <= maxIndex; i++) {
                    float x = float(tmp1.at<cv::Point>(i).x)*1.0079 / w * org_w;
                    float y = float(tmp1.at<cv::Point>(i).y)*1.0105 / h * org_h;
                    ps.push_back(Pointf{x, y, label, float(0), 1, prob_mask.at<float>(tmp1.at<cv::Point>(i).y,tmp1.at<cv::Point>(i).x)});
                }
            } else {
                for (size_t i = minIndex; i < int(tmp1.rows); i++) {
                    float x = float(tmp1.at<cv::Point>(i).x)*1.0079 / w * org_w;
                    float y = float(tmp1.at<cv::Point>(i).y)*1.0105 / h * org_h;
                    ps.push_back(Pointf{x, y, label, float(0), 1, prob_mask.at<float>(tmp1.at<cv::Point>(i).y,tmp1.at<cv::Point>(i).x)});
                }
                for (size_t i = 0; i <= maxIndex; i++) {
                    float x = float(tmp1.at<cv::Point>(i).x)*1.0079 / w * org_w;
                    float y = float(tmp1.at<cv::Point>(i).y)*1.0105 / h * org_h;
                    ps.push_back(Pointf{x, y, label, float(0), 1, prob_mask.at<float>(tmp1.at<cv::Point>(i).y,tmp1.at<cv::Point>(i).x)});
                }
            }
        }
        else
        {
            mask_ditan = mask;
                
            for (size_t i = 0; i < int(tmp1.rows); i++) 
            {
                if(tmp1.at<cv::Point>(i).x > min_width && tmp1.at<cv::Point>(i).x < max_width
                && tmp1.at<cv::Point>(i).y > min_height && tmp1.at<cv::Point>(i).y < max_height)
                {
                    x = float(tmp1.at<cv::Point>(i).x)*1.0079 / w * org_w;
                    y = float(tmp1.at<cv::Point>(i).y)*1.0105 / h * org_h;
                    ps.push_back(Pointf{x, y, label, float(0), 1, prob_mask.at<float>(tmp1.at<cv::Point>(i).y,tmp1.at<cv::Point>(i).x)});
                }
            }
        } 
    }
}

/**
 * @Autor: zhangdongsheng
 * @description: 创建det蒙板
 * @param {*}
 * @return {*}
 */
void create_mask(cv::Mat& mask1, int h, int w, int input_h, int input_w, std::vector<BBox>& boxes, float threshold, int class1_label, int class2_label) {
    // 遍历所有BBox
    for (auto &box : boxes) {
        // 判断是否为目标类别且score是否大于阈值
        if (box.label == class1_label && box.score > threshold) {
            // float转int,计算矩形框坐标
            int xmin = static_cast<int>(box.xmin / input_w * w);
            int ymin = static_cast<int>(box.ymin / input_h * h);
            int xmax = static_cast<int>(box.xmax / input_w * w);
            int ymax = static_cast<int>(box.ymax / input_h * h);
            std::cout << "xmin: " << xmin << "  ymin: " << ymin << "  xmax: " << xmax << "  ymax: " <<  ymax << std::endl;

            // // 创建矩形框对象
            for(int ii = MAX(xmin,0); ii < MIN(xmax, w); ii ++) {
                for(int jj = MAX(ymin, h/2); jj <  MIN(ymax, h); jj ++) {
                    mask1.at<uchar>(jj, ii) = 255;
                }
            }
        }
    }
}

/**
 * @Autor: zhangdongsheng
 * @description: 获取所有类别的轮廓点
 * @param {*}
 * @return {*}
 */
static std::vector<int>  carpet_mask;

void find_classes_contours(cv::Mat &mask_ditan, result_counter& p1, std::vector<BBox>& result_boxes, uint8_t *data, int h, int w, int input_h, int input_w, int org_h, int org_w, int& cm_distance) {
    int numclass1  = 0;
    int numclass2  = 0; 
    int numclass3  = 0; // 右侧地毯个数
    int numclass4  = 0; // 左侧地毯个数

    cv::Mat mask(h, w, CV_32FC(7), data);    // 将数据存储为cv::Mat格式(输出概率) 
    // cv::Mat mask_argmax = cv::Mat::zeros(h, w, CV_8UC1);   // 存放模型的原始分割结果(输出类别)
    // cv::Mat prob_mask   = cv::Mat::zeros(h, w, CV_32FC1);       
    cv::Mat class1      = cv::Mat::zeros(h, w, CV_8UC1); // 128x128
    // cv::Mat class1_box  = cv::Mat::zeros(h, w, CV_8UC1); // 128x128
    int max_h      = cm_distance * h / org_h;
    int min_width  = 3,         max_width  = 252;
    int min_height = max_h + 1, max_height = 188;
    // create_mask(class1_box, h, w, org_h, org_w, result_boxes, 0, 0, 1);
    for (int j = max_h ; j < h-2; j++) {  // 循环遍历矩阵的每个像素点
        for (int i = 2; i < w-2; i++) {
            cv::Vec<float, 7> pixel_mask_f = mask.at<cv::Vec<float, 7>>(j, i); // 获取当前像素点在通道上的信息
            int   max_idx = -1;
            float max_val = -1000;
            float min_val = 9999;
            float sum     = -1;
            float prob_mask = -1;
            for(int w = 0 ; w < pixel_mask_f.rows; w++) 
            {
                sum = sum + (float)pixel_mask_f[w];
                if(pixel_mask_f[w] > max_val) 
                {
                    max_val = (float)pixel_mask_f[w];
                    max_idx = w;
                }
                if(pixel_mask_f[w] < min_val) 
                {
                    min_val = (float)pixel_mask_f[w];
                }
            }
            // prob_mask.at<float>(j, i) = (max_val+ std::abs(min_val))/(sum + 7 * std::abs(min_val)); // 归一化到 0-1 之间
            prob_mask = (max_val+ std::abs(min_val))/(sum + 7 * std::abs(min_val));
            // 根据阈值筛选后的分割结果,赋值给各类灰度图
            if (max_idx == 2) {
                class1.at<uchar>(j, i) = 255;
                numclass1 ++;
                if(i > 2.5 * w / 5)
                {
                    numclass3 ++; 
                }
                else
                {
                    numclass4 ++; 
                }

                if((i % int((j-94)/5 + 1)  == 0) && (j % int((j-94)/25 + 1) == 0))
                {
                    float x = float(i)*1.0079 / w * org_w;
                    float y = float(j)*1.0105 / h * org_h;
                    p1.ps.push_back(Pointf{x, y, 2, float(0), 1, prob_mask});
                    p1.id = 1;
                }
                    // if((class1.at<uchar>(j, i) != class1.at<uchar>(j - 1, i) && class1.at<uchar>(j - 1, i) == 255))
                    // {
                    //     float x = float(i)*1.0079 / w * org_w;
                    //     float y = float(j)*1.0105 / h * org_h;
                    //     p1.ps.push_back(Pointf{x, y, 2, float(0), 1, prob_mask});
                    //     p1.id = 1;
                    //     continue;
                    // }
            }
            if (max_idx == 6) {
                class1.at<uchar>(j, i) = 200;

                if((i % int((j-94)/2 + 1)  == 0) && (j % int((j-94)/10 + 1) == 0))
                {
                    float x = float(i)*1.0079 / w * org_w;
                    float y = float(j)*1.0105 / h * org_h;
                    p1.ps.push_back(Pointf{x, y, 6, float(0), 1, prob_mask});
                    p1.id = 1;
                }
                    // if((class1.at<uchar>(j, i) != class1.at<uchar>(j - 1, i) && class1.at<uchar>(j - 1, i) == 200))
                    // {
                    //     float x = float(i)*1.0079 / w * org_w;
                    //     float y = float(j)*1.0105 / h * org_h;
                    //     p1.ps.push_back(Pointf{x, y, 6, float(0), 1, prob_mask});
                    //     p1.id = 1;
                    // }
            }
            if (max_idx == 4) {
                class1.at<uchar>(j, i) = 125;
                numclass2 ++;
            }
        }
    }

    if(numclass1 > 700)
    {
        if(numclass4 > 0 && numclass3 > 0)
        {
            class1.at<uchar>(0, 0) = 50; // 地毯在双边
        }
        else
        {
            class1.at<uchar>(0, 0) = 0; // 地毯在双边
        }
        carpet_mask.push_back(1);
        mask_ditan = class1;
    }
    else
    {
        if(numclass4 > 0 && numclass3 > 0)
        {
            carpet_mask.push_back(1);
            class1.at<uchar>(0, 0) = 50; // 地毯在双边
            mask_ditan = class1; 
 
        }
        else
        {
            class1.at<uchar>(0, 0) = 0; // 地毯在双边
            carpet_mask.push_back(0);
            p1.ps.clear();
            p1.id = 0;
        }

    }
    if(carpet_mask.size() > 3)
    {
        carpet_mask.erase(carpet_mask.begin());
    }

    // 寻找某类别的全轮廓
    // if(numclass1 > 400 && std::accumulate(carpet_mask.begin(),carpet_mask.end(),0) >= 2) {
    //     find_contours(mask_ditan, p1.ps, class1, prob_mask, h, w, org_h, org_w, 2., 25, 1, false, p1.id);
    // }
    // if(numclass1 > 700 )     //  输出地毯全轮廓，标签为 9
    // {
    //     find_contours(mask_ditan, p1.ps, class1, prob_mask, h, w, org_h, org_w, 0., 25, 1, false, p1.id);
    // }
    //  输出U形椅底座下轮廓
    // if(numclass2 > 0) {
    //     find_contours(mask_ditan, p1.ps, class2, prob_mask, h, w, org_h, org_w, 4., 10, 1, true, p1.id);
    // }
}

static float deqnt_affine_to_f32(float qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

// 零拷贝 增加int2float
void find_classes_contours_zero_copy(cv::Mat &mask_ditan, result_counter& p1, std::vector<BBox>& result_boxes, int8_t *data, int h, int w, int input_h, int input_w, int org_h, int org_w, int& cm_distance, int32_t zp, float scale) {
    int numclass1  = 0;
    int numclass2  = 0; 
    int numclass3  = 0; // 右侧地毯个数
    int numclass4  = 0; // 左侧地毯个数

    std::vector<Pointf> tmp_ps;

    cv::Mat mask(h, w, CV_MAKETYPE(CV_8S, 7), data);    // 将数据存储为cv::Mat格式(输出概率) 
    cv::Mat mask_argmax = cv::Mat::zeros(h, w, CV_8UC1);   // 存放模型的原始分割结果(输出类别)
    cv::Mat prob_mask   = cv::Mat::zeros(h, w, CV_32FC1);       
    cv::Mat class1      = cv::Mat::zeros(h, w, CV_8UC1); // 128x128
    cv::Mat class1_box  = cv::Mat::zeros(h, w, CV_8UC1); // 128x128
    int max_h      = cm_distance * h / org_h;
    int min_width  = 3,         max_width  = 252;
    int min_height = max_h + 1, max_height = 188;
    float output_float;
    // create_mask(class1_box, h, w, org_h, org_w, result_boxes, 0, 0, 1);
    for (int j = max_h ; j < h-2; j++) {  // 循环遍历矩阵的每个像素点
        for (int i = 2; i < w-2; i++) {
            cv::Vec<int8_t, 7> pixel_mask_f = mask.at<cv::Vec<int8_t, 7>>(j, i); // 获取当前像素点在通道上的信息
            int   max_idx = -1;
            float max_val = -1000;
            float min_val = 9999;
            float sum     = -1;
            for(int w = 0 ; w < pixel_mask_f.rows; w++) 
            {
                output_float = deqnt_affine_to_f32(pixel_mask_f[w], zp, scale);
                sum = sum + output_float;
                if(output_float > max_val) 
                {
                    max_val = output_float;
                    max_idx = w;
                }
                if(output_float < min_val) 
                {
                    min_val = output_float;
                }
            }
            prob_mask.at<float>(j, i) = (max_val+ std::abs(min_val))/(sum + 7 * std::abs(min_val)); // 归一化到 0-1 之间
            // 根据阈值筛选后的分割结果,赋值给各类灰度图
            if (max_idx == 2) {
                class1.at<uchar>(j, i) = 255;
                numclass1 ++;
                if(i > 2.5 * w / 5)
                {
                    numclass3 ++; 
                }
                else
                {
                    numclass4 ++; 
                }

                if((i % int((j-94)/5 + 1)  == 0) && (j % int((j-94)/25 + 1) == 0))
                {
                    float x = float(i)*1.0079 / w * org_w;
                    float y = float(j)*1.0105 / h * org_h;
                    p1.ps.push_back(Pointf{x, y, 2, float(0), 1, prob_mask.at<float>(j, i)});
                    p1.id = 1;
                }
                if(i > min_width && i < max_width && j > min_height && j < max_height)
                {
                    if((class1.at<uchar>(j, i) != class1.at<uchar>(j - 1, i) && class1.at<uchar>(j - 1, i) == 255))
                    {
                        float x = float(i)*1.0079 / w * org_w;
                        float y = float(j)*1.0105 / h * org_h;
                        p1.ps.push_back(Pointf{x, y, 2, float(0), 1, prob_mask.at<float>(j, i)});
                        p1.id = 1;
                    }
                }
            }
            if (max_idx == 6) {
                class1.at<uchar>(j, i) = 200;

                if((i % int((j-94)/2 + 1)  == 0) && (j % int((j-94)/10 + 1) == 0))
                {
                    float x = float(i)*1.0079 / w * org_w;
                    float y = float(j)*1.0105 / h * org_h;
                    p1.ps.push_back(Pointf{x, y, 6, float(0), 1, prob_mask.at<float>(j, i)});
                    p1.id = 1;
                }
                if(i > min_width && i < max_width && j > min_height && j < max_height)
                {
                    if((class1.at<uchar>(j, i) != class1.at<uchar>(j - 1, i) && class1.at<uchar>(j - 1, i) == 200))
                    {
                        float x = float(i)*1.0079 / w * org_w;
                        float y = float(j)*1.0105 / h * org_h;
                        p1.ps.push_back(Pointf{x, y, 6, float(0), 1, prob_mask.at<float>(j, i)});
                        p1.id = 1;
                    }
                }
            }
            if (max_idx == 4) {
                class1.at<uchar>(j, i) = 125;
                numclass2 ++;
                float x = float(i)*1.0079 / w * org_w;
                float y = float(j)*1.0105 / h * org_h;
                tmp_ps.push_back(Pointf{x, y, 4, float(0), 1, prob_mask.at<float>(j, i)});
                p1.id = 1;
            }
        }
    }

    if(numclass1 > 700)
    {
        if(numclass4 > 0 && numclass3 > 0)
        {
            class1.at<uchar>(0, 0) = 50; // 地毯在双边
        }
        else
        {
            class1.at<uchar>(0, 0) = 0; // 地毯在双边
        }
        carpet_mask.push_back(1);
        mask_ditan = class1;
    }
    else
    {
        if(numclass4 > 0 && numclass3 > 0)
        {
            carpet_mask.push_back(1);
            class1.at<uchar>(0, 0) = 50; // 地毯在双边
            mask_ditan = class1; 
 
        }
        else
        {
            class1.at<uchar>(0, 0) = 0; // 地毯在双边
            carpet_mask.push_back(0);
            p1.ps.clear();
            p1.id = 0;
        }

    }
    if(carpet_mask.size() > 3)
    {
        carpet_mask.erase(carpet_mask.begin());
    }

    std::map<int, Pointf> bottomContour = extractBottomContourFromPoints(tmp_ps);
    for (const auto& pair : bottomContour) {
        p1.ps.push_back(pair.second);
    }

    // 寻找某类别的全轮廓
    // if(numclass1 > 400 && std::accumulate(carpet_mask.begin(),carpet_mask.end(),0) >= 2) {
    //     find_contours(mask_ditan, p1.ps, class1, prob_mask, h, w, org_h, org_w, 2., 25, 1, false, p1.id);
    // }
    // if(numclass1 > 700 )     //  输出地毯全轮廓，标签为 9
    // {
    //     find_contours(mask_ditan, p1.ps, class1, prob_mask, h, w, org_h, org_w, 0., 25, 1, false, p1.id);
    // }
    //  输出U形椅底座下轮廓
    // if(numclass2 > 0) {
    //     find_contours(mask_ditan, p1.ps, class2, prob_mask, h, w, org_h, org_w, 4., 10, 1, true, p1.id);
    // }
}

std::map<int, Pointf> extractBottomContourFromPoints(const std::vector<Pointf>& points) {
    std::map<int, Pointf> xToBottomPoint;
    
    if (points.empty()) return xToBottomPoint;
    
    for (const auto& point : points) {
        int x_int = static_cast<int>(point.x);
        
        if (xToBottomPoint.find(x_int) == xToBottomPoint.end() || 
            point.y > xToBottomPoint[x_int].y) {
            xToBottomPoint[x_int] = point;
        }
    }
    
    return xToBottomPoint;
}

void find_classes_contours_fast_zero_copy(cv::Mat &mask_ditan, result_counter& p1, std::vector<BBox>& result_boxes, int8_t *data, int h, int w, int input_h, int input_w, int org_h, int org_w, int& cm_distance) {
    int numclass1  = 0;
    int numclass2  = 0; 
    int numclass3  = 0; // 右侧地毯个数
    int numclass4  = 0; // 左侧地毯个数
    
    cv::Mat class1      = cv::Mat::zeros(h, w, CV_8UC1); // 128x128
    int max_h      = cm_distance * h / org_h;
    int min_width  = 3,         max_width  = 252;
    int min_height = max_h + 1, max_height = 188;

    data += max_h * 256 * 7 + 2 * 7;

    for (int j = max_h ; j < h-2; j++) {  // 循环遍历矩阵的每个像素点
        for (int i = 2; i < w-2; i++) {
            float prob_mask = -1;

            if((i % int((j-94)/5 + 1)  == 0) && (j % int((j-94)/25 + 1) == 0))
            {
                char max_idx = std::max_element(data, data + 7) - data;
                
                // 根据阈值筛选后的分割结果,赋值给各类灰度图
                if (max_idx == 2) {
                    class1.at<uchar>(j, i) = 255;
                    numclass1 ++;
                    if(i > 2.5 * w / 5)
                    {
                        numclass3 ++; 
                    }
                    else
                    {
                        numclass4 ++; 
                    }

                    float x = float(i)*1.0079 / w * org_w;
                    float y = float(j)*1.0105 / h * org_h;
                    p1.ps.push_back(Pointf{x, y, 2, float(0), 1, prob_mask});
                    p1.id = 1;

                }
                if (max_idx == 6) {
                    class1.at<uchar>(j, i) = 200;

                    float x = float(i)*1.0079 / w * org_w;
                    float y = float(j)*1.0105 / h * org_h;
                    p1.ps.push_back(Pointf{x, y, 6, float(0), 1, prob_mask});
                    p1.id = 1;

                }
                if (max_idx == 4) {
                    class1.at<uchar>(j, i) = 125;
                    numclass2 ++;
                }
            }
            data += 7;
        }
        data += 2 * 7 + 2 * 7;  // (2 ~ w-2)

    }

    if(numclass1 > 700)
    {
        if(numclass4 > 0 && numclass3 > 0)
        {
            class1.at<uchar>(0, 0) = 50; // 地毯在双边
        }
        else
        {
            class1.at<uchar>(0, 0) = 0; // 地毯在双边
        }
        carpet_mask.push_back(1);
        mask_ditan = class1;
    }
    else
    {
        if(numclass4 > 0 && numclass3 > 0)
        {
            carpet_mask.push_back(1);
            class1.at<uchar>(0, 0) = 50; // 地毯在双边
            mask_ditan = class1; 
 
        }
        else
        {
            class1.at<uchar>(0, 0) = 0; // 地毯在双边
            carpet_mask.push_back(0);
            p1.ps.clear();
            p1.id = 0;
        }

    }
    if(carpet_mask.size() > 3)
    {
        carpet_mask.erase(carpet_mask.begin());
    }

    // 寻找某类别的全轮廓
    // if(numclass1 > 400 && std::accumulate(carpet_mask.begin(),carpet_mask.end(),0) >= 2) {
    //     find_contours(mask_ditan, p1.ps, class1, prob_mask, h, w, org_h, org_w, 2., 25, 1, false, p1.id);
    // }
    // if(numclass1 > 700 )     //  输出地毯全轮廓，标签为 9
    // {
    //     find_contours(mask_ditan, p1.ps, class1, prob_mask, h, w, org_h, org_w, 0., 25, 1, false, p1.id);
    // }
    //  输出U形椅底座下轮廓
    // if(numclass2 > 0) {
    //     find_contours(mask_ditan, p1.ps, class2, prob_mask, h, w, org_h, org_w, 4., 10, 1, true, p1.id);
    // }
}

}
