
#include "utils.h"
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"


namespace sweeper_ai
{
    //y用RGA的方式进行 rezise
    int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
    {
        im_rect src_rect;
        im_rect dst_rect;
        memset(&src_rect, 0, sizeof(src_rect));
        memset(&dst_rect, 0, sizeof(dst_rect));
        size_t img_width  = image.cols;
        size_t img_height = image.rows;
        if (image.type() != CV_8UC3)
        {
            printf("source image type is %d!\n", image.type());
            return -1;
        }
        size_t target_width  = target_size.width;
        size_t target_height = target_size.height;
        src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
        int ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret)
        {
            fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
            return -1;
        }
        IM_STATUS STATUS = imresize(src, dst);
        return 0;
    }

    //图像 rezise+底部padding
    void static_resize(cv::Mat& img, cv::Mat& resize_image, const int& INPUT_W , const int& INPUT_H) 
    {
        float r     = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows;
        cv::resize(img, img, cv::Size(unpad_w, unpad_h));
        img.copyTo(resize_image(cv::Rect(0, 0, img.cols, img.rows)));
    }

    void static_resize_top_bottom(cv::Mat& img, cv::Mat& resize_image, const int& INPUT_W , const int& INPUT_H) 
    {
        float r      = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
        int unpad_w  = r * img.cols;
        int unpad_h  = r * img.rows;
        cv::resize(img, img, cv::Size(unpad_w, unpad_h));
        int pad_up   = (INPUT_H - unpad_h) / 2;
        int pad_down = INPUT_H - unpad_h - pad_up;
        // 将图像复制到输出图像中央的部分，上下各padding一半
        img.copyTo(resize_image(cv::Rect(0, pad_up, img.cols, img.rows)));
    }


    void eco_resize(cv::Mat& bgr, cv::Mat& resize_image, const int& output_w , const int& output_h, const EcoResizeTypeS& resize_type) 
    {
        #ifdef D_DEBUG
            std::cout << "resize_type = " << resize_type << "- resize" << std::endl;
        #endif
        // resize 的方式 0 默认resize ， 1 使用 底部padding+resize 的方式， 2使用上下padding+resize的方式
        if (EM_NORMAL_RESIZE   == resize_type)
        {
            cv::Mat bgr_clone = bgr.clone();
            ////这里bgr和resize_image的内存地址不一样,做了一次拷贝动作
            cv::resize(bgr_clone, resize_image, cv::Size(output_w, output_h));
        }
        else if (EM_PAD_RESIZE == resize_type)
        {
            static_resize(bgr, resize_image, output_w, output_h);
        }
        else if (EM_PAD_RESIZE_TOP_BOTTOM == resize_type)
        {
            static_resize_top_bottom(bgr, resize_image, output_w, output_h);
        }
        else if (EM_RGA_NORMAL_RESIZE == resize_type)
        {
            rga_buffer_t src;
            rga_buffer_t dst;
            memset(&src, 0, sizeof(src));
            memset(&dst, 0, sizeof(dst));
            int ret;
            ret = resize_rga(src, dst, bgr, resize_image, cv::Size(output_w, output_h));
            if (ret != 0)
            {
                fprintf(stderr, "resize with rga error\n");
            }
            // cv::imwrite("./output_pic/resize_image_.jpg", resize_image);
        }
    }

    int rga_copy(rga_buffer_t &src, rga_buffer_t &dst, size_t img_width, size_t img_height, void *src_img, void *dst_img)
    {
        src = wrapbuffer_virtualaddr(src_img, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr(dst_img, img_width, img_height, RK_FORMAT_RGB_888);
        IM_STATUS STATUS = imcopy(src, dst);
        return STATUS;
    }

    //  rknn_tensor_attr　ｌｏｇ信息输出
    static void dump_tensor_attr(rknn_tensor_attr *attr)
    {
        printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
    }

    //  读取模型 fp:模型文件　　ofst：数据偏移量　　sz：模型所占内存大小
    static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
    {
        unsigned char *data;
        int ret;

        data = NULL;

        if (NULL == fp)
        {
            return NULL;
        }

        ret = fseek(fp, ofst, SEEK_SET);
        if (ret != 0)
        {
            printf("blob seek failure.\n");
            return NULL;
        }

        data = (unsigned char *)malloc(sz);
        if (data == NULL)
        {
            printf("buffer malloc failure.\n");
            return NULL;
        }
        ret = fread(data, 1, sz, fp);
        return data;
    }

    //　读取目标检测算法模型　　filename：模型路径　　　model_size：模型大小
    unsigned char *load_model(const char *filename, int *model_size)
    {
        FILE *fp;
        unsigned char *data;

        fp = fopen(filename, "rb");
        if (NULL == fp)
        {
            printf("Open file %s failed.\n", filename);
            return NULL;
        }

        fseek(fp, 0, SEEK_END);
        int size = ftell(fp);

        data = load_data(fp, 0, size);

        fclose(fp);

        *model_size = size;
        return data;
    }

    //　字符分割
    void str_split(std::string &all_str, const std::string delimit, std::vector<float> &result)
    {
        size_t pos = all_str.find(delimit);
        all_str += delimit;
        while (pos != std::string::npos)
        {
            result.push_back(atof(all_str.substr(0, pos).c_str()));
            all_str = all_str.substr(pos + 1);
            pos = all_str.find(delimit);
        }
    }

    //　打印信息并且获取模型输入输出节点维度与反量化参数
    int rknn_get_ctx_attr(EcoRknnModelParams &modelparams)
    {
        int ret(0);
        //　从句柄中确定推断引擎版本
        rknn_sdk_version version;                      
        ret = rknn_query(modelparams.ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0)
        {
            printf("rknn_query rknn_sdk_version error ret=%d\n", ret);
            return ret;
        }
        // printf("sdk version: %s driver version: %s\n", version.api_version,
        //     version.drv_version);


        // 输入节点　input_attrs　属性
        rknn_tensor_attr input_attrs[modelparams.io_num.n_input];    
        memset(input_attrs, 0, sizeof(input_attrs));

        for (int i = 0; i < modelparams.io_num.n_input; i++)
        {
            input_attrs[i].index = i;
            //　获取输入节点属性
            ret = rknn_query(modelparams.ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0)
            {
                printf("rknn_query input_attrs error ret = %d\n", ret);
                return ret;
            }

            //　获取模型输入节点的输入大小
            if (input_attrs[i].fmt == RKNN_TENSOR_NCHW)
            {
                // printf("model is NCHW input fmt\n");
                (modelparams.nmodelinputchannel_)[i] = input_attrs[i].dims[1];
                (modelparams.nmodelinputheight_)[i]  = input_attrs[i].dims[2];
                (modelparams.nmodelinputweith_)[i]   = input_attrs[i].dims[3];
            }
            else
            {
                // printf("model is NHWC input fmt\n");
                (modelparams.nmodelinputheight_)[i]  = input_attrs[i].dims[1];
                (modelparams.nmodelinputweith_)[i]   = input_attrs[i].dims[2];
                (modelparams.nmodelinputchannel_)[i] = input_attrs[i].dims[3];
            }
            /////下面这个函数是用来表示模型输入端是int类型还是float类型的
            #ifdef D_DEBUG
                dump_tensor_attr(&(input_attrs[i]));
            #endif      
            // std::cout<<"**********************"<<std::endl;
            dump_tensor_attr(&(input_attrs[i]));  
        }


        //　输出节点属性　output_attrs　　属性
        rknn_tensor_attr output_attrs[modelparams.io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < modelparams.io_num.n_output; i++)
        {
            output_attrs[i].index = i;
            // 获取输出节点属性
            ret = rknn_query(modelparams.ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            // ret = rknn_query(modelparams.ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0)
            {
                printf("rknn_query output_attrs  error ret=%d\n", ret);
                return ret;
            }

            //　获取模型输入节点的输入大小
            if (output_attrs[i].fmt == RKNN_TENSOR_NCHW)
            {
                // printf("model is NCHW output fmt\n");
                (modelparams.nmodeloutputchannel_)[i] = output_attrs[i].dims[1];
                (modelparams.nmodeloutputheight_)[i]  = output_attrs[i].dims[2];
                (modelparams.nmodeloutputweith_)[i]   = output_attrs[i].dims[3];
            }
            else
            {
                // printf("model is NHWC output fmt\n");
                (modelparams.nmodeloutputheight_)[i]  = output_attrs[i].dims[1];
                (modelparams.nmodeloutputweith_)[i]   = output_attrs[i].dims[2];
                (modelparams.nmodeloutputchannel_)[i] = output_attrs[i].dims[3];
            }
            modelparams.out_scales.push_back(output_attrs[i].scale);
            modelparams.out_zps.push_back(output_attrs[i].zp);
            
            // dump_tensor_attr(&(output_attrs[i]));
            /////下面这句代码是用来输出模型输出端是int类型还是float类型的
            #ifdef D_DEBUG
                dump_tensor_attr(&(output_attrs[i]));
            #endif
            dump_tensor_attr(&(output_attrs[i]));
        }
        return ret;
    }


    //　零拷贝 提前申请内存 打印信息并且获取模型输入输出节点维度与反量化参数
    int rknn_get_ctx_attr_zero_copy(EcoRknnModelParams &modelparams)
    {
        int ret(0);
        //　从句柄中确定推断引擎版本
        rknn_sdk_version version;                      
        ret = rknn_query(modelparams.ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0)
        {
            printf("rknn_query rknn_sdk_version error ret=%d\n", ret);
            return ret;
        }

        // Get Model Input Info
        rknn_tensor_attr input_attrs[modelparams.io_num.n_input];    
        memset(input_attrs, 0, sizeof(input_attrs));
        for (int i = 0; i < modelparams.io_num.n_input; i++) {
            input_attrs[i].index = i;
            ret = rknn_query(modelparams.ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0)
            {
                printf("rknn_query input_attrs error ret = %d\n", ret);
                return ret;
            }

            //　获取模型输入节点的输入大小
            if (input_attrs[i].fmt == RKNN_TENSOR_NCHW)
            {
                // printf("model is NCHW input fmt\n");
                (modelparams.nmodelinputchannel_)[i] = input_attrs[i].dims[1];
                (modelparams.nmodelinputheight_)[i]  = input_attrs[i].dims[2];
                (modelparams.nmodelinputweith_)[i]   = input_attrs[i].dims[3];
            }
            else
            {
                // printf("model is NHWC input fmt\n");
                (modelparams.nmodelinputheight_)[i]  = input_attrs[i].dims[1];
                (modelparams.nmodelinputweith_)[i]   = input_attrs[i].dims[2];
                (modelparams.nmodelinputchannel_)[i] = input_attrs[i].dims[3];
            }

            dump_tensor_attr(&(input_attrs[i]));
        }

        // default input type is int8 (normalize and quantize need compute in outside)
        // if set uint8, will fuse normalize and quantize to npu
        input_attrs[0].type = RKNN_TENSOR_UINT8;
        modelparams.input_mems[0] = rknn_create_mem(modelparams.ctx, input_attrs[0].size_with_stride);

        // Set input tensor memory
        ret = rknn_set_io_mem(modelparams.ctx, modelparams.input_mems[0], &input_attrs[0]);
        if (ret < 0) 
        {
            modelparams.input_mems[0] = NULL;
            printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return ret;
        }

        // Get Model Output Info
        rknn_tensor_attr output_attrs[modelparams.io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < modelparams.io_num.n_output; i++) {
            output_attrs[i].index = i;
            ret = rknn_query(modelparams.ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) 
            {
                printf("rknn_query fail! ret=%d\n", ret);
                return ret;
            }

            //　获取模型输入节点的输入大小
            if (output_attrs[i].fmt == RKNN_TENSOR_NCHW)
            {
                // printf("model is NCHW output fmt\n");
                (modelparams.nmodeloutputchannel_)[i] = output_attrs[i].dims[1];
                (modelparams.nmodeloutputheight_)[i]  = output_attrs[i].dims[2];
                (modelparams.nmodeloutputweith_)[i]   = output_attrs[i].dims[3];
            }
            else
            {
                // printf("model is NHWC output fmt\n");
                (modelparams.nmodeloutputheight_)[i]  = output_attrs[i].dims[1];
                (modelparams.nmodeloutputweith_)[i]   = output_attrs[i].dims[2];
                (modelparams.nmodeloutputchannel_)[i] = output_attrs[i].dims[3];
            }
            modelparams.out_scales.push_back(output_attrs[i].scale);
            modelparams.out_zps.push_back(output_attrs[i].zp);

            dump_tensor_attr(&(output_attrs[i]));
        }

        // Set output tensor memory
        // modelparams.output_mems = (rknn_tensor_mem**)malloc(modelparams.io_num.n_output * sizeof(rknn_tensor_mem*));
        for (uint32_t i = 0; i < modelparams.io_num.n_output; ++i) {
            modelparams.output_mems[i] = rknn_create_mem(modelparams.ctx, output_attrs[i].size_with_stride);
            ret = rknn_set_io_mem(modelparams.ctx, modelparams.output_mems[i], &output_attrs[i]);
            if (ret < 0) 
            {
                modelparams.output_mems[i] = NULL;
                printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
                return ret;
            }
        }

        return ret;
    }

    // static void read_inner(int camid, CameraInfo &inner_para, rapidjson::Value &cam_param)
    EcoEStatus read_inner(CameraInfo &inner_para, std::string RGBD1_cam_yaml_path)
    {
        EcoEStatus ecoestatus(EStatus_Success);
        const char* keys[] = {
            "Camera.fx",
            "Camera.fy",
            "Camera.cx",
            "Camera.cy",
            "Camera.k1",
            "Camera.k2",
            "Camera.p1",
            "Camera.p2",
            "Camera.k3",
            "Camera.k4",
            "Camera.k5",
            "Camera.k6",
        };

        double paraCam[12] = {};
        // 打开 JSON 文件
        std::ifstream ifs_json(RGBD1_cam_yaml_path);
        if (!ifs_json.is_open()) 
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout << "Error opening inner json file "<< std::endl;
            return ecoestatus;
        }
        else 
        {
            // 将文件内容解析为 JSON 文档
            rapidjson::IStreamWrapper isw(ifs_json);
            rapidjson::Document document;
            document.ParseStream(isw);
            // 关闭文件
            ifs_json.close();

            if (document.HasParseError()) {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "Error parsing JSON file." << std::endl;
                return ecoestatus;
            }
            // 获取 JSON 文档中的参数
            if (document.HasMember(keys[0]) && document.HasMember(keys[1]) && document.HasMember(keys[2]) && document.HasMember(keys[3]) &&
                document.HasMember(keys[4]) && document.HasMember(keys[5]) && document.HasMember(keys[6]) && document.HasMember(keys[7]) &&
                document.HasMember(keys[8]) && document.HasMember(keys[9]) && document.HasMember(keys[10]) && document.HasMember(keys[11]))
            {
                inner_para.RGB.fx = document[keys[0]].GetFloat();
                inner_para.RGB.fy = document[keys[1]].GetFloat();
                inner_para.RGB.cx = document[keys[2]].GetFloat();
                inner_para.RGB.cy = document[keys[3]].GetFloat();

                inner_para.RGB.k1 = document[keys[4]].GetFloat();
                inner_para.RGB.k2 = document[keys[5]].GetFloat();
                inner_para.RGB.p1 = document[keys[6]].GetFloat();
                inner_para.RGB.p2 = document[keys[7]].GetFloat();

                inner_para.RGB.k3 = document[keys[8]].GetFloat();
                inner_para.RGB.k4 = document[keys[9]].GetFloat();
                inner_para.RGB.k5 = document[keys[10]].GetFloat();
                inner_para.RGB.k6 = document[keys[11]].GetFloat();

            }                
            else 
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "Json key not found: " << std::endl;
                return ecoestatus;
            }

        }

        return ecoestatus;

    }

    EcoEStatus read_ir_inner(CameraInfo &inner_para, std::string RGBD1_cam_yaml_path)
    {
        EcoEStatus ecoestatus(EStatus_Success);

        // 打开 JSON 文件
        std::ifstream ifs_json(RGBD1_cam_yaml_path);
        if (!ifs_json.is_open()) 
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout << "Error opening inner json file "<< std::endl;
            return ecoestatus;
        }
        else 
        {
            // 将文件内容解析为 JSON 文档
            rapidjson::IStreamWrapper isw(ifs_json);
            rapidjson::Document document;
            document.ParseStream(isw);
            // 关闭文件
            ifs_json.close();

            if (document.HasParseError()) {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "Error parsing JSON file." << std::endl;
                return ecoestatus;
            }
            // 获取 JSON 文档中的参数
            if (document.HasMember("camera") && document.HasMember("discoeff"))
            {
                std::vector<float> IOU_tmpVec;
                rapidjson::Value IOU_threshold_of_each = document["camera"].GetArray();   
                for (rapidjson::SizeType j = 0; j < IOU_threshold_of_each.Size(); j++)
                {
                    IOU_tmpVec.push_back(IOU_threshold_of_each[j].GetFloat());
                }  

                inner_para.IR.fx = IOU_tmpVec[0];
                inner_para.IR.fy = IOU_tmpVec[4];
                inner_para.IR.cx = IOU_tmpVec[2];
                inner_para.IR.cy = IOU_tmpVec[5];
                inner_para.ir_rangle = 90.0 / 180.0 * M_PI;
                inner_para.ir_Height = 6.48;

                std::vector<float> discoeff_tmpVec;
                rapidjson::Value discoeff_of_each = document["discoeff"].GetArray();   
                for (rapidjson::SizeType j = 0; j < discoeff_of_each.Size(); j++)
                {
                    discoeff_tmpVec.push_back(discoeff_of_each[j].GetFloat());
                }  

                inner_para.IR.k1 = discoeff_tmpVec[0];
                inner_para.IR.k2 = discoeff_tmpVec[1];
                inner_para.IR.p1 = discoeff_tmpVec[2];
                inner_para.IR.p2 = discoeff_tmpVec[3];
                inner_para.IR.k3 = discoeff_tmpVec[4];
                inner_para.IR.k4 = discoeff_tmpVec[5];
                inner_para.IR.k5 = discoeff_tmpVec[6];
                inner_para.IR.k6 = discoeff_tmpVec[7];
            }                
            else 
            {
                ecoestatus = EStatus_InvalidParameter;
                std::cout << "Json key not found: " << std::endl;
                return ecoestatus;
            }
        }
        return ecoestatus;
    }
    ///读取外参参数--暂时没用，暂时留存
    static void read_outer(int camid, CameraInfo &outer_para, rapidjson::Value &cam_param)
    {
        if(cam_param.HasMember("x"))
        {
            outer_para.Length = cam_param["x"].GetFloat();
        }
        if(cam_param.HasMember("z"))
        {
            outer_para.Height = cam_param["z"].GetFloat();
        }
        if(cam_param.HasMember("pitch"))
        {
            outer_para.rangle = 90 - cam_param["pitch"].GetFloat() / PI * 180;
        }

        std::cout <<  "camid = " <<  camid     << "  Length = "  <<  outer_para.Length
            << "  Height = "  << outer_para.Height  << "  angle = "  <<  outer_para.rangle   << std::endl;   

    }


     EcoEStatus load_caminfo(int ncamid, std::string RGBD1_cam_yaml_path, CameraInfo& RGBDup)
    {
        EcoEStatus ecoestatus(EStatus_Success);

        if(-1 == access(RGBD1_cam_yaml_path.c_str(), 0))
        {
            ecoestatus = EStatus_InvalidParameter;
            std::cout << "have no cam_para file " << RGBD1_cam_yaml_path << std::endl;
            return ecoestatus;
        }
        if(ncamid == 0)
        {
            ////在这里写上读取内参json文件的代码
            ecoestatus = read_inner(RGBDup, RGBD1_cam_yaml_path);
        }
        else
        {
            ecoestatus = read_ir_inner(RGBDup, RGBD1_cam_yaml_path);
        }

        return ecoestatus;
    }



    // // hwc 转 chw
    void nhwc_to_nchw(int8_t ** src, int8_t ** det, int weith_, int height_, int channel_)
    {
        // nhwc 转 nchw
        int stride = weith_ * height_;
        int out_count = stride * channel_;
        for (size_t i = 0; i != stride; ++i)
        {
            for (size_t c = 0; c != channel_; ++c)
            {
                (*det)[c * stride + i] = (*src)[i * channel_ + c];
                // std::cout << "src["<< i * channel_ + c  <<"] = " <<  (*src)[i * channel_ + c] << std::endl; 
                // std::cout << "det["<< c * stride + i <<"] = " <<  (*det)[c * stride + i] << std::endl; 
            }    
        }
        memcpy((*src), (*det), sizeof(int8_t) * weith_ * height_ * channel_);
    }

    // // chw 转 hwc
    void nchw_to_nhwc(int8_t ** src, int8_t ** det, int weith_, int height_, int channel_)
    {
        int stride = weith_ * height_;
        for (size_t c = 0; c != channel_; ++c)
        {
            int t = c * stride;
            for (size_t i = 0; i != stride; ++i)
            {
                (*det)[i * channel_ + c] = (*src)[t + i];
                // std::cout << "src["<< t + i  <<"] = " << (int) (*src)[t + i] << std::endl; 
                // std::cout << "det["<< i * channel_ + c <<"] = " <<  (*det)[i * channel_ + c] << std::endl; 
            }
        } 
        memcpy((*src), (*det), sizeof(int8_t) * weith_ * height_ * channel_);   
    }


 void lds2pixel(std::vector< std::vector<float> >& lds_map, int64_t& image_time, EcoAInterfaceDeebotStatus_t& ImagePose, 
    EcoAInterfaceLdsData_t& LdsData, EcoAInterfaceSlData_t& SLSData, CamerainnerInfo& paramCam, std::vector<float>& Tcl)
    {
        if (Tcl.size() < 12)
        {
            // Tcl = { 1, 0, 0, 0, 0, -1, 0, 1, 0, 26.0, 22.5,  -24.0 };
            Tcl = { 1, 0, 0, 0, 0, -1, 0, 1, 0, 11.1, 13.67, -23.88};   //
        }

        std::vector<float> point_lds(3,0);
        std::vector<float> point_cam(3,0);
        double angle = (-57) * M_PI / 180;

        int col(-1), row(-1); 
        // float dist(-1);

        for (int i = LDS_POINT_CNT - 1; i > 0; i--) 
        {

            // 计算新坐标
            point_lds[0] = LdsData.ldsPoint[i].x * cos(angle) - LdsData.ldsPoint[i].y * sin(angle);
            point_lds[1] = LdsData.ldsPoint[i].x * sin(angle) + LdsData.ldsPoint[i].y * cos(angle);

            if (point_lds[1] > 0.1)
            {

                // if (dist < 10000) 
                // {
                    // 点云转相机坐标系
                    point_cam = { Tcl[0] * point_lds[0] + Tcl[1] * point_lds[1] + Tcl[9],
                                  Tcl[3] * point_lds[0] + Tcl[4] * point_lds[1] + Tcl[10],
                                  Tcl[6] * point_lds[0] + Tcl[7] * point_lds[1] + Tcl[11] };

                    // paramCam：相机内参，{cx, cy, fx ,fy}
                    // col = int((paramCam[2] / 1.5) * point_cam[0] / point_cam[2] + (paramCam[0] - 320) + 0.5);
                    // row = int((paramCam[3] / 1.5) * point_cam[1] / point_cam[2] + (paramCam[1] - 60)  + 0.5);

                    col = int((paramCam.fx) * point_cam[0] / point_cam[2] + (paramCam.cx) + 0.5);
                    row = int((paramCam.fy) * point_cam[1] / point_cam[2] + (paramCam.cy) + 0.5);

                    if(col < 0 || col >= 1280 || row < 0 || row >= 960)
                    {
                        continue;
                    }           
                    else
                    {   
                        lds_map[int(col / 128)].push_back(row);
                        lds_map[int(col / 128)].push_back(col);
                        lds_map[int(col / 128)].push_back(LdsData.ldsPoint[i].x);
                        lds_map[int(col / 128)].push_back(LdsData.ldsPoint[i].y);
                        lds_map[int(col / 128)].push_back(LdsData.status.x);
                        lds_map[int(col / 128)].push_back(LdsData.status.y);
                        lds_map[int(col / 128)].push_back(LdsData.status.Qz);

                        // std::cout << "lds_map:  row = " << row << " col = " << col << " x = " << point_lds[0] << " y = " << point_lds[1] << std::endl;                    
                    }
                // }
            }
        }
    }


    // 判断结果中是否存在地毯
    bool brug_seg(std::vector<EcoKeyPoint>& maskdata)
    {
        for (int i = 0; i < maskdata.size(); i++)
        {
            if(maskdata[i].label == EM_OUT_CARPET)
            {
                return true;
            }
        }
        return false;
    }

    // 给接地线赋值语义信息  U 形椅底座以及地毯信息
    bool brug_seg_remove(EcoKeyPoint& point_mask, cv::Mat& rug_mask, cv::Point3f& max_keypoint, cv::Point3f& min_keypoint, cv::Mat& single_rug_mask)
    {
        // single_rug_mask 中地毯的值是 125
        // 接地线语义是电线，直接输出
        if(point_mask.label == EM_OUT_LINE)  
        {
            return true;
        }

        uchar midmask = rug_mask.at<uchar>((point_mask.mappos.y / 5) - 3, point_mask.mappos.x / 5);

        int nUflag          = 0;
        int nRugflag        = 0;
        int nsingle_rugflag = 0;

        // 如果在U形椅正下方，输出U形椅语义信息
        if(rug_mask.at<uchar>(point_mask.mappos.y / 5 - 2, point_mask.mappos.x / 5) == 125 || rug_mask.at<uchar>(point_mask.mappos.y / 5 - 1, point_mask.mappos.x / 5) == 125)
        {
            point_mask.inlabel = 4;
            point_mask.label   = EM_OUT_UCHAIR_BASE;
            return true;
        }

        // 如果在地毯正下方，输出地毯语义信息
        // if(rug_mask.at<uchar>(point_mask.mappos.y / 5 - 2, point_mask.mappos.x / 5) == 255 || rug_mask.at<uchar>(point_mask.mappos.y / 5 - 1, point_mask.mappos.x / 5) == 255)
        if(point_mask.label == EM_OUT_CARPET_EDGE) 
        {
            // point_mask.inlabel = 9;
            point_mask.label   = EM_OUT_CARPET;
            max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
            max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
            min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
            min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点
            // if(point_mask.keypoint.y > 0)
            // {
            //     max_keypoint.z ++;
            // }
            // else
            // {
            //     min_keypoint.z ++;
            // }
            single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
            return true;
        }

        // 如果在流苏正下方，输出流苏语义信息
        if(rug_mask.at<uchar>(point_mask.mappos.y / 5 - 2, point_mask.mappos.x / 5) == 200 || rug_mask.at<uchar>(point_mask.mappos.y / 5 - 1, point_mask.mappos.x / 5) == 200)
        {
            point_mask.inlabel = 9;
            point_mask.label   = EM_OUT_TASSELS;
            max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
            max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
            min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
            min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点
            // if(point_mask.keypoint.y > 0)
            // {
            //     max_keypoint.z ++;
            // }
            // else
            // {
            //     min_keypoint.z ++;
            // }
            single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
            return true;
        }




        // // 如果在地毯BEV 视角正下方/左右 8CM 以内，输出地毯语义信息
        // if(single_rug_mask.at<uchar>(point_mask.keypoint.x / 4 + 1, (0 - point_mask.keypoint.y) / 4 + 40) == 125 
        // || (single_rug_mask.at<uchar>(point_mask.keypoint.x / 4 + 2, (0 - point_mask.keypoint.y) / 4 + 40) == 125
        // && (single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40 - 2) == 125
        // || single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40 + 2) == 125)))
        // {
        //     point_mask.inlabel = 9;
        //     point_mask.label   = EM_OUT_CARPET;
        //     if(single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) != 125)
        //     {
        //         single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
        //     }
        //     max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
        //     max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
        //     min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
        //     min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点
        //     // if(point_mask.keypoint.y > 0)
        //     // {
        //     //     max_keypoint.z ++;
        //     // }
        //     // else
        //     // {
        //     //     min_keypoint.z ++;
        //     // }
        //     return true; 
        // }

        // 如果在地毯边角 U形椅边界 则需要
        for (int i = -2; i <= 2; i += 2)
        {
            for(int j = -2; j <= 0; j += 2)
            {
                if((point_mask.mappos.y / 5 + j > 0) && (point_mask.mappos.y / 5 + j < rug_mask.rows) && (point_mask.mappos.x / 5 + i > 0) && (point_mask.mappos.x / 5 + i < rug_mask.cols))
                {
                    // 如果在U形状
                    if(rug_mask.at<uchar>(point_mask.mappos.y / 5 + j, point_mask.mappos.x / 5 + i) == 125)
                    {
                        nUflag ++;
                        if(nUflag > 1)
                        {
                            point_mask.inlabel = 4;
                            point_mask.label   = EM_OUT_UCHAIR_BASE;
                            return true;
                        }
                    }

                    // if(rug_mask.at<uchar>(point_mask.mappos.y / 5 + j, point_mask.mappos.x / 5 + i) == 255) 
                    // {
                    //     nRugflag ++;
                    //     if(nRugflag > 1)
                    //     {
                    //         point_mask.inlabel = 9;
                    //         point_mask.label   = EM_OUT_CARPET;
                    //         if(single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) != 125)
                    //         {
                    //             single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
                    //         }
                    //         max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
                    //         max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
                    //         min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
                    //         min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点
                    //         // if(point_mask.keypoint.y > 0)
                    //         // {
                    //         //     max_keypoint.z ++;
                    //         // }
                    //         // else
                    //         // {
                    //         //     min_keypoint.z ++;
                    //         // }
                    //         return true;
                    //     } 
                    // }

                    // if(j >= 0 && i >= 0 && single_rug_mask.at<uchar>(point_mask.keypoint.x / 4 + j, (0 - point_mask.keypoint.y) / 4 + i) == 125) 
                    // {
                    //     nsingle_rugflag ++;
                    //     if(nsingle_rugflag > 1)
                    //     {
                    //         point_mask.inlabel = 9;
                    //         point_mask.label   = EM_OUT_TRASH_CAN;
                    //         if(single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) != 125)
                    //         {
                    //             single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
                    //         }
                    //         max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
                    //         max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
                    //         min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
                    //         min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点
                    //         // if(point_mask.keypoint.y > 0)
                    //         // {
                    //         //     max_keypoint.z ++;
                    //         // }
                    //         // else
                    //         // {
                    //         //     min_keypoint.z ++;
                    //         // }
                    //         return true;
                    //     } 
                    // }
                }
            }
        }
        return true;    // 保留
    }

    // 地毯点和接地点融合仅在bev空间进行
    bool brug_seg_remove_bev(EcoKeyPoint& point_mask, cv::Mat& rug_mask, cv::Point3f& max_keypoint, cv::Point3f& min_keypoint, cv::Mat& single_rug_mask)
    {
        // single_rug_mask 中地毯的值是 125
        // 接地线语义是电线，直接输出
        if(point_mask.label == EM_OUT_LINE || point_mask.inlabel == 10)  
        {
            return true;
        }

        uchar midmask = rug_mask.at<uchar>((point_mask.mappos.y / 5) - 3, point_mask.mappos.x / 5);

        int nUflag          = 0;
        int nRugflag        = 0;
        int nsingle_rugflag = 0;

        // 如果在U形椅正下方，输出U形椅语义信息
        if(rug_mask.at<uchar>(point_mask.mappos.y / 5 - 2, point_mask.mappos.x / 5) == 125 || rug_mask.at<uchar>(point_mask.mappos.y / 5 - 1, point_mask.mappos.x / 5) == 125)
        {
            point_mask.inlabel = 4;
            point_mask.label   = EM_OUT_UCHAIR_BASE;
            return true;
        }

        // 如果在地毯正下方，输出地毯语义信息
        // if(rug_mask.at<uchar>(point_mask.mappos.y / 5 - 2, point_mask.mappos.x / 5) == 255 || rug_mask.at<uchar>(point_mask.mappos.y / 5 - 1, point_mask.mappos.x / 5) == 255)
        if(point_mask.label == EM_OUT_CARPET_EDGE) 
        {
            // point_mask.inlabel = 9;
            point_mask.label   = EM_OUT_CARPET;
            max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
            max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
            min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
            min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点

            // single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
            return true;
        }

        // 如果在流苏正下方，输出流苏语义信息
        if(rug_mask.at<uchar>(point_mask.mappos.y / 5 - 2, point_mask.mappos.x / 5) == 200 || rug_mask.at<uchar>(point_mask.mappos.y / 5 - 1, point_mask.mappos.x / 5) == 200)
        {
            point_mask.inlabel = 60;  // 设置为60在可视化时，用粉色表示被赋予流苏语义的障碍物点
            point_mask.label   = EM_OUT_TASSELS;
            max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
            max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
            min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
            min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点

            // single_rug_mask.at<uchar>(point_mask.keypoint.x / 4, (0 - point_mask.keypoint.y) / 4 + 40) = 0;
            return true;
        }


        // 障碍物点在地毯的下边界 -> 赋地毯语义
        if(((int)point_mask.keypoint.x / 4 + 2 < 25) && ((int)point_mask.keypoint.x / 4 - 2 >= 0) && ((int)(0 - point_mask.keypoint.y) / 4 + 1 + 40 < 80) && ((int)(0 - point_mask.keypoint.y) / 4 + 1 + 40 >= 0))  // 确保不超出范围
        {
            // 障碍物点的上方栅格是地毯
            if(single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 + 1, (int)(0 - point_mask.keypoint.y) / 4 + 40) == 125 
            || (single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 + 2, (int)(0 - point_mask.keypoint.y) / 4 + 40) == 125
                && (single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4, (int)(0 - point_mask.keypoint.y) / 4 + 40 + 2) == 125 || single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4, (int)(0 - point_mask.keypoint.y) / 4 + 40 - 2) == 125)))
            {
                // 并且下方栅格不是地毯 => 下边界
                if(single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 - 1, (int)(0 - point_mask.keypoint.y) / 4 + 40) != 125 
                && (single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 - 2, (int)(0 - point_mask.keypoint.y) / 4 + 40) != 125 
                    && (single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 - 2, (int)(0 - point_mask.keypoint.y) / 4 + 40 + 1) != 125 || single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 - 2, (int)(0 - point_mask.keypoint.y) / 4 + 40 - 1) != 125)))
                {
                    point_mask.inlabel = 90;  // 设置为90在可视化时用黄色表示被赋予地毯语义的障碍物点
                    // point_mask.label   = EM_OUT_CARPET;
                    if(single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4, (int)(0 - point_mask.keypoint.y) / 4 + 40) != 125)
                    {
                        single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4, (int)(0 - point_mask.keypoint.y) / 4 + 40) = 0;
                    }


                    max_keypoint.x = max_keypoint.x > point_mask.keypoint.x ? max_keypoint.x : point_mask.keypoint.x;  // 取出距离最远的地毯边缘接地点
                    max_keypoint.y = max_keypoint.y > point_mask.keypoint.y ? max_keypoint.y : point_mask.keypoint.y;  // 取出距离最远的地毯边缘接地点
                    min_keypoint.x = min_keypoint.x < point_mask.keypoint.x ? min_keypoint.x : point_mask.keypoint.x;  // 取出距离最近的地毯边缘接地点
                    min_keypoint.y = min_keypoint.y < point_mask.keypoint.y ? min_keypoint.y : point_mask.keypoint.y;  // 取出距离最近的地毯边缘接地点

                    return true; 
                }
            }
        }

        // 障碍物点在地毯的上边界 -> 保留该点
        if((int)point_mask.keypoint.x / 4 + 2 < 25 && (int)point_mask.keypoint.x / 4 - 1 >= 0 && (int)(0 - point_mask.keypoint.y) / 4 + 40 + 2 < 80 && (int)(0 - point_mask.keypoint.y) / 4 + 40 >= 0)  // 确保不超出范围
        {
            // 障碍物点的上栅格不是地毯
            if(single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 + 1, (int)(0 - point_mask.keypoint.y) / 4 + 40) != 125)
            {
                // 障碍物点的下栅格是地毯
                if(single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 - 1, (int)(0 - point_mask.keypoint.y) / 4 + 40) == 125 && single_rug_mask.at<uchar>((int)point_mask.keypoint.x / 4 - 2, (int)(0 - point_mask.keypoint.y) / 4 + 40) == 125)
                {
                    return true;
                }     
            }

        }

        // 如果在地毯边角 U形椅边界 则需要
        for (int i = -2; i <= 2; i += 2)
        {
            for(int j = -2; j <= 0; j += 2)
            {
                if((point_mask.mappos.y / 5 + j > 0) && (point_mask.mappos.y / 5 + j < rug_mask.rows) && (point_mask.mappos.x / 5 + i > 0) && (point_mask.mappos.x / 5 + i < rug_mask.cols))
                {
                    // 如果在U形状
                    if(rug_mask.at<uchar>(point_mask.mappos.y / 5 + j, point_mask.mappos.x / 5 + i) == 125)
                    {
                        nUflag ++;
                        if(nUflag > 1)

                        {
                            point_mask.inlabel = 4;
                            point_mask.label   = EM_OUT_UCHAIR_BASE;
                            return true;
                        }
                    }

                }
            }
        }

        point_mask.inlabel = 91;  // 设置为91，在可视化使用深红色表示被消掉的障碍物点
        return true;    // 这里为了可视化返回true，实际使用应 return false
    }

    std::time_t getTimeStamp()
    {
        std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());//获取当前时间点
        std::time_t timestamp =  tp.time_since_epoch().count(); //计算距离1970-1-1,00:00的时间长度
        return timestamp;
    }

    int checkPoseInPureTextureArea(EcoAInterfaceAreas_t* spotAreas, EcoAInterfaceDeebotStatus_t& pose) 
    {
        printf("[debug]222指针 = %p\n", spotAreas);

        // 检查参数有效性
        if (!spotAreas) {
            std::cout << "spotAreas error!" << std::endl;
            return 0;
        }

        // 如果没有区域数据，直接返回0
        if (spotAreas->spotAreas_len <= 0) {
            std::cout << "spotAreas_len: " << spotAreas->spotAreas_len << std::endl;
            return 0;
        }

        // 使用位姿中的x和y坐标
        cv::Point2f pt(pose.x, pose.y);

        // 遍历所有区域
        for (int i = 0; i < spotAreas->spotAreas_len; ++i) {
            const EcoAInterfaceSpotAreas_t& area = spotAreas->spotAreas[i];

            // 跳过无效区域
            std::cout << "area:" << i << " area.dot_len: " << area.dot_len << std::endl;
            if (!area.dot || area.dot_len <= 2) {
                continue;
            }

            // 创建轮廓点向量
            std::vector<cv::Point2f> contour;
            contour.clear();

            for (int j = 0; j < area.dot_len; ++j) {
                contour.push_back(cv::Point2f(area.dot[j].x, area.dot[j].y));
            }

            // 检查点是否在当前轮廓内
            if (!contour.empty()) {
                double result = cv::pointPolygonTest(contour, pt, false);
                if (result >= 0) {
                    // 点在当前轮廓内，检查texture
                    std::cout << "[ground_material] Point is in area " << i << ", texture = " << (int)area.texture << std::endl;
                    return (area.texture == 3) ? 1 : 0;
                }
            }
        }

        // 点不在任何轮廓内
        return 0;
    }
}


