//
// Created by Lonqi on 2017/11/18.
//
#pragma once

#ifndef __MTCNN_NCNN_H__
#define __MTCNN_NCNN_H__
#include "ncnn/net.h"
#include <string>
#include <vector>
#include <time.h>
#include <map>
#include <iostream>



struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    float ppoint[10];
    float regreCoord[4];
};

class MTCNN {

public:
	MTCNN();
	MTCNN(const std::string &model_path);
    MTCNN(const std::vector<std::string> param_files, const std::vector<std::string> bin_files);
    ~MTCNN();
	
	void init(const std::string& model_path);
	void SetMinFace(int minSize);
    void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
	void detectMaxFace(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
	float getRnetScore(ncnn::Mat& ncnn_img, Bbox& bbox);
  //  void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale);
	void nmsTwoBoxs(std::vector<Bbox> &boundingBox_, std::vector<Bbox> &previousBox_, const float overlap_threshold,std::string modelname = "Union");
    void nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname="Union");
    void refine(std::vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
	void extractMaxFace(std::vector<Bbox> &boundingBox_);

	void PNet(float scale);
    void PNet();
    void RNet();
    void ONet();

    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;

    const float nms_threshold[3] = {0.5f, 0.7f, 0.7f};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
	const int MIN_DET_SIZE = 12;
	std::vector<Bbox> firstPreviousBbox_, secondPreviousBbox_, thirdPrevioussBbox_;
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    int img_w, img_h;

private://部分可调参数
	const float threshold[3] = { 0.8f, 0.8f, 0.6f };
	int minsize = 40;
	const float pre_facetor = 0.709f;
	
};


#endif //__MTCNN_NCNN_H__
