/*************************************************************************
	> File Name: imgDitis.h
	> Author: Xuxiansong
	> Mail: 2808595125@163.com 
	> Created Time: 2017年07月19日 星期三 19时49分10秒
 ************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <string>
#include <dirent.h>

#pragma once

#define IMGDITIS_HOG_WINDOW_SIZE      80  
#define IMGDITIS_HOG_BLOCK_SIZE       40  
#define IMGDITIS_HOG_BLOCK_STRIDE     20  
#define IMGDITIS_HOG_CELL_SIZE        20  
#define IMGDITIS_HOG_BIN_COUNTS       9   
// 28
// 14
// 7 
// 7 
// 9 
#define IMGDITIS_SEG_IMG_WIDTH       80
#define IMGDITIS_SEG_IMG_HEIGHT      80


class ImgDitis{

public:
    ImgDitis(const std::string& modelPath):mpHOG(NULL)
    {
        if(modelPath.empty()) std::cout<<" need trianning data>>>>>>>>>>>>>>\n";
        msvmClassifier.load(modelPath.c_str());

        //need to change param
        mpHOG=new cv::HOGDescriptor(cv::Size(IMGDITIS_HOG_WINDOW_SIZE,IMGDITIS_HOG_WINDOW_SIZE), cv::Size(IMGDITIS_HOG_BLOCK_SIZE,IMGDITIS_HOG_BLOCK_SIZE),cv::Size(IMGDITIS_HOG_BLOCK_STRIDE,IMGDITIS_HOG_BLOCK_STRIDE), cv::Size(IMGDITIS_HOG_CELL_SIZE,IMGDITIS_HOG_CELL_SIZE),IMGDITIS_HOG_BIN_COUNTS);
    }

    ~ImgDitis(){if(mpHOG)delete mpHOG;}

    //main func
    void img2NUMS(const cv::Mat& img, std::vector<int>& nums);


private:
    int img2NUM(const cv::Mat& img, int& num);
    void segmentNUM(const cv::Mat& img, std::vector<cv::Mat>& vnum_imgs);
    cv::SVM msvmClassifier;
    cv::HOGDescriptor *mpHOG;
    
};




class TrainningHOG{
public:
    TrainningHOG(const std::string& saving_xml_file):msaving_files(saving_xml_file),mpHOG(NULL)
    {
        if(mpHOG==NULL)
            mpHOG=new cv::HOGDescriptor(cv::Size(IMGDITIS_HOG_WINDOW_SIZE,IMGDITIS_HOG_WINDOW_SIZE), cv::Size(IMGDITIS_HOG_BLOCK_SIZE,IMGDITIS_HOG_BLOCK_SIZE),cv::Size(IMGDITIS_HOG_BLOCK_STRIDE,IMGDITIS_HOG_BLOCK_STRIDE), cv::Size(IMGDITIS_HOG_CELL_SIZE,IMGDITIS_HOG_CELL_SIZE),IMGDITIS_HOG_BIN_COUNTS);
    }

    ~TrainningHOG(){if(mpHOG)delete mpHOG;}


    void trainning(const std::vector<std::string>& trainningImags, const std::vector<int>& associ_nums);


private:
    std::string     msaving_files;
    cv::HOGDescriptor *mpHOG;
};


//common func
void segment(const cv::Mat& im, std::vector<cv::Mat>& vnum_imgs);

typedef bool(*isYES)(const std::string& );
void taitsDir(std::string dir_path, std::vector<std::string>& vfull, isYES func= [](const std::string&)->bool{return 1;});



