/*************************************************************************
	> File Name: imgDitis.cpp
	> Author: Xuxiansong
	> Mail: 2808595125@163.com 
	> Created Time: 2017年07月19日 星期三 19时49分26秒
 ************************************************************************/

#include <iostream>
#include "imgDitis.h"

using namespace std;
using namespace cv;


int ImgDitis::img2NUM(const Mat& img, int& num)
{
    vector<float> desp;
    mpHOG->compute(img, desp);

    //for debug
    //cout<< "ImgDitis: HOG desp size: "<< desp.size()<<endl;

    cv::Mat matdesp=cv::Mat::zeros(1, desp.size(),CV_32FC1);
    for(size_t i=0; i< desp.size(); ++i){
        matdesp.at<float>(i)=desp[i];
    }
    num=msvmClassifier.predict(matdesp);

    return num;
}

void ImgDitis::segmentNUM(const cv::Mat& img, std::vector<cv::Mat>& vnum_imgs)
{
    vnum_imgs.clear();
    segment(img, vnum_imgs);
}

void ImgDitis::img2NUMS(const Mat& img, vector<int>& nums)
{
    nums.clear();
    vector<Mat> vnum_imgs;
    segmentNUM(img, vnum_imgs);

   // //for debug
   // if(vnum_imgs.size()==4){
   //     cv::imshow("1", vnum_imgs[0]);
   //     cv::imshow("2", vnum_imgs[1]);
   //     cv::imshow("3", vnum_imgs[2]);
   //     cv::imshow("4", vnum_imgs[3]);
   // }

    for(auto im: vnum_imgs){
        int tmp;
        nums.push_back(img2NUM(im,tmp));
    }
}

void TrainningHOG::trainning(const vector<string>& vtrainningImags, const vector<int>& vassoci_nums)
{
    vector<vector<float> > vvdesp(vtrainningImags.size());
    for(size_t i=0; i< vtrainningImags.size(); ++i){
        Mat img= imread(vtrainningImags[i]);
        mpHOG->compute(img, vvdesp[i]);
    }

    //for debug
    //cout<<"TrainningHOG HOG desp size: "<< vvdesp[0].size()<<endl;

    Mat traningData=cv::Mat::zeros(vtrainningImags.size(), vvdesp[0].size(), CV_32FC1);
    Mat resData= cv::Mat::zeros(vtrainningImags.size(), 1, CV_32FC1);
    for(int i=0; i< traningData.rows; ++i){
        for(int j=0;j < traningData.cols; ++j){
            traningData.at<float>(i, j)=vvdesp[i][j];
        }
        resData.at<float>(i)=vassoci_nums[i];
    }


    CvSVM svm;
    CvSVMParams param;
    CvTermCriteria criteria;
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );
    cout<<"start trainning ...\n";
    svm.train( traningData, resData, Mat(), Mat(), param );
    cout<<"finished trainning and saving files to "<< msaving_files<<endl;
    svm.save(msaving_files.c_str());

}


void segment(const cv::Mat& im, std::vector<cv::Mat>& vsubimgs)
{
    cv::Mat img=im.clone();
    cv::floodFill(img, cv::Point(0,0), cv::Scalar(0,0,0), 0, cv::Scalar(100,100,100), cv::Scalar(255,255,255));
    cv::floodFill(img, cv::Point(im.cols-10,im.rows-10), cv::Scalar(0,0,0));
    cv::erode(img, img, cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(5,5)));

    if(img.channels() ==3)cv::cvtColor(img, img, CV_BGR2GRAY);
    if(img.rows > img.cols){cv::transpose(img, img); cv::flip(img,img,0);}


    int width=9, counts_th=30;
    vector<int> vlines;
    for(int j=width; j< img.cols; ++j){
        cv::Mat rect(img,cv::Rect(j-width,0,width, img.rows));
        int counts= cv::countNonZero(rect);
        if(counts < counts_th){
            vlines.push_back(j-width/2+1);
            j+=2*width;
        }
    }

    // remove outlier of vlines
    for(size_t i=2; i< vlines.size();){
        cv::Mat rect0(img, cv::Rect(vlines[i-2],0, vlines[i-1]-vlines[i-2], img.rows));
        cv::Mat rect1(img, cv::Rect(vlines[i-1],0, vlines[i]-vlines[i-1], img.rows));
        if(cv::countNonZero(rect0) < counts_th && cv::countNonZero(rect1) < counts_th){
            vlines.erase(vlines.begin()+i-1);
            continue;
        }
        ++i;
    }
    if(vlines.size() < 5){ vlines.clear(); return; }
    cv::Mat rect0(img, cv::Rect(vlines[0],0, vlines[1]-vlines[0], img.rows));
    cv::Mat rect1(img, cv::Rect(vlines[vlines.size()-2],0, vlines[vlines.size()-1]-vlines[vlines.size()-2], img.rows));
    if(cv::countNonZero(rect0) <counts_th) vlines.erase(vlines.begin());
    if(cv::countNonZero(rect1) <counts_th) vlines.erase(vlines.end()-1);
    if(vlines.size() != 5){ vlines.clear(); return; }
    for(size_t i=1; i< vlines.size(); ++i){
        cv::Mat rect0(img, cv::Rect(vlines[i-1],20, vlines[i]-vlines[i-1], img.rows-50));
        if(cv::countNonZero(rect0) < counts_th) continue;
        vsubimgs.push_back(rect0.clone());
    }

    //expand border
    for(size_t i=0; i< vsubimgs.size(); ++i){
        int win_width=vsubimgs[i].rows;
        int oriW=vsubimgs[i].cols;
        int leftW= win_width-oriW;
        cv::copyMakeBorder(vsubimgs[i], vsubimgs[i], 0,0, leftW/2, leftW/2+ leftW%2,cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        if(vsubimgs[i].cols != 80)
          cv::resize(vsubimgs[i], vsubimgs[i],cv::Size(80,80));
    }

//    //for debug
//    cout<<"       "<< vlines.size()<<endl;
//    for(int i=0; i< vlines.size(); ++i){
//        cv::line(img, cv::Point(vlines[i],0), cv::Point(vlines[i], img.rows-1),cv::Scalar(255,255,255),2);
//    }
//    cv::imshow("img", img);

}


void taitsDir(std::string dir_path,std::vector<string>& vfull, isYES func)
{
    if(dir_path[dir_path.length()-1]!='/')dir_path+="/";

    DIR* dp;
    struct dirent *dir;

    if((dp=opendir(dir_path.c_str())) == NULL) {
        cerr<< "open dirent failed!\n";
        return;
    }

    while((dir=readdir(dp)) !=NULL){
        string name=dir->d_name;
        if(func(name))
          vfull.push_back(dir_path+name);
    }
}



