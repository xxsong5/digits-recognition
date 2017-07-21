/*************************************************************************
	> File Name: detectNUMs.cpp
	> Author: Xuxiansong
	> Mail: 2808595125@163.com 
	> Created Time: 2017年07月20日 星期四 15时18分26秒
 ************************************************************************/

#include <iostream>
#include "imgDitis.h"


using namespace std;

int main(int argc , char** argv)
{
    
    string test_image_ph, number_model_ph;
    if( argc < 2){
        cerr<< "need test-images & number_model_path\n";
        exit(0);
    }else if(argc==2){
        number_model_ph="../trainData/number_model.xml";
    }else{
        number_model_ph=argv[2];
    }
    test_image_ph=argv[1];


    vector<int> vres;
    cv::Mat img= cv::imread(test_image_ph.c_str());
    ImgDitis detector(number_model_ph);
    detector.img2NUMS(img,vres);


    cout<< "result: ";
    for(auto i: vres){
        cout<<i<<" ";
    }
    cout<<endl;
    cv::imshow("img", img);
    cv::waitKey(0);


    return 0;
}



