/*************************************************************************
	> File Name: getNumImage.cpp
	> Author: Xuxiansong
	> Mail: 2808595125@163.com 
	> Created Time: 2017年07月20日 星期四 15时17分34秒
 ************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "imgDitis.h"

using namespace std;


int main( int argc , char** argv)
{
    if( argc !=2){
        cerr<< "need raw images' path as param\n";
        exit(0);
    }


    vector<string> vimgs;
    taitsDir(argv[1], vimgs, [](const string& name){
                if(name.length()>4 && name[name.length()-5] == 'h') return true;
                else return false;
                });


    for(size_t i=0; i< vimgs.size(); ++i){
        cv::Mat im=cv::imread(vimgs[i]);

        std::vector<cv::Mat> vsubimgs;
        segment(im ,vsubimgs);

        if(vsubimgs.size()==4){
            cv::imshow("num0", vsubimgs[0]);
            cv::imshow("num1", vsubimgs[1]);
            cv::imshow("num2", vsubimgs[2]);
            cv::imshow("num3", vsubimgs[3]);
        }else continue;
        cv::imshow("img", im);
        char key=cv::waitKey(0);

        if(key == 's' && vsubimgs.size()==4){
            char name[100]={0};
            static int counts=0;
            for(size_t j=0; j<vsubimgs.size(); ++j){
                sprintf(name, "trainData/img%05d_0.png", ++counts);
                cv::imwrite(name,vsubimgs[j]);
            }
            cout<< "saved: "<<name<<"\n";
        }else{
            cout<< "igorned"<<"\n";
        }
    }


    return 0;
}





