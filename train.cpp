/*************************************************************************
	> File Name: train.cpp
	> Author: Xuxiansong
	> Mail: 2808595125@163.com 
	> Created Time: 2017年07月20日 星期四 15时17分48秒
 ************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "imgDitis.h"

using namespace std;



int main(int argc, char** argv)
{
    if( argc !=2){
        cerr<< "need training num-images' path as param\n";
        exit(0);
    }


    vector<string> vimgs;
    vector<int> vassoNUMs;
    taitsDir(argv[1], vimgs, [](const string& name){
                if(name.length()>5 && name[name.length()-6]=='_' && name[0]=='i') return true;
                else return false;
                });
    for(size_t i=0; i< vimgs.size(); ++i){
        char num= vimgs[i][vimgs[i].length()-5];
        char num0=vimgs[i][vimgs[i].length()-6];
        num0= num0== '_' ? '0': '1';
        vassoNUMs.push_back((num-'0')+10*(num0-'0'));
    }


    TrainningHOG trainner("trainData/number_model.xml");
    trainner.trainning(vimgs, vassoNUMs);

    return 0;
}




