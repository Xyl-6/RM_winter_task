#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

//对摄像头进行  中值、均值、高斯滤波



//中值  medianBlur
void test01(){
    VideoCapture capture(0);
    Mat frame;
    while(1){
        capture >> frame;
        medianBlur(frame,frame,9);
        imshow("medianBlur",frame);
        waitKey(30);
    }
}



//均值 blur
void test02(){
    VideoCapture capture(0);
    Mat frame;
    while(1){
        capture >> frame;
        blur(frame,frame,Size(20,20));
        imshow("Blur",frame);
        waitKey(30);
    }
}


//高斯 GaussianBlur
void test03(){
    VideoCapture capture(0);
    Mat frame;
    while(1){
        capture >> frame;
        GaussianBlur(frame,frame,Size(15,15),5,5);
        imshow("GaussianBlur",frame);
        waitKey(30);
    }
}



int main(){
    // test01();
    // test02();
    test03();
    return 0;
}