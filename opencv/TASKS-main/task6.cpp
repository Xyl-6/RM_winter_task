#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void DrawCircle(Mat& img,Point center,int r){
  
    circle(img,center,r,Scalar(0,0,0),3,8);
}

void DrawLine(Mat& img ,Point start, Point end){
    
    line(img,start,end,Scalar(0,0,255),2,8);
}

void DrawRect(Mat& img,int x , int y , int width ,int height){
 
    rectangle(img,Rect(x,y,width,height),Scalar(255,0,0),3,8);
}

int main(){
    Mat img = imread("../pictures/1.png");
    Point center(500,500);
    Point p1(300,700);
    Point p2(900,50);

    DrawCircle(img,center,200);
    DrawLine(img,p1,p2);
    DrawRect(img,300,400,500,400);
    imshow("Drawing",img);
    waitKey(0);
    return 0;
}