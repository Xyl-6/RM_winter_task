#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
using namespace std;
using namespace cv;

//单纯的调用摄像头
void test01(){
    VideoCapture capture(0);
    Mat frame;
    if(!capture.isOpened()){
        printf("无法调用摄像头");
        return ;
    }
    while(1){
        capture >> frame;
        imshow("video",frame);
        waitKey(30);
    }
}

//调用canny后的摄像头
void test02(){
    VideoCapture capture(0);
    while(1){
        Mat frame;
        capture >> frame;
        cvtColor(frame,frame,COLOR_BGR2GRAY);
        blur(frame,frame,Size(7,7));
        Canny(frame,frame,20,60,3);
        imshow("webcam",frame);
        waitKey(30);
    }
}



//计时的连招
//double t = getTickCount()              getTickCount() 返回程序运行到这里所用的时间
// .......
//t = (getTickCount()-t) / getTickFrequency()
//调用摄像头并显示帧数
void test03(){
    Mat frame;
    VideoCapture capture(0);
    double fps;  //记录帧数

    char string[10];
    namedWindow("Camera FPS");
    while(1){
        //记录加载一帧用的时间 算出帧频
        double t = getTickCount();
        capture >> frame;
        t = (getTickCount()-t) / getTickFrequency();
        fps = 1.0  / t;
        //-------------------------------------------
        sprintf(string,"%.2f",fps);   //帧率保留两位小数
        std::string fpsSrting = "FPS :";
        fpsSrting += string ;   

        //输出帧率信息
        putText(
            frame,                //输出在frame矩阵上
            fpsSrting,            //输出的string内容
            Point(10,30),         //文字内容的其实位置  以左下角为原点
            FONT_HERSHEY_COMPLEX, //字体类型
            0.8,                  //字号
            Scalar(0,0,0)         //颜色
        );
        //-----------------------------------------------


        //获取照片 
        int key = waitKey(30);
        if(key != -1 && key == 'k'){
            imwrite("../Captureimgs/1.jpg",frame);  //按 k 拍照
            imshow("Captured Img",frame);
        }

        imshow("Camera FPS",frame);

    }    
}

int main(){
    // test01();
    // test02();
    test03();
    return 0;
}