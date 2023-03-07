#include <opencv2/opencv.hpp>
using namespace cv;

//练习1
//用 Mat::at.() 模板函数访问单个像素
void test01(){
    Mat img = imread("../pictures/1.png");
    imshow("test01",img);
    int rowNumber = img.rows;
    int colNumebr = img.cols;
    for(int i = 0;i < rowNumber; i++){
        for(int j = 0; j < colNumebr; j++){
            //访问24位彩色图像Vec3b     灰度图像是uchar
            img.at<Vec3b>(i,j)[0] = (img.at<Vec3b>(i,j)[0]+img.at<Vec3b>(i,j)[1]+img.at<Vec3b>(i,j)[2])/3;//bule通道
            img.at<Vec3b>(i,j)[1] = (img.at<Vec3b>(i,j)[0]+img.at<Vec3b>(i,j)[1]+img.at<Vec3b>(i,j)[2])/3;//green通道
            img.at<Vec3b>(i,j)[2] = (img.at<Vec3b>(i,j)[0]+img.at<Vec3b>(i,j)[1]+img.at<Vec3b>(i,j)[2])/3;//red通道
        }
    }
    imshow("lalala1",img);

}


//练习二
void test02(){
    Mat img = imread("../pictures/1.png");
    imshow("test01",img);
    int rowNumber = img.rows;
    int colNumebr = img.cols;
    
    uchar threshold = 100; 
    for(int i = 0;i < rowNumber; i++){
        for(int j = 0; j < colNumebr; j++){
            
            int average = (img.at<Vec3b>(i,j)[0]+img.at<Vec3b>(i,j)[1]+img.at<Vec3b>(i,j)[2])/3;
            
            if(average > threshold){
                img.at<Vec3b>(i,j)[0] = 255;
                img.at<Vec3b>(i,j)[1] = 255;
                img.at<Vec3b>(i,j)[2] = 255;
            }else{
                img.at<Vec3b>(i,j)[0] = 0;
                img.at<Vec3b>(i,j)[1] = 0;
            }
        }
    }
    
    imshow("lalala2",img);

}


int main(){
    test01();
    test02();
    waitKey(0);
    return 0;
}