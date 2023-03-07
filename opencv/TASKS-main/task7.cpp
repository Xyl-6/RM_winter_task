#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

void Gamma(Mat& channel);


//-----------Gamma 矫正-----------------
void  GammaCorrection01(){
    Mat img  = imread("../pictures/7-1.png");
    double gamma = 1/1.3;
    //----------将测试图像的通道数改为 1--------------
    if (img.channels() != 1){
		cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);    // cvtColor() 转换色彩空间  此处转换称为 灰度图像
	}
    //----------------------------------------------
    //深拷贝创建和原图大小一样的图片
    Mat gammaimg;
    gammaimg = img.clone();
    int rowNumber = gammaimg.rows;
    int colNumber = gammaimg.cols;
    //遍历拷贝后的图片的每一个像素  用at方法
    for(int i = 0;  i <  rowNumber; i++){
        for(int j = 0;j < colNumber; j++){
            //进行 Gamma变换
            int gray = img.at<uchar>(i,j);
            gammaimg.at<uchar>(i,j) = pow( gray , gamma);
        }
    }
    normalize(gammaimg, gammaimg, 0, 255, cv::NORM_MINMAX);
    imshow("orgimg",img);
    imshow("gammaimg",gammaimg);
    waitKey(0);
}




//--------彩色图像的gamma矫正-----------------
// 先分离图层再矫正
void GammaCorrection02(){
    Mat img = imread("../pictures/7-1.png");
    // Mat img = imread("./pictures/7-2.jpg");
    vector<Mat> mv;
    split(img,mv);
    imshow("orgimg",img);
    
    //对三个通道进行操作
    Gamma(mv[0]);
    Gamma(mv[1]);
    Gamma(mv[2]);

    Mat dst;       //矫正后的图片
    merge(mv,dst);
    imshow("dst",dst);
    waitKey(0);
}

// 对单个通道进行gamma矫正
void Gamma(Mat& channel){
    int bins = 256;
    float *hist = new float[bins];
    memset(hist,0,sizeof(float)*bins);

	for (float i = 0; i < bins; i++){ 
		float k = i / 255;                                //归一化
		hist[int(i)] = pow(k, 0.65);                       //预补偿
		hist[int(i)] = 255 * hist[int(i)];                //反归一化 并将结果存入 gamma 像素查找表 hist中
    }

    int rowNumber = channel.rows;
    int colNumber = channel.cols;

    for(int i=0 ; i<rowNumber ;i++){
        for(int j=0; j<colNumber ; j++){
            int a = channel.at<uchar>(i,j);
            channel.at<uchar>(i,j) = hist[a];
            }
    }

    delete[] hist;
}




int main(){
    GammaCorrection02();
    return 0;
}