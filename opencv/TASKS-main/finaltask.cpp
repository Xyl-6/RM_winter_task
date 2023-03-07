#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <vector>
using namespace std;
using namespace cv;


void test01(){
    Mat img = imread("../pictures/final1.jpg");
    Mat b_img = Mat::zeros(img.size(),img.type());   //二值化后的图像
    Mat c_img = img.clone();                         //去噪
    blur(img,c_img,Size(20,20));
    threshold(c_img,b_img,130,255,THRESH_BINARY_INV);
    cvtColor(b_img,b_img,COLOR_BGR2GRAY);
    
    Mat lables, status ,centroids ;
    int count = connectedComponentsWithStats(b_img,lables,status,centroids,8);
    cout << count << endl;
    

    //------------------------------
    int flag[100] = {0} ; //初值为0
    // 遍历每个连通域
    for(int i=0 ; i<count ;i++){
        int width = status.at<int>(i,2);
        int height = status.at<int>(i,3);
        if(fabs(width-height)<=12 && width>30){
            flag[i] = 1;
        }
    }

    //开始遍历图片
    int rowNum = img.rows;
    int colNum = img.cols;
    for(int i=0 ; i<rowNum ;i++){
        for(int j=0 ; j<colNum ;j++){
            int a = lables.at<int>(i,j);
            if(flag[a] == 1){            //B   G   R
                img.at<Vec3b>(i,j) = Vec3b(0,255,255);
            }
        }
    }
    imshow("img1",img);
}




void test02(){
    Mat img = imread("../pictures/final2.jpg");
    // imshow("img",img);
    //---------------------------------------
    // Mat dst1 = img.clone();
    Mat element = getStructuringElement(MORPH_RECT,Size(10,10));
    // morphologyEx(dst1,dst1,MORPH_OPEN,element);
    // imshow("dst1",dst1);
    //----------------------------------------
    //this effect is better
    Mat dst2 = img.clone();
    morphologyEx(dst2,dst2,MORPH_CLOSE,element);
    Mat binary;
    blur(dst2,dst2,Size(7,7));
    threshold(dst2,binary,160,255,THRESH_BINARY);
    cvtColor(binary,binary,COLOR_BGR2GRAY);
    imshow("dst2",binary);
    Mat lables,status,centroids;
    int cnt = connectedComponentsWithStats(binary,lables,status,centroids,8,4);
    int x = status.at<int>(2,0);
    int y = status.at<int>(2,1);
    int width = status.at<int>(2,2);
    int height = status.at<int>(2,3);
    rectangle(img,Rect(x,y,width,height),Scalar(0,0,255),5,8);
    imshow("img",img);
    //----------------------------------------


}





void test03(){
    Mat img = imread("../pictures/final3.jpg");
    Mat b_img;
    Mat img_gray;

    medianBlur(img,img,7);
    threshold(img,b_img,110,255,THRESH_BINARY);
    // imshow("erzhi",b_img);
    cvtColor(b_img,b_img,COLOR_BGR2GRAY);
    // imshow("b_img",b_img);
    inRange(b_img,60,100,b_img);
    // imshow("img",b_img);

    Mat lables,status,centroids;
    int count = connectedComponentsWithStats(b_img,lables,status,centroids,8);
    //给面积最大的连通域画框即可
    int maxarea = status.at<int>(1,4);
    for(int i=2 ; i<count ; i++){
        int area = status.at<int>(i,4);
        if(area >= maxarea){
            maxarea = area;
        }
    }

    for(int i=0 ; i<count ; i++){
        if(maxarea == status.at<int>(i,4)){
            rectangle(img,Rect(status.at<int>(i,0),status.at<int>(i,1),status.at<int>(i,2),status.at<int>(i,3)),Scalar(0,0,255),2,8);
            break;
        }
    }
    imshow("img3",img);
}



int main(){
    // test01();
    // test02();
    test03();
    waitKey(0);
    return 0;
}