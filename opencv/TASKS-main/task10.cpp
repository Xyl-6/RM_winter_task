#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <vector>
using namespace cv;
using namespace std;


//图像的二值化 将灰度的值改为 0 或 255 或阈值  
//globle二值化方法  threshold()
//连通域分析通常处理的是二值化后的图像 
void test01(){
    Mat img = imread("../pictures/10.png");
    Mat bi_Img = img.clone();   //二值化后的图像

    threshold(img,bi_Img,100,255,THRESH_BINARY);
    cvtColor(bi_Img,bi_Img,COLOR_BGR2GRAY);
    Mat lables;
    //connectedComponents()  这个函数的第一个参数要单通道图像  
    //第二个参数 lables：图像上每一像素的标记 不同连通域用不同数字表示
    int count = connectedComponents(bi_Img,lables,8);  //记录连通域数量
    // imshow("dst",lables);  wrong!!   此时的lable 是记录连通域标号的矩阵

    //------------上色-----------------
    //vector容器用来储存颜色  并初始化颜色
    vector<Vec3b>color(count);       //每个连通域对应的颜色
    RNG rng(10086);  //随机数生成器
    for(int i=0 ; i < count ; i++){
        color[i]  = Vec3b(rng.uniform(0,256),rng.uniform(0,256),rng.uniform(0,256));
    }
    Mat output = Mat::zeros(bi_Img.size(),img.type());
    //遍历 bi_Img 这张图片  在 output 输出
    int rowNum = bi_Img.rows;
    int colNum = bi_Img.cols;
    for(int i=0 ; i < rowNum ; i++){
        for(int j=0 ; j < colNum ; j++){
            int a = lables.at<int>(i,j);
            if(a == 0){
                continue;
            }else{
                output.at<Vec3b>(i,j) = color[a]; 
            }
        }
    }

    imshow("output",output);
}



//连通域标记  绘制外接四边形  输出硬币个数
void test02(){
    Mat img = imread("../pictures/10.png");
    Mat bi_Img = img.clone();   //二值化后的图像
    threshold(img,bi_Img,90,255,THRESH_BINARY);
    cvtColor(bi_Img,bi_Img,COLOR_BGR2GRAY);
    Mat lables,output;
    //----------开始标记-------------
    //status 连通域的信息[x,y,width,height,area]   连通域的中心点centroids [x,y]
    Mat status, centroids;  
    int count = connectedComponentsWithStats(bi_Img,lables,status,centroids,8,CV_16U);
    
    for(int i=0 ; i<count ;i++){
        int x = status.at<int>(i,0);
        int y = status.at<int>(i,1);
        int width = status.at<int>(i,2);
        int height = status.at<int>(i,3);
        rectangle(img,Rect(x,y,width,height),Scalar(0,0,255),1,8,0);
    }
    cout << count-1 << endl;
    imshow("img",img);

}



int main(){
    // test01();
    test02();
    waitKey(0);
    return 0;
}