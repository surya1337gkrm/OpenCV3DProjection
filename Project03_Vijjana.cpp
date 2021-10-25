#define _USE_MATH_DEFINES

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<sstream>
#include<fstream>
#include <cmath>
#pragma warning(disable:4996)

using namespace cv;
using namespace std;

Mat depth, color, xyz_cloud, rgb_cloud, camMatrix, RxMatrix, RyMatrix, Pt_cloud, V_Img, tempV;

//for arbitrary rotation
Mat T, R1, R2, R3, ar_ptcloud;

short int f = 524;
short int cx = 316;
short int cy = 256;

int c = 0;
int j = 0;

int deg = 0;
int deg2 = 0;
float tx = 0;
float ty = 0;
float tz = 0;

double degInRadians = 0;
double deg2InRadians = 0;

short int Z, X, Y;
uchar b, g, r;

int x, y;

string rotationType = "x";
bool isArRotation = false;



void genPtCloud(short int X, short int Y, short int Z) {
    xyz_cloud.ptr<float>(0)[c] = float(X);
    xyz_cloud.ptr<float>(1)[c] = float(Y);
    xyz_cloud.ptr<float>(2)[c] = float(Z);
    xyz_cloud.ptr<float>(3)[c] = float(1);
}

void genRGBCloud(uchar b, uchar g, uchar r) {
    rgb_cloud.ptr<uchar>(0)[c] = b;
    rgb_cloud.ptr<uchar>(1)[c] = g;
    rgb_cloud.ptr<uchar>(2)[c] = r;
}

void fillVImg(int x, int y) {
    if (x >= 0 && x < V_Img.cols && y >= 0 && y < V_Img.rows) {
        V_Img.ptr<uchar>(y)[3 * x] = rgb_cloud.ptr<uchar>(0)[j];
        V_Img.ptr<uchar>(y)[3 * x + 1] = rgb_cloud.ptr<uchar>(1)[j];
        V_Img.ptr<uchar>(y)[3 * x + 2] = rgb_cloud.ptr<uchar>(2)[j];
    }
}

void genCamMatrix() {
    camMatrix.ptr<float>(0)[0] = f;
    camMatrix.ptr<float>(0)[2] = cx;
    camMatrix.ptr<float>(1)[1] = f;
    camMatrix.ptr<float>(1)[2] = cy;
    camMatrix.ptr<float>(2)[2] = 1;
}

void genRxMatrix(){
    RxMatrix.ptr<float>(0)[3] = tx;
    RxMatrix.ptr<float>(1)[1] = float(cos(degInRadians));
    RxMatrix.ptr<float>(1)[2] = float(-sin(degInRadians));
    RxMatrix.ptr<float>(1)[3] = ty;
    RxMatrix.ptr<float>(2)[1] = float(sin(degInRadians));
    RxMatrix.ptr<float>(2)[2] = float(cos(degInRadians));
    RxMatrix.ptr<float>(2)[3] = tz;
}

void genRyMatrix() {
    RyMatrix.ptr<float>(0)[0] = float(cos(degInRadians));
    RyMatrix.ptr<float>(0)[2] = float(sin(degInRadians));
    RyMatrix.ptr<float>(0)[3] = tx;
    RyMatrix.ptr<float>(1)[3] = ty;
    RyMatrix.ptr<float>(2)[0] = float(-sin(degInRadians));
    RyMatrix.ptr<float>(2)[2] = float(cos(degInRadians));
    RyMatrix.ptr<float>(2)[3] = tz;
}

void genARImage() {

    genCamMatrix();
    genRxMatrix();
    genRyMatrix();


    R2.ptr<float>(0)[0] = float(319 / 399.02);
    R2.ptr<float>(0)[1] = float(-239.7 / 399.02);
    R2.ptr<float>(1)[1] = float(319 / 399.02);
    R2.ptr<float>(1)[0] = float(239.7 / 399.02);

    R3.ptr<float>(1)[1] = float(cos(deg2InRadians));
    R3.ptr<float>(1)[2] = float(-sin(deg2InRadians));
    R3.ptr<float>(2)[1] = float(sin(deg2InRadians));
    R3.ptr<float>(2)[2] = float(cos(deg2InRadians));

    ar_ptcloud = T.inv() * (R1.inv() * (R2.inv() * (R3 * (R2 * (R1 * (T * xyz_cloud))))));
    Pt_cloud = camMatrix * (RxMatrix*ar_ptcloud);

}

void genVImage() {

    genCamMatrix();
    genRxMatrix();
    genRyMatrix();

    for (int y = 0; y < color.rows; y++) {
        for (int x = 0; x < color.cols; x++) {

            Z = depth.ptr<short int>(y)[x];
            X = ((x - cx) * Z) / f;
            Y = ((y - cy) * Z) / f;

            b = color.ptr<uchar>(y)[3 * x];
            g = color.ptr<uchar>(y)[3 * x + 1];
            r = color.ptr<uchar>(y)[3 * x + 2];
            c++;
            genPtCloud(X, Y, Z); 
            genRGBCloud(b, g, r);
        }
    }

    if (rotationType == "x") {
        Pt_cloud = (camMatrix * (RxMatrix * xyz_cloud));
    }
    else if (rotationType == "y") {
        Pt_cloud = (camMatrix * (RyMatrix * xyz_cloud));
    }
    else if (rotationType == "ar") {
        genARImage();
    }

    for (int i = 0; i < 307200; i++) {
        if (Pt_cloud.ptr<float>(2)[i] != 0) {
            x = int(Pt_cloud.ptr<float>(0)[i] / Pt_cloud.ptr<float>(2)[i]);
            y = int(Pt_cloud.ptr<float>(1)[i] / Pt_cloud.ptr<float>(2)[i]);

        }
        else {

            x = 0;
            y = 0;
        }

        fillVImg(x, y);
        j++;
    }
}


int main(int argc, char** argv)
{

    depth.create(480, 640, CV_16UC1);
    color = imread("C:/Sem01_Fall2021/Advanced Computer Vision/OpenCV3DProjection/color.jpg");

    FILE* fp = fopen("C:/Sem01_Fall2021/Advanced Computer Vision/OpenCV3DProjection/depth.dat", "rb");

    fread(depth.data, 2, 640 * 480, fp);
    fclose(fp);

    xyz_cloud = Mat::ones(4, 307200, CV_32FC1);
    rgb_cloud.create(3, 307200, CV_8UC1);

    camMatrix = Mat::zeros(3, 4, CV_32FC1);
    RxMatrix = Mat::eye(4, 4, CV_32FC1);
    RyMatrix = Mat::eye(4, 4, CV_32FC1);

    V_Img = Mat::zeros(color.rows, color.cols, CV_8UC3);

    //translation matrix 1
    T = Mat::eye(4, 4, CV_32FC1);
    T.ptr<float>(0)[3] = float(-0.3);
    T.ptr<float>(1)[3] = float(-1.0);

    //rotation matrix 1
    R1 = Mat::eye(4, 4, CV_32FC1); //coz,aplha is 0deg.

    //rotation matrix 2
    R2 = Mat::eye(4, 4, CV_32FC1);
    
    //rotation matrix 3
    R3 = Mat::eye(4, 4, CV_32FC1);
    

    while (1) {

        V_Img = Mat::zeros(color.rows, color.cols, CV_8UC3);
        

        if (c >= 307200) {
            c = 0;
        }
        if (j >= 307200) {
            j = 0;
        }

        genVImage();

        //V_Img = Mat::zeros(color.rows, color.cols, CV_8UC3);

        imshow("depth", depth);
        imshow("color", color);
        imshow("VImg", V_Img);

        uchar a = waitKey(1);
        degInRadians = deg * M_PI / 180;
        deg2InRadians = deg2 * M_PI / 180;

        if (a == 27) {
            break;
        }
        else if (a == 'a') {
           
                tx = tx - 10;
            
            
        }
        else if (a == 'd') {
            
                tx = tx + 10;
            
             
        }
        else if (a == 'w') {
            
                ty = ty - 10;
           
        }
        else if (a == 's') {
            
                ty = ty + 10;
           
        }
        else if (a == 'z') {
           
             f = f + 30;
            
        }
        else if (a == 'x') {
            
                f = f - 30;
            
        }
        else if (a == 'r') {
            f = 524;
            tx = 0;
            ty = 0;
            deg = 0;
            deg2 = 0;
            rotationType = "x";
        }
        else if (a == 'o') {
            printf("Image Saved.");
            imwrite("vImg.jpg", V_Img);
        }
        else if (a == '1') {
            deg = deg + 5;
            rotationType = "x";

        }
        else if (a == '2') {
            deg = deg - 5;
            rotationType = "x";
        }
        else if (a == '3') {
            deg = deg + 5;
            rotationType = "y";
        }
        else if (a == '4') {
            deg = deg - 5;
            rotationType = "y";
        }
        else if (a == '5') {
            deg2 = deg2 + 5;
            rotationType = "ar";
            
        }
        else if (a == '6') {
            deg2 = deg2 - 5;
            rotationType = "ar";
        }

    }

    return 1;
}