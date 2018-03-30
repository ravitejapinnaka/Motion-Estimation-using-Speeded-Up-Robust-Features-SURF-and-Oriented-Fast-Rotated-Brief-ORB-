
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
int main( int argc, char* argv[] )
{
    
    double t = (double)getTickCount();
    cv::VideoCapture capture("/Users/kakarlakeerthy/Desktop/prj1/prj1/input.mov"); // input video present in input folder
    
    double fps = capture.get( cv::CAP_PROP_FPS );// obtaining  frames per second
    
    cv::Size size(
                  (int)capture.get( cv::CAP_PROP_FRAME_WIDTH ),
                  (int)capture.get( cv::CAP_PROP_FRAME_HEIGHT )
                  );// getting width and height of video
    
    
    cv::Mat bgr_frame1,frame_gray1, img_keypoints,bgr_frame2, frame_gray2,gray2, img_keypoints2, bgr_frame3,frame_gray3,img_keypoints3;
    
    std::vector<KeyPoint> keypoints1 , keypoints2,keypoints3;
    cv::Size winSize = Size(11,11);
    int maxLevel = 3;
    double minEigThreshold = 1e-4;
    cv::TermCriteria criteria= TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,30,0.01);
    Mat descriptors, err;
    vector<uchar> status;
    CV_OUT std::vector<Point2f> points2f1,points2f2, points2f3;
    int length = int(capture.get(CAP_PROP_FRAME_COUNT));
    const std::vector< int > & 	keypointIndexes = std::vector< int >();
    const std::vector< int > & 	keypointIndexes2 = std::vector< int >();
    cv::VideoWriter writer; // video writer object
    
    capture >> bgr_frame1; // taking first frame from the video and storing in bgr_frame1
    Ptr<Feature2D> f2d = SURF::create(7000); //for SURF, we set hessianThreshold = 7000 to get a limited number of keypoints
    cv::cvtColor( bgr_frame1, frame_gray1, cv::COLOR_BGR2GRAY);// converting bgr_frame1 to gray image and storing in frame_gray1
    f2d->detect(frame_gray1, keypoints1);// detecting keypoints for first frame using SURF algorithm
    
    KeyPoint::convert(keypoints1,points2f1,keypointIndexes);// converting keypoints of first frame to Points2f type
    
    Mat imgLines = Mat::zeros( frame_gray1.size(), CV_8UC3 );// creating an empty image with size equals to frame_gray1 size.
    // char c = cv::waitKey(fps);
    capture >> bgr_frame2;// get second frame from the video
    cv::cvtColor( bgr_frame2, frame_gray2, cv::COLOR_BGR2GRAY);//converting bgr_frame2 to gray image and storing in frame_gray2
    f2d->detect(frame_gray2, keypoints2);// detecting keypoints for second frame using SURF algorithm
    
    KeyPoint::convert(keypoints2,points2f2,keypointIndexes2);//converting keypoints of 2nd frame to Points2f type
    length = length -1;
    writer.open("/Users/kakarlakeerthy/Desktop/Proj1/output.mp4", CV_FOURCC('M','J','P','G'), fps, size );// output video path and type
    for(;;) // infinite loop
        
    {
        length = length -1;
        if (length == 2 )
            break;
        calcOpticalFlowPyrLK(frame_gray1,frame_gray2, points2f1,points2f2,status,err,winSize,maxLevel,criteria, minEigThreshold); //calculating optical flow for two frames
        
        frame_gray2.copyTo(frame_gray1); // copying content of frame_gray2 to frame_gray1
        
        for(size_t i=0; i<points2f2.size(); i++)//
        {
            if(status[i])
            {
                if (i == 0)
                {
                    line(imgLines,points2f1[i],points2f2[i],Scalar(0,0,255),4);
                }
                if ( i == 1)
                {
                    line(imgLines,points2f1[i],points2f2[i],Scalar(0,255,0),4);
                }
                if ( i ==2 )
                {
                    line(imgLines,points2f1[i],points2f2[i],Scalar(255,0,0),4);
                }
                if ( i==3)
                {
                    line(imgLines,points2f1[i],points2f2[i],Scalar(255,255,255),4);
                }
                
                points2f1[i].x = points2f2[i].x;
                points2f1[i].y = points2f2[i].y;
            }
            
        }
        
        bgr_frame2 = bgr_frame2 + imgLines; // adding the original image and the image with lines
        
        
        imshow("eee",bgr_frame2); // displaying the image after addition
        
        writer << bgr_frame2; // saving bgr_frame2 to writer
        //    if( c == 27)  break;
        
        char c = cv::waitKey(fps);
        capture >> bgr_frame2;   // capturing a new frame from the input video
        cv::cvtColor( bgr_frame2, frame_gray2, cv::COLOR_BGR2GRAY); //converting bgr_frame2 to gray image and storing in frame_gray2
        f2d->detect(frame_gray2, keypoints2);// detecting keypoints for frame_gray2 using SURF algorithm
        KeyPoint::convert(keypoints2,points2f2,keypointIndexes2);//converting keypoints2 to Points2f type
        //   if( c == 27)  break;
    }
        t = ((double)getTickCount() - t)/getTickFrequency();
        std::cout << "Times passed in seconds: " << t << std::endl;
    
    
    return 0;
    
}

















