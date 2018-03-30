
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

int main( int argc, char** argv )
{
      double t = (double)getTickCount();
    cv::VideoCapture capture("/Users/kakarlakeerthy/Desktop/prj1/prj1/input.mov"); // video input
    
    double fps = capture.get( cv::CAP_PROP_FPS );// obtaining  frames per second
    
    cv::Size size(
                  (int)capture.get( cv::CAP_PROP_FRAME_WIDTH ),
                  (int)capture.get( cv::CAP_PROP_FRAME_HEIGHT )
                  );// getting width and height of video
    cv::VideoWriter writer; // video writer object
    
    
    cv::Mat bgr_frame1,frame_gray1, img_keypoints,bgr_frame2, frame_gray2,gray2, img_keypoints2, bgr_frame3,frame_gray3,img_keypoints3, img_1, img_2;
    
    std::vector<KeyPoint> keypoints1 , keypoints2,keypoints3;
    
    const std::vector< int > & 	keypointIndexes2 = std::vector< int >();
    
    Mat descriptors, err;
    vector<uchar> status;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    int length = int(capture.get(CAP_PROP_FRAME_COUNT));
    capture >> bgr_frame1;// taking first frame from the video and storing in bgr_frame1
    Ptr<Feature2D> f2d = SURF::create(7000);//for SURF, we set HessainThreshold to get 4 keypoints
    cv::cvtColor( bgr_frame1, img_1, cv::COLOR_BGR2GRAY);// converting bgr_frame1 to gray image and storing in img_1
    f2d->detect(img_1, keypoints_1);// detecting keypoints for first frame using SURF algorithm
    
    f2d->compute(img_1, keypoints_1, descriptors_1);// computing descriptors for the corresponding keypoints
  
    
    
    char c = cv::waitKey(fps);
    capture >> bgr_frame2;// get second frame from the video
    cv::cvtColor( bgr_frame2, img_2, cv::COLOR_BGR2GRAY);//converting bgr_frame2 to gray image and storing in img_2
    
    f2d->detect(img_2, keypoints_2);// detecting keypoints for second frame using SURF algorithm
    
    f2d->compute(img_2, keypoints_2, descriptors_2); // computing descriptors for keypoints_2
    
    Mat imgLines = Mat::zeros( img_1.size(), CV_8UC3 );// creating an empty image with size equals to frame_gray1 size.
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    writer.open("/Users/kakarlakeerthy/Desktop/saikkk/output1.mp4", CV_FOURCC('M','J','P','G'), fps, size );// output video path and type
    
    
    for(;;) // infinite loop
    {
        length = length-1;
        if(length == 3) break;
        
        matcher.match( descriptors_1, descriptors_2, matches ); //Matching descriptor vectors using FLANN matcher
        
        double max_dist = 0; double min_dist = 100;
        
        // calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        
        //Draw only "good" matches (i.e. whose distance is less than 2*min_dist, or a small arbitary value ( 0.02 ) in the event that min_dist is very small)
        std::vector< DMatch > good_matches;
        
        for( int i = 0; i < descriptors_1.rows; i++ )
        { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
        }
        
        std::vector<Point2f> points2f1,points2f2, points2f3;
        
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //Get the keypoints from the good matches
            points2f1.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
            points2f2.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
        }
        
        descriptors_1 = descriptors_2;
        keypoints_1 = keypoints_2;
        for(size_t i=0; i<points2f2.size(); i++)
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
           // line(imgLines,points2f1[i],points2f2[i],Scalar(20,80,150),10);
            
        }
        
        bgr_frame2 = bgr_frame2 + imgLines;// adding the original image and the image with lines
        
        
        imshow("eee",bgr_frame2); // displaying the image after addition
        writer << bgr_frame2; // saving bgr_frame2 to writer
        c = cv::waitKey(fps);
        capture >> bgr_frame2;// capturing a new frame from the input video
        cv::cvtColor( bgr_frame2, img_2, cv::COLOR_BGR2GRAY);//c onverting bgr_frame2 to gray image and storing in img_2
        f2d->detect(img_2, keypoints_2);// detecting keypoints for frame_gray2 using SURF algorithm
        f2d->compute(img_2, keypoints_2, descriptors_2);//computing descriptors for keypoints_2
        
        
        
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
        std::cout << "Times passed in seconds: " << t << std::endl;
    
    return 0;
}
