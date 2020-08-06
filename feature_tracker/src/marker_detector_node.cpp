#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include "parameters.h"
#include "tic_toc.h"

ros::Publisher pub_marker;

double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;
cv::Mat cameraMatrix;
cv::Mat distCoeffs;

cv::Mat draw(cv::Mat& inputImage, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, std::vector<cv::Vec3d> &rvecs, std::vector<cv::Vec3d>& tvecs)
{
    cv::Mat outputImage = inputImage.clone();
    for (int i = 0; i < rvecs.size(); ++i) {
        auto rvec = rvecs[i];
        auto tvec = tvecs[i];
        cv::aruco::drawAxis(outputImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
        std::cout << "t: " << tvec << std::endl;
    }
    return outputImage;
}


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    TicToc t_r;
    /// TODO: marker detection
    cv::Mat inputImage = ptr->image; 
    ROS_INFO("input image size: %d, %d", inputImage.cols, inputImage.rows);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds);

    cv::aruco::estimatePoseSingleMarkers(markerCorners, MARKER_SIDE, cameraMatrix, distCoeffs, rvecs, tvecs);
    cv::Mat poseDraw = draw(inputImage, cameraMatrix, distCoeffs, rvecs, tvecs);
    cv::imshow("marker pose", poseDraw);
    cv::waitKey(10);

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr marker_info(new sensor_msgs::PointCloud);
        marker_info->header = img_msg->header;
        marker_info->header.frame_id = "world";

        sensor_msgs::ChannelFloat32 id_of_marker;
        id_of_marker.name = "id_of_marker";
        sensor_msgs::ChannelFloat32 rt_of_marker;
        rt_of_marker.name = "rt_of_marker";

        for(size_t i = 0; i != markerCorners.size(); ++i)
        {
            int makerId = markerIds[i];
            id_of_marker.values.push_back(makerId);

            cv::Vec3d& r = rvecs[i];
            cv::Vec3d& t = tvecs[i];
            rt_of_marker.values.push_back(r[0]);
            rt_of_marker.values.push_back(r[1]);
            rt_of_marker.values.push_back(r[2]);
            rt_of_marker.values.push_back(t[0]);
            rt_of_marker.values.push_back(t[1]);
            rt_of_marker.values.push_back(t[2]);

            std::vector<cv::Point2f>& corners = markerCorners[i];
            for(size_t j = 0; j != corners.size(); ++j)
            {
                geometry_msgs::Point32 pt;
                pt.x = corners[j].x;
                pt.y = corners[j].y;
                pt.z = 0;
                marker_info->points.push_back(pt);
            }
        }
        marker_info->channels.push_back(id_of_marker);
        marker_info->channels.push_back(rt_of_marker);
        
        ROS_DEBUG("publish %f, at %f", marker_info->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
        {
            pub_marker.publish(marker_info);
        }
    }
    ROS_INFO("whole marker detetction processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "marker_detector");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    cameraMatrix = (cv::Mat_<double>(3, 3) << FX , 0, CX, 0, FY, CY, 0, 0, 1.0);
    distCoeffs = (cv::Mat_<double>(5, 1) << K1, K2, 0, P1, P2);


    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_marker = n.advertise<sensor_msgs::PointCloud>("marker", 1000);

    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?
