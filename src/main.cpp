#include <iostream>
#include <opencv2/opencv.hpp>

int main() {

    // std::ifstream file("opencv_project/resources/test_image.png");
    // if (!file) {
    // std::cerr << "File not found!" << std::endl;
    // }

    // 定义图像路径
    std::string imagePath = "/home/wangjunhao/opencv_project/resources/test_image.png";

    // 使用 OpenCV 读取图像
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    // 检查图像是否成功加载
    if (image.empty()) {
        std::cerr << "Error: Could not load image at " << imagePath << std::endl;
        return -1;
    }

// #图像颜色空间转换

// 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

     cv::Mat hsvImage;
    cv::Mat channels[3];

    // 色调 (Hue) 通道，初始化为0
    channels[0] = cv::Mat::zeros(grayImage.size(), CV_8UC1);

    // 饱和度 (Saturation) 通道，初始化为0
    channels[1] = cv::Mat::zeros(grayImage.size(), CV_8UC1);

    // 亮度 (Value) 通道，直接使用灰度图像
    channels[2] = grayImage;

    // 合并通道创建 HSV 图像
    cv::merge(channels, 3, hsvImage);

        // 将 HSV 图像转换为 BGR 图像以便正确显示
    cv::Mat bgrImage;
    cv::cvtColor(hsvImage, bgrImage, cv::COLOR_HSV2BGR);

    // 显示 BGR 图像****
    cv::imshow("HSV Image with Gray Appearance", bgrImage);
 std::string outputPath = "/home/wangjunhao/opencv_project/resources/ans1.png"; // 修改为实际的保存路径
  cv::imwrite(outputPath, bgrImage);//保存到resources里面

//   // 显示原图、灰度图和 HSV 图像
//     cv::imshow("Original Image", image);

    // cv::imshow("Gray Image", grayImage);

    // cv::imshow("HSV Image", hsvImage);
    
//#应用各种滤波操作

   // 应用均值滤波
    cv::Mat meanFiltered;
    cv::blur(image, meanFiltered, cv::Size(5, 5)); // 5x5 的均值滤波器

    // 应用高斯滤波
    cv::Mat gaussianFiltered;
    cv::GaussianBlur(meanFiltered, gaussianFiltered, cv::Size(5, 5), 0); // 5x5 的高斯滤波器   
    
    
    //     // 显示原图、均值滤波和高斯滤波图像
    // cv::imshow("Original Image", image);
    cv::imshow("Mean Filtered Image", meanFiltered);
    // //*****
    cv::imshow("Gaussian Filtered Image", gaussianFiltered);

    outputPath = "/home/wangjunhao/opencv_project/resources/ans2.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, meanFiltered);//保存到resources里面
    outputPath = "/home/wangjunhao/opencv_project/resources/ans3.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, gaussianFiltered);//保存到resources里面

//# 特征提取

// 对图像进行高斯模糊，减少噪声
    cv::Mat smoothedImage;
    cv::GaussianBlur(image, smoothedImage, cv::Size(5, 5), 0);

    // 将图像转换为 HSV 色彩空间
    cv::cvtColor(smoothedImage, hsvImage, cv::COLOR_BGR2HSV);

    // 提取红色区域
    cv::Mat mask1, mask2;
    cv::inRange(hsvImage, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask1);    // 红色的下界
    cv::inRange(hsvImage, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask2); // 红色的上界
    cv::Mat redMask = mask1 | mask2;

    // 应用形态学操作清理掩码
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(redMask, redMask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    // 在掩码中寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(redMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓和外接矩形
    cv::Mat resultImage = image.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        // 绘制轮廓
        cv::drawContours(resultImage, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);

        // 对轮廓进行近似处理，减少点数
        std::vector<cv::Point> approxContour;
        cv::approxPolyDP(contours[i], approxContour, 5, true);
        
        // 计算并绘制外接矩形
        cv::Rect boundingBox = cv::boundingRect(approxContour);
        cv::rectangle(resultImage, boundingBox, cv::Scalar(255, 0, 0), 2);

        // 计算轮廓的面积
        double area = cv::contourArea(contours[i]);
        std::cout << "Contour " << i << " 面积: " << area << std::endl;
    }

    // 高亮区域处理：灰度化、二值化、膨胀、腐蚀
    cv::Mat binaryImage, dilatedImage, erodedImage;

    // 将红色区域的掩码应用于原图
    cv::Mat redRegion;
    cv::bitwise_and(image, image, redRegion, redMask);

    // 灰度化
    cv::cvtColor(redRegion, grayImage, cv::COLOR_BGR2GRAY);

    // 自适应阈值处理
    cv::adaptiveThreshold(grayImage, binaryImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);

    // 膨胀
    cv::dilate(binaryImage, dilatedImage, kernel, cv::Point(-1, -1), 2);

    // 腐蚀
    cv::erode(dilatedImage, erodedImage, kernel, cv::Point(-1, -1), 2);

    // 漫水填充
    cv::Mat floodFillImage = erodedImage.clone();
    if (!contours.empty() && !contours[0].empty()) {
        cv::Point seedPoint = contours[0][0];
        cv::floodFill(floodFillImage, seedPoint, cv::Scalar(255), 0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY);
    }

    // 将处理结果应用于原图
    cv::Mat finalResult = resultImage.clone();
    redRegion.copyTo(finalResult, floodFillImage);

    // // 显示结果********
    cv::imshow("Final Result", finalResult);
    
    outputPath = "/home/wangjunhao/opencv_project/resources/ans4.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, finalResult);//保存到resources里面


 // 创建一张空白图像用于绘制形状和文字
    cv::Mat drawingImage = cv::Mat::zeros(500, 500, CV_8UC3);

    // 绘制圆形
    cv::circle(drawingImage, cv::Point(150, 150), 50, cv::Scalar(0, 255, 0), -1); // 填充绿色圆
    cv::circle(drawingImage, cv::Point(150, 150), 50, cv::Scalar(0, 0, 255), 2);  // 红色外轮廓

    // 绘制方形
    cv::rectangle(drawingImage, cv::Point(250, 100), cv::Point(350, 200), cv::Scalar(255, 0, 0), -1); // 填充蓝色矩形
    cv::rectangle(drawingImage, cv::Point(250, 100), cv::Point(350, 200), cv::Scalar(0, 0, 255), 2);  // 红色外轮廓

    // 绘制文字
    cv::putText(drawingImage, "OpenCV Text", cv::Point(50, 300), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
   

    // 显示绘制图像
    cv::imshow("Drawing Shapes and Text", drawingImage);
    outputPath = "/home/wangjunhao/opencv_project/resources/ans5.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, drawingImage);//保存到resources里面

     // 绘制红色外轮廓
    cv::Mat imageWithContours = image.clone();
    cv::drawContours(imageWithContours, contours, -1, cv::Scalar(0, 0, 255), 2);

    // 绘制红色的 bounding box
    cv::Mat imageWithBoundingBox = imageWithContours.clone();
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Rect boundingBox = cv::boundingRect(contours[i]);
        cv::rectangle(imageWithBoundingBox, boundingBox, cv::Scalar(0, 0, 255), 2);
    }

    // // // 显示结果
    // // cv::imshow("Original Image", image);
    cv::imshow("Image with Contours", imageWithContours);
    cv::imshow("Image with Bounding Box", imageWithBoundingBox);

    outputPath = "/home/wangjunhao/opencv_project/resources/ans6.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, imageWithContours);//保存到resources里面
    outputPath = "/home/wangjunhao/opencv_project/resources/ans7.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, imageWithBoundingBox);//保存到resources里面

// 图像旋转 35 度
    double angle = 35.0;
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0); // 图像中心
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotatedImage;
    cv::warpAffine(image, rotatedImage, rotationMatrix, image.size());

    // 图像裁剪为左上角 1/4
    int newWidth = rotatedImage.cols / 2;
    int newHeight = rotatedImage.rows / 2;
    cv::Rect roi(0, 0, newWidth, newHeight); // 定义左上角 1/4 的区域
    cv::Mat croppedImage = image(roi);

    // // // 显示结果
    // // cv::imshow("Original Image", image);
    cv::imshow("Rotated Image", rotatedImage);
    cv::imshow("Cropped Image", croppedImage);
    outputPath = "/home/wangjunhao/opencv_project/resources/ans8.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, rotatedImage);//保存到resources里面
    outputPath = "/home/wangjunhao/opencv_project/resources/ans9.png"; // 修改为实际的保存路径
    cv::imwrite(outputPath, croppedImage);//保存到resources里面

    // 等待用户按键
    cv::waitKey(0);

    // 释放资源
    cv::destroyAllWindows();

    return 0;
}
