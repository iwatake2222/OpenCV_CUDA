#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#define NUM_REPEAT 10000

int main()
{
	cv::Mat imgSrc = cv::imread(RESOURCE_DIR"lena.jpg");
	cv::imshow("imgSrc", imgSrc);

	{
		cv::Mat imgDst;
		const auto& t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < NUM_REPEAT; i++) cv::resize(imgSrc, imgDst, cv::Size(300, 300));
		const auto& t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> timeSpan = t1 - t0;
		printf("CPU = %.3lf [msec]\n", timeSpan.count() * 1000.0 / NUM_REPEAT);
		cv::imshow("CPU", imgDst);
	}

	{
		cv::cuda::GpuMat imgGpuSrc, imgGpuDst;
		cv::Mat imgDst;
		const auto& t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < NUM_REPEAT; i++) {
			imgGpuSrc.upload(imgSrc);
			cv::cuda::resize(imgGpuSrc, imgGpuDst, cv::Size(300, 300));
			imgGpuDst.download(imgDst);
		}
		const auto& t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> timeSpan = t1 - t0;
		printf("GPU = %.3lf [msec]\n", timeSpan.count() * 1000.0 / NUM_REPEAT);
		cv::imshow("GPU", imgDst);
	}

	cv::waitKey(0);
	return 0;
}

