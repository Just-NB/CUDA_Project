#pragma once
#include <Windows.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

class IpCPU {
public:
	Mat imgResize(Mat origin, int targetWidth, int targetHeight);
	Mat imgCrop(Mat origin, int sx, int sy, int ex, int ey);
	Mat imgConnect(Mat left, Mat right);
	Mat imgBarrelDistortion(Mat origin, double k);
};


Mat IpCPU::imgResize(Mat origin, int targetWidth, int targetHeight) {
	Mat res(targetHeight, targetWidth, CV_8UC3);

	double cols =((double)origin.cols/(double)targetWidth);
	double rows = ((double)origin.rows/(double)targetHeight);


	for (int i = 0; i < targetHeight; i++) {
		for (int j = 0; j < targetWidth; j++) {
			if(j*cols<origin.cols && i*rows < origin.rows) res.at<Vec3b>(i, j) = origin.at<Vec3b>(i*rows, j*cols);
		}
	}

	return res;
}

Mat IpCPU::imgCrop(Mat origin, int sx, int sy, int ex, int ey) {
	Mat res(abs(ey - sy), abs(ex - sx), CV_8UC3);

	int targetHeight = abs(ey - sy);
	int targetWidth = abs(ex - sx);

	for (int i = 0; i < targetHeight; i++) {
		for (int j = 0; j < targetWidth; j++) {
			res.at<Vec3b>(i, j) = origin.at<Vec3b>(sy+i, sx+j);
		}
	}

	return res;
}


Mat IpCPU::imgConnect(Mat left, Mat right) {
	Mat res(MAX(left.rows, right.rows), left.cols + right.cols, CV_8UC3);

	for (int i = 0; i < res.rows; i++) {
		for (int j = 0; j < res.cols; j++) {
			if (j < left.cols) { if(i<left.rows)res.at<Vec3b>(i, j) = left.at<Vec3b>(i, j); }
			else { if (i < right.rows) res.at<Vec3b>(i, j) = right.at<Vec3b>(i, j); }
		}
	}

	return res;
}


//k = 왜곡률 이 알고리즘에선 k를 사용하지 않음
Mat IpCPU::imgBarrelDistortion(Mat origin, double k) {
	Mat res(origin.rows, origin.cols, CV_8UC3);
	res.setTo(0);

	//파라미터 a,b,c가 음수로가면 먼지점이 중심에서 멀리 이동 양수로가면 멀리있는 점이 가운데로 이동함
	//구글카드보드로 보면 핀쿠션왜곡현상이 일어나므로 술통왜곡을 만들어야함
	//즉, 멀리있는 점이 가운데로 모이는 형태가 됨 따라서 A,B 둘 다 양수로 설정한다.

	//가장 바깥 쪽 픽셀에만 영향을 주는 파라미터 A
	double pA = 0.007715;
	//균일하게 영향을 주는 파라미터 B
	double pB = 0.026731;
	//파라미터 C(이게 뭘 의미하는 지는 모르겠음 전반적인 왜곡률이 결정되는 듯)
	double pC = 0.0;
	//이미지의 스케일값(pA+pB+pC+pD = 1 이 되도록해야 원본 크기의 스케일이 됨)
	double pD = 1.0 - pA - pB - pC;
	
	double cx = origin.cols / 2.0;
	double cy = origin.rows / 2.0;

	for (int y = 0; y < origin.rows; y++) {
		for (int x = 0; x < origin.cols; x++) {
			int d = MIN(cx, cy);

			double dx = (x - cx) / d;
			double dy = (y - cy) / d;

			double dr = sqrt(dx*dx + dy * dy);
			double sr = (pA*dr*dr*dr + pB * dr*dr + pC * dr + pD)*dr;
			double factor = abs(dr / sr);

			double srcXd = cx + (dx*factor*d);
			double srcYd = cy + (dy*factor*d);

			int nx = (int)srcXd;
			int ny = (int)srcYd;

			if(nx >= 0 && ny >= 0 && nx < res.cols && ny < res.rows)
				res.at<Vec3b>(ny, nx) = origin.at<Vec3b>(y, x);
		}
	}
	
	//중앙값 보간 (0점 보간)
	for (int y = res.rows*10/100; y < res.rows*90/100; y++) {
		for (int x = res.cols * 25 / 100; x < res.cols * 75 / 100; x++) {
			if(res.at<Vec3b>(y,x)[0] == 0)res.at<Vec3b>(y, x) = origin.at<Vec3b>(y, x);
		}
	}

	return res;
}