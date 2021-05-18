#include <Windows.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ipCPU.h"
#include "KernelCall.h"
#include "DS_timer.h"

using namespace std;
using namespace cv;

#define MODE_CPU_CALC 0
#define MODE_GPU_CALC 1
#define MODE_MAPPING 2

uchar *cuRc, *cuBc, *cuGc, *cuDestR, *cuDestB, *cuDestG;
uint *ch,*map_,*Dest;
cv::cuda::GpuMat cvOrigin; cv::cuda::GpuMat cvRes;
string FileSelector();

int main() {
	cout << "You only File Select *.avi file" << endl;
	cout << "---------------------------------------------------------" << endl;
	cout << " 1. cpu mode " << endl;
	cout << " 2. gpu mode " << endl;
	cout << " 3. mapping mode " << endl;
	cout << "---------------------------------------------------------" << endl;
	int mode = 0;
	while (true) {
		char c;
		cin >> c;
		if (c == '1') {
			cout << "you choose mode CPU" << endl;
			mode = MODE_CPU_CALC;
			break;
		}
		else if (c == '2') {
			cout << "you choose mode GPU" << endl;
			mode = MODE_GPU_CALC;
			break;
		}
		else if (c == '3') {
			cout << "you choose mode Mapping" << endl;
			mode = MODE_MAPPING;
			break;
		}
		else {
			cout << "u must be input 1 or 2 or 3" <<endl;
		}
	}

	string fname = FileSelector();
	DS_timer timer(5);
	timer.setTimerName(0, "Rendering for 1 frame");
	timer.setTimerName(1, "Distortion(CPU) for 1 frame");
	timer.setTimerName(2, "GPU Rendering for 1 frame");
	timer.setTimerName(3, "Distortion Map Create");
	timer.setTimerName(4, "Total Rendered");


	while (fname.empty()) {
		string fname = FileSelector();
	}

	VideoCapture tagetVideo(fname);

	Mat img;
	tagetVideo >> img;
	cout << "Original Image(first Frame) width = " << img.cols << ", height = " << img.rows << endl;
	cout << tagetVideo.get(CAP_PROP_FRAME_COUNT);

	//device memory allocation
	cudaMalloc(&cuRc, sizeof(uchar)*img.cols*img.rows);
	cudaMalloc(&cuBc, sizeof(uchar)*img.cols*img.rows);
	cudaMalloc(&cuGc, sizeof(uchar)*img.cols*img.rows);
	cudaMalloc(&cuDestR, sizeof(uchar)* 800 * 480);
	cudaMalloc(&cuDestB, sizeof(uchar)* 800 * 480);
	cudaMalloc(&cuDestG, sizeof(uchar)*800*480);
	cudaMalloc(&ch, sizeof(uint)*img.cols*img.rows);
	cudaMalloc(&Dest, sizeof(uint) * 800 * 480);
	cudaMalloc(&map_, sizeof(uint) * 800 * 480);
	IpCPU imageProcessing;

	timer.onTimer(0);
	Mat img1, img2;
	img1 = imageProcessing.imgCrop(img, 0, 0, img.cols - EYEWIDTH, img.rows);
	timer.onTimer(1);
	img1 = imageProcessing.imgBarrelDistortion(img1, 0.055);
	timer.offTimer(1);
	img1 = imageProcessing.imgResize(img1, 400, 480);
	img2 = imageProcessing.imgCrop(img, EYEWIDTH, 0, img.cols, img.rows);
	img2 = imageProcessing.imgBarrelDistortion(img2, 0.055);
	img2 = imageProcessing.imgResize(img2, 400, 480);
	Mat target = imageProcessing.imgConnect(img1, img2);
	timer.offTimer(0);
	cout << "width = " << target.cols << ", height = " << target.rows << endl;

	timer.onTimer(2);
	//Mat img3 = KernelCall(img, cuRc, cuBc, cuGc, cuDestR, cuDestB, cuDestG);
	Mat img3 = KernelCall2(img, ch, Dest);
	timer.offTimer(2);

	timer.onTimer(3);
	uint *maps = KernelCall3(img, map_);
	timer.offTimer(3);

	VideoWriter writer;
	double fps = 60.0;
	writer.open("output.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(800,480), true);
	if (!writer.isOpened()) {
		cout << "Error " << endl;
		return 1;
	}

	clock_t start, end1, end2;
	timer.onTimer(4);
	start = clock();
	int cnt = 0;
	while (waitKey(2) != 27) {
		imshow("imageread2", target);
		tagetVideo >> img;
		if (img.empty()) { break; }
		//cpu code 
		if (mode == MODE_CPU_CALC) {
			img1 = imageProcessing.imgCrop(img, 0, 0, img.cols - EYEWIDTH, img.rows);
			img1 = imageProcessing.imgBarrelDistortion(img1, 0.055);
			img1 = imageProcessing.imgResize(img1, 400, 480);
			img2 = imageProcessing.imgCrop(img, EYEWIDTH, 0, img.cols, img.rows);
			img2 = imageProcessing.imgBarrelDistortion(img2, 0.055);
			img2 = imageProcessing.imgResize(img2, 400, 480);
			target = imageProcessing.imgConnect(img1, img2);
		}
		else if (mode == MODE_GPU_CALC) {
			//gpu code
			//target = KernelCall(img, cuRc, cuBc, cuGc, cuDestR, cuDestB, cuDestG);
			target = KernelCall2(img, ch, Dest);
		}
		else if (mode == MODE_MAPPING) {
			target = mapping(img, maps); //gpu로 할 수 있을까?
			//target = mappingInGpu(img, map_, cvOrigin, cvRes);
		}
		
		writer.write(target);
		cnt++;
		end1 = clock();
		if (end1 - start > 1000) {
			cout << "fps = " << cnt << endl;
			start = clock();
			cnt = 0;
		}
		/*
		*/
	}
	timer.offTimer(4);

	writer.release();
	cudaFree(&cuRc); cudaFree(&cuBc); cudaFree(&cuGc); cudaFree(&cuDestR); cudaFree(&cuDestG); cudaFree(&cuDestB);

	timer.printTimer();
	system("pause");
}


string FileSelector() {
	char filename[MAX_PATH];

	OPENFILENAME ofn;
	ZeroMemory(&filename, sizeof(filename));
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "Video File\0*.avi\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = "Select a File";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	if (GetOpenFileNameA(&ofn))
	{
		std::cout << "You chose the file \"" << filename << "\"\n";
	}
	else
	{
		// All this stuff below is to tell you exactly how you messed up above. 
		// Once you've got that fixed, you can often (not always!) reduce it to a 'user cancelled' assumption.
		switch (CommDlgExtendedError())
		{
		case CDERR_DIALOGFAILURE: std::cout << "CDERR_DIALOGFAILURE\n";   break;
		case CDERR_FINDRESFAILURE: std::cout << "CDERR_FINDRESFAILURE\n";  break;
		case CDERR_INITIALIZATION: std::cout << "CDERR_INITIALIZATION\n";  break;
		case CDERR_LOADRESFAILURE: std::cout << "CDERR_LOADRESFAILURE\n";  break;
		case CDERR_LOADSTRFAILURE: std::cout << "CDERR_LOADSTRFAILURE\n";  break;
		case CDERR_LOCKRESFAILURE: std::cout << "CDERR_LOCKRESFAILURE\n";  break;
		case CDERR_MEMALLOCFAILURE: std::cout << "CDERR_MEMALLOCFAILURE\n"; break;
		case CDERR_MEMLOCKFAILURE: std::cout << "CDERR_MEMLOCKFAILURE\n";  break;
		case CDERR_NOHINSTANCE: std::cout << "CDERR_NOHINSTANCE\n";     break;
		case CDERR_NOHOOK: std::cout << "CDERR_NOHOOK\n";          break;
		case CDERR_NOTEMPLATE: std::cout << "CDERR_NOTEMPLATE\n";      break;
		case CDERR_STRUCTSIZE: std::cout << "CDERR_STRUCTSIZE\n";      break;
		case FNERR_BUFFERTOOSMALL: std::cout << "FNERR_BUFFERTOOSMALL\n";  break;
		case FNERR_INVALIDFILENAME: std::cout << "FNERR_INVALIDFILENAME\n"; break;
		case FNERR_SUBCLASSFAILURE: std::cout << "FNERR_SUBCLASSFAILURE\n"; break;
		default: std::cout << "You cancelled.\n";
		}
	}

	return string(filename);
}