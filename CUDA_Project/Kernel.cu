#include "KernelCall.h"

//인자로 받은 메모리주소가 아닌 값은 자동으로 constant 메모리에 올라감
__global__ void kernelProcessing(uchar *rc, uchar *gc, uchar *bc, uchar *destR, uchar *destG, uchar *destB, int rows, int cols, int eyeWidth);
__global__ void kernelProcessing2(uint *ch, uint *dest, int rows, int cols, int eyeWidth);
__global__ void kernelProcessing3(uint *dest, int rows, int cols, int eyeWidth);
__global__ void kernelMapping(cv::cuda::GpuMat cvOrigin, cv::cuda::GpuMat cvRes, int rows, int cols, uint *gpuMap);

cv::Mat KernelCall(cv::Mat origin, uchar *cuRc, uchar *cuBc, uchar *cuGc, uchar *cuDestR, uchar *cuDestB, uchar *cuDestG) {
	int size = origin.rows*origin.cols;
	uchar *rChannel = new uchar[size];
	uchar *gChannel = new uchar[size];
	uchar *bChannel = new uchar[size];
	uchar *resR = new uchar[800 * 480];
	uchar *resB = new uchar[800 * 480];
	uchar *resG = new uchar[800 * 480];

	DS_timer timer_a(4);

	timer_a.setTimerName(0, "array divide");
	timer_a.setTimerName(1, "memcpy cpu to device");
	timer_a.setTimerName(2, "array merge");
	timer_a.setTimerName(3, "memcpy device to cpu");
	//overhead

	timer_a.onTimer(0);
	for (int y = 0; y < origin.rows; y++) {
		for (int x = 0; x < origin.cols; x++) {
			rChannel[y*origin.cols + x] = origin.at<cv::Vec3b>(y, x)[2];
			gChannel[y*origin.cols + x] = origin.at<cv::Vec3b>(y, x)[1];
			bChannel[y*origin.cols + x] = origin.at<cv::Vec3b>(y, x)[0];
		}
	}
	timer_a.offTimer(0);

	//uchar *dest = new uchar[800*480*3];

	//memcopy to host to device
	timer_a.onTimer(1);
	cudaMemcpy(cuRc, rChannel, sizeof(uchar)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuBc, bChannel, sizeof(uchar)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuGc, gChannel, sizeof(uchar)*size, cudaMemcpyHostToDevice);
	timer_a.offTimer(1);

	//kernel dimension
	dim3 blockDim(32, 16);
	dim3 gridDim(ceil((float)origin.cols / 32), ceil((float)origin.rows / 16));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	cudaEventRecord(start);
	

	kernelProcessing << <gridDim, blockDim >> > (cuRc, cuGc, cuBc, cuDestR, cuDestG, cuDestB, origin.rows, origin.cols, 80);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout <<"per frame calc"<< time << " ms " << std::endl << std::endl;
	
	timer_a.onTimer(3);
	cudaMemcpy(resR, cuDestR, sizeof(uchar)* 800 * 480, cudaMemcpyDeviceToHost);
	cudaMemcpy(resB, cuDestB, sizeof(uchar)* 800 * 480, cudaMemcpyDeviceToHost);
	cudaMemcpy(resG, cuDestG, sizeof(uchar)* 800 * 480, cudaMemcpyDeviceToHost);

	cudaMemset(cuDestR, 0, sizeof(uchar)* 800 * 480);
	cudaMemset(cuDestB, 0, sizeof(uchar)* 800 * 480);
	cudaMemset(cuDestG, 0, sizeof(uchar)* 800 * 480);
	timer_a.offTimer(3);

	cv::Mat resM(480, 800, CV_8UC3);
	resM.setTo(0);

	timer_a.onTimer(2);
	for (int y = 0; y < resM.rows; y++) {
		for (int x = 0; x < resM.cols; x++) {
			resM.at<cv::Vec3b>(y, x)[0] = (resB[y*resM.cols + x]!= 0)? resB[y*resM.cols + x]: resB[y*resM.cols + x-1]/2+ resB[y*resM.cols + x + 1]/2;
			resM.at<cv::Vec3b>(y, x)[1] = (resG[y*resM.cols + x]!= 0) ? resG[y*resM.cols + x] : resG[y*resM.cols + x - 1]/2 + resG[y*resM.cols + x + 1]/2;
			resM.at<cv::Vec3b>(y, x)[2] = (resR[y*resM.cols + x]!= 0) ? resR[y*resM.cols + x] : resR[y*resM.cols + x - 1]/2 + resR[y*resM.cols + x + 1]/2;
		}
	}
	timer_a.offTimer(2);

	timer_a.printTimer();

	//imshow("resM", resM);

	delete[] rChannel; delete[] bChannel; delete[] gChannel; delete[] resR; delete[] resG; delete[] resB;

	return resM;
}
cv::Mat KernelCall2(cv::Mat origin, uint *ch, uint *Dest) {
	int size = origin.rows*origin.cols;
	uint *Channel = new uint[size];
	uint *res = new uint[800 * 480];

	DS_timer timer_a(4);

	timer_a.setTimerName(0, "array divide");
	timer_a.setTimerName(1, "memcpy cpu to device");
	timer_a.setTimerName(2, "array merge");
	timer_a.setTimerName(3, "memcpy device to cpu");
	//overhead

	timer_a.onTimer(0);
	for (int y = 0; y < origin.rows; y++) {
		for (int x = 0; x < origin.cols; x++) {
			Channel[y*origin.cols + x] = origin.at<cv::Vec3b>(y, x)[0]<<16;
			Channel[y*origin.cols + x] |= origin.at<cv::Vec3b>(y, x)[1]<<8;
			Channel[y*origin.cols + x] |= origin.at<cv::Vec3b>(y, x)[2];
		}
	}
	timer_a.offTimer(0);

	//memcopy to host to device
	timer_a.onTimer(1);
	cudaMemcpy(ch, Channel, sizeof(uint)*size, cudaMemcpyHostToDevice);
	timer_a.offTimer(1);

	//kernel dimension
	dim3 blockDim(32, 16);
	dim3 gridDim(ceil((float)origin.cols / 32), ceil((float)origin.rows / 256));

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	cudaEventRecord(start);


	kernelProcessing2 << <gridDim, blockDim >> > (ch, Dest, origin.rows, origin.cols, 80);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "per frame calc" << time << " ms " << std::endl;

	timer_a.onTimer(3);
	cudaMemcpy(res, Dest, sizeof(uint) * 800 * 480, cudaMemcpyDeviceToHost);
	
	cudaMemset(Dest, 0, sizeof(uint) * 800 * 480);
	timer_a.offTimer(3);

	cv::Mat resM(480, 800, CV_8UC3);
	resM.setTo(0);

	timer_a.onTimer(2);
	for (int y = 0; y < resM.rows; y++) {
		for (int x = 0; x < resM.cols; x++) {
			resM.at<cv::Vec3b>(y, x)[0] = (res[y*resM.cols + x] & 0xFF0000)>>16;
			resM.at<cv::Vec3b>(y, x)[1] = (res[y*resM.cols + x] & 0x00FF00)>>8;
			resM.at<cv::Vec3b>(y, x)[2] = (res[y*resM.cols + x] & 0x0000FF);
		}
	}
	timer_a.offTimer(2);

	timer_a.printTimer();

	//imshow("resM", resM);

	delete[] Channel; delete[] res;

	return resM;
}
uint* KernelCall3(cv::Mat origin, uint *map_) {
	uint *res = new uint[800 * 480];

	dim3 blockDim(32, 16);
	dim3 gridDim(ceil((float)origin.cols / 32), ceil((float)origin.rows / 16));

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	cudaEventRecord(start);


	kernelProcessing3 << <gridDim, blockDim >> > (map_, origin.rows, origin.cols, 80);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "per frame calc " << time << " ms " << std::endl;

	cudaMemcpy(res, map_, sizeof(uint) * 800 * 480, cudaMemcpyDeviceToHost);

	return res;
}
cv::Mat mapping(cv::Mat origin, uint *map_) {
	cv::Mat resM(480, 800, CV_8UC3);
	resM.setTo(0);

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	for (int y = 0; y < resM.rows; y++) {
		for (int x = 0; x < resM.cols; x++) {
			int dx = 0; int dy = 0;
			if (origin.cols > origin.rows) {
				dx = map_[y*resM.cols + x] / origin.cols;
				dy = map_[y*resM.cols + x] % origin.cols;
			}
			else {
				dy = map_[y*resM.cols + x] / (origin.rows + 1);
				dx = map_[y*resM.cols + x] % (origin.rows + 1);
			}
			resM.at<cv::Vec3b>(y, x) = origin.at<cv::Vec3b>(dy, dx);
		}
	}
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

	std::chrono::microseconds micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double millis = ((double)(micro.count())) / 1000;

	std::cout <<"Mapping : " << millis << " ms" << std::endl;

	return resM;
}
cv::Mat mappingInGpu(cv::Mat origin, uint *gpuMap, cv::cuda::GpuMat cvOrigin, cv::cuda::GpuMat cvRes) {
	cv::Mat resM(480, 800, CV_8UC3);
	resM.setTo(0);

	cvOrigin.upload(origin);
	cvRes.upload(resM);


	dim3 blockDim(16, 16);
	dim3 gridDim(ceil((float)resM.cols / 16), ceil((float)resM.rows / 16));

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	kernelMapping << <gridDim, blockDim >> > (cvOrigin, cvRes, origin.rows, origin.cols, gpuMap);
	cudaDeviceSynchronize();

	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

	std::chrono::microseconds micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double millis = ((double)(micro.count())) / 1000;

	std::cout << "Mapping : " << millis << " ms" << std::endl;

	cvRes.download(resM);

	return resM;
}

__global__ void kernelProcessing(uchar *rc, uchar *gc, uchar *bc, uchar *destR, uchar *destG, uchar *destB, int rows, int cols, int eyeWidth) {
	//croping is just memcpy
	/*not use croping loop*/
	//barrel Distortion
	double pA = 0.007715;
	double pB = 0.026731;
	double pC = 0.0;
	double pD = 1 - pA - pB - pC;

	double cx = (double)cols / 2;
	double cy = (double)rows / 2;

	int idxX = blockDim.x * blockIdx.x + threadIdx.x;
	int idxY = (blockDim.y * blockIdx.y + threadIdx.y)*16 ;

	if (idxX > cols || idxY > rows) return;

	int d = MIN(cx, cy);

	for (int i = 0; i < 16; i++) {


		double dx = (idxX - cx) / d;
		double dy = (idxY+i - cy) / d;

		double dr = sqrt(dx*dx + dy * dy);
		double sr = (pA*dr*dr*dr + pB * dr*dr + pC * dr + pD)*dr;
		double factor = abs(dr / sr);

		double srcXd = cx + (dx*factor*d);
		double srcYd = cy + (dy*factor*d);

		int nx = ceil(srcXd);
		int ny = ceil(srcYd);

		double dCol = (double)cols / 400.0;
		double dRow = (double)rows / 480.0;

		//left 0 to cols-eyeWidth, right eyeWidth to cols
		if (nx < 0 || ny < 0 || nx >= cols || ny >= rows || idxX % (int)dCol != 0 || (idxY + i) % (int)dRow != 0) return;
		if ((idxY + i) > rows) return;
		if (idxX < cols - eyeWidth) {
			destR[(int)(ny / dRow) * 800 + (int)(nx / dCol)] = rc[(idxY + i)*cols + idxX];
			destG[(int)(ny / dRow) * 800 + (int)(nx / dCol)] = gc[(idxY + i)*cols + idxX];
			destB[(int)(ny/dRow)* 800 + (int)(nx/dCol)] = bc[(idxY + i)*cols + idxX];
		}
		if (idxX > eyeWidth) {
			destR[(int)(ny / dRow) * 800 + (int)(nx / dCol) + 400] = rc[(idxY + i)*cols + idxX];
			destG[(int)(ny / dRow) * 800 + (int)(nx / dCol) + 400] = gc[(idxY + i)*cols + idxX];
			destB[(int)(ny / dRow)* 800 + (int)(nx / dCol) + 400] = bc[(idxY + i)*cols + idxX];
		}
		__syncthreads();
	}
	__syncthreads();
	//resizing

	//merge 이것도 같이


	//return
}
__global__ void kernelProcessing2(uint *ch, uint *dest, int rows, int cols, int eyeWidth) {
	//croping is just memcpy
	/*not use croping loop*/
	//barrel Distortion
	double pA = 0.007715;
	double pB = 0.026731;
	double pC = 0.0;
	double pD = 1 - pA - pB - pC;

	double cx = (double)cols / 2;
	double cy = (double)rows / 2;

	int idxX = blockDim.x * blockIdx.x + threadIdx.x;
	int idxY = (blockDim.y * blockIdx.y + threadIdx.y)*16;

	if (idxX > cols || idxY > rows) return;

	int d = MIN(cx, cy);

	for (int i = 0; i < 16; i++) {
		double dx = (idxX - cx) / d;
		double dy = (idxY + i - cy) / d;

		double dr = sqrt(dx*dx + dy * dy);
		double sr = (pA*dr*dr*dr + pB * dr*dr + pC * dr + pD)*dr;
		double factor = abs(dr / sr);

		double srcXd = cx + (dx*factor*d);
		double srcYd = cy + (dy*factor*d);

		int nx = ceil(srcXd);
		int ny = ceil(srcYd);

		double dCol = (double)cols / 400.0;
		double dRow = (double)rows / 480.0;

		//left 0 to cols-eyeWidth, right eyeWidth to cols
		if (nx < 0 || ny < 0 || nx >= cols || ny >= rows || idxX % (int)dCol != 0 || (idxY + i) % (int)dRow != 0) return;
		if ((idxY + i) > rows) return;
		if (idxX < cols - eyeWidth) {
			dest[(int)(ny / dRow) * 800 + (int)(nx / dCol)] = ch[(idxY + i)*cols + idxX];
		}
		if (idxX > eyeWidth) {
			dest[(int)(ny / dRow) * 800 + (int)(nx / dCol) + 400] = ch[(idxY + i)*cols + idxX];
		}
		__syncthreads();
	}
	__syncthreads();


	//return
}
__global__ void kernelProcessing3(uint *dest, int rows, int cols, int eyeWidth) {
	double pA = 0.007715;
	double pB = 0.026731;
	double pC = 0.0;
	double pD = 1 - pA - pB - pC;

	double cx = (double)cols / 2;
	double cy = (double)rows / 2;

	int idxX = blockDim.x * blockIdx.x + threadIdx.x;
	int idxY = blockDim.y * blockIdx.y + threadIdx.y;

	if (idxX > cols || idxY > rows) return;

	int d = MIN(cx, cy);

	double dx = (idxX - cx) / d;
	double dy = (idxY - cy) / d;

	double dr = sqrt(dx*dx + dy * dy);
	double sr = (pA*dr*dr*dr + pB * dr*dr + pC * dr + pD)*dr;
	double factor = abs(dr / sr);

	double srcXd = cx + (dx*factor*d);
	double srcYd = cy + (dy*factor*d);

	int nx = ceil(srcXd);
	int ny = ceil(srcYd);

	double dCol = (double)cols / 400.0;
	double dRow = (double)rows / 480.0;

	//left 0 to cols-eyeWidth, right eyeWidth to cols
	if (nx < 0 || ny < 0 || nx >= cols || ny >= rows || idxX % (int)dCol != 0 || idxY % (int)dRow != 0) return;
	if (idxX < cols - eyeWidth) {
		dest[(int)(ny / dRow) * 800 + (int)(nx / dCol)] = (cols>rows)?idxX*cols+idxY:idxY*(rows+1)+idxX;
	}
	if (idxX > eyeWidth) {
		dest[(int)(ny / dRow) * 800 + (int)(nx / dCol) + 400] = (cols>rows)?idxX*cols+idxY:idxY*(rows+1)+idxX;
	}
	__syncthreads();
}

__global__ void kernelMapping(cv::cuda::GpuMat cvOrigin, cv::cuda::GpuMat cvRes, int rows, int cols, uint *gpuMap) {
	int idxX = blockDim.x * blockIdx.x + threadIdx.x;
	int idxY = blockDim.y * blockIdx.y + threadIdx.y;

	if (idxX > cols || idxY > rows) return;
	
	int dx = 0; int dy = 0;
	if (cols > rows) {
		dx = gpuMap[idxY*cvRes.cols + idxX] / cvOrigin.cols;
		dy = gpuMap[idxY*cvRes.cols + idxX] % cvOrigin.cols;
	}
	else {
		dx = gpuMap[idxY*cvRes.cols + idxX] / (cvOrigin.rows + 1);
		dy = gpuMap[idxY*cvRes.cols + idxX] % (cvOrigin.rows + 1);
	}

	//cvRes.ptr(idxY)[idxX] = cvOrigin.ptr(dy)[dx];
	cvRes.data[idxY*800+idxX] = cvOrigin.data[dy * cvOrigin.cols + dx];
	__syncthreads();
}