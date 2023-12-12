#include "../include/check_visibility.h"

void check(bool* visible, bool** gt, unsigned short verts_rows) {
	bool error = false;
	for(unsigned short i = 0; i < verts_rows; i++) {
		if (visible[i] != gt[i][0]) {
			error = true;
			std::cerr << "Error in vertex " << i << std::endl;
			std::cerr << "Expected: " << gt[i][0] << " - Obtained: " << visible[i] << std::endl;
		}
	}
	if(!error)
		std::cout << "All vertices are correctly classified" << std::endl;
}

bool* check_visibility(
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	bool** gt) 
{

	double camera_location[3] = { 0, 0, 0 };

	double** V1 = new double* [meshes_rows];
	double** V2 = new double* [meshes_rows];
	double** V3 = new double* [meshes_rows];
	for (unsigned short i = 0; i < meshes_rows; i++) {
		V1[i] = new double[columns];
		V2[i] = new double[columns];
		V3[i] = new double[columns];

		for (unsigned short j = 0; j < columns; j++) {
			V1[i][j] = verts[meshes[i][0]][j];
			V2[i][j] = verts[meshes[i][1]][j];
			V3[i][j] = verts[meshes[i][2]][j];
		}
	}

	/*
	std::cout << "First 5 V1:" << std::endl;
	for (unsigned short i = 0; i < 5; i++)
		std::cout << V1[i][0] << ", " << V1[i][1] << ", " << V1[i][2] << std::endl;

	std::cout << "First 5 V2:" << std::endl;
	for (unsigned short i = 0; i < 5; i++)
		std::cout << V2[i][0] << ", " << V2[i][1] << ", " << V2[i][2] << std::endl;

	std::cout << "First 5 V3:" << std::endl;
	for (unsigned short i = 0; i < 5; i++)
		std::cout << V3[i][0] << ", " << V3[i][1] << ", " << V3[i][2] << std::endl;
	*/

	bool* flag = new bool[meshes_rows], *visible = new bool[verts_rows];

	//the t in the matlab code was be replaced by the v to mantain the same name used in the fastRayTriangleIntersection function
	double* t = new double[meshes_rows], *u = new double[meshes_rows], *v = new double[meshes_rows];

	//visible = sequential_code(camera_location, verts, verts_rows, meshes, meshes_rows, columns, V1, V2, V3, flag, t, u, v, visible);
	//check(visible, gt, verts_rows);
	
	visible = parallel_code(camera_location, verts, verts_rows, meshes, meshes_rows, columns, V1, V2, V3, flag, t, u, v, visible);
	check(visible, gt, verts_rows);

	for (unsigned short i = 0; i < meshes_rows; i++) {
		delete[] V1[i];
		delete[] V2[i];
		delete[] V3[i];
	}
	delete[] V1;
	delete[] V2;
	delete[] V3;

	delete[] t;
	delete[] u;
	delete[] v;
	delete[] flag;

	return visible;
}


bool* sequential_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible)
{

	for (unsigned short i = 0; i < verts_rows; i++) {
		fastRayTriangleIntersection(camera_location, verts[i], V1, V2, V3, meshes_rows, columns, Exclusive, Segment, TwoSided, false, flag, t, u, v);

		visible[i] = true;
		for (unsigned short j = 0; j < meshes_rows; j++)
			if (flag[j]) {
				visible[i] = false;
				break;
			}

		/*
		std::cout << "First 5 v:" << std::endl;
		for (unsigned short i = 0; i < 5; i++)
			//std::cout << v[i] << std::endl;
		*/
	}
	return visible;
}


bool* parallel_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible)
{
	// Creating the device variables
	double* d_camera_location, * d_verts,
		** d_meshes, ** d_V1, ** d_V2, ** d_V3,
		* d_t, * d_u, * d_v;
	bool* d_flag, * d_visible;

	//TODO remove this (is just for testing)
	meshes_rows = 32;
	verts_rows = 32;

	// Initialize CUDA
	cudaError_t cudaStatus = cudaSetDevice(0);  // You can set the GPU device index as needed
	if (cudaStatus != cudaSuccess)
		std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;
	

	int device;
	cudaGetDevice(&device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	unsigned short maxStreams = deviceProp.asyncEngineCount;
	std::cout << "Maximum number of CUDA streams on GPU " << device << ": " << maxStreams << std::endl;


	std::cout << "Before Allocation\n";

	// Allocate memory for device variables and copy data from host to device
	cudaMalloc((void**)&d_camera_location, columns * sizeof(double));
	cudaMalloc((void**)&d_verts, verts_rows * columns * sizeof(double));

	cudaMalloc((void**)&d_meshes, meshes_rows * columns * sizeof(unsigned short));
	cudaMalloc((void**)&d_V1, meshes_rows * columns * sizeof(double));
	cudaMalloc((void**)&d_V2, meshes_rows * columns * sizeof(double));
	cudaMalloc((void**)&d_V3, meshes_rows * columns * sizeof(double));
	cudaMalloc((void**)&d_flag, meshes_rows * sizeof(bool));

	cudaMalloc((void**)&d_t, meshes_rows * sizeof(double));
	cudaMalloc((void**)&d_u, meshes_rows * sizeof(double));
	cudaMalloc((void**)&d_v, meshes_rows * sizeof(double));
	cudaMalloc((void**)&d_visible, meshes_rows * sizeof(bool));

	std::cout << "After Allocation\n";

	// Creating the streams
	unsigned short SegSize = 32, batch = (unsigned short)std::round(meshes_rows / SegSize);
	dim3 blockDim(SegSize, 1);
	dim3 gridDim(batch, 1);


	cudaStream_t* meshes_streams = new cudaStream_t[maxStreams];
	for (unsigned short i = 0; i < maxStreams; i++)
		cudaStreamCreate(&meshes_streams[i]);


	for (unsigned short i = 0; i < verts_rows; i++) {
		visible[i] = true;

		for (unsigned short j = 0; j < meshes_rows && visible[i]; j += SegSize) {
			unsigned short stream = j % maxStreams;
			cudaMemcpyAsync(d_camera_location, camera_location, columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			std::cout << "After copyng camera_location\n";

			cudaMemcpyAsync(d_verts + i, verts[i], columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			std::cout << "After copying verts[" << i << "]\n";
			cudaMemcpyAsync(d_meshes + j * SegSize, meshes + j * SegSize, SegSize * columns * sizeof(unsigned short), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V1 + j * SegSize, V1 + j * SegSize, SegSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V2 + j * SegSize, V2 + j * SegSize, SegSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V3 + j * SegSize, V3 + j * SegSize, SegSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			
			kernel_fastRayTriangleIntersection <<<gridDim, blockDim, 0, meshes_streams[stream] >>>
				(d_camera_location, d_verts + i,
					d_V1 + j * SegSize, d_V2 + j * SegSize, d_V3 + j * SegSize,
					SegSize,
					BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
					d_flag + j, d_t + j, d_u + j, d_v + j
					);
			
			cudaMemcpyAsync(flag + j * SegSize, d_flag + j * SegSize, SegSize * sizeof(bool), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			cudaMemcpyAsync(t + j * SegSize, d_t + j * SegSize, SegSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			cudaMemcpyAsync(u + j * SegSize, d_u + j * SegSize, SegSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			cudaMemcpyAsync(v + j * SegSize, d_v + j * SegSize, SegSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);

			for (unsigned short k = j; k < j + SegSize; k++)
				if (flag[k]) {
					visible[i] = false;
					break;
				}
		}
	}
	//Free memory
	cudaFree(d_camera_location);

	delete[] d_camera_location;
	delete[] d_verts;
	delete[] d_V1;
	delete[] d_V2;
	delete[] d_V3;
	delete[] d_flag;
	delete[] d_t;
	delete[] d_u;
	delete[] d_v;
	delete[] d_visible;

	return visible;
}
