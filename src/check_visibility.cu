#include "../include/check_visibility.h"
#include "../include/fastRayTriangleIntersection.h"

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

bool* sequential_code(double** verts, unsigned short verts_rows, unsigned short** meshes, unsigned short meshes_rows, unsigned short columns, double** V1, double** V2, double** V3, bool* flag, double* t, double* u, double* v, bool* visible, double* camera_location) {
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

bool* parallel_code(double** verts, unsigned short verts_rows, unsigned short** meshes, unsigned short meshes_rows, unsigned short columns, double** V1, double** V2, double** V3, bool* flag, double* t, double* u, double* v, bool* visible, double* camera_location) {
	// Creating the streams
	cudaStream_t* streams = new cudaStream_t[verts_rows];
	// Creating the events
	cudaEvent_t* start = new cudaEvent_t[verts_rows];
	cudaEvent_t* stop = new cudaEvent_t[verts_rows];
	
	for (unsigned short i = 0; i < verts_rows; i++) {
		cudaStreamCreate(&streams[i]);
		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);
	}
	
	// Creating the device variables
	double** d_camera_location = new double* [verts_rows];
	double** d_verts = new double* [verts_rows];
	double** d_V1 = new double* [meshes_rows];
	double** d_V2 = new double* [meshes_rows];
	double** d_V3 = new double* [meshes_rows];
	bool** d_flag = new bool* [meshes_rows];
	double** d_t = new double* [meshes_rows];
	double** d_u = new double* [meshes_rows];
	double** d_v = new double* [meshes_rows];
	bool** d_visible = new bool* [verts_rows];
	
	// Allocate memory for device variables and copy data from host to device
	for (unsigned short i = 0; i < verts_rows; i++) {
		cudaMalloc((void**)&d_verts[i], columns * sizeof(double));
		cudaMalloc((void**)&d_camera_location[i], columns * sizeof(double));
		cudaMalloc((void**)&d_visible[i], sizeof(bool));
		cudaMemcpyAsync(d_verts[i], verts[i], columns * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_camera_location[i], camera_location, columns * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
	}
	
	for (unsigned short i = 0; i < meshes_rows; i++) {
		cudaMalloc((void**)&d_V1[i], columns * sizeof(double));
		cudaMalloc((void**)&d_V2[i], columns * sizeof(double));
		cudaMalloc((void**)&d_V3[i], columns * sizeof(double));
		cudaMalloc((void**)&d_flag[i], sizeof(bool));
		cudaMalloc((void**)&d_t[i], sizeof(double));
		cudaMalloc((void**)&d_u[i], sizeof(double));
		cudaMalloc((void**)&d_v[i], sizeof(double));
		
		cudaMemcpyAsync(d_V1[i], V1[i], columns * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_V2[i], V2[i], columns * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_V3[i], V3[i], columns * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_flag[i], &flag[i], sizeof(bool), cudaMemcpyHostToDevice, streams[i]);
	}
	
	// Calling the kernel
	for (unsigned short i = 0; i < verts_rows; i++) {
		cudaEventRecord(start[i], streams[i]);
		kernel_fastRayTriangleIntersection<<<1, meshes_rows, 0, streams[i]>>>(d_camera_location[i], d_verts[i], d_V1, d_V2, d_V3, meshes_rows, columns, Exclusive, Segment, TwoSided, false, d_flag[i], d_t[i], d_u[i], d_v[i]);
		cudaEventRecord(stop[i], streams[i]);
	}
	
	// Copying the results back to the host
	for (unsigned short i = 0; i < verts_rows; i++) {
		cudaMemcpyAsync(&visible[i], d_visible[i], sizeof(bool), cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(&flag[i], d_flag[i], sizeof(bool), cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(&t[i], d_t[i], sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(&u[i], d_u[i], sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(&v[i], d_v[i], sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
	}
	
	// Synchronize streams and clean up
	for (unsigned short i = 0; i < verts_rows; i++) {
		cudaStreamSynchronize(streams[i]);
		cudaStreamDestroy(streams[i]);
		cudaEventDestroy(start[i]);
		cudaEventDestroy(stop[i]);
		cudaFree(d_camera_location[i]);
		cudaFree(d_verts[i]);
		cudaFree(d_visible[i]);
	}
	
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


bool* check_visibility(double** verts, unsigned short verts_rows, unsigned short** meshes, unsigned short meshes_rows, unsigned short columns, bool** gt) {
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

	bool* flag = new bool[meshes_rows], * visible = new bool[verts_rows];

	//the t in the matlab code was be replaced by the v to mantain the same name used in the fastRayTriangleIntersection function
	double* t = new double[meshes_rows], * u = new double[meshes_rows], * v = new double[meshes_rows];
	
	visible = sequential_code(verts, verts_rows, meshes, meshes_rows, columns, gt);
	check(visible, gt, verts_rows);
	
	visible = parallel_code(verts, verts_rows, meshes, meshes_rows, columns, gt);
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
