#ifndef CHECK_VISIBILITY_H
#define CHECK_VISIBILITY_H

#include <iostream>
#include "../src/fastRayTriangleIntersection.cuh"

bool* check_visibility(double** verts,
	unsigned short verts_rows, // Number of vertices
	unsigned short** meshes,
	unsigned short meshes_rows, // Number of meshes
	unsigned short columns,
	bool** gt); // Number of columns in the mesh and verts matrices

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
	double * camera_location,
	double ** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double ** V1, double ** V2, double ** V3,
	bool * flag,
	double * t, double * u, double * v, bool * visible)
{
	// Creating the device variables
	double * d_camera_location, * d_verts,
			** d_meshes, ** d_V1, ** d_V2, ** d_V3,
			* d_t, * d_u, * d_v;
	bool* d_flag, *d_visible;

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

	// Creating the streams
	unsigned short SegSize = 32, batch = std::round(meshes_rows / SegSize);
	dim3 blockDim(SegSize, 1);
	dim3 gridDim(batch, 1);

	cudaStream_t* meshes_streams = new cudaStream_t[batch];
	for (unsigned short i = 0; i < batch; i++)
		cudaStreamCreate(&meshes_streams[i]);

	cudaMemcpy(d_camera_location, camera_location, columns * sizeof(double), cudaMemcpyHostToDevice);
	for (unsigned short i = 0; i < verts_rows; i++) {
		visible[i] = true;

		cudaMemcpy(d_verts + i, verts[i], columns * sizeof(double), cudaMemcpyHostToDevice);		
		
		for (unsigned short j = 0; j < meshes_rows && visible[i]; j += SegSize) {
			cudaMemcpyAsync(d_meshes + j * SegSize, meshes + j * SegSize, SegSize * columns * sizeof(unsigned short), cudaMemcpyHostToDevice, meshes_streams[j]);
			cudaMemcpyAsync(d_V1 + j * SegSize, V1 + j * SegSize, SegSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[j]);
			cudaMemcpyAsync(d_V2 + j * SegSize, V2 + j * SegSize, SegSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[j]);
			cudaMemcpyAsync(d_V3 + j * SegSize, V3 + j * SegSize, SegSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[j]);

			kernel_fastRayTriangleIntersection <<<gridDim, blockDim, 0, meshes_streams[j]>>>
				(d_camera_location, d_verts + i,
					d_V1 + j * SegSize, d_V2 + j * SegSize, d_V3 + j * SegSize,
					SegSize, columns,
					Exclusive, Segment, TwoSided, false,
					d_flag + j, d_t + j, d_u + j, d_v + j
				);
			
			cudaMemcpyAsync(flag + j * SegSize, d_flag + j * SegSize, SegSize * sizeof(bool), cudaMemcpyDeviceToHost, meshes_streams[j]);
			cudaMemcpyAsync(t + j * SegSize, d_t + j * SegSize, SegSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[j]);
			cudaMemcpyAsync(u + j * SegSize, d_u + j * SegSize, SegSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[j]);
			cudaMemcpyAsync(v + j * SegSize, d_v + j * SegSize, SegSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[j]);
		
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



#endif CHECK_VISIBILITY_H