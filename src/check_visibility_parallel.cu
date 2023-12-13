#include "../include/check_visibility_parallel.cuh"

bool* check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double** V1, double** V2, double **V3, unsigned short V_rows,
	bool* flag, 
	double* t, double* u, double* v, bool* visible)
{
	// Creating the device variables
	double *d_camera_location, * d_vert,
		** d_V1, ** d_V2, ** d_V3,
		* d_t, * d_u, * d_v;

	bool* d_flag;

	// Initialize CUDA
	cudaError_t cudaStatus = cudaSetDevice(0);  // You can set the GPU device index as needed
	if (cudaStatus != cudaSuccess)
		std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;

	int device;
	cudaGetDevice(&device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	unsigned short maxStreams = deviceProp.asyncEngineCount;
	//std::cout << "Maximum number of CUDA streams on GPU " << device << ": " << maxStreams << std::endl;
	//TODO remove
	maxStreams = 1;
	unsigned short segSize = 32, segsNumber = (unsigned short)std::round(V_rows / segSize);

	// Allocate memory for device variables and copy data from host to device
	cudaMalloc((void**)&d_camera_location, COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_vert, COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V1, segSize * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V2, segSize * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V3, segSize * COLUMNS_SIZE * sizeof(double));
	
	cudaMalloc((void**)&d_flag, segSize * sizeof(bool));
	cudaMalloc((void**)&d_t, segSize * sizeof(double));
	cudaMalloc((void**)&d_u, segSize * sizeof(double));
	cudaMalloc((void**)&d_v, segSize * sizeof(double));

	std::cout << "After Allocation\n";

	// Creating the streams
	dim3 blockDim(segSize, 1);
	dim3 gridDim(segsNumber, 1);


	cudaStream_t* meshes_streams = new cudaStream_t[maxStreams];
	for (unsigned short i = 0; i < maxStreams; i++)
		cudaStreamCreate(&meshes_streams[i]);


	double vert[COLUMNS_SIZE];
	for (unsigned short verts_row = 0, segs_index = 0, stream = 0, V_row = 0; verts_row < verts_rows; verts_row++) {
		visible[verts_row] = true;

		//std::cout << "verts[" << i << "] =\t";
		for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
			vert[col] = verts[verts_row][col];

		cudaMemcpyAsync(d_camera_location, camera_location, COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[0]);
		cudaMemcpyAsync(d_vert, vert, COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[0]);
		
		for (segs_index = 0; segs_index < segsNumber && visible[verts_row]; segs_index++) {
			stream = segs_index % maxStreams;
			V_row = segs_index * segSize;

			cudaMemcpyAsync(d_V1, V1 + V_row, segSize * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V2, V2 + V_row, segSize * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V3, V3 + V_row, segSize * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			
			//std::cout << "After copying all, j = " << j << "\n";
			
			kernel_fastRayTriangleIntersection<<<gridDim, blockDim, 0, meshes_streams[stream]>>>(
				d_camera_location, d_vert,
				d_V1, d_V2, d_V3, segSize,
				BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
				d_flag, d_t, d_u, d_v);
			
			//std::cout << "After kernel, j = " << j << "\n";

			cudaMemcpyAsync(flag + V_row, d_flag, segSize * sizeof(bool), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			cudaMemcpyAsync(t + V_row, d_t, segSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			cudaMemcpyAsync(u + V_row, d_u, segSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			cudaMemcpyAsync(v + V_row, d_v, segSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			
			//TODO check
			cudaStreamSynchronize(meshes_streams[stream]);

			for (unsigned short k = V_row; k < V_row + segSize; k++)
				if (flag[k]) {
					visible[verts_row] = false;
					std::cout << "bene!\n";
					break;
				}

			//std::cout << "After checking visibility, visible[" << i << "] = " << visible[i] << "\n";
		}
	}

	//Free memory
	cudaFree(d_camera_location);
	cudaFree(d_vert);
	cudaFree(d_V1);
	cudaFree(d_V2);
	cudaFree(d_V3);

	cudaFree(d_flag);
	cudaFree(d_t);
	cudaFree(d_u);
	cudaFree(d_v);

	std::cout << "After freeing all\n";

	return visible;
}
