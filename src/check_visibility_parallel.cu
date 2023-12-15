#include "../include/check_visibility_parallel.cuh"


void copy_V_arrays_to_device(double** V1, double** V2, double** V3, unsigned short V_rows,
	double* d_V1, double* d_V2, double* d_V3) {

	double h_V1[V_BLOCK_SIZE], h_V2[V_BLOCK_SIZE], h_V3[V_BLOCK_SIZE];

	unsigned short V_row, d_V_row;
	for (V_row = 0, d_V_row = 0; V_row < V_rows - BLOCK_ROWS_SIZE + 1; V_row += BLOCK_ROWS_SIZE, d_V_row += V_BLOCK_SIZE) {
		for (unsigned short row = 0; row < BLOCK_ROWS_SIZE; row++)
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++) {
				h_V1[row * COLUMNS_SIZE + col] = V1[V_row + row][col];
				h_V2[row * COLUMNS_SIZE + col] = V2[V_row + row][col];
				h_V3[row * COLUMNS_SIZE + col] = V3[V_row + row][col];
			}

		cudaMemcpy(d_V1 + d_V_row, h_V1, V_BLOCK_SIZE * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_V2 + d_V_row, h_V2, V_BLOCK_SIZE * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_V3 + d_V_row, h_V3, V_BLOCK_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	}
	unsigned short V_last_row_size = V_rows - V_row;
	for (unsigned short row = 0; row < V_last_row_size; row++)
		for (unsigned short col = 0; col < COLUMNS_SIZE; col++) {
			h_V1[row * COLUMNS_SIZE + col] = V1[V_row + row][col];
			h_V2[row * COLUMNS_SIZE + col] = V2[V_row + row][col];
			h_V3[row * COLUMNS_SIZE + col] = V3[V_row + row][col];
		}
	cudaMemcpy(d_V1 + d_V_row, h_V1, V_last_row_size * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2 + d_V_row, h_V2, V_last_row_size * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V3 + d_V_row, h_V3, V_last_row_size * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
}



void check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double** V1, double** V2, double** V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible) // Output variables
{
	// Initialize CUDA
	cudaError_t cudaStatus = cudaSetDevice(1);  // You can set the GPU device index as needed
	if (cudaStatus != cudaSuccess)
		std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;

	// Creating the device variables
	double* d_camera_location, * d_vert,
		* d_V1, * d_V2, * d_V3,
		* d_t, * d_u, * d_v;

	bool* d_flag;

	// Allocate memory for device variables and copy data from host to device
	cudaMalloc((void**)&d_camera_location, COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_vert, COLUMNS_SIZE * sizeof(double));

	cudaMalloc((void**)&d_V1, V_rows * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V2, V_rows * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V3, V_rows * COLUMNS_SIZE * sizeof(double));

	cudaMalloc((void**)&d_flag, BLOCK_ROWS_SIZE * sizeof(bool));
	cudaMalloc((void**)&d_t, BLOCK_ROWS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_u, BLOCK_ROWS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_v, BLOCK_ROWS_SIZE * sizeof(double));

	dim3 blockDim(1, BLOCK_ROWS_SIZE);
	dim3 gridDim(std::ceil(V_rows / BLOCK_ROWS_SIZE), 1);

	cudaMemcpy(d_camera_location, camera_location, COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	copy_V_arrays_to_device(V1, V2, V3, V_rows, d_V1, d_V2, d_V3);

	for (unsigned short verts_row = 0; verts_row < verts_rows; verts_row++) {
		visible[verts_row] = true;

		cudaMemcpy(d_vert, verts[verts_row], COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
		for (unsigned short V_row = 0, d_V_row = 0; V_row < V_rows - BLOCK_ROWS_SIZE + 1; V_row += BLOCK_ROWS_SIZE, d_V_row += V_BLOCK_SIZE) {
			//std::cout << "V_row = " << V_row << "\n";

			fastRayTriangleIntersection_parallel << <gridDim, blockDim, 0 >> > (
				d_camera_location, d_vert,
				d_V1 + d_V_row, d_V2 + d_V_row, d_V3 + d_V_row, BLOCK_ROWS_SIZE,
				BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
				d_flag, d_t, d_u, d_v);

			cudaDeviceSynchronize();

			//std::cout << "After kernel, V_row = " << V_row << "\n";

			cudaMemcpy(flag + V_row, d_flag, BLOCK_ROWS_SIZE * sizeof(bool), cudaMemcpyDeviceToHost);
			//cudaMemcpy(t + V_row, d_t, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaMemcpy(u + V_row, d_u, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaMemcpy(v + V_row, d_v, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

			//std::cout << "After copying back all, V_row = " << V_row << "\n";

			for (unsigned short k = V_row; k < V_row + BLOCK_ROWS_SIZE; k++)
				if (flag[k]) {
					visible[verts_row] = false;
					break;
				}
		}
		if (verts_row % 100 == 0)
			std::cout << "Visible[" << verts_row << "] = " << visible[verts_row] << "\n";
	}

	cudaFree(d_camera_location), cudaFree(d_vert);
	cudaFree(d_V1), cudaFree(d_V2), cudaFree(d_V3);
	cudaFree(d_flag), cudaFree(d_t), cudaFree(d_u), cudaFree(d_v);
}


void check_visibility_parallel_code_streams(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double** V1, double** V2, double** V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible) // Output variables
{
	// Initialize CUDA
	cudaError_t cudaStatus = cudaSetDevice(1);  // You can set the GPU device index as needed
	if (cudaStatus != cudaSuccess)
		std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;

	int device;
	cudaGetDevice(&device);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	unsigned short streams_size = deviceProp.asyncEngineCount;

	cudaStream_t* streams = new cudaStream_t[streams_size];
	for (unsigned short i = 0; i < streams_size; i++)
		cudaStreamCreate(&streams[i]);

	// Creating the device variables
	double* d_camera_location, * d_vert,
		* d_V1, * d_V2, * d_V3,
		* d_t, * d_u, * d_v;

	bool* d_flag;

	// Allocate memory for device variables and copy data from host to device
	cudaMalloc((void**)&d_camera_location, COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_vert, COLUMNS_SIZE * sizeof(double));

	cudaMalloc((void**)&d_V1, V_rows * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V2, V_rows * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V3, V_rows * COLUMNS_SIZE * sizeof(double));

	cudaMalloc((void**)&d_flag, BLOCK_ROWS_SIZE * sizeof(bool));
	cudaMalloc((void**)&d_t, BLOCK_ROWS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_u, BLOCK_ROWS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_v, BLOCK_ROWS_SIZE * sizeof(double));

	dim3 blockDim(1, BLOCK_ROWS_SIZE);
	dim3 gridDim(std::ceil(V_rows / BLOCK_ROWS_SIZE), 1);

	cudaMemcpy(d_camera_location, camera_location, COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	copy_V_arrays_to_device(V1, V2, V3, V_rows, d_V1, d_V2, d_V3);

	unsigned short stream = 0;
	for (unsigned short verts_row = 0; verts_row < verts_rows; verts_row++) {
		visible[verts_row] = true;
		stream = verts_row % streams_size;

		cudaMemcpyAsync(d_vert, verts[verts_row], COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice, streams[stream]);
		for (unsigned short V_row = 0, d_V_row = 0; V_row < V_rows - BLOCK_ROWS_SIZE + 1; V_row += BLOCK_ROWS_SIZE, d_V_row += V_BLOCK_SIZE) {
			//std::cout << "V_row = " << V_row << "\n";

			fastRayTriangleIntersection_parallel<<<gridDim, blockDim, 0, streams[stream]>>>(
				d_camera_location, d_vert,
				d_V1 + d_V_row, d_V2 + d_V_row, d_V3 + d_V_row, BLOCK_ROWS_SIZE,
				BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
				d_flag, d_t, d_u, d_v);

			//std::cout << "After kernel, V_row = " << V_row << "\n";

			cudaMemcpyAsync(flag + V_row, d_flag, BLOCK_ROWS_SIZE * sizeof(bool), cudaMemcpyDeviceToHost, streams[stream]);
			//cudaMemcpyAsync(t + V_row, d_t, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[stream]);
			//cudaMemcpyAsync(u + V_row, d_u, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[stream]);
			//cudaMemcpyAsync(v + V_row, d_v, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[stream]);

			//std::cout << "After copying back all, V_row = " << V_row << "\n";

			for (unsigned short k = V_row; k < V_row + BLOCK_ROWS_SIZE; k++)
				if (flag[k]) {
					visible[verts_row] = false;
					break;
				}
		}

		if (verts_row % 100 == 0)
			std::cout << "Visible[" << verts_row << "] = " << visible[verts_row] << "\n";
	}

	cudaFree(d_camera_location), cudaFree(d_vert);
	cudaFree(d_V1), cudaFree(d_V2), cudaFree(d_V3);
	cudaFree(d_flag), cudaFree(d_t), cudaFree(d_u), cudaFree(d_v);
}
