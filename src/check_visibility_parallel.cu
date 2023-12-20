#include "../include/check_visibility_parallel.cuh"

double check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double* h_verts, unsigned short verts_rows,
	double* h_V1, double* h_V2, double* h_V3, unsigned short V_rows,
	bool* h_flag, double* h_t, double* h_u, double* h_v, bool* visible) // Output variables
{
	cudaError_t cudaStatus = cudaSetDevice(1);  // Setting the GPU device index as needed
	if (cudaStatus != cudaSuccess)
		std::cerr << "cudaSetDevice failed! Error: " << cudaGetErrorString(cudaStatus) << std::endl;

	double* d_camera_location, * d_verts,
		* d_V1, * d_V2, * d_V3,
		* d_t, * d_u, * d_v;

	bool* d_flag;

	unsigned short* d_visible, * h_visible; // Using h_visible to copy back from device to host (Pinned memory)

	cudaMalloc((void**)&d_camera_location, COLUMNS_SIZE * sizeof(double));

	cudaMalloc((void**)&d_verts, verts_rows * COLUMNS_SIZE * sizeof(double));
	cudaMallocHost((void**)&h_visible, verts_rows * sizeof(unsigned short));
	cudaMalloc((void**)&d_visible, verts_rows * sizeof(unsigned short));

	cudaMalloc((void**)&d_V1, V_rows * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V2, V_rows * COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_V3, V_rows * COLUMNS_SIZE * sizeof(double));

	cudaMalloc((void**)&d_flag, BLOCK_ROWS_SIZE * sizeof(bool));
	cudaMalloc((void**)&d_t, BLOCK_ROWS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_u, BLOCK_ROWS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_v, BLOCK_ROWS_SIZE * sizeof(double));

	cudaMemcpy(d_camera_location, camera_location, COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_verts, h_verts, verts_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	for (unsigned short verts_row = 0; verts_row < verts_rows; verts_row++)
		h_visible[verts_row] = 0;
	cudaMemcpy(d_visible, h_visible, verts_rows * sizeof(unsigned short), cudaMemcpyHostToDevice);

	cudaMemcpy(d_V1, h_V1, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V3, h_V3, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	

	dim3 blockDim(1, BLOCK_ROWS_SIZE);
	dim3 gridDim(std::ceil(V_rows / BLOCK_ROWS_SIZE), 1);


	timer::Timer<timer::DEVICE> dev_TM;
	dev_TM.start();

	unsigned short V_row, d_V_row;
	for (unsigned short verts_row = 0, d_verts_row = 0; verts_row < verts_rows; verts_row++, d_verts_row += COLUMNS_SIZE) {
		for (V_row = 0, d_V_row = 0; V_row < V_rows; V_row += BLOCK_ROWS_SIZE, d_V_row += V_BLOCK_SIZE) {
			//std::cout << "V_row = " << V_row << "\n";

			fastRayTriangleIntersection_parallel_with_check << <gridDim, blockDim, 0>> > (
				d_camera_location, d_verts + d_verts_row,
				d_V1 + d_V_row, d_V2 + d_V_row, d_V3 + d_V_row, BLOCK_ROWS_SIZE,
				BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
				d_flag, d_t, d_u, d_v,
				d_visible + verts_row);	
		}

		fastRayTriangleIntersection_parallel_with_check << <gridDim, blockDim, 0 >> > (
			d_camera_location, d_verts + d_verts_row,
			d_V1 + d_V_row, d_V2 + d_V_row, d_V3 + d_V_row, V_rows - V_row,
			BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
			d_flag, d_t, d_u, d_v,
			d_visible + verts_row);
	}

	dev_TM.stop();
	dev_TM.print("MoellerTrumboreIntersectionAlgorithm device:   ");


	cudaMemcpy(h_visible, d_visible, verts_rows * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	for(unsigned short i = 0; i < verts_rows; i++)
		visible[i] = h_visible[i] == 0;

	cudaFree(d_camera_location), cudaFree(d_verts);
	cudaFree(d_V1), cudaFree(d_V2), cudaFree(d_V3);
	cudaFree(d_flag), cudaFree(d_t), cudaFree(d_u), cudaFree(d_v);
	cudaFree(d_visible);
	cudaFreeHost(h_visible);

	return dev_TM.duration();
}
