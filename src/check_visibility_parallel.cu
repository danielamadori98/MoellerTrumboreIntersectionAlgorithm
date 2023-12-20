#include "../include/check_visibility_parallel.cuh"


void copy_verts_to_device(double** verts, unsigned short verts_rows, double* d_vert) {
	double h_vert[COLUMNS_SIZE];

	for (unsigned short verts_row = 0, d_V_row = 0; verts_row < verts_rows; verts_row++, d_V_row += COLUMNS_SIZE) {
		for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
			h_vert[col] = verts[verts_row][col];

		cudaMemcpy(d_vert + d_V_row, h_vert, COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	}
}

//Zero error
double check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double* h_V1, double* h_V2, double* h_V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible) // Output variables
{
	cudaError_t cudaStatus = cudaSetDevice(1);  // Setting the GPU device index as needed
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
	unsigned short* d_visible, * h_visible; // Using h_visible to copy back from device to host (Pinned memory)
	cudaMallocHost((void**)&h_visible, verts_rows * sizeof(unsigned short));

	cudaMalloc((void**)&d_visible, verts_rows * sizeof(unsigned short));

	cudaMalloc((void**)&d_camera_location, COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_vert, verts_rows * COLUMNS_SIZE * sizeof(double));

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
	
	cudaMemcpy(d_V1, h_V1, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V3, h_V3, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	
	copy_verts_to_device(verts, verts_rows, d_vert);

	for(unsigned short verts_row = 0; verts_row < verts_rows; verts_row++)
		h_visible[verts_row] = 0;
	cudaMemcpy(d_visible, h_visible, verts_rows * sizeof(unsigned short), cudaMemcpyHostToDevice);


	timer::Timer<timer::DEVICE> dev_TM;
	dev_TM.start();

	for (unsigned short verts_row = 0, d_verts_row = 0; verts_row < verts_rows; verts_row++, d_verts_row += COLUMNS_SIZE) {

		unsigned short V_row, d_V_row;
		for (V_row = 0, d_V_row = 0; V_row < V_rows; V_row += BLOCK_ROWS_SIZE, d_V_row += V_BLOCK_SIZE) {
			//std::cout << "V_row = " << V_row << "\n";

			fastRayTriangleIntersection_parallel_with_check << <gridDim, blockDim, 0>> > (
				d_camera_location, d_vert + d_verts_row,
				d_V1 + d_V_row, d_V2 + d_V_row, d_V3 + d_V_row, BLOCK_ROWS_SIZE,
				BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
				d_flag, d_t, d_u, d_v,
				d_visible + verts_row);	
		}

		fastRayTriangleIntersection_parallel_with_check << <gridDim, blockDim, 0 >> > (
			d_camera_location, d_vert + d_verts_row,
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

	cudaFree(d_camera_location), cudaFree(d_vert);
	cudaFree(d_V1), cudaFree(d_V2), cudaFree(d_V3);
	cudaFree(d_flag), cudaFree(d_t), cudaFree(d_u), cudaFree(d_v);
	cudaFree(d_visible);
	cudaFreeHost(h_visible);

	return dev_TM.duration();
}


//Zero error (sometimes)
void check_visibility_parallel_code_streams(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double* h_V1, double* h_V2, double* h_V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible) // Output variables
{
	cudaError_t cudaStatus = cudaSetDevice(1);  // Setting the GPU device index as needed
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
	unsigned short *d_visible, *h_visible; // Using h_visible to copy back from device to host (Pinned memory)
	cudaMallocHost((void**)&h_visible, streams_size * sizeof(unsigned short));

	cudaMalloc((void**)&d_visible, sizeof(unsigned short));

	cudaMalloc((void**)&d_camera_location, COLUMNS_SIZE * sizeof(double));
	cudaMalloc((void**)&d_vert, verts_rows * COLUMNS_SIZE * sizeof(double));

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

	cudaMemcpy(d_V1, h_V1, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V3, h_V3, V_rows * COLUMNS_SIZE * sizeof(double), cudaMemcpyHostToDevice);

	copy_verts_to_device(verts, verts_rows, d_vert);

	for (unsigned short verts_row = 0, d_verts_row = 0; verts_row < verts_rows; verts_row++, d_verts_row += COLUMNS_SIZE) {
		visible[verts_row] = true;
		
		for (unsigned short V_row = 0, d_V_row = 0; V_row < V_rows;) {
			//std::cout << "V_row = " << V_row << "\n";

			for (unsigned short stream = 0; stream < streams_size; stream++, V_row += BLOCK_ROWS_SIZE, d_V_row += V_BLOCK_SIZE) {
				unsigned short remaing_rows = V_rows - V_row;
				if (remaing_rows >= BLOCK_ROWS_SIZE)
					remaing_rows = BLOCK_ROWS_SIZE;
				
				fastRayTriangleIntersection_parallel_with_check << <gridDim, blockDim, 0, streams[stream] >> > (
					d_camera_location, d_vert + d_verts_row,
					d_V1 + d_V_row, d_V2 + d_V_row, d_V3 + d_V_row, remaing_rows,
					BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
					d_flag, d_t, d_u, d_v,
					d_visible);

				/*
				cudaMemcpyAsync(flag + V_row, d_flag, BLOCK_ROWS_SIZE * sizeof(bool), cudaMemcpyDeviceToHost, streams[stream]);
				cudaMemcpyAsync(t + V_row, d_t, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[stream]);
				cudaMemcpyAsync(u + V_row, d_u, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[stream]);
				cudaMemcpyAsync(v + V_row, d_v, BLOCK_ROWS_SIZE * sizeof(double), cudaMemcpyDeviceToHost, streams[stream]);
				*/
				cudaMemcpyAsync(h_visible + stream, d_visible, sizeof(unsigned short), cudaMemcpyDeviceToHost, streams[stream]);
				//std::cout << "h_visible = " << h_visible[stream] << "\n";
				//cudaStreamSynchronize(streams[stream]);
			}
			// TODO: fix cudaStreamSyncronize work just if you put it twice: in the strems loop and after it (IDK why)
			for(unsigned short stream = 0; stream < streams_size; stream++){
				cudaStreamSynchronize(streams[stream]);
				if (h_visible[stream] > 0) 
					visible[verts_row] = false;
			}
		}

		if (verts_row % 100 == 0)
			std::cout << "Visible[" << verts_row << "] = " << visible[verts_row] << "\n";
	}

	cudaFree(d_camera_location), cudaFree(d_vert);
	cudaFree(d_V1), cudaFree(d_V2), cudaFree(d_V3);
	cudaFree(d_flag), cudaFree(d_t), cudaFree(d_u), cudaFree(d_v);
	cudaFree(d_visible);
	cudaFreeHost(h_visible);
}


