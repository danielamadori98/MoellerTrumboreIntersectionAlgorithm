#include "../include/check_visibility_parallel.cuh"

bool* check_visibility_parallel_code(
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
	unsigned short segSize = 32, segNumber = (unsigned short)std::round(meshes_rows / segSize);
	dim3 blockDim(segSize, 1);
	dim3 gridDim(segNumber, 1);


	cudaStream_t* meshes_streams = new cudaStream_t[maxStreams];
	for (unsigned short i = 0; i < maxStreams; i++)
		cudaStreamCreate(&meshes_streams[i]);

	for (unsigned short i = 0; i < verts_rows; i++) {
		visible[i] = true;

		for (unsigned short j = 0; j < segNumber && visible[i]; j++) {
			unsigned short stream = j % maxStreams;
			cudaMemcpyAsync(d_camera_location, camera_location, columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			//std::cout << "After copyng camera_location\n";
			cudaMemcpyAsync(d_verts + i, verts[i], columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			//std::cout << "After copying verts[" << i << "]\n";
			
			cudaMemcpyAsync(d_meshes + j * segSize, meshes + j * segSize, segSize * columns * sizeof(unsigned short), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V1 + j * segSize, V1 + j * segSize, segSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V2 + j * segSize, V2 + j * segSize, segSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			cudaMemcpyAsync(d_V3 + j * segSize, V3 + j * segSize, segSize * columns * sizeof(double), cudaMemcpyHostToDevice, meshes_streams[stream]);
			//std::cout << "After copying all, j = " << j << "\n";

			kernel_fastRayTriangleIntersection << <gridDim, blockDim, 0, meshes_streams[stream] >> >
				(d_camera_location, d_verts + i,
					d_V1 + j * segSize, d_V2 + j * segSize, d_V3 + j * segSize,
					segSize,
					BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
					d_flag + j, d_t + j, d_u + j, d_v + j
					);

			//std::cout << "After kernel, j = " << j << "\n";

			cudaMemcpyAsync(flag + j * segSize, d_flag + j * segSize, segSize * sizeof(bool), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			//std::cout << "After flag, j = " << j << "\n";
			cudaMemcpyAsync(t + j * segSize, d_t + j * segSize, segSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			//std::cout << "After t, j = " << j << "\n";
			cudaMemcpyAsync(u + j * segSize, d_u + j * segSize, segSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			//std::cout << "After u, j = " << j << "\n";
			cudaMemcpyAsync(v + j * segSize, d_v + j * segSize, segSize * sizeof(double), cudaMemcpyDeviceToHost, meshes_streams[stream]);
			//std::cout << "After v, j = " << j << "\n";
			
			for (unsigned short k = j * segSize; k < j * segSize + segSize; k++)
				if (flag[k]) {
					visible[i] = false;
					break;
				}
		}
	}

	std::cout << "After kernel\n";

	//Free memory
	cudaFree(d_camera_location);

	std::cout << "After freeing d_camera_location\n";

	/*delete[] d_camera_location;
	delete[] d_verts;
	delete[] d_V1;
	delete[] d_V2;
	delete[] d_V3;
	delete[] d_flag;
	delete[] d_t;
	delete[] d_u;
	delete[] d_v;
	delete[] d_visible;
	*/
	std::cout << "After freeing all\n";

	return visible;
}
