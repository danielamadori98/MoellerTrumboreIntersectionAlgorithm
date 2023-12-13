#include "../include/fastRayTriangleIntersection_parallel.cuh"

__device__ void cross(double* a, double* b, double* result) {
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = a[2] * b[0] - a[0] * b[2];
	result[2] = a[0] * b[1] - a[1] * b[0];
}

__global__ void kernel_fastRayTriangleIntersection(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double** V1, double** V2, double** V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
	bool* intersect, double* t, double* u, double* v) 
{
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < rows) {
		double eps = 1e-5, zero;
		
		switch (border) {
			case BORDER_NORMAL:
				zero = 0.0;
				break;
			case BORDER_INCLUSIVE:
				zero = eps;
				break;
			case BORDER_EXCLUSIVE:
				zero = -eps;
				break;
			//default:
				//TODO: Handle error
		}
		
		__shared__ double edge1[ROWS_SIZE][COLUMNS_SIZE], edge2[ROWS_SIZE][COLUMNS_SIZE],
			tvec[ROWS_SIZE][COLUMNS_SIZE], pvec[ROWS_SIZE][COLUMNS_SIZE], det[ROWS_SIZE];
		
		for (unsigned short i = 0; i < COLUMNS_SIZE; i++) {
			edge1[row][i] = V2[row][i] - V1[row][i];
			edge2[row][i] = V3[row][i] - V1[row][i];
			tvec[row][i] = orig[i] - V1[row][i];
		}
		
		pvec[row][0] = dir[1] * edge2[row][2] - dir[2] * edge2[row][1];
		pvec[row][1] = dir[2] * edge2[row][0] - dir[0] * edge2[row][2];
		pvec[row][2] = dir[0] * edge2[row][1] - dir[1] * edge2[row][0];
		
		det[row] = 0;
		for (unsigned short i = 0; i < COLUMNS_SIZE; i++)
			det[row] += edge1[row][i] * pvec[row][i];
		
		if (planeType == PLANE_TYPE_TWOSIDED)
			intersect[row] = abs(det[row]) > eps;
		else if (planeType == PLANE_TYPE_ONESIDED)
			intersect[row] = det[row] > eps;
		else {
			// Handle error
			return;
		}
		
		if (!intersect[row])
			u[row] = NAN;
		else {
			u[row] = 0;
			for (unsigned short i = 0; i < COLUMNS_SIZE; i++)
				u[row] += tvec[row][i] * pvec[row][i];
			
			u[row] /= det[row];
		}
		
		//__syncthreads();

		if (fullReturn) {
			__shared__ double qvec[COLUMNS_SIZE];
			if (!intersect[row])
				v[row] = NAN, t[row] = NAN;
			else {
				qvec[0] = tvec[row][1] * edge1[row][2] - tvec[row][2] * edge1[row][1];
				qvec[1] = tvec[row][2] * edge1[row][0] - tvec[row][0] * edge1[row][2];
				qvec[2] = tvec[row][0] * edge1[row][1] - tvec[row][1] * edge1[row][0];
				
				v[row] = t[row] = 0;
				for (unsigned short i = 0; i < COLUMNS_SIZE; i++){
					v[row] += dir[i] * qvec[i];
					t[row] += edge2[row][i] * qvec[i];
				}
				
				v[row] /= det[row];
				t[row] /= det[row];
				
				intersect[row] = u[row] >= -zero && v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
			}

			//__syncthreads();

		} else {
			__shared__ double qvec[COLUMNS_SIZE];

			intersect[row] = intersect[row] && u[row] >= -zero && u[row] <= 1 + zero;

			if (!intersect[row])
				v[row] = NAN;
			else {
				qvec[0] = tvec[row][1] * edge1[row][2] - tvec[row][2] * edge1[row][1];
				qvec[1] = tvec[row][2] * edge1[row][0] - tvec[row][0] * edge1[row][2];
				qvec[2] = tvec[row][0] * edge1[row][1] - tvec[row][1] * edge1[row][0];
				
				v[row] = 0;
				for (unsigned short i = 0; i < COLUMNS_SIZE; i++)
					v[row] += dir[i] * qvec[i];
				v[row] /= det[row];
				
				if (lineType == LINE_TYPE_LINE)
					t[row] = NAN;
				else {
					t[row] = 0;
					for (unsigned short i = 0; i < COLUMNS_SIZE; i++)
						t[row] += edge2[row][i] * qvec[i];
					t[row] /= det[row];
				}
				
				intersect[row] = v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
			}

			//__syncthreads();
		}
	
		switch (lineType) {
		case LINE_TYPE_LINE:// Nothing to do
			break;
		case LINE_TYPE_RAY:
				intersect[row] = intersect[row] && t[row] >= -zero;
			break;
		case LINE_TYPE_SEGMENT:
				intersect[row] = intersect[row] && t[row] >= -zero && t[row] <= 1.0 + zero;
			break;
		//default:
			//TODO: std::cerr << "LineType parameter must be either 'line', 'ray' or 'segment'\n";
		}
	}
	
}