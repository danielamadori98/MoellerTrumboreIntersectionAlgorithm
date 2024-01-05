#include "../include/fastRayTriangleIntersection_parallel.cuh"

__global__ void fastRayTriangleIntersection_parallel_full_return(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType,
	bool* intersect, double* t, double* u, double* v,
	unsigned int* visible)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		col = blockIdx.x * blockDim.x + threadIdx.x;

	if (*visible == 0 && row < rows && col < 1) {
		const double eps = 1e-5;
		double zero;

		switch (border) {
		case BORDER_NORMAL:
			zero = 0.0;
			break;
		case BORDER_INCLUSIVE:
			zero = eps;
		case BORDER_EXCLUSIVE:
			zero = -eps;
			break;
		default:
			printf("Error: border must be either BORDER_NORMAL, BORDER_INCLUSIVE or BORDER_EXCLUSIVE\n");
			return;
		}

		__shared__ double edge1[BLOCK_ROWS_SIZE * COLUMNS_SIZE], edge2[BLOCK_ROWS_SIZE * COLUMNS_SIZE],
			tvec[BLOCK_ROWS_SIZE * COLUMNS_SIZE],
			det[BLOCK_ROWS_SIZE];

		for (col = 0; col < COLUMNS_SIZE; col++) {
			edge1[row * COLUMNS_SIZE + col] = V2[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			edge2[row * COLUMNS_SIZE + col] = V3[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			tvec[row * COLUMNS_SIZE + col] = orig[col] - V1[row * COLUMNS_SIZE + col];
		}

		{// Scope: First Parst: pvec is used only in this block, limiting its scope
			__shared__ double pvec[BLOCK_ROWS_SIZE * COLUMNS_SIZE];

			pvec[row * COLUMNS_SIZE] = dir[1] * edge2[row * COLUMNS_SIZE + 2] - dir[2] * edge2[row * COLUMNS_SIZE + 1];
			pvec[row * COLUMNS_SIZE + 1] = dir[2] * edge2[row * COLUMNS_SIZE] - dir[0] * edge2[row * COLUMNS_SIZE + 2];
			pvec[row * COLUMNS_SIZE + 2] = dir[0] * edge2[row * COLUMNS_SIZE + 1] - dir[1] * edge2[row * COLUMNS_SIZE];

			det[row] = edge1[row * COLUMNS_SIZE] * pvec[row * COLUMNS_SIZE];
			for (col = 1; col < COLUMNS_SIZE; col++)
				det[row] += edge1[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];


			if (planeType == PLANE_TYPE_TWOSIDED)
				intersect[row] = abs(det[row]) > eps;
			else if (planeType == PLANE_TYPE_ONESIDED)
				intersect[row] = det[row] > eps;
			else {
				printf("Error: planeType must be either PLANE_TYPE_TWOSIDED or PLANE_TYPE_ONESIDED\n");
				return;
			}

			if (!intersect[row])
				u[row] = NAN;
			else {
				u[row] = tvec[row * COLUMNS_SIZE] * pvec[row * COLUMNS_SIZE];
				for (col = 1; col < COLUMNS_SIZE; col++)
					u[row] += tvec[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];

				u[row] /= det[row];
			}
		}//pvec is not used anymore

		// Second Part: qvec is used only in this block, limiting its scope
		__shared__ double qvec[BLOCK_ROWS_SIZE * COLUMNS_SIZE];
		
		if (!intersect[row])
			v[row] = NAN, t[row] = NAN;
		else {
			qvec[row] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
			qvec[row + 1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE] - tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 2];
			qvec[row + 2] = tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE];

			v[row] = dir[0] * qvec[row];
			t[row] = edge2[row * COLUMNS_SIZE] * qvec[row];
			for (col = 1; col < COLUMNS_SIZE; col++) {
				v[row] += dir[col] * qvec[row + col];
				t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[row + col];
			}

			v[row] /= det[row];
			t[row] /= det[row];

			intersect[row] = u[row] >= -zero && v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
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
		default:
			printf("Error: lineType must be either LINE_TYPE_LINE, LINE_TYPE_RAY or LINE_TYPE_SEGMENT\n");
			return;
		}

		//__syncthreads();

		if (intersect[row])
			atomicAdd(visible, 1);
			//(*visible)++;		
	}
}


__global__ void fastRayTriangleIntersection_parallel(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType,
	unsigned int* visible)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		col = blockIdx.x * blockDim.x + threadIdx.x;

	if (*visible == 0 && row < rows && col < 1) {
		const double eps = 1e-5;
		double zero;

		switch (border) {
		case BORDER_NORMAL:
			zero = 0.0;
			break;
		case BORDER_INCLUSIVE:
			zero = eps;
		case BORDER_EXCLUSIVE:
			zero = -eps;
			break;
		default:
			printf("Error: border must be either BORDER_NORMAL, BORDER_INCLUSIVE or BORDER_EXCLUSIVE\n");
			return;
		}

		__shared__ double edge1[BLOCK_ROWS_SIZE * COLUMNS_SIZE], edge2[BLOCK_ROWS_SIZE * COLUMNS_SIZE],
			tvec[BLOCK_ROWS_SIZE * COLUMNS_SIZE],
			det[BLOCK_ROWS_SIZE]; // Scope: All Parts

		for (col = 0; col < COLUMNS_SIZE; col++) {
			edge1[row * COLUMNS_SIZE + col] = V2[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			edge2[row * COLUMNS_SIZE + col] = V3[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			tvec[row * COLUMNS_SIZE + col] = orig[col] - V1[row * COLUMNS_SIZE + col];
		}

		__shared__ bool intersect[BLOCK_ROWS_SIZE]; // Scope: All Parts
		__shared__ double u[BLOCK_ROWS_SIZE]; // Scope: All Parts

		{// Scope: First Parst: pvec is used only in this block, limiting its scope
			__shared__ double pvec[BLOCK_ROWS_SIZE * COLUMNS_SIZE];

			pvec[row * COLUMNS_SIZE] = dir[1] * edge2[row * COLUMNS_SIZE + 2] - dir[2] * edge2[row * COLUMNS_SIZE + 1];
			pvec[row * COLUMNS_SIZE + 1] = dir[2] * edge2[row * COLUMNS_SIZE] - dir[0] * edge2[row * COLUMNS_SIZE + 2];
			pvec[row * COLUMNS_SIZE + 2] = dir[0] * edge2[row * COLUMNS_SIZE + 1] - dir[1] * edge2[row * COLUMNS_SIZE];

			det[row] = edge1[row * COLUMNS_SIZE] * pvec[row * COLUMNS_SIZE];
			for (col = 1; col < COLUMNS_SIZE; col++)
				det[row] += edge1[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];


			if (planeType == PLANE_TYPE_TWOSIDED)
				intersect[row] = abs(det[row]) > eps;
			else if (planeType == PLANE_TYPE_ONESIDED)
				intersect[row] = det[row] > eps;
			else {
				printf("Error: planeType must be either PLANE_TYPE_TWOSIDED or PLANE_TYPE_ONESIDED\n");
				return;
			}

			
			if (!intersect[row])
				u[row] = NAN;
			else {
				u[row] = tvec[row * COLUMNS_SIZE] * pvec[row * COLUMNS_SIZE];
				for (col = 1; col < COLUMNS_SIZE; col++)
					u[row] += tvec[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];

				u[row] /= det[row];
			}
			
			intersect[row] = intersect[row] && u[row] >= -zero && u[row] <= 1 + zero;

		}//pvec is not used anymore
		
		
		// Second Part: t, v, qvec are used only in this block, limiting their scope
		__shared__ double t[BLOCK_ROWS_SIZE], v[BLOCK_ROWS_SIZE], qvec[BLOCK_ROWS_SIZE * COLUMNS_SIZE];

		if (intersect[row]){
			qvec[row] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
			qvec[row + 1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE] - tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 2];
			qvec[row + 2] = tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE];
		
			v[row] = dir[0] * qvec[row];
			for (col = 1; col < COLUMNS_SIZE; col++)
				v[row] += dir[col] * qvec[row + col];

			v[row] /= det[row];

			if (lineType == LINE_TYPE_LINE)
				t[row] = NAN;
			else {
				t[row] = edge2[row * COLUMNS_SIZE] * qvec[row];
				for (col = 1; col < COLUMNS_SIZE; col++)
					t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[row + col];

				t[row] /= det[row];
			}

			intersect[row] = v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
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
		default:
			printf("Error: lineType must be either LINE_TYPE_LINE, LINE_TYPE_RAY or LINE_TYPE_SEGMENT\n");
			return;
		}

		if (intersect[row])
			atomicAdd(visible, 1);	
	}
}
