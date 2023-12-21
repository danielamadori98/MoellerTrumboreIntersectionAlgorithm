#include "../include/fastRayTriangleIntersection_parallel.cuh"

//TODO update this function with the new changes of the .._with_check function
__global__ void fastRayTriangleIntersection_parallel(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double* V1, double* V2, double* V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
	bool* intersect, double* t, double* u, double* v) 
{
	int row = blockIdx.y * blockDim.y + threadIdx.y,
		col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows && col < 1) {
		double eps = 1e-5, zero;
		
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
				
		double edge1[ROWS_SIZE * COLUMNS_SIZE], edge2[ROWS_SIZE * COLUMNS_SIZE],
			tvec[ROWS_SIZE * COLUMNS_SIZE], pvec[ROWS_SIZE * COLUMNS_SIZE],
			det[ROWS_SIZE] = {0};
		
		for (col = 0; col < COLUMNS_SIZE; col++) {
			edge1[row * COLUMNS_SIZE + col] = V2[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			edge2[row * COLUMNS_SIZE + col] = V3[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			tvec[row * COLUMNS_SIZE + col]	= orig[col] - V1[row * COLUMNS_SIZE + col];
		}

		pvec[row * COLUMNS_SIZE]	 = dir[1] * edge2[row * COLUMNS_SIZE + 2] - dir[2] * edge2[row * COLUMNS_SIZE + 1];
		pvec[row * COLUMNS_SIZE + 1] = dir[2] * edge2[row * COLUMNS_SIZE] - dir[0] * edge2[row * COLUMNS_SIZE + 2];
		pvec[row * COLUMNS_SIZE + 2] = dir[0] * edge2[row * COLUMNS_SIZE + 1] - dir[1] * edge2[row * COLUMNS_SIZE];

		for(col = 0; col < COLUMNS_SIZE; col++)
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
			u[row] = 0;
			for (col = 0; col < COLUMNS_SIZE; col++)
				u[row] += tvec[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];

			u[row] /= det[row];
		}

		double qvec[COLUMNS_SIZE];

		if (fullReturn) {
			if (!intersect[row])
				v[row] = NAN, t[row] = NAN;
			else {
				qvec[0] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
				qvec[1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE] - tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 2];
				qvec[2] = tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE];
				
				v[row] = t[row] = 0;
				for(col = 0; col < COLUMNS_SIZE; col++){
					v[row] += dir[col] * qvec[col];
					t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[col];
				}

				v[row] /= det[row];
				t[row] /= det[row];
				
				intersect[row] = u[row] >= -zero && v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
			}

		} else {
			intersect[row] = intersect[row] && u[row] >= -zero && u[row] <= 1 + zero;

			if (!intersect[row])
				v[row] = NAN;
			else {
				qvec[0] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
				qvec[1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE] - tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 2];
				qvec[2] = tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE];

				v[row] = 0;
				for(col = 0; col < COLUMNS_SIZE; col++)
					v[row] += dir[col] * qvec[col];

				v[row] /= det[row];
				
				if (lineType == LINE_TYPE_LINE)
					t[row] = NAN;
				else {
					t[row] = 0;
					for(col = 0; col < COLUMNS_SIZE; col++)
						t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[col];
	
					t[row] /= det[row];
				}
				
				intersect[row] = v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
			}
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
		}
	}
}

__global__ void fastRayTriangleIntersection_parallel_with_check(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType, const bool fullReturn,
	bool* intersect, double* t, double* u, double* v,
	unsigned short* visible)
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

		double edge1[ROWS_SIZE * COLUMNS_SIZE], edge2[ROWS_SIZE * COLUMNS_SIZE],
			tvec[ROWS_SIZE * COLUMNS_SIZE], pvec[ROWS_SIZE * COLUMNS_SIZE],
			det[ROWS_SIZE];

		for (col = 0; col < COLUMNS_SIZE; col++) {
			edge1[row * COLUMNS_SIZE + col] = V2[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			edge2[row * COLUMNS_SIZE + col] = V3[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
			tvec[row * COLUMNS_SIZE + col] = orig[col] - V1[row * COLUMNS_SIZE + col];
		}
		
		pvec[row * COLUMNS_SIZE]	 = dir[1] * edge2[row * COLUMNS_SIZE + 2] - dir[2] * edge2[row * COLUMNS_SIZE + 1];
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

		double qvec[COLUMNS_SIZE];

		if (fullReturn) {
			if (!intersect[row])
				v[row] = NAN, t[row] = NAN;
			else {
				qvec[0] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
				qvec[1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE] - tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 2];
				qvec[2] = tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE];
				
				v[row] = dir[0] * qvec[0];
				t[row] = edge2[row * COLUMNS_SIZE] * qvec[0];;
				for (col = 1; col < COLUMNS_SIZE; col++) {
					v[row] += dir[col] * qvec[col];
					t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[col];
				}

				v[row] /= det[row];
				t[row] /= det[row];

				intersect[row] = u[row] >= -zero && v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
			}

		}
		else {
			intersect[row] = intersect[row] && u[row] >= -zero && u[row] <= 1 + zero;

			if (!intersect[row])
				v[row] = NAN;
			else {
				qvec[0] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
				qvec[1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE] - tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 2];
				qvec[2] = tvec[row * COLUMNS_SIZE] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE];
				
				v[row] = dir[0] * qvec[0];
				for (col = 1; col < COLUMNS_SIZE; col++)
					v[row] += dir[col] * qvec[col];

				v[row] /= det[row];

				if (lineType == LINE_TYPE_LINE)
					t[row] = NAN;
				else {
					t[row] = edge2[row * COLUMNS_SIZE] * qvec[0];
					for (col = 1; col < COLUMNS_SIZE; col++)
						t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[col];

					t[row] /= det[row];
				}

				intersect[row] = v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
			}
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
			(*visible)++;		
	}
}