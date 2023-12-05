#include "../include/fastRayTriangleIntersection.h"
#include <iostream>

#include <cuda_runtime.h>

__device__ void cross(double* a, double* b, double* result) {
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = a[2] * b[0] - a[0] * b[2];
	result[2] = a[0] * b[1] - a[1] * b[0];
}

__global__ void kernel_fastRayTriangleIntersection(double** orig, double** dir, double*** V1, double*** V2, double*** V3,
                                                   unsigned short rows, unsigned short columns, Border border, LineType lineType,
                                                   PlaneType planeType, bool fullReturn, bool** intersect,
                                                   double** t, double** u, double** v) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < rows) {
		double eps = 1e-5, zero;
		
		switch (border) {
			case Normal:
				zero = 0.0;
				break;
			case Inclusive:
				zero = eps;
				break;
			case Exclusive:
				zero = -eps;
				break;
			default:
				// Handle error
				return;
		}
		
		double edge1[3], edge2[3], tvec[3], pvec[3], det = 0;
		
		for (int j = 0; j < columns; j++) {
			edge1[j] = V2[i][j] - V1[i][j];
			edge2[j] = V3[i][j] - V1[i][j];
			tvec[j] = orig[i][j] - V1[i][j];
		}
		
		pvec[0] = dir[i][1] * edge2[2] - dir[i][2] * edge2[1];
		pvec[1] = dir[i][2] * edge2[0] - dir[i][0] * edge2[2];
		pvec[2] = dir[i][0] * edge2[1] - dir[i][1] * edge2[0];
		
		for (int j = 0; j < columns; j++)
			det += edge1[j] * pvec[j];
		
		if (planeType == TwoSided)
			intersect[i] = abs(det) > eps;
		else if (planeType == OneSided)
			intersect[i] = det > eps;
		else {
			// Handle error
			return;
		}
		
		if (!intersect[i])
			u[i] = NAN;
		else {
			u[i] = 0;
			for (int j = 0; j < columns; j++)
				u[i] += tvec[j] * pvec[j];
			
			u[i] /= det;
		}
		
		if (fullReturn) {
			double qvec[3];
			if (!intersect[i])
				v[i] = NAN, t[i] = NAN;
			else {
				qvec[0] = tvec[1] * edge1[2] - tvec[2] * edge1[1];
				qvec[1] = tvec[2] * edge1[0] - tvec[0] * edge1[2];
				qvec[2] = tvec[0] * edge1[1] - tvec[1] * edge1[0];
				
				v[i] = t[i] = 0;
				for (int j = 0; j < columns; j++) {
					v[i] += dir[i][j] * qvec[j];
					t[i] += edge2[j] * qvec[j];
				}
				
				v[i] /= det;
				t[i] /= det;
				
				intersect[i] = u[i] >= -zero && v[i] >= -zero && u[i] + v[i] <= 1.0 + zero;
			}
		} else {
			double qvec[3];
			if (!intersect[i])
				v[i] = NAN;
			else {
				qvec[0] = tvec[1] * edge1[2] - tvec[2] * edge1[1];
				qvec[1] = tvec[2] * edge1[0] - tvec[0] * edge1[2];
				qvec[2] = tvec[0] * edge1[1] - tvec[1] * edge1[0];
				
				v[i] = 0;
				for (int j = 0; j < columns; j++)
					v[i] += dir[i][j] * qvec[j];
				v[i] /= det;
				
				if (lineType == Line)
					t[i] = NAN;
				else {
					t[i] = 0;
					for (int j = 0; j < columns; j++)
						t[i] += edge2[j] * qvec[j];
					t[i] /= det;
				}
				
				intersect[i] = v[i] >= -zero && u[i] + v[i] <= 1.0 + zero;
			}
		}
	}
}