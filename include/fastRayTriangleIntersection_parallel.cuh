#ifndef FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_H
#define FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_H

#include <iostream>

#define BORDER_NORMAL 0
#define BORDER_INCLUSIVE 1
#define BORDER_EXCLUSIVE 2

#define LINE_TYPE_RAY 0
#define LINE_TYPE_LINE 1
#define LINE_TYPE_SEGMENT 2

#define PLANE_TYPE_TWOSIDED 0
#define PLANE_TYPE_ONESIDED 1


#define ROW_SIZE 32
#define COL_SIZE 3

__global__ void kernel_fastRayTriangleIntersection(
		double orig[COL_SIZE], double dir[COL_SIZE],
		double* V1[COL_SIZE], double* V2[COL_SIZE], double* V3[COL_SIZE],
		unsigned short rows,
		unsigned short border, unsigned short lineType, unsigned short planeType,
		bool fullReturn,
		bool* intersect, double* t, double* u, double* v);//Returning values

#endif
