#ifndef FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_HPP
#define FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_HPP

#include "fastRayTriangleIntersection.hpp"
#include <iostream>//TODO : remove

#define ROWS_SIZE 32

__global__ void fastRayTriangleIntersection_parallel(
		double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
		double* V1, double* V2, double* V3, unsigned short rows,
		unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
		bool* intersect, double* t, double* u, double* v);//Returning values


__global__ void fastRayTriangleIntersection_parallel_with_check(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double* V1, double* V2, double* V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
	bool* intersect, double* t, double* u, double* v,
	unsigned short* visible);

#endif
