#ifndef FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_HPP
#define FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_HPP

#include "fastRayTriangleIntersection.hpp"
#include <iostream>

#define ROWS_SIZE 32

__global__ void fastRayTriangleIntersection_parallel(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double* V1, double* V2, double* V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
	bool* intersect, double* t, double* u, double* v);//Returning values


__global__ void fastRayTriangleIntersection_parallel_with_check(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType, const bool fullReturn,
	bool* intersect, double* t, double* u, double* v,
	unsigned short* visible);//Returning values

#endif
