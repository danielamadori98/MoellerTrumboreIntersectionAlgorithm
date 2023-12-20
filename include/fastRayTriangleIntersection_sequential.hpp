#ifndef FAST_RAY_TRIANGLE_INTERSECTION_SEQUENTIAL_HPP
#define FAST_RAY_TRIANGLE_INTERSECTION_SEQUENTIAL_HPP

#include "fastRayTriangleIntersection.hpp"
#include <iostream>

void fastRayTriangleIntersection_sequential(
		double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
		double* V1, double* V2, double* V3, unsigned short rows,
		unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
		bool* intersect, double* t, double* u, double* v); // Returning values

#endif
