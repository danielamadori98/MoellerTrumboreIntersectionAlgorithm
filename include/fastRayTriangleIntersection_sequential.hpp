#ifndef FAST_RAY_TRIANGLE_INTERSECTION_SEQUENTIAL_HPP
#define FAST_RAY_TRIANGLE_INTERSECTION_SEQUENTIAL_HPP

#include "fastRayTriangleIntersection.hpp"
#include <iostream>

/* Space cost
*
* First Part :
* pvec		= COLUMNS_SIZE * rows * sizeof(double)
*
* Second Part:
* qvec		= COLUMNS_SIZE * sizeof(double)
*
* Both:
* edge1		= COLUMNS_SIZE * rows * sizeof(double)
* edge2		= COLUMNS_SIZE * rows * sizeof(double)
* tvec		= COLUMNS_SIZE * rows * sizeof(double)
* det		= rows * sizeof(double)
* TOT(COLUMNS_SIZE, rows) = sizeof(double) * rows * (3 * COLUMNS_SIZE + 1)
*
* Total First Part(COLUMNS_SIZE, rows)  = rows * [4 COLUMNS_SIZE + 1 * sizeof(double)]
* Total Second Part(COLUMNS_SIZE, rows) = {rows * [3 COLUMNS_SIZE + 1] + COLUMNS_SIZE} * sizeof(double)
*
*/

void fastRayTriangleIntersection_sequential(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double* V1, double* V2, double* V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType,
	bool* intersect, double* t, double* u, double* v); // Output variables



/* Space cost : (All double except intersect)
*
* First Part :
* pvec		= COLUMNS_SIZE * rows * sizeof(double)
*
* Second Part:
* t			= rows * sizeof(double)
* v			= rows * sizeof(double)
* qvec		= COLUMNS_SIZE * sizeof(double)
* TOT(COLUMNS_SIZE, rows) = (COLUMNS_SIZE + 2 * rows) * sizeof(double)
*
* Both:
* edge1		= COLUMNS_SIZE * rows * sizeof(double)
* edge2		= COLUMNS_SIZE * rows * sizeof(double)
* tvec		= COLUMNS_SIZE * rows * sizeof(double)
* det		= rows * sizeof(double)
* u			= rows * sizeof(double)
* TOT(COLUMNS_SIZE, rows) = sizeof(double) * rows * (3 * COLUMNS_SIZE + 2)
*
*
* Total First Part(COLUMNS_SIZE, rows)  = rows * [4(COLUMNS_SIZE + 2) * sizeof(double)]
* Total Second Part(COLUMNS_SIZE, rows) = {rows * [3COLUMNS_SIZE + 4] + 2 rows} * sizeof(double)
*
*/

void fastRayTriangleIntersection_sequential(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double* V1, double* V2, double* V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType,
	bool* intersect); // Output variables

#endif
