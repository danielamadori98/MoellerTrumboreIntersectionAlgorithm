#ifndef FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_HPP
#define FAST_RAY_TRIANGLE_INTERSECTION_PARALLEL_HPP

#include "fastRayTriangleIntersection.hpp"
#include <iostream>

/* Shared Memory of actual GPU :
* NVIDIA GeForce RTX 2070 SUPER: 48 KB = 49.152 bytes
*/

#define BLOCK_ROWS_SIZE 320

// x: SHARED_MEM_SIZE / MAX_SPACE_COST
// y: multiple used to round down x to the nearest multiple of y
#define MAX_BLOCK_ROWS_SIZE(x, y) ((x) / (y) * (y))


/* Space cost
*
* First Part :
* pvec		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
*
* Second Part:
* qvec		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* 
* Both:
* edge1		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* edge2		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* tvec		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* det		= BLOCK_ROWS_SIZE * sizeof(double)
* TOT(COLUMNS_SIZE, BLOCK_ROWS_SIZE) = sizeof(double) * BLOCK_ROWS_SIZE * (3 * COLUMNS_SIZE + 1)
*
* Total First Part(COLUMNS_SIZE, BLOCK_ROWS_SIZE)  = BLOCK_ROWS_SIZE * [4(COLUMNS_SIZE + 1) * sizeof(double) + sizeof(bool)]
* Total Second Part(COLUMNS_SIZE, BLOCK_ROWS_SIZE) = BLOCK_ROWS_SIZE * [4(COLUMNS_SIZE + 1)	* sizeof(double) + sizeof(bool)]
*
*/

#define SPACE_COST_FIRST_PART_FULL_RETURN  (4 * (COLUMNS_SIZE + 1) * sizeof(double) + sizeof(bool))
#define SPACE_COST_SECOND_PART_FULL_RETURN (4 * (COLUMNS_SIZE + 1) * sizeof(double) + sizeof(bool))

#define MAX_SPACE_COST_FULL_RETURN (SPACE_COST_FIRST_PART_FULL_RETURN > SPACE_COST_SECOND_PART_FULL_RETURN ? SPACE_COST_FIRST_PART_FULL_RETURN : SPACE_COST_SECOND_PART_FULL_RETURN)


__global__ void fastRayTriangleIntersection_parallel(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType,
	bool* intersect, double* t, double* u, double* v, unsigned int* visible); // Output variables


/* Space cost : (All double except intersect)
*
* First Part :
* pvec		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* 
* Second Part:
* t			= BLOCK_ROWS_SIZE * sizeof(double)
* v			= BLOCK_ROWS_SIZE * sizeof(double)
* qvec		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* TOT(COLUMNS_SIZE, BLOCK_ROWS_SIZE) = (COLUMNS_SIZE + 2) * BLOCK_ROWS_SIZE * sizeof(double)
* 
* Both:
* edge1		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* edge2		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* tvec		= COLUMNS_SIZE * BLOCK_ROWS_SIZE * sizeof(double)
* det		= BLOCK_ROWS_SIZE * sizeof(double)
* u			= BLOCK_ROWS_SIZE * sizeof(double)
* intersect = BLOCK_ROWS_SIZE * sizeof(bool)
* TOT(COLUMNS_SIZE, BLOCK_ROWS_SIZE) = sizeof(double) * BLOCK_ROWS_SIZE * (3 * COLUMNS_SIZE + 2) + sizeof(bool) * BLOCK_ROWS_SIZE
*
* 
* Total First Part(COLUMNS_SIZE, BLOCK_ROWS_SIZE)  = BLOCK_ROWS_SIZE * [4(COLUMNS_SIZE + 1/2) * sizeof(double) + sizeof(bool)]
* Total Second Part(COLUMNS_SIZE, BLOCK_ROWS_SIZE) = BLOCK_ROWS_SIZE * [4(COLUMNS_SIZE + 1)	  * sizeof(double) + sizeof(bool)]
* 
*/

#define SPACE_COST_FIRST_PART  (4 * (COLUMNS_SIZE + 1/2) * sizeof(double) + sizeof(bool))
#define SPACE_COST_SECOND_PART (4 * (COLUMNS_SIZE + 1)	 * sizeof(double) + sizeof(bool))

#define MAX_SPACE_COST (SPACE_COST_FIRST_PART > SPACE_COST_SECOND_PART ? SPACE_COST_FIRST_PART : SPACE_COST_SECOND_PART)

__global__ void fastRayTriangleIntersection_parallel(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType,
	unsigned int* visible); // Output variable

__global__ void fastRayTriangleIntersection_parallel_dyn(
	const double* orig, const double* dir,
	const double* V1, const double* V2, const double* V3, const unsigned short rows,
	const unsigned short max_block_rows_size,
	const unsigned short border, const unsigned short lineType, const unsigned short planeType,
	unsigned int* visible); // Output variable


#endif
