#ifndef CHECK_VISIBILITY_PARALLEL_H
#define CHECK_VISIBILITY_PARALLEL_H

#include "fastRayTriangleIntersection_parallel.cuh"

#define COLUMNS_SIZE 3

void check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double **verts, unsigned short verts_rows,
	double** V1, double** V2, double** V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

#endif