#ifndef CHECK_VISIBILITY_PARALLEL_CUH
#define CHECK_VISIBILITY_PARALLEL_CUH

#include "fastRayTriangleIntersection_parallel.cuh"
#include "lib/Timer.cuh"
#include <iostream>

double check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double* verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

double check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double* verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* visible); // Output variable

double check_visibility_parallel_code_dyn(
	double camera_location[COLUMNS_SIZE],
	double* verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* visible); // Output variable

#endif