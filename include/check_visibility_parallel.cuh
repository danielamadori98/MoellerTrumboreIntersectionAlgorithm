#ifndef CHECK_VISIBILITY_PARALLEL_CUH
#define CHECK_VISIBILITY_PARALLEL_CUH

#include "fastRayTriangleIntersection_parallel.cuh"
#include "lib/Timer.cuh"
#include <iostream>
//#include "lib/CheckError.cuh" //TODO FIX: problem with multiple definitions of CheckError, SAFE_CALL, etc.

#define BLOCK_ROWS_SIZE 32

#define V_BLOCK_SIZE (BLOCK_ROWS_SIZE * COLUMNS_SIZE)

double check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double **verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

void check_visibility_parallel_code_streams(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

#endif