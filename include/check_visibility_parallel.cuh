#ifndef CHECK_VISIBILITY_PARALLEL_H
#define CHECK_VISIBILITY_PARALLEL_H

#include "fastRayTriangleIntersection_parallel.cuh"

//#include "lib/CheckError.cuh" //TODO FIX: problem with multiple definitions of CheckError, SAFE_CALL, etc.

#define COLUMNS_SIZE 3

#define BLOCK_ROWS_SIZE 32

#define V_BLOCK_SIZE (BLOCK_ROWS_SIZE * COLUMNS_SIZE)


void check_visibility_parallel_code(
	double camera_location[COLUMNS_SIZE],
	double **verts, unsigned short verts_rows,
	double** V1, double** V2, double** V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

void check_visibility_parallel_code_streams(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double** V1, double** V2, double** V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

#endif