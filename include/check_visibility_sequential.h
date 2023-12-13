#ifndef CHECK_VISIBILITY_SEQUENTIAL_H
#define CHECK_VISIBILITY_SEQUENTIAL_H

#include "fastRayTriangleIntersection_sequential.h"

#define COLUMNS_SIZE 3

bool* check_visibility_sequential_code(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double** V1, double** V2, double** V3, unsigned short V_rows,
	bool* flag,
	double* t, double* u, double* v, bool* visible);

#endif