#ifndef CHECK_VISIBILITY_PARALLEL_H
#define CHECK_VISIBILITY_PARALLEL_H

#include <iostream>
#include "fastRayTriangleIntersection_parallel.cuh"

bool* check_visibility_parallel_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible);

#endif