#ifndef CHECK_VISIBILITY_SEQUENTIAL_H
#define CHECK_VISIBILITY_SEQUENTIAL_H

#include "fastRayTriangleIntersection_sequential.h"

bool* check_visibility_sequential_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible);

#endif