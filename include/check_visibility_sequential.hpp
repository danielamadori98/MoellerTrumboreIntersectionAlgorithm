#ifndef CHECK_VISIBILITY_SEQUENTIAL_HPP
#define CHECK_VISIBILITY_SEQUENTIAL_HPP

#include "fastRayTriangleIntersection_sequential.hpp"

void check_visibility_sequential_code(
	double camera_location[COLUMNS_SIZE],
	double* verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible); // Output variables

#endif