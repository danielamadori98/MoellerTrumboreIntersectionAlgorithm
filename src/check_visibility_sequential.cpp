#include "../include/check_visibility_sequential.hpp"

void check_visibility_sequential_code(
	double camera_location[COLUMNS_SIZE],
	double** verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible) // Output variables
{

	for (unsigned short i = 0; i < verts_rows; i++) {
		fastRayTriangleIntersection_sequential(
			camera_location, verts[i],
			V1, V2, V3, V_rows,
			BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWOSIDED, false,
			flag, t, u, v);

		visible[i] = true;
		for (unsigned short j = 0; j < V_rows; j++)
			if (flag[j]) {
				visible[i] = false;
				break;
			}
	}
}
