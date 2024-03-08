#include "../include/check_visibility_sequential.hpp"

void check_visibility_sequential_code(
	double camera_location[COLUMNS_SIZE],
	double* verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* flag, double* t, double* u, double* v, bool* visible) // Output variables
{

	for (unsigned short row = 0; row < verts_rows; row++) {
		fastRayTriangleIntersection_sequential(
			camera_location, verts + row * COLUMNS_SIZE,
			V1, V2, V3, V_rows,
			BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWO_SIDED,
			flag, t, u, v);

		visible[row] = true;
		for (unsigned short v_row = 0; v_row < V_rows; v_row++)
			if (flag[v_row]) {
				visible[row] = false;
				break;
			}
	}
}


void check_visibility_sequential_code(
	double camera_location[COLUMNS_SIZE],
	double* verts, unsigned short verts_rows,
	double* V1, double* V2, double* V3, unsigned short V_rows,
	bool* visible) // Output variable
{

	bool* flag = new bool[V_rows];
	for (unsigned short row = 0; row < verts_rows; row++) {
		fastRayTriangleIntersection_sequential(
			camera_location, verts + row * COLUMNS_SIZE,
			V1, V2, V3, V_rows,
			BORDER_EXCLUSIVE, LINE_TYPE_SEGMENT, PLANE_TYPE_TWO_SIDED,
			flag);

		visible[row] = true;
		for (unsigned short v_row = 0; v_row < V_rows; v_row++)
			if (flag[v_row]) {
				visible[row] = false;
				break;
			}
	}

	delete[] flag;
}