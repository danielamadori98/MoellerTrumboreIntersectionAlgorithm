#include "../include/check_visibility_sequential.h"

bool* check_visibility_sequential_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible)
{

	for (unsigned short i = 0; i < verts_rows; i++) {
		fastRayTriangleIntersection(camera_location, verts[i], V1, V2, V3, meshes_rows, columns, Exclusive, Segment, TwoSided, false, flag, t, u, v);

		visible[i] = true;
		for (unsigned short j = 0; j < meshes_rows; j++)
			if (flag[j]) {
				visible[i] = false;
				break;
			}

		/*
		std::cout << "First 5 v:" << std::endl;
		for (unsigned short i = 0; i < 5; i++)
			//std::cout << v[i] << std::endl;
		*/
	}
	return visible;
}
