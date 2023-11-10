#include "../include/check_visibility.h"
#include "../include/fastRayTriangleIntersection.h"


bool* check_visibility(double** verts, unsigned short verts_rows, unsigned short** meshes, unsigned short meshes_rows, unsigned short columns) {
	double camera_location[3] = { 0, 0, 0 };

	double** V1 = new double* [meshes_rows];
	double** V2 = new double* [meshes_rows];
	double** V3 = new double* [meshes_rows];
	for (unsigned short i = 0; i < meshes_rows; i++) {
		V1[i] = new double[columns];
		V2[i] = new double[columns];
		V3[i] = new double[columns];

		for (unsigned short j = 0; j < columns; j++) {
			V1[i][j] = verts[meshes[i][0]][j];
			V2[i][j] = verts[meshes[i][1]][j];
			V3[i][j] = verts[meshes[i][2]][j];
		}
	}

	/*
	std::cout << "First 5 V1:" << std::endl;
	for (unsigned short i = 0; i < 5; i++)
		std::cout << V1[i][0] << ", " << V1[i][1] << ", " << V1[i][2] << std::endl;

	std::cout << "First 5 V2:" << std::endl;
	for (unsigned short i = 0; i < 5; i++)
		std::cout << V2[i][0] << ", " << V2[i][1] << ", " << V2[i][2] << std::endl;

	std::cout << "First 5 V3:" << std::endl;
	for (unsigned short i = 0; i < 5; i++)
		std::cout << V3[i][0] << ", " << V3[i][1] << ", " << V3[i][2] << std::endl;
	*/

	bool* flag = new bool[meshes_rows], * visible = new bool[verts_rows];

	//the t in the matlab code was be replaced by the v to mantain the same name used in the fastRayTriangleIntersection function
	double* t = new double[meshes_rows], * u = new double[meshes_rows], * v = new double[meshes_rows];

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

	for (unsigned short i = 0; i < meshes_rows; i++) {
		delete[] V1[i];
		delete[] V2[i];
		delete[] V3[i];
	}
	delete[] V1;
	delete[] V2;
	delete[] V3;

	delete[] t;
	delete[] u;
	delete[] v;
	delete[] flag;

	return visible;
}
