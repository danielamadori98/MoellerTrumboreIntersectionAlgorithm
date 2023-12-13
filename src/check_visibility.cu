#include "../include/check_visibility.h"

void check(bool* visible, bool** gt, unsigned short verts_rows) {
	bool error = false;
	for(unsigned short i = 0; i < verts_rows; i++) {
		if (visible[i] != gt[i][0]) {
			error = true;
			std::cerr << "Error in vertex " << i << std::endl;
			std::cerr << "Expected: " << gt[i][0] << " - Obtained: " << visible[i] << std::endl;
			return; //Remove this line to print all errors
		}
	}
	if(!error)
		std::cout << "All vertices are correctly classified" << std::endl;
}

bool* check_visibility(
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	bool** gt){

	double camera_location[COLUMNS_SIZE] = { 0, 0, 0 };

	double** V1 = new double* [meshes_rows];
	double** V2 = new double* [meshes_rows];
	double** V3 = new double* [meshes_rows];

	for (unsigned short i = 0; i < meshes_rows; i++) {
		V1[i] = new double[COLUMNS_SIZE];
		V2[i] = new double[COLUMNS_SIZE];
		V3[i] = new double[COLUMNS_SIZE];

		for (unsigned short j = 0; j < COLUMNS_SIZE; j++) {
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

	//Output variables

	bool* flag = new bool[meshes_rows], *visible = new bool[verts_rows];
	//the t in the matlab code was be replaced by the v to mantain the same name used in the fastRayTriangleIntersection function
	double* t = new double[meshes_rows], *u = new double[meshes_rows], *v = new double[meshes_rows];


	Timer<DEVICE> TM_device;
	Timer<HOST>   TM_host;

	TM_host.start();
	//check_visibility_sequential_code(camera_location, verts, verts_rows, V1, V2, V3, meshes_rows, flag, t, u, v, visible);
	TM_host.stop();
	TM_host.print("MatrixTranspose host:   ");
	//check(visible, gt, verts_rows);

	TM_device.start();
	check_visibility_parallel_code(camera_location, verts, verts_rows, V1, V2, V3, meshes_rows, flag, t, u, v, visible);
	TM_device.stop();
	CHECK_CUDA_ERROR
		TM_device.print("MatrixTranspose device: ");

	std::cout << std::setprecision(1) << "Speedup: " << TM_host.duration() / TM_device.duration() << "x\n\n";
	check(visible, gt, verts_rows);

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
