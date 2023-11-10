﻿#include "../include/readCSV.h"
#include "../include/check_visibility.h"


int main() {
	std::string path = "../../../",
		vertsFilename = path + "verts.csv",
		meshesFilename = path + "meshes.csv",
		ground_truth = path + "visible.csv";

	unsigned short verts_rows = 0, meshes_rows = 0, ground_truth_rows = 0, columns = 3;

	double** verts = read_CSV_double(vertsFilename, columns, ',', verts_rows);

	unsigned short** meshes = read_CSV_unsigned_short(meshesFilename, columns, ',', meshes_rows);

	bool** gt = read_CSV_bool(ground_truth, 1, ',', ground_truth_rows);

	if(verts == nullptr || meshes == nullptr || gt == nullptr)
		return -1;

	if (verts_rows == 0 || meshes_rows == 0 || ground_truth_rows == 0) {
		std::cerr << "Empty files" << std::endl;
		return -1;
	}

	std::cout << "Number of Verts rows: " << verts_rows << std::endl;
	std::cout << "Number of Meshes rows: " << meshes_rows << std::endl;

	if (verts_rows != ground_truth_rows) {
		std::cerr << "Number of rows in Verts and Ground Truth files are different" << std::endl;
		return -1;
	}
	
	bool* visible = check_visibility(verts, verts_rows, meshes, meshes_rows, columns);

	// Check the result
	bool error = false;
	for(unsigned short i = 0; i < verts_rows; i++) {
		if (visible[i] != gt[i][0]) {
			error = true;
			std::cerr << "Error in vertex " << i << std::endl;
			std::cerr << "Expected: " << gt[i][0] << " - Obtained: " << visible[i] << std::endl;
		}
	}
	if(!error)
		std::cout << "All vertices are correctly classified" << std::endl;


	for (unsigned short i = 0; i < verts_rows; i++)
		delete[] verts[i];
	delete[] verts;
	
	for (unsigned short i = 0; i < meshes_rows; i++)
		delete[] meshes[i];
	delete[] meshes;
	
	for (unsigned short i = 0; i < ground_truth_rows; i++)
		delete[] gt[i];
	delete[] gt;
	

	return 0;
}