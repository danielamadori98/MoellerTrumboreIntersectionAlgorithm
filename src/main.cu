﻿#include "../include/readCSV.hpp"
#include "../include/check_visibility.hpp"

int main() {
	std::string path = "../",
		vertsFilename = path + "rotated_verts.csv",
		meshesFilename = path + "meshes.csv",
		ground_truth = path + "ground_truth.csv";

	unsigned short verts_rows = 0, meshes_rows = 0, ground_truth_rows = 0;

	double** verts = read_CSV_double(vertsFilename, COLUMNS_SIZE, ',', verts_rows);

	unsigned short** meshes = read_CSV_unsigned_short(meshesFilename, COLUMNS_SIZE, ',', meshes_rows);

	bool** gt = read_CSV_bool(ground_truth, 1, ',', ground_truth_rows);

	if(verts == nullptr || meshes == nullptr || gt == nullptr)
		return -1;

	if (verts_rows == 0 || meshes_rows == 0 || ground_truth_rows == 0) {
		std::cerr << "Empty files" << std::endl;
		return -1;
	}
	
	if (verts_rows != ground_truth_rows) {
		std::cerr << "Number of rows in Verts and Ground Truth files are different" << std::endl;
		return -1;
	}

	
	bool* visible = check_visibility(verts, verts_rows, meshes, meshes_rows, gt);

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
