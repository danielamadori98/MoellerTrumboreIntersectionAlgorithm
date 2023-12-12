#ifndef CHECK_VISIBILITY_H
#define CHECK_VISIBILITY_H

#include <iostream>
#include "../src/fastRayTriangleIntersection.cuh"

void check(bool* visible, bool** gt, unsigned short verts_rows);

bool* check_visibility(
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	bool** gt);

bool* check_visibility(double** verts,
	unsigned short verts_rows, // Number of vertices
	unsigned short** meshes,
	unsigned short meshes_rows, // Number of meshes
	unsigned short columns,
	bool** gt); // Number of columns in the mesh and verts matrices

bool* sequential_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible);

bool* parallel_code(
	double* camera_location,
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	unsigned short columns,
	double** V1, double** V2, double** V3,
	bool* flag,
	double* t, double* u, double* v, bool* visible);

#endif