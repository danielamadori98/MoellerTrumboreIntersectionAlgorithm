#ifndef CHECK_VISIBILITY_H
#define CHECK_VISIBILITY_H

#include "check_visibility_parallel.cuh"
#include "check_visibility_sequential.h"

#include "lib/Timer.cuh"

void check_results(bool* visible, bool** gt, unsigned short verts_rows);

bool* check_visibility(
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	bool** gt);

#endif