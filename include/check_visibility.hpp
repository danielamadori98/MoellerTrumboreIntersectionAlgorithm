#ifndef CHECK_VISIBILITY_HPP
#define CHECK_VISIBILITY_HPP

#include "check_visibility_parallel.cuh"
#include "check_visibility_sequential.hpp"


// Remove the following comment to get full return

//# define FULL_RETURN 0


void check_results(bool* visible, bool** gt, unsigned short verts_rows);

void check_visibility(
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	bool** gt);

#endif