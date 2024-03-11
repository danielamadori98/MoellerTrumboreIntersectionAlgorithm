#include "../include/check_visibility.hpp"

void check_results(bool* visible, bool** gt, unsigned short verts_rows) {
	for(unsigned short i = 0; i < verts_rows; i++)
		if (visible[i] != gt[i][0]) {
			std::cerr << "Error in vertex " << i << std::endl;
			std::cerr << "Expected: " << gt[i][0] << " - Obtained: " << visible[i] << std::endl;
			//return; //Comment this line to print all errors
		}
}

void check_visibility(
	double** verts, unsigned short verts_rows,
	unsigned short** meshes, unsigned short meshes_rows,
	bool** gt){

	double camera_location[COLUMNS_SIZE] = { 0, 0, 0 };

	double* h_verts = new double[verts_rows * COLUMNS_SIZE];

	for (unsigned short row = 0; row < verts_rows; row++)
		for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
				h_verts[row * COLUMNS_SIZE + col] = verts[row][col];
	
	double	* V1 = new double [meshes_rows * COLUMNS_SIZE],
			* V2 = new double[meshes_rows * COLUMNS_SIZE],
			* V3 = new double[meshes_rows * COLUMNS_SIZE];

	for (unsigned short row = 0; row < meshes_rows; row++)
		for (unsigned short col = 0; col < COLUMNS_SIZE; col++) {
			V1[row * COLUMNS_SIZE + col] = verts[meshes[row][0]][col];
			V2[row * COLUMNS_SIZE + col] = verts[meshes[row][1]][col];
			V3[row * COLUMNS_SIZE + col] = verts[meshes[row][2]][col];
		}

	//Output variables
	bool* visible = new bool[verts_rows];

	//Timer
	double time_h, time_d;

	timer::Timer<timer::HOST> host_TM;


	//Check if FULL_RETURN is defined
#ifdef FULL_RETURN

	bool* flag = new bool[meshes_rows];
	
	//Output variables for FULL_RETURN
	//the t in the matlab code was be replaced by the v to mantain the same name used in the fastRayTriangleIntersection function
	double	* t = new double[meshes_rows], 
			* u = new double[meshes_rows],
			* v = new double[meshes_rows];

	host_TM.start();
	check_visibility_sequential_code(camera_location, h_verts, verts_rows, V1, V2, V3, meshes_rows, flag, t, u, v, visible);
	host_TM.stop();
	time_h = host_TM.duration();
	host_TM.print("Host:   ");
	check_results(visible, gt, verts_rows);
	
	time_d = check_visibility_parallel_code(camera_location, h_verts, verts_rows, V1, V2, V3, meshes_rows, flag, t, u, v, visible);
	check_results(visible, gt, verts_rows);

	delete[] t;
	delete[] u;
	delete[] v;
	delete[] flag;

#else

	host_TM.start();
	check_visibility_sequential_code(camera_location, h_verts, verts_rows, V1, V2, V3, meshes_rows, visible);
	host_TM.stop();
	time_h = host_TM.duration();
	host_TM.print("Host:   ");
	check_results(visible, gt, verts_rows);

	time_d = check_visibility_parallel_code(camera_location, h_verts, verts_rows, V1, V2, V3, meshes_rows, visible);
	check_results(visible, gt, verts_rows);

#endif

	delete[] h_verts;

	delete[] V1;
	delete[] V2;
	delete[] V3;

	delete[] visible;

	std::cout << "Speedup: " << std::fixed << std::setprecision(5) <<  time_h / time_d  << " x\n\n";
}
