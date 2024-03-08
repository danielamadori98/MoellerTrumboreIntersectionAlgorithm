#include "../include/fastRayTriangleIntersection_sequential.hpp"

void fastRayTriangleIntersection_sequential(
		double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
		double* V1, double* V2, double* V3, unsigned short rows,
		unsigned short border, unsigned short lineType, unsigned short planeType,
		bool* intersect, double* t, double* u, double* v){

	// Settings defaults values
	double eps = 1e-5, zero;

	switch (border) {
	case BORDER_NORMAL:
		zero = 0.0;
		break;
	case BORDER_INCLUSIVE:
		zero = eps;
	case BORDER_EXCLUSIVE:
		zero = -eps;
		break;
	default:
		std::cerr << "Border parameter must be either 'normal', 'inclusive' or 'exclusive'\n";
		return;
	}

	double *edge1 = new double[rows * COLUMNS_SIZE],
			*edge2 = new double[rows * COLUMNS_SIZE],
			*tvec = new double[rows * COLUMNS_SIZE],
			*det = new double[rows];

	
	{// Scope: First Parst: pvec is used only in this block, limiting its scope
		double* pvec = new double[rows * COLUMNS_SIZE];

		/*
		%% Find faces parallel to the ray
		edge1 = vert1-vert0;          % find vectors for two edges sharing vert0
		edge2 = vert2-vert0;
		tvec  = orig -vert0;          % vector from vert0 to ray origin
		pvec  = cross(dir, edge2,2);  % begin calculating determinant - also used to calculate U parameter
		det   = sum(edge1.*pvec,2);   % determinant of the matrix M = dot(edge1,pvec)
		*/
		for (unsigned short row = 0; row < rows; row++) {
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++) {
				edge1[row * COLUMNS_SIZE + col] = V2[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
				edge2[row * COLUMNS_SIZE + col] = V3[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
				tvec[row * COLUMNS_SIZE + col] = orig[col] - V1[row * COLUMNS_SIZE + col];
			}

			pvec[row * COLUMNS_SIZE + 0] = dir[1] * edge2[row * COLUMNS_SIZE + 2] - dir[2] * edge2[row * COLUMNS_SIZE + 1];
			pvec[row * COLUMNS_SIZE + 1] = dir[2] * edge2[row * COLUMNS_SIZE + 0] - dir[0] * edge2[row * COLUMNS_SIZE + 2];
			pvec[row * COLUMNS_SIZE + 2] = dir[0] * edge2[row * COLUMNS_SIZE + 1] - dir[1] * edge2[row * COLUMNS_SIZE + 0];

			det[row] = 0;
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
				det[row] += edge1[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];
		}

		/*
		switch planeType
			case 'two sided' % treats triangles as two sided
			angleOK = (abs(det) > eps);% if determinant is near zero then ray lies in the plane of the triangle
			case 'one sided' % treats triangles as one sided
			angleOK = (det > eps);
		otherwise
			error('Triangle parameter must be either "one sided" or "two sided"');
		end
		*/

		if (planeType == PLANE_TYPE_TWO_SIDED)
			for (unsigned short row = 0; row < rows; row++)
				intersect[row] = abs(det[row]) > eps;

		else if (planeType == PLANE_TYPE_ONE_SIDED)
			for (unsigned short row = 0; row < rows; row++)
				intersect[row] = det[row] > eps;

		else {
			std::cerr << "PlaneType parameter must be either 'two sided' or 'one sided'\n";
			delete[] edge1;
			delete[] edge2;
			delete[] tvec;
			delete[] pvec;
			delete[] det;

			return;
		}

		/*
			%% Different behavior depending on one or two sided triangles
			det(~angleOK) = nan;              % change to avoid division by zero
			u = sum(tvec.*pvec,2)./det;    % 1st barycentric coordinate
		*/
		for (unsigned short row = 0; row < rows; row++) {
			if (!intersect[row])
				u[row] = NAN;
			else {
				u[row] = 0;
				for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
					u[row] += tvec[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];

				u[row] /= det[row];
			}
		}
		delete[] pvec;
	}//pvec is not used anymore

	// Second Part: qvec is used only in this block, limiting its scope
	double qvec[COLUMNS_SIZE];

	//Calculate all variables for all line/triangle pairs
	/*
		qvec = cross(tvec, edge1,2);    % prepare to test V parameter
		v    = sum(dir  .*qvec,2)./det; % 2nd barycentric coordinate
		t    = sum(edge2.*qvec,2)./det; % 'position on the line' coordinate

		% test if line/plane intersection is within the triangle
		ok   = (angleOK & u>=-zero & v>=-zero & u+v<=1.0+zero);
	*/
	for (unsigned short row = 0; row < rows; row++) {
		if (!intersect[row])
			v[row] = NAN, t[row] = NAN;

		else {
			qvec[0] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
			qvec[1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 0] - tvec[row * COLUMNS_SIZE + 0] * edge1[row * COLUMNS_SIZE + 2];
			qvec[2] = tvec[row * COLUMNS_SIZE + 0] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 0];

			v[row] = t[row] = 0;
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++) {
				v[row] += dir[col] * qvec[col];
				t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[col];
			}

			v[row] /= det[row];
			t[row] /= det[row];

			intersect[row] = u[row] >= -zero && v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
		}
	}
	
	delete[] edge1;
	delete[] edge2;
	delete[] tvec;
	delete[] det;

	/*
		%% Test where along the line the line/plane intersection occurs
		switch lineType
		  case 'line'      % infinite line
			intersect = ok;
		  case 'ray'       % ray is bound on one side
			intersect = (ok & t>=-zero); % intersection on the correct side of the origin
		  case 'segment'   % segment is bound on two sides
			intersect = (ok & t>=-zero & t<=1.0+zero); % intersection between origin and destination
		  otherwise
			error('lineType parameter must be either "line", "ray" or "segment"');
		end
	*/

	switch (lineType) {
	case LINE_TYPE_LINE:// Nothing to do
		break;
	case LINE_TYPE_RAY:
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = intersect[i] && t[i] >= -zero;
		break;
	case LINE_TYPE_SEGMENT:
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = intersect[i] && t[i] >= -zero && t[i] <= 1.0 + zero;
		break;
	default:
		std::cerr << "LineType parameter must be either 'line', 'ray' or 'segment'\n";
	}
}

void fastRayTriangleIntersection_sequential(
	double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
	double* V1, double* V2, double* V3, unsigned short rows,
	unsigned short border, unsigned short lineType, unsigned short planeType,
	bool* intersect) {

	// Settings defaults values
	double eps = 1e-5, zero;

	switch (border) {
	case BORDER_NORMAL:
		zero = 0.0;
		break;
	case BORDER_INCLUSIVE:
		zero = eps;
	case BORDER_EXCLUSIVE:
		zero = -eps;
		break;
	default:
		std::cerr << "Border parameter must be either 'normal', 'inclusive' or 'exclusive'\n";
		return;
	}

	double* edge1 = new double[rows * COLUMNS_SIZE],
		* edge2 = new double[rows * COLUMNS_SIZE],
		* tvec = new double[rows * COLUMNS_SIZE],
		* det = new double[rows],
		* u = new double[rows];

	{// Scope: First Parst: pvec is used only in this block, limiting its scope
		double* pvec = new double[rows * COLUMNS_SIZE];
			/*
				%% Find faces parallel to the ray
				edge1 = vert1-vert0;          % find vectors for two edges sharing vert0
				edge2 = vert2-vert0;
				tvec  = orig -vert0;          % vector from vert0 to ray origin
				pvec  = cross(dir, edge2,2);  % begin calculating determinant - also used to calculate U parameter
				det   = sum(edge1.*pvec,2);   % determinant of the matrix M = dot(edge1,pvec)
			*/
		for (unsigned short row = 0; row < rows; row++) {
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++) {
				edge1[row * COLUMNS_SIZE + col] = V2[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
				edge2[row * COLUMNS_SIZE + col] = V3[row * COLUMNS_SIZE + col] - V1[row * COLUMNS_SIZE + col];
				tvec[row * COLUMNS_SIZE + col] = orig[col] - V1[row * COLUMNS_SIZE + col];
			}

			pvec[row * COLUMNS_SIZE + 0] = dir[1] * edge2[row * COLUMNS_SIZE + 2] - dir[2] * edge2[row * COLUMNS_SIZE + 1];
			pvec[row * COLUMNS_SIZE + 1] = dir[2] * edge2[row * COLUMNS_SIZE + 0] - dir[0] * edge2[row * COLUMNS_SIZE + 2];
			pvec[row * COLUMNS_SIZE + 2] = dir[0] * edge2[row * COLUMNS_SIZE + 1] - dir[1] * edge2[row * COLUMNS_SIZE + 0];

			det[row] = 0;
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
				det[row] += edge1[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];
		}

		/*
		switch planeType
			case 'two sided' % treats triangles as two sided
			angleOK = (abs(det) > eps);% if determinant is near zero then ray lies in the plane of the triangle
			case 'one sided' % treats triangles as one sided
			angleOK = (det > eps);
		otherwise
			error('Triangle parameter must be either "one sided" or "two sided"');
		end
		*/

		if (planeType == PLANE_TYPE_TWO_SIDED)
			for (unsigned short row = 0; row < rows; row++)
				intersect[row] = abs(det[row]) > eps;

		else if (planeType == PLANE_TYPE_ONE_SIDED)
			for (unsigned short row = 0; row < rows; row++)
				intersect[row] = det[row] > eps;

		else {
			std::cerr << "PlaneType parameter must be either 'two sided' or 'one sided'\n";
			delete[] edge1;
			delete[] edge2;
			delete[] tvec;
			delete[] pvec;
			delete[] det;

			return;
		}

		/*
			%% Different behavior depending on one or two sided triangles
			det(~angleOK) = nan;              % change to avoid division by zero
			u = sum(tvec.*pvec,2)./det;    % 1st barycentric coordinate
		*/
		for (unsigned short row = 0; row < rows; row++) {
			if (!intersect[row])
				u[row] = NAN;
			else {
				u[row] = 0;
				for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
					u[row] += tvec[row * COLUMNS_SIZE + col] * pvec[row * COLUMNS_SIZE + col];

				u[row] /= det[row];
			}
		}

		delete[] pvec;
	}//pvec is not used anymore


	 // Second Part: t, v, qvec are used only in this block, limiting their scope
	double* t = new double[rows],
		* v = new double[rows];
	double qvec[COLUMNS_SIZE];

	/*
		ok = (angleOK & u>=-zero & u<=1.0+zero); % mask
		% if all line/plane intersections are outside the triangle than no intersections
		qvec = cross(tvec(ok,:), edge1(ok,:),2); % prepare to test V parameter
		v(ok,:) = sum(dir(ok,:).*qvec,2) ./ det(ok,:); % 2nd barycentric coordinate
		if (~strcmpi(lineType,'line')) % 'position on the line' coordinate
			t(ok,:) = sum(edge2(ok,:).*qvec,2)./det(ok,:);
		end
		% test if line/plane intersection is within the triangle
		ok = (ok & v>=-zero & u+v<=1.0+zero);
	*/
	for (unsigned short row = 0; row < rows; row++) {
		intersect[row] = intersect[row] && u[row] >= -zero && u[row] <= 1 + zero;

		if (!intersect[row])
			v[row] = NAN;
		else {
			qvec[0] = tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 2] - tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 1];
			qvec[1] = tvec[row * COLUMNS_SIZE + 2] * edge1[row * COLUMNS_SIZE + 0] - tvec[row * COLUMNS_SIZE + 0] * edge1[row * COLUMNS_SIZE + 2];
			qvec[2] = tvec[row * COLUMNS_SIZE + 0] * edge1[row * COLUMNS_SIZE + 1] - tvec[row * COLUMNS_SIZE + 1] * edge1[row * COLUMNS_SIZE + 0];


			v[row] = 0;
			for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
				v[row] += dir[col] * qvec[col];
			v[row] /= det[row];


			if (lineType == LINE_TYPE_LINE)
				t[row] = NAN;

			else {
				t[row] = 0;
				for (unsigned short col = 0; col < COLUMNS_SIZE; col++)
					t[row] += edge2[row * COLUMNS_SIZE + col] * qvec[col];

				t[row] /= det[row];
			}

			intersect[row] = v[row] >= -zero && u[row] + v[row] <= 1.0 + zero;
		
		}
	}

	delete[] edge1;
	delete[] edge2;
	delete[] tvec;
	delete[] det;
	delete[] t;
	delete[] v;

	/*
		%% Test where along the line the line/plane intersection occurs
		switch lineType
		  case 'line'      % infinite line
			intersect = ok;
		  case 'ray'       % ray is bound on one side
			intersect = (ok & t>=-zero); % intersection on the correct side of the origin
		  case 'segment'   % segment is bound on two sides
			intersect = (ok & t>=-zero & t<=1.0+zero); % intersection between origin and destination
		  otherwise
			error('lineType parameter must be either "line", "ray" or "segment"');
		end
	*/

	switch (lineType) {
	case LINE_TYPE_LINE:// Nothing to do
		break;
	case LINE_TYPE_RAY:
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = intersect[i] && t[i] >= -zero;
		break;
	case LINE_TYPE_SEGMENT:
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = intersect[i] && t[i] >= -zero && t[i] <= 1.0 + zero;
		break;
	default:
		std::cerr << "LineType parameter must be either 'line', 'ray' or 'segment'\n";
	}
}
