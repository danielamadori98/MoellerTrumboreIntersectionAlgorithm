#include "../include/fastRayTriangleIntersection_sequential.hpp"

void fastRayTriangleIntersection_sequential(
		double orig[COLUMNS_SIZE], double dir[COLUMNS_SIZE],
		double** V1, double** V2, double** V3, unsigned short rows,
		unsigned short border, unsigned short lineType, unsigned short planeType, bool fullReturn,
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

	/*
		%% Find faces parallel to the ray
		edge1 = vert1-vert0;          % find vectors for two edges sharing vert0
		edge2 = vert2-vert0;
		tvec  = orig -vert0;          % vector from vert0 to ray origin
		pvec  = cross(dir, edge2,2);  % begin calculating determinant - also used to calculate U parameter
		det   = sum(edge1.*pvec,2);   % determinant of the matrix M = dot(edge1,pvec)
	*/

	//Dyn allocation of edge1, edge2, tvec, pvec, det
	/*double* edge1[] = new double[rows][COLUMNS_SIZE], * edge2 = new double[rows][COLUMNS_SIZE],
		*tvec = new double[rows][COLUMNS_SIZE], *pvec = new double[rows][COLUMNS_SIZE],
		*det = new double[rows];
	*/
	double(*edge1)[COLUMNS_SIZE] = new double[rows][COLUMNS_SIZE];
	double(*edge2)[COLUMNS_SIZE] = new double[rows][COLUMNS_SIZE];
	double(*tvec)[COLUMNS_SIZE] = new double[rows][COLUMNS_SIZE];
	double(*pvec)[COLUMNS_SIZE] = new double[rows][COLUMNS_SIZE];
	double* det = new double[rows];

	for (unsigned short i = 0; i < rows; i++) {
		for (unsigned short j = 0; j < COLUMNS_SIZE; j++) {
			edge1[i][j] = V2[i][j] - V1[i][j];
			edge2[i][j] = V3[i][j] - V1[i][j];
			tvec[i][j] = orig[j] - V1[i][j];
		}

		pvec[i][0] = dir[1] * edge2[i][2] - dir[2] * edge2[i][1];
		pvec[i][1] = dir[2] * edge2[i][0] - dir[0] * edge2[i][2];
		pvec[i][2] = dir[0] * edge2[i][1] - dir[1] * edge2[i][0];

		det[i] = 0;
		for (unsigned short j = 0; j < COLUMNS_SIZE; j++)
			det[i] += edge1[i][j] * pvec[i][j];
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

	if (planeType == PLANE_TYPE_TWOSIDED)
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = abs(det[i]) > eps;

	else if (planeType == PLANE_TYPE_ONESIDED)
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = det[i] > eps;

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
	for (unsigned short i = 0; i < rows; i++) {
		if (!intersect[i])
			u[i] = NAN;
		else {
			u[i] = 0;
			for (unsigned short j = 0; j < COLUMNS_SIZE; j++)
				u[i] += tvec[i][j] * pvec[i][j];

			u[i] /= det[i];
		}
	}

	double qvec[COLUMNS_SIZE];
	if (fullReturn) { //Calculate all variables for all line/triangle pairs
		/*
			qvec = cross(tvec, edge1,2);    % prepare to test V parameter
			v    = sum(dir  .*qvec,2)./det; % 2nd barycentric coordinate
			t    = sum(edge2.*qvec,2)./det; % 'position on the line' coordinate

			% test if line/plane intersection is within the triangle
			ok   = (angleOK & u>=-zero & v>=-zero & u+v<=1.0+zero);
		*/
		for (unsigned short i = 0; i < rows; i++) {
			if (!intersect[i])
				v[i] = NAN, t[i] = NAN;

			else {
				qvec[0] = tvec[i][1] * edge1[i][2] - tvec[i][2] * edge1[i][1];
				qvec[1] = tvec[i][2] * edge1[i][0] - tvec[i][0] * edge1[i][2];
				qvec[2] = tvec[i][0] * edge1[i][1] - tvec[i][1] * edge1[i][0];

				v[i] = t[i] = 0;
				for (unsigned short j = 0; j < COLUMNS_SIZE; j++) {
					v[i] += dir[j] * qvec[j];
					t[i] += edge2[i][j] * qvec[j];
				}

				v[i] /= det[i];
				t[i] /= det[i];

				intersect[i] = u[i] >= -zero && v[i] >= -zero && u[i] + v[i] <= 1.0 + zero;
			}
		}
	}
	else {
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
		for (unsigned short i = 0; i < rows; i++) {
			intersect[i] = intersect[i] && u[i] >= -zero && u[i] <= 1 + zero;

			if (!intersect[i])
				v[i] = NAN;
			else {
				qvec[0] = tvec[i][1] * edge1[i][2] - tvec[i][2] * edge1[i][1];
				qvec[1] = tvec[i][2] * edge1[i][0] - tvec[i][0] * edge1[i][2];
				qvec[2] = tvec[i][0] * edge1[i][1] - tvec[i][1] * edge1[i][0];

				v[i] = 0;
				for (unsigned short j = 0; j < COLUMNS_SIZE; j++)
					v[i] += dir[j] * qvec[j];
				v[i] /= det[i];


				if (lineType == LINE_TYPE_LINE)
					t[i] = NAN;

				else {
					t[i] = 0;
					for (unsigned short j = 0; j < COLUMNS_SIZE; j++)
						t[i] += edge2[i][j] * qvec[j];

					t[i] /= det[i];
				}

				intersect[i] = v[i] >= -zero && u[i] + v[i] <= 1.0 + zero;
			}
		}
	}

	delete[] edge1;
	delete[] edge2;
	delete[] tvec;
	delete[] pvec;
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