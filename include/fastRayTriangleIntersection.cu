#include "../include/fastRayTriangleIntersection.h"
#include <iostream>

void fastRayTriangleIntersection(double* orig, double* dir, double** V1, double** V2, double** V3, unsigned short rows, unsigned short columns, Border border, LineType lineType, PlaneType planeType, bool fullReturn, bool* intersect, double* t, double* u, double* v) {
	double eps = 1e-5, zero;// Settings defaults values


	switch (border) {// Read user preferences: lineType, border
	case Normal:
		zero = 0.0;
		break;
	case Inclusive:
		zero = eps;
		break;
	case Exclusive:
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
	
	double** edge1 = new double*[rows], ** edge2 = new double* [rows],
		** tvec = new double* [rows], ** pvec = new double* [rows],
		*det = new double[rows];

	for (unsigned short i = 0; i < rows; i++) {
		edge1[i] = new double[columns];
		edge2[i] = new double[columns];
		tvec[i] = new double[columns];
		pvec[i] = new double[columns];

		for (unsigned short j = 0; j < columns; j++) {
			edge1[i][j] = V2[i][j] - V1[i][j];
			edge2[i][j] = V3[i][j] - V1[i][j];
			tvec[i][j] = orig[j] - V1[i][j];
		}

		pvec[i][0] = dir[1] * edge2[i][2] - dir[2] * edge2[i][1];
		pvec[i][1] = dir[2] * edge2[i][0] - dir[0] * edge2[i][2];
		pvec[i][2] = dir[0] * edge2[i][1] - dir[1] * edge2[i][0];

		det[i] = 0;
		for (unsigned short j = 0; j < columns; j++)
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
	if (planeType == TwoSided)
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = abs(det[i]) > eps;

	else if (planeType == OneSided)
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = det[i] > eps;

	else {
		std::cerr << "PlaneType parameter must be either 'two sided' or 'one sided'\n";
		for (unsigned short i = 0; i < rows; i++) {
			delete[] edge1[i];
			delete[] edge2[i];
			delete[] tvec[i];
			delete[] pvec[i];
		}
		delete[] edge1;
		delete[] edge2;
		delete[] tvec;
		delete[] pvec;

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
			for (unsigned short j = 0; j < columns; j++)
				u[i] += tvec[i][j] * pvec[i][j];

			u[i] /= det[i];
		}
	}

	if (fullReturn) { //Calculate all variables for all line/triangle pairs
		/*
			qvec = cross(tvec, edge1,2);    % prepare to test V parameter
			v    = sum(dir  .*qvec,2)./det; % 2nd barycentric coordinate
			t    = sum(edge2.*qvec,2)./det; % 'position on the line' coordinate

			% test if line/plane intersection is within the triangle
			ok   = (angleOK & u>=-zero & v>=-zero & u+v<=1.0+zero);
		*/
		double *qvec = new double[columns]; //Check qvec size, and also if needed fullReturn part

		for (unsigned short i = 0; i < rows; i++) {
			if (!intersect[i])
				v[i] = NAN, t[i] = NAN;

			else {
				qvec[0] = tvec[i][1] * edge1[i][2] - tvec[i][2] * edge1[i][1];
				qvec[1] = tvec[i][2] * edge1[i][0] - tvec[i][0] * edge1[i][2];
				qvec[2] = tvec[i][0] * edge1[i][1] - tvec[i][1] * edge1[i][0];

				v[i] = t[i] = 0;
				for (unsigned short j = 0; j < columns; j++) {
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
		double *qvec = new double[columns];

		for (unsigned short i = 0; i < rows; i++) {
			intersect[i] = intersect[i] && u[i] >= -zero && u[i] <= 1 + zero;

			if (!intersect[i])
				v[i] = NAN;
			else {
				qvec[0] = tvec[i][1] * edge1[i][2] - tvec[i][2] * edge1[i][1];
				qvec[1] = tvec[i][2] * edge1[i][0] - tvec[i][0] * edge1[i][2];
				qvec[2] = tvec[i][0] * edge1[i][1] - tvec[i][1] * edge1[i][0];

				v[i] = 0;
				for (unsigned short j = 0; j < columns; j++)
					v[i] += dir[j] * qvec[j];
				v[i] /= det[i];


				if (lineType == Line)
					t[i] = NAN;

				else {
					t[i] = 0;
					for (unsigned short j = 0; j < columns; j++)
						t[i] += edge2[i][j] * qvec[j];

					t[i] /= det[i];
				}

				intersect[i] = v[i] >= -zero && u[i] + v[i] <= 1.0 + zero;
			}
		}
	}

	for (unsigned short i = 0; i < rows; i++) {
		delete[] edge1[i];
		delete[] edge2[i];
		delete[] tvec[i];
		delete[] pvec[i];
	}
	delete[] edge1;
	delete[] edge2;
	delete[] tvec;
	delete[] pvec;


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
	case Line:// Nothing to do
		break;
	case Ray:
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = intersect[i] && t[i] >= -zero;
		break;
	case Segment:
		for (unsigned short i = 0; i < rows; i++)
			intersect[i] = intersect[i] && t[i] >= -zero && t[i] <= 1.0 + zero;
		break;
	default:
		std::cerr << "LineType parameter must be either 'line', 'ray' or 'segment'\n";
	}
}