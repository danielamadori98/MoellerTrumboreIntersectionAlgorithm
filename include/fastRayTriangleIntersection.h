#ifndef FAST_RAY_TRIANGLE_INTERSECTION_H
#define FAST_RAY_TRIANGLE_INTERSECTION_H

enum Border {
	Normal,     // Triangle is exactly as defined
	Inclusive,  // Triangle is marginally larger
	Exclusive   // Triangle is marginally smaller
};

enum LineType {
	Ray,      // Infinite (on one side) ray coming out of origin
	Line,     // Infinite (on both sides) line
	Segment   // Line segment bounded on both sides
};

enum PlaneType {
	TwoSided,  // Treats triangles as two sided
	OneSided   // Treats triangles as one sided
};

void fastRayTriangleIntersection(double* orig, double* dir,
								double** V1, double** V2, double** V3,
								unsigned short rows, unsigned short columns,
								Border border, LineType lineType, PlaneType planeType,
								bool fullReturn,
								bool* intersect, double* t, double* u, double* v);//Returning values


#endif FAST_RAY_TRIANGLE_INTERSECTION_H