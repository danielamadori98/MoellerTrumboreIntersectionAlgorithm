# Parallelization (CUDA) of the Möller–Trumbore intersection algorithm in a Human Shape Estimation context

### Problem statement:
Given a set P of 6890 3D points and a set M of 13776 meshes, find for each point p of P if the segment that connects (0,0,0) to p does not intersect any mesh of M.

#### Definition: Triangle mesh
A triangle mesh is a common representation of a 3D object or surface in computer graphics and 3D modeling. It is composed of a collection of triangles (three-sided polygons) that are connected to one another to form the shape of the object. Each triangle consists of three vertices (points in 3D space) and their associated edges (lines connecting the vertices). Triangle meshes are widely used because they are simple to work with and provide a good balance between accuracy and efficiency in rendering and modeling 3D objects. They are especially useful for representing complex shapes like characters, landscapes, and other 3D models in video games, animations, and simulations.

### Performance analysis:
Kernel Size: 320
Speedup: 5.0x
