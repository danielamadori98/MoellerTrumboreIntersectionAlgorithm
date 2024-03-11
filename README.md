# Parallelization (CUDA) of the Möller–Trumbore intersection algorithm in a Human Shape Estimation context

### Problem statement:
Given a set P of 6890 3D points and a set M of 13776 meshes, find for each point p of P if the segment that connects (0,0,0) to p does not intersect any mesh of M.

#### Definition: Triangle mesh
A triangle mesh is a common representation of a 3D object or surface in computer graphics and 3D modeling. It is composed of a collection of triangles (three-sided polygons) that are connected to one another to form the shape of the object. Each triangle consists of three vertices (points in 3D space) and their associated edges (lines connecting the vertices). Triangle meshes are widely used because they are simple to work with and provide a good balance between accuracy and efficiency in rendering and modeling 3D objects. They are especially useful for representing complex shapes like characters, landscapes, and other 3D models in video games, animations, and simulations.

## Project Structure

- **src/**: Contains the source code with CUDA and C++ implementations.
    - `main.cu`: Main program file orchestrating the workflow.
    - `check_visibility*.cu/cpp`: Implementations of visibility checking algorithms.
- **include/**: Header files for the project functions and definitions.
- **build/**: Directory for compiled binaries and build files (generated after build).
- **out/**: Output directory for storing execution results.

## Compilation Instructions

Ensure you have the CUDA toolkit and CMake installed on your system. Follow these steps to compile the project:

1. Navigate to the project root directory.
2. Create a build directory and enter it:
   ```bash
   mkdir build && cd build
   ```
3. Configure the project with CMake:
   ```bash
   cmake ..
   ```
4. Build the project:
   ```bash
   make
   ```

The executable will be in the `build` directory.

## Execution Instructions

Run the executable from the `build` directory, providing the necessary CSV file paths as arguments:

```bash
./MoellerTrumboreIntersectionAlgorithm verts.csv meshes.csv ground_truth.csv

```

The ground_truth.csv file is needed just to check the correctness, it was used during the code development.

### Performance analysis:
Kernel Size: 352
Speedup: 5.0x
