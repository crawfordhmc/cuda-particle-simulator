This program is used to simulate the movement of particles in a square surface.

An input file is read in main.cpp, the file follows strictly the structure shown:
    N - number of particles
    L - size of the square (µm)
    r – Radius of the particle (µm)
    S – Number of steps in simulation
    print or perf - If word print appears, the position of each particle needs to be printed at each step. Otherwise perf should appear.
    (Optional) On a single line for each particle, seperated by spaces:
    i - index of the particle (from 0 to N-1)
    x - initial horizontal position of the particle (from r to L - r)
    Y - initial vertical position of the particle (from r to L - r)
    Vx - initial horizontal velocity of the particle (positive or negative)
    Vy - initial vertical position of the particle (positive or negative)

This program has no input checking - please make sure values are physically possible before running.
The default number of threads run is 512
If you are getting CUDA errors it is because your machine does not have the resources to run the number of threads specified in the program.
The output is printed in the command window.

To compile and run:
nvcc -std=c++11 -o run main.cu
./run [number of threads, optional] < input

input3 was used for testing stats.