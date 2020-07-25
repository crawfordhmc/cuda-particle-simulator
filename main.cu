#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>
#include <fstream>

#include <ctime>
#include <random>
using namespace std;

//use seed from time to generate random values
std::mt19937 rng(time(0));

//used to set time to collision as a high value to indicate no collision
int const NO_VALUE = std::numeric_limits<int>::max();

//type of collision
enum CollType{NOCOLL, LEFTWALL, RIGHTWALL, TOPWALL, BOTTOMWALL, PARTCOLL};

//used to return what kind of collision happened with default values if
//a collision did not happen
struct Collision{
    //colliding particle #1 (only one colliding if hitting a wall)
    int partId1 = NO_VALUE;
    //colliding particle #2
    int partId2 = NO_VALUE;
    //tells if we collided with a particle, with which wall
    //or didn't collide
    CollType type = NOCOLL;
    //time it took to collide
    double t = NO_VALUE;
};

typedef enum {
    MODE_PRINT,
    MODE_PERF
} simulation_mode_t;

//used to present particles in the simulation
struct particle_t {
    int i = -1;
    double x = NO_VALUE;
    double y = NO_VALUE;
    double vx = NO_VALUE;
    double vy = NO_VALUE;
    int p_collisions = 0;
    int w_collisions = 0;
    bool collided = false ;
    bool stopPart = false;
};


__constant__ int l, r, s;
__constant__ int n;

__managed__ particle_t* particles;
__managed__ Collision wallCol;
__managed__ Collision partCol;
 
int host_n;

// device method to atomically set minimum float value
__device__ __forceinline__ float atomicMinD (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
} 
// device method to calculate wall collision time
__device__ Collision calcCollisionTime(particle_t &p1, int iL, int iR,  double t){
	   Collision rCol;
	   double l = iL;
	   double r = iR;
	   if(p1.vx < 0){
	   	   double t_help = (l - r - (l - p1.x))/(-p1.vx);
		   if(t_help >= 0. && t_help < t){
		   	     rCol.t = t_help;
			     rCol.type = LEFTWALL;
			     rCol.partId1 = p1.i;
		   }
	   }
	   if(p1.vx > 0){
	   	    double t_help = (l - r - p1.x)/p1.vx;
		    if(t_help >= 0. && t_help < t){
		    	      rCol.t = t_help;
			      rCol.type = RIGHTWALL;
			      rCol.partId1 = p1.i;
		    }
	   }
	   if(p1.vy < 0){
	   	    double t_help = (l - r - (l - p1.y))/(-p1.vy);
		    if(t_help >= 0. && t_help < t && t_help < rCol.t){
		    	      rCol.t = t_help;
			      rCol.type = BOTTOMWALL;
			      rCol.partId1 = p1.i;
		    }
	   }
	   if(p1.vy > 0){
	   	    double t_help = (l - r - p1.y)/p1.y;
		    if(t_help >= 0. && t_help < t && t_help < rCol.t){
		    	      rCol.t = t_help;
			      rCol.type = TOPWALL;
			      rCol.partId1 = p1.i;
		    }
	   }
	   return rCol;
}

// device method to calculate particle collision time
__device__ Collision calcCollisionTime(particle_t &p1, particle_t &p2, double t){
	   Collision rCol;
	   double a = (p2.vx-p1.vx)*(p2.vx-p1.vy) + (p2.vy-p1.vy)*(p2.vy-p1.vy);
	   double b = 2.0*((p2.x-p1.x)*(p2.vx-p1.vx) + (p2.y-p1.y)*(p2.vy-p1.vy));
	   double c = (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y) - (2*r)*(2*r);
	   double det = b*b - 4*a*c;
	   if (a != 0){
	      double t_help = (-b - sqrt(det))/(2.0*a);
	      if (t_help >= 0. && t_help < t){
	      	 rCol.type = PARTCOLL;
		 rCol.t = t_help;
		 rCol.partId1 = p1.i;
		 rCol.partId2 = p2.i;
	      }
     	   }
	   return rCol;
}

// Method updates fastest wall or particle collision depending on thread block
__global__ void find_first_collisions(int num_threads, double t, int offset)
{
    __shared__ int helpN; // collision indexes that only gets a value in block 2
    int newN = NO_VALUE;

    Collision retCol; // fastest collision within thread

    __shared__ float blocktime; // fastest time in block
    if (threadIdx.x == 0)
	blocktime = NO_VALUE;
	helpN = NO_VALUE;
    //barrier to ensure blocktime is initialized to a comparable default before continuing
    __syncthreads();

    int index = threadIdx.x;
    if (n > num_threads){
        index += offset;
    }

    //wall collision, checked by 1st block
    if(blockIdx.x == 0 && index < n){
        //If particle is stopped at a wall it can be ignored
        if(!particles[index].stopPart){
            retCol = calcCollisionTime(particles[index], l, r, t);

	    // comparing to remove collisions that are too slow to need atomic comparing
            if (retCol.type != NOCOLL && (float) retCol.t < blocktime) {
                atomicMinD(&blocktime, (float) retCol.t);
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    //Block2 used to calculate particle-particle collisions
    else if(blockIdx.x == 1 && index < n){

        //if hasn't collided yet
        if(!particles[index].collided){

            //find collision time with each particle
            for(int j=0; j<n; j++){
		if (particles[index].i == particles[j].i) {
		    continue;
		}
                //neither particle has collided yet
                if(!particles[index].collided && !particles[j].collided){
		    retCol = calcCollisionTime(particles[index], particles[j], t);
                }
            }
        }

        if (retCol.type == PARTCOLL && (float) retCol.t < blocktime){
            // atomic compare & replace fastest block time to found time if sooner
            // float conversion needed to further convert into int for atomicMin
            atomicMinD(&blocktime, (float) retCol.t);
        }
    }
    // barrier must be done outside of conditional
    __syncthreads();

    // if there was a particle collision
    // and this thread's collision time = the block minimum
    if(retCol.type == PARTCOLL && (float) retCol.t == blocktime){
        // calculate and compare index of collision
        newN = retCol.partId1 + retCol.partId2;
	//if new N is the same as current (reverse of existing collision or different collision with the same col index), void N 
        if(atomicMin(&helpN, newN) == newN){
	    newN = NO_VALUE;
	} else {
	  //printf("th%i p%i t%f N%i compared with N%i\n", threadIdx.x, index, blocktime, newN, helpN);
	}
    }
    // else if particle did collide with a wall (retCol has a value)
    // and thread's collision time = the block minimum
    else if (retCol.type != NOCOLL && (float) retCol.t == blocktime){
        // if kernel is being run the first time (shortcircuits to save global access)
        // or global wall collision is later
        if(offset == 0 || wallCol.t > retCol.t){
            // update rest of wallCol struct
	    wallCol.t = retCol.t;
            wallCol.type = retCol.type;
            wallCol.partId1 = retCol.partId1;
        }
    }
    // make sure all the fastest threads in particle block have compared indexes
    __syncthreads();
    // if this thread has a collision index and it's the chosen one
    // not accounting for identical indexes here, only 1 col should execute
    if (newN != NO_VALUE && newN == helpN) {
        // if kernel is being run the first time (shortcircuits to save global access)
        // or global particle collision is later
        if(offset == 0 || partCol.t > retCol.t){
            // update partCol
            partCol.type = PARTCOLL;
            partCol.t = retCol.t;
	    partCol.partId1 = retCol.partId1;
	    partCol.partId2 = retCol.partId2;
	    //printf("partCol updated\n");
        }
    }
}

// advances particles in parallel
__global__ void advanceParticlesP(int num_threads, double t, int offset){
    int i = threadIdx.x + blockIdx.x*num_threads + offset;
    if (i < n && !particles[i].stopPart){
        particles[i].y += particles[i].vy*t;	
        particles[i].x += particles[i].vx*t;
    }
}

__host__ void checkCudaErrors() {
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }
}

__host__ void print_particles(int step)
{
    int i;
    for (i = 0; i < host_n; i++) {
        printf("%d %d %f %f %f %f\n", step, i, particles[i].x, particles[i].y,
               particles[i].vx, particles[i].vy);
    }
}

__host__ void print_statistics(int num_step)
{
    int i;
    for (i = 0; i < host_n; i++) {
        printf("%d %d %f %f %f %f %d %d\n", num_step, i, particles[i].x,
               particles[i].y, particles[i].vx, particles[i].vy,
               particles[i].p_collisions, particles[i].w_collisions);
    }
}

// returns soonest out of fastest wall and particle collisions
__host__ Collision whichCollision()
{
    //If both collisions happen, choose the earlier one
    if(wallCol.type != NOCOLL && partCol.type == PARTCOLL){

        //NOW WE PREFER WALLCOL IF BOTH T ARE SAME
        if(wallCol.t <= partCol.t){
            return wallCol;
        }
        else {
            return partCol;
        }
    }

    //only one happened
    if(wallCol.type != NOCOLL){
        return wallCol;
    }
    return partCol;

    //if no collisions happened the returned partCol
    //will have default info, this is checked later and no collision is done
}

__host__ void resetUnifiedCols(){
    wallCol.type = NOCOLL;
    wallCol.t = NO_VALUE;
    wallCol.partId1 = NO_VALUE;
    partCol.type = NOCOLL;
    partCol.t = NO_VALUE;
    partCol.partId1 = NO_VALUE;
    partCol.partId2 = NO_VALUE;
}

__host__ void collideParticles(particle_t &p1, particle_t &p2)
{
    //printf("colliding particles %i and %i, at %fx%f and %fx%f\n", p1.i, p2.i, p1.x, p1.y, p2.x, p2.y);

    //projecting the velocity unit normal and unit tangent vectors onto the unit
    //normal and unit tangent vectors, which is done by taking the dot product
    double vnx = (p2.x - p1.x);
    double vny = (p2.y - p1.y);
    double length = std::sqrt(vnx*vnx + vny*vny);
    double vunx = 0;
    double vuny = 0;
    if (length != 0){
    	vunx = vnx/length;
    	vuny = vny/length;
    }
    double vutx = -vuny;
    double vuty = vunx;
    //printf("normal unit %fx%f, length %f, tangent unit %fx%f\n", vunx, vuny, length, vutx, vuty);
    //dot products
    double v1n = vunx*p1.vx + vuny*p1.vy;
    double v2n = vunx*p2.vx + vuny*p2.vy;
    double v1t = vutx*p1.vx + vuty*p1.vy;
    double v2t = vutx*p2.vx + vuty*p2.vy;
    //double v1n = vun.dot(p1.v);
    //double v2n = vun.dot(p2.v);
    //double v1t = vut.dot(p1.v);
    //double v2t = vut.dot(p2.v);

    //find the new tangential velocities (after the collision).
    double v1tPrime = v1t;
    double v2tPrime = v2t;

    //find the new normal velocities.
    //this is where we use the one-dimensional collision formulas.
    //for m1 = m2 we get
    double v1nPrime = v2n;
    double v2nPrime = v1n;

    //convert the scalar normal and tangential velocities into vectors.
    //multiply the unit normal vector by the scalar normal velocity
    double v_v1nPrimex = v1nPrime*vunx;
    double v_v1nPrimey = v1nPrime*vuny;
    double v_v2nPrimex = v2nPrime*vunx;
    double v_v2nPrimey = v2nPrime*vuny;

    double v_v1tPrimex = v1tPrime*vutx;
    double v_v1tPrimey = v1tPrime*vuty;
    double v_v2tPrimex = v2tPrime*vutx;
    double v_v2tPrimey = v2tPrime*vuty;

    //the final velocity vectors by adding the normal and tangential
    //components for each object
    p1.vx = v_v1nPrimex + v_v1tPrimex;
    p1.vy = v_v1nPrimey + v_v1tPrimey;
    p2.vx = v_v2nPrimex + v_v2tPrimex;
    p2.vy = v_v2nPrimey + v_v2tPrimey;

    //info that collided in cycle and up the collision count
    p1.collided = true;
    p1.p_collisions += 1;
    p2.collided = true;
    p2.p_collisions += 1;
}


//collides given particle with the given wall (changes the velociy)
__host__ void collideWall(int i, CollType col)
{
    //info that collided in cycle and up the collision count
    particles[i].collided = true;
    particles[i].w_collisions += 1;

    //fabs returns the absolute value
    //depending on which wall is hit, either vx or vy will
    //change sign
    if(col == LEFTWALL){
        particles[i].vx = fabs(particles[i].vx);
    }

    else if (col == RIGHTWALL){
        particles[i].vx = -fabs(particles[i].vx);
    }

    else if (col == TOPWALL){
        particles[i].vy = -fabs(particles[i].vy);
    }

    else if (col == BOTTOMWALL){
        particles[i].vy = fabs(particles[i].vy);
    }

}

///////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    int i, x, y, vx, vy;
    int num_blocks, num_threads;
    int step, offset;
    int host_l, host_r, host_s;
    simulation_mode_t mode;
    char mode_buf[6];

    if (argc == 2){
    	num_threads = atoi(argv[1]);
    } else if (argc == 1){
	num_threads = 512;
    } else {
        printf("Usage:\n%s (optional) num_threads < input\n", argv[0]);
        return 1;
    }

    scanf("%d", &host_n);
    scanf("%d", &host_l);
    scanf("%d", &host_r);
    scanf("%d", &host_s);
    scanf("%5s", mode_buf);

    // blocks to address all particles with the given number of threads = ratio rounded up
    num_blocks = ceil((float)num_threads/(float)host_n);

    printf("%d particles, %d threads per block, %d blocks.\n", host_n, num_threads, num_blocks);

    cudaMallocManaged(&particles, sizeof(particle_t) * host_n);

    for (i = 0; i < host_n; i++) {
        particles[i].i = -1;
        particles[i].p_collisions = 0;
        particles[i].w_collisions = 0;
    }

    i = 0; //added
    while (scanf("%d %d %d %d %d", &i, &x, &y, &vx, &vy) != EOF) {
        particles[i].i = i;
        particles[i].x = x;
        particles[i].y = y;
        particles[i].vx = vx;
        particles[i].vy = vy;
    }

    if (particles[0].i == -1) {
        //generate a distribution for random values in given range
        std::uniform_int_distribution<int> dist_pos(host_r,host_l-host_r);

        std::uniform_int_distribution<int> dist_v(host_l/(8*host_r), host_l/4);

        //Generate particles and store into data structure
        for(int j = 0; j < host_n; j++){

            x=dist_pos(rng);
            y=dist_pos(rng);
            vx=dist_v(rng);
            vy=dist_v(rng);

            particles[j].i = j;
            particles[j].x = x;
            particles[j].y = y;
            particles[j].vx = vx;
            particles[j].vy = vy;
        }
    }

    mode = strcmp(mode_buf, "print") == 0 ? MODE_PRINT : MODE_PERF;

        /* Copy to GPU constant memory */
    cudaMemcpyToSymbol(n, &host_n, sizeof(n));
    cudaMemcpyToSymbol(l, &host_l, sizeof(l));
    cudaMemcpyToSymbol(r, &host_r, sizeof(r));
    cudaMemcpyToSymbol(s, &host_s, sizeof(s));

    //printf("particles%i size%i radius%i steps%i\n", host_n, host_l, host_r, host_s);
        //simulation of step
    for (step = 0; step < host_s; step++) {
        if (mode == MODE_PRINT || step == 0) {
           print_particles(step);
        }

        for (i = 0; i < host_n; i++) {
      	  particles[i].collided = false;
          particles[i].stopPart = false;
        }

        double tMoved = 0.0;
        Collision deviceCol;

            //loop while within the step
        while(tMoved < 1.0){

            // if number of particles is less than threads:
        
            // one block for each calculation will suffice
            // one block of N threads for wall collisions, one for particle
            // kernel is called with 2 blocks, numthreads in 1D each block
            // if block1/block2 statement seperates wall and particle collisions
            // syncthreads breaks up block seperation, then it returns to update wall/partCol

            // if particles exceeds threads:
            // a new grid is created with code to start at offset of particles already addressed.
            // each section of [numthreads] particles is calculated sequentially.
            for (offset = 0; offset < host_n; offset += num_threads) {
                /* Call the kernel */
                find_first_collisions<<<2, num_threads>>>(num_threads, 1.0-tMoved, offset);
                //printf("Step %i tMoved %lf\n", step, tMoved);
                /* Barrier */
                cudaDeviceSynchronize();
		        checkCudaErrors();
            }

            deviceCol = whichCollision();
	    resetUnifiedCols();

            //we had a collision
            if(deviceCol.type != NOCOLL && deviceCol.t >= 0){

	    	    //printf("returned collision type %i in %f\n", deviceCol.type, deviceCol.t);

                //move particles until collision time, using particle index offset if threads < n
                for (offset = 0; offset < host_n; offset += num_blocks*num_threads){
                    advanceParticlesP<<<num_blocks, num_threads>>>(num_threads, deviceCol.t, offset);
                    cudaDeviceSynchronize();
                    checkCudaErrors();
//printf("advancing: t%d b%d time%lf, offset%d\n", num_threads, num_blocks, deviceCol.t, offset);
                }
		//advanceParticles(deviceCol.t);
                tMoved += deviceCol.t;

                    //Collision between 2 particles
                if(deviceCol.type == PARTCOLL){
                    collideParticles(particles[deviceCol.partId1], particles[deviceCol.partId2]);
                }

                    //wall collision
                else {
                    //If particle hasn't collided yet
                    if(!particles[deviceCol.partId1].collided){
                        collideWall(deviceCol.partId1, deviceCol.type);
                    }

                    //If it has, we need to stop it's movement at the wall
                    else{
                        particles[deviceCol.partId1].stopPart = true;
                    }
                }
            }
            else {
                //no remaining collisions so advance to end of step
		for (offset = 0; offset < host_n; offset += num_blocks*num_threads){
                    advanceParticlesP<<<num_blocks, num_threads>>>(num_threads, 1 - tMoved, offset);
                    cudaDeviceSynchronize();
                    checkCudaErrors();
            	}
		tMoved = 1;
            }
    	}
    }
    print_statistics(host_s);
    return 0;
}









