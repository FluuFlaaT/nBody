#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> 

#define NumP 500 														/*Number of Particles*/
#define dt 5.0												/*Timestep in days*/
#define Ndt 5000														/*Number of Timesteps*/
#define G 6.67E-11 														/*Gravitational Constant*/
#define e 1E13 															/*Epsilon Value*/
#define prec 50												/*Set Output Precision: 1-Full Precision, >1-Less Precise*/
#define size 1.0E16 												/*Universe Size*/
#define a 0.0															/*a(t) expansion factor. Start value a(t=0).*/
#define M 2.0E28

double *masses, *pos, *vel, *new_pos, *new_vel, *data, *dpos, *force, *Fx, *Fy, *Fz, *kin_en, *pot_en;

__device__ int posvel_encoder_gpu(int t1, int t2)
{
    return (t1 * 3 + t2);
}

__device__ int dpos_encoder_gpu(int t1, int t2, int t3)
{
    return (t1 * NumP * 4 + t2 * 4 + t3);
}

__device__ int data_encoder_gpu(int t1, int t2, int t3)
{
    return (t1 * Ndt * 6 + t2 * 6 + t3);
}

__device__ int force_encoder_gpu(int t1, int t2, int t3)
{
    return (t1 * NumP * 3 + t2 * 3 + t3);
}

int posvel_encoder(int t1, int t2)
{
    return (t1 * 3 + t2);
}

int dpos_encoder(int t1, int t2, int t3)
{
    return (t1 * NumP * 4 + t2 * 4 + t3);
}

int data_encoder(int t1, int t2, int t3)
{
    return (t1 * Ndt * 6 + t2 * 6 + t3);
}

int force_encoder(int t1, int t2, int t3)
{
    return (t1 * NumP * 3 + t2 * 3 + t3);
}

__global__
void Calc(double* Fx, double* Fy, double* Fz, double* pot_en, double* masses, double* dpos, double* force, double* pos)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = 0; i < 3; i++){
        Fx[i] = 0.0;
        Fy[i] = 0.0;
        Fz[i] = 0.0;
        pot_en[i] = 0.0;
    }

    for(int i = index; i < NumP - 1; i += stride)
    {
        for(int j = i + 1; j < NumP; j ++)
        {
            for(int k = 0; k < 3; k++)
            {
                dpos[dpos_encoder_gpu(i, j, k)] = pos[posvel_encoder_gpu(i, k)] - pos[posvel_encoder_gpu(j, k)];
                dpos[dpos_encoder_gpu(j,i,k)] = - dpos[dpos_encoder_gpu(i,j,k)];
            }

            dpos[dpos_encoder_gpu(i,j,3)] = sqrt( pow(dpos[dpos_encoder_gpu(i,j,0)],2) + pow(dpos[dpos_encoder_gpu(i,j,1)], 2) + pow(dpos[dpos_encoder_gpu(i,j,2)],2));
            dpos[dpos_encoder_gpu(j,i,3)] = dpos[dpos_encoder_gpu(i,j,3)];

            /*in Fx*/
            force[force_encoder_gpu(i,j,0)] = - (G * masses[i] * masses[j] * dpos[dpos_encoder_gpu(i,j,0)]) / pow(sqrt (pow(dpos[dpos_encoder_gpu(i,j,3)], 2))+ pow(e, 2), 3);
            force[force_encoder_gpu(j,i,0)] = - force[force_encoder_gpu(i,j,0)];
            Fx[i] = Fx[i] + force[force_encoder_gpu(i,j,0)];
            Fx[j] = Fx[j] + force[force_encoder_gpu(j,i,0)];
            

            /*in Fy*/
            force[force_encoder_gpu(i,j,1)] = - (G * masses[i] * masses[j] * dpos[dpos_encoder_gpu(i,j,1)]) / pow( sqrt(pow(dpos[dpos_encoder_gpu(i,j,3)],2) + pow(e,2)) ,3);
            force[force_encoder_gpu(j,i,1)] = - force[force_encoder_gpu(i,j,1)];
            Fy[i] = Fy[i] + force[force_encoder_gpu(i,j,1)];
            Fy[j] = Fy[j] + force[force_encoder_gpu(j,i,1)];

            /*in Fz*/
            force[force_encoder_gpu(i,j,2)] = - (G * masses[i] * masses[j] * dpos[dpos_encoder_gpu(i,j,2)]) / pow(sqrt (pow(dpos[dpos_encoder_gpu(i,j,3)], 2))+ pow(e, 2), 3);
            force[force_encoder_gpu(j,i,2)] = - force[force_encoder_gpu(i,j,2)];
            Fx[i] = Fx[i] + force[force_encoder_gpu(i,j,2)];
            Fx[j] = Fx[j] + force[force_encoder_gpu(j,i,2)];

            pot_en[i] = pot_en[i] - (G * masses[i] * masses[j])/(dpos[dpos_encoder_gpu(i,j,3)]);
        }
    }
}

int ImportParticles(double* masses, double* pos, double* vel)
{
    FILE *particles_file;
	particles_file = fopen("particles.txt", "r");
	
	char line[100];
	int index = 0;
	
	fgets (line, sizeof line, particles_file); /*Skip first line*/
		
	while (fgets (line, sizeof line, particles_file))
	{
		//printf ("\n Line - %s \n", line);
		char *result = NULL;
		result = strtok(line, ",");
		masses[index] = atof(result);
		
		result = strtok( NULL, "," );
		pos[index * 3 + 0] = atof(result);
		
		result = strtok( NULL, "," );
		pos[index * 3 + 1] = atof(result);
		
		result = strtok( NULL, "," );
		pos[index * 3 + 2] = atof(result);
		
		result = strtok( NULL, "," );
		vel[index * 3 + 0] = atof(result);
		
		result = strtok( NULL, "," );
		vel[index * 3 + 1] = atof(result);
		
		result = strtok( NULL, "," );
		vel[index * 3 +2] = atof(result);
		
		index++;
	}
	
	fclose(particles_file);
	
	return 0;
}

int Iterate(int particle_i,int timestep)										/*Particle iteration function*/
{
	data[data_encoder(timestep, particle_i, 5)] = pot_en[particle_i];					/*Add potetial energy to data array*/
	
	/*Calculate new velocity in each direction and add to new_vel array*/
    new_vel[posvel_encoder(particle_i, 0)] = vel[posvel_encoder(particle_i, 0)] + ((Fx[particle_i] * dt * 3600 * 24)/(masses[particle_i]));
    new_vel[posvel_encoder(particle_i, 1)] = vel[posvel_encoder(particle_i, 1)] + ((Fx[particle_i] * dt * 3600 * 24)/(masses[particle_i]));
    new_vel[posvel_encoder(particle_i, 2)] = vel[posvel_encoder(particle_i, 2)] + ((Fx[particle_i] * dt * 3600 * 24)/(masses[particle_i]));
	
	/*Calculate new x positon and add to data array*/
    new_pos[posvel_encoder(particle_i, 0)] = (1.0 + a) * (pos[posvel_encoder(particle_i, 0)] + (new_vel[posvel_encoder(particle_i, 0)]* dt * 3600 * 24));

	/*printf ("Pos[i][0] - %.2e \t New_Pos[i][0] - %.2e \t New_Pos/(1.0+a) - %.2e \n",pos[particle_i][0],new_pos[particle_i][0],new_pos[particle_i][0]/(1.0+a));*/
    data[data_encoder(timestep, particle_i, 0)] = ( new_pos[posvel_encoder(particle_i, 0)] / pow((1.0+a),(timestep+1)));
	
	/*Calculate new y positon and add to data array*/
	new_pos[posvel_encoder(particle_i, 1)] = (1.0+a) * (pos[posvel_encoder(particle_i, 1)] + (new_vel[posvel_encoder(particle_i, 1)] * dt * 3600 * 24));
	data[data_encoder(timestep, particle_i, 1)] = (new_pos[posvel_encoder(particle_i, 1)] / pow((1.0+a),(timestep+1)));
	
	/*Calculate new z positon and add to data array*/
    new_pos[posvel_encoder(particle_i, 2)] = (1.0+a) * (pos[posvel_encoder(particle_i, 2)] + (new_vel[posvel_encoder(particle_i, 2)] * dt * 3600 * 24));
	data[data_encoder(timestep, particle_i, 2)] = (new_pos[posvel_encoder(particle_i, 2)] / pow((1.0+a),(timestep+1)));
	
	/*Calculate velocities half a timestep for kinetic energy calcs*/
	double vel_x = vel[posvel_encoder(particle_i, 0)] + ((Fx[particle_i] * dt * 3600 * 24 * 0.5)/(masses[particle_i]));
	double vel_y = vel[posvel_encoder(particle_i, 1)] + ((Fx[particle_i] * dt * 3600 * 24 * 0.5)/(masses[particle_i]));
	double vel_z = vel[posvel_encoder(particle_i, 2)] + ((Fx[particle_i] * dt * 3600 * 24 * 0.5)/(masses[particle_i]));
	double vel_abs = sqrt(pow(vel_x,2) + pow(vel_y,2) + pow(vel_z,2));
	
	/*Calculate kinectic energy and add to kin_en array*/
	kin_en[particle_i] = 0.5*masses[particle_i]*pow(vel_abs,2);
	data[data_encoder(timestep, particle_i, 4)] = kin_en[particle_i];

	return 0;
}	

int main()
{
    cudaMallocManaged(&masses, sizeof(double) * NumP);
    cudaMallocManaged(&pos, sizeof(double) * NumP * 3);
    cudaMallocManaged(&vel, sizeof(double) * NumP * 3);
    cudaMallocManaged(&new_pos, sizeof(double) * NumP * 3);
    cudaMallocManaged(&new_vel, sizeof(double) * NumP * 3);
    cudaMallocManaged(&data, sizeof(double) * Ndt * Ndt * 6);
    cudaMallocManaged(&dpos, sizeof(double) * NumP * NumP * 4);
    cudaMallocManaged(&force, sizeof(double) * NumP * NumP * 3);
    cudaMallocManaged(&Fx, sizeof(double) * NumP);
    cudaMallocManaged(&Fy, sizeof(double) * NumP);
    cudaMallocManaged(&Fz, sizeof(double) * NumP);
    cudaMallocManaged(&kin_en, sizeof(double) * NumP);
    cudaMallocManaged(&pot_en, sizeof(double) * NumP);

    double *mass, *ppos, *vels;
    mass = (double *)malloc(NumP*sizeof(double));
    ppos = (double *)malloc(NumP*sizeof(double)*3);
    vels = (double *)malloc(NumP*sizeof(double)*3);
    ImportParticles(mass, ppos, vels);
    cudaMemcpy(masses, mass, NumP*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pos, ppos, NumP*sizeof(double)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(vel, vels, NumP*sizeof(double)*3, cudaMemcpyHostToDevice);

/*
    printf("------FINISHED MEMORY COPY------\n");

    for(int i = 0; i < NumP; i++)
    {
        printf("pos: %e %e %e\n", pos[posvel_encoder(i, 0)], pos[posvel_encoder(i, 1)], pos[posvel_encoder(i, 2)]);
    }
*/

    int t;
    for(t = 0; t < Ndt; t++)
    {
        Calc<<<32,32>>>(Fx, Fy, Fz, pot_en, masses, dpos, force, pos);
        //printf("Finished calc in round %d\n", t);
        int particle_i;

        for(particle_i = 0; particle_i < NumP; particle_i++)
        {
            data[t * Ndt * 6 + particle_i * 6 + 3] = t * dt;
            Iterate(particle_i, t);
        }
    }

    cudaDeviceSynchronize();

    for(int i = 0; i < NumP; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            pos[posvel_encoder(i,j)] = new_pos[posvel_encoder(i,j)];
            vel[posvel_encoder(i,j)] = new_vel[posvel_encoder(i,j)];
        }
    }

    printf ("NumP,dt,Ndt,Precision, , ,\n");
    printf ("%i,%f,%i,%i,%e,%e,\n", NumP, dt, Ndt, prec, size, e);
    printf("x,y,z,t,Ek,Ep,\n");

    for(int i = 0; i < Ndt; i += prec)
    {
        for(int j = 0; j < NumP; j++)
        {
            printf("%e,%e,%e,%e,%e,%e,\n", data[data_encoder(i,j,0)], data[data_encoder(i,j,1)], data[data_encoder(i,j,2)], data[data_encoder(i,j,3)], data[data_encoder(i,j,4)], data[data_encoder(i,j,5)]);
        }
    }

    return 0;
}