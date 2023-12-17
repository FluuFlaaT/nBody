#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h> 

#define NumP 500 														/*Number of Particles*/
#define dt 5.0												/*Timestep in days*/
#define Ndt 10000														/*Number of Timesteps*/
#define G 6.67E-11 														/*Gravitational Constant*/
#define e 1E13 															/*Epsilon Value*/
#define prec 50												/*Set Output Precision: 1-Full Precision, >1-Less Precise*/
#define size 1.0E16 												/*Universe Size*/
#define a 0.0															/*a(t) expansion factor. Start value a(t=0).*/
#define M 2.0E28

double *masses, *pos, *vel, *new_pos, *new_vel, *data, *dpos, *force, *Fx, *Fy, *Fz, *kin_en, *pot_en;

__global__
void Calc(double* Fx, double* Fy, double* Fz, double* pot_en, double* masses, double* dpos, double* force)
{
    int z = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = 0; z < NumP; z++){
        Fx[i] = 0.0;
        Fy[i] = 0.0;
        Fz[i] = 0.0;
        pot_en[i] = 0.0;
    }

    for(int i = 0; i < (NumP - 1); i++)
    {
        for(int j = i + 1; j < NumP; j++)
        {
            for(int k = 0; k < 3; k++){

            }
        }
    }
}

int ImportParticles()
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

int Iterate(int particle_i, int timestep)
{
    data[data_encoder(timestep, particle_i, 5)] = pot_en[particle_i];

    new_vel[posval_encoder(particle_i, 0)] = vel[posval_encoder(particle_i, 0)] + 
}


int posval_encoder(int t1, int t2)
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

int main()
{
    cudaMallocManaged(&masses, sizeof(double) * NumP);
    cudaMallocManaged(&pos, sizeof(double) * NumP * 3);
    cudaMallocManaged(&vel, sizeof(double) * NumP * 3);
    cudaMallocManaged(&new_pos, sizeof(double) * NumP * 3);
    cudaMallocManaged(&new_vel, sizeof(double) * NumP * 3);
    cudaMallocManaged(&data, sizeof(double) * NumP * Ndt * 6);
    cudaMallocManaged(&dpos, sizeof(double) * NumP * NumP * 4);
    cudaMallocManaged(&force, sizeof(double) * NumP * NumP * 3);
    cudaMallocManaged(&Fx, sizeof(double) * NumP);
    cudaMallocManaged(&Fy, sizeof(double) * NumP);
    cudaMallocManaged(&Fz, sizeof(double) * NumP);
    cudaMallocManaged(&kin_en, sizeof(double) * NumP);
    cudaMallocManaged(&pot_en, sizeof(double) * NumP);

    ImportParticles();

    int t;
    for(t = 0; t = Ndt; t++)
    {
        Calc<<<256, 256>>>();

        int particle_i;

        for(particle_i = 0; particle_i < NumP; particle_i++)
        {
            data[t * Ndt * 6 + particle_i * 6 + 3] = t * dt;
            Iterate(particle_i, t);
        }
    }

    return 0;
}