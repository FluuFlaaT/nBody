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
		vel[index * 3 + 2] = atof(result);
		
		index++;

		printf("%d %d %d %d %d %d %d\n", masses[index], pos[index * 3 + 0],pos[index * 3 + 1], pos[index * 3 + 2],pos[index * 3 + 3],pos[index * 3 + 4],pos[index * 3 + 5]);
	}
	
	fclose(particles_file);
	
	return 0;
}

int main()
{
    double *masses, *pos, *vel, *new_pos, *new_vel, *data, *dpos, *force, *Fx, *Fy, *Fz, *kin_en, *pot_en;

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

    ImportParticles(masses, pos, vel);

    return 0;
}