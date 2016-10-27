/*! \file grid_fft.c
 *  \brief Function definitions for dealing with FFT's on grids.
 *
 *  Note that FFTW3 fftws are unnormalized, and are therefore multiplied by N 
 *  In fftw3, plans are apparently intended to be local.  They will be created
 *  and destroyed within the functions that perform the forward and reverse transforms.
 *  Real arrays have n elements, while complex arrays have n/2+1 (rounded down) elements 
 *  for out of place transforms. For inplace transforms, real arrays have 2*(n/2+1) elements.
 *
 *  An nx x ny x nz array (in row major order) array will have an output array of size
 *  nx x ny x (nz/2+1) after an r2c transform.  For an inplace transform, the input array
 *  must be padded to size nx x ny x 2*(nz/2+1) 
 *
 *  Section 4.8 of the fftw3 manual has what FFTW3 really computes.*/
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fftw3-mpi.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_sf_trig.h>
#include "grid_fft.h"


/*! \fn void Copy_FFTW_Grid_Info(FFTW_Grid_Info source, FFTW_Grid_Info *dest)
 *  \brief Copy an FFTW_Grid_info struct. */
void Copy_FFTW_Grid_Info(FFTW_Grid_Info source, FFTW_Grid_Info *dest)
{

	//save grid dimensions
	(*dest).nx = source.nx;
	(*dest).ny = source.ny;
	(*dest).nz = source.nz;
	(*dest).nz_complex = source.nz_complex;

	//save this process's local sizes
	(*dest).nx_local       = source.nx_local;
	(*dest).nx_local_start = source.nx_local_start;
	(*dest).n_local_real_size   = source.n_local_real_size;
	(*dest).n_local_complex_size   = source.n_local_complex_size;

	//save physical grid info
	(*dest).BoxSize        = source.BoxSize;
	(*dest).dx   = source.dx;
	(*dest).dy   = source.dy;
	(*dest).dz   = source.dz;
	(*dest).dV   = source.dV;
	(*dest).dVk  = source.dVk;

	//save number of dimensions
	(*dest).ndim = source.ndim;

	//save fftw3 plans
	(*dest).plan  = source.plan;
	(*dest).iplan = source.iplan;
}

/*! \fn int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for fftw grid based on coordinates i,j,k. */
int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
{
	int jj;	//wrap safe y index
	int kk;	//wrap safe z index

	//wrap safely in y and z
	jj = j;
	if(jj<0)
		jj += grid_info.ny;
	if(jj>(grid_info.ny-1))
		jj -= grid_info.ny;

	kk = k;
	if(kk<0)
		kk += grid_info.nz;
	if(kk>(grid_info.nz-1))
		kk -= grid_info.nz;

	//see page 61 of fftw3 manual
	return (i*grid_info.ny + jj)*(2*(grid_info.nz/2+1)) + kk;
}

/*! \fn int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info)
 *  \brief Given a position, return the grid index. */
int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info)
{
	int i = (int) (x/grid_info.dx) - grid_info.nx_local_start;	//integer index along x direction
	int j = (int) (y/grid_info.dy);					//integer index along y direction
	int k = (int) (z/grid_info.dz);					//integer index along z direction

	//if the position is not within this slab, then
	//return -1
	if(i < 0 || i >= grid_info.nx_local)
		return -1;

	//return the ijk of this position
	return grid_ijk(i,j,k,grid_info);	
}

/*! \fn int grid_complex_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for complex fftw grid based on coordinates i,j,k. */
int grid_complex_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
{
	//no wrapping
	//complex are nx x ny x nz
	//return (i*grid_info.ny + j)*grid_info.nz + k;
	return (i*grid_info.ny + j)*(grid_info.nz/2+1) + k;

}


/*! \fn int grid_complex_from_real_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for complex fftw grid created from r2c based on coordinates i,j,k. */
int grid_complex_from_real_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
{
	//no wrapping, p60 of fftw3 manual
	//complex are nx x ny x (nz/2 +1)
	return (i*grid_info.ny + j)*(grid_info.nz/2+1) + k;
}

/*! \fn void initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, MPI_Comm world);
 *  \brief Function to determine local grid sizes for parallel FFT. */
void initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, MPI_Comm world)
{
	ptrdiff_t nx_local;	
	ptrdiff_t nx_local_start;	
	ptrdiff_t n_local_complex_size;	

	//set the z-index size
	grid_info->nz_complex = grid_info->nz/2 + 1;

	//find the local sizes for complex arrays
	n_local_complex_size = fftw_mpi_local_size_3d(grid_info->nx, grid_info->ny, grid_info->nz/2+1, world, &nx_local, &nx_local_start);

	//remember the size
	grid_info->nx_local       = nx_local;
	grid_info->nx_local_start = nx_local_start;
	grid_info->n_local_complex_size = n_local_complex_size;
	grid_info->n_local_real_size    = 2*n_local_complex_size;


	if(!(nx_local>0))
	{
	    printf("*****************************************\n");
	    printf("WARNING!\n");
	    printf("nx_local = %d on at least 1 processor\n",(int) nx_local);
	    printf("Many functions implicitly assume nx_local>0\n");
	    printf("Try a different (lower) nprocs if possible.\n");
	    printf("*****************************************\n");
	    fflush(stdout);
	}
}

/*! \fn double *allocate_real_fftw_grid_sized(int n_size)
 *  \brief Allocates a pre-sized 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid_sized(int n_size)
{
	double *data;

	//allocate data
	data = fftw_alloc_real(n_size);

	//return data
	return data;
}

/*! \fn double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info)
{
	double *data;

	//allocate data
	data = fftw_alloc_real(grid_info.n_local_real_size);

	//return data
	return data;
}

/*! \fn fftw_complex *allocate_complex_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a 3-d complex grid for use with fftw.*/
fftw_complex *allocate_complex_fftw_grid(FFTW_Grid_Info grid_info)
{
	fftw_complex *cdata;

	//allocate data
	cdata = fftw_alloc_complex(grid_info.n_local_complex_size);

	//return data	
	return cdata;
}


/*! \fn double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info);
 *  \brief Allocates a field[ndim][n_local_real_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info)
{
	double **data;

	//allocate the field
	data = new double *[nd];

	//each field element is an fftw grid
	for(int i=0;i<nd;i++)
		data[i] = allocate_real_fftw_grid(grid_info);

	//return the field
	return data;
}

/*! \fn void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info)
 *  \brief De-allocates a field[ndim][n_local_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info)
{
	//free field elements
	for(int i=0;i<nd;i++)
		fftw_free(field[i]);
	//free field pointer
	delete field;
}


/*! \fn double ***allocate_tensor_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a tensor[ndim][ndim][n_local_real_size] (of dimension ndim*ndim) of  3-d real grids for use with fftw.*/
double ***allocate_tensor_fftw_grid(FFTW_Grid_Info grid_info)
{

	//tensor data
	double ***data;

	//allocate the tensor
	data = new double **[grid_info.ndim];

	for(int i=0;i<grid_info.ndim;i++)
	{
		data[i] = new double *[grid_info.ndim];
		for(int j=0;j<grid_info.ndim;j++)
			data[i][j] = allocate_real_fftw_grid(grid_info);
	}

	//return the tensor
	return data;
}



/*! \fn void deallocate_tensor_fftw_grid(double ***tensor, FFTW_Grid_Info grid_info)
 *  \brief De-allocates a tensor[ndim][n_local_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
void deallocate_tensor_fftw_grid(double ***tensor, FFTW_Grid_Info grid_info)
{
	//free tensor elements
	for(int i=0;i<grid_info.ndim;i++)
	{
		for(int j=0;j<grid_info.ndim;j++)
			free(tensor[i][j]);
		delete[] tensor[i];
	}
	//free tensor pointer
	delete tensor;
}



/*! \fn void inverse_transform_fftw_grid(double *work, fftw_complex *cdata, FFTW_Grid_Info grid_info)
 *  \brief Takes a forward transform, already re-normalized by 1./(nx*ny*nz), and returns a properly normalized inverse transform */
void inverse_transform_fftw_grid(double *work, fftw_complex *cwork, FFTW_Grid_Info grid_info)
{
	//do inverse transform
	fftw_execute(grid_info.iplan);
}


/*! \fn void forward_transform_fftw_grid(double *work, fftw_complex *cwork, FFTW_Grid_Info grid_info)
 *  \brief Produces a forward transform normalized by 1./(nx*ny*nz). */
void forward_transform_fftw_grid(double *work, fftw_complex *cwork, FFTW_Grid_Info grid_info)
{
	//grid index
	int ijk;

	//normalization
	double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

	//the fftw plan is contained within grid_info

	//perform forward fftw
	fftw_execute(grid_info.plan);

	//normalize transform
	for(int i=0;i<grid_info.nx_local;++i)
		for(int j=0;j<grid_info.ny;++j)
			for(int k=0;k<grid_info.nz_complex;++k)
			{
				//find i,j,k element
				ijk = grid_complex_ijk(i,j,k,grid_info);

				//rescale the complex data
				cwork[ijk][0]*=scale;
				cwork[ijk][1]*=scale;
			}
}


void output_fft_grid(char *output_fname, double *data, FFTW_Grid_Info grid_info, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int myid, int numprocs, MPI_Comm world)
{

	//in this version, all processors send their data to the root process, who then outputs the data

	FILE *fp;


	int     ijk;
	int     ijk_out;
	double *xout;

	int  initial_flag  = 0;
	int  total_initial = 0;


	int  error_flag  = 0;


	int  nx_max;
	int  nx_min;

	int  ny_max;
	int  ny_min;

	int  nz_max;
	int  nz_min;


	int nx_out;
	int ny_out;
	int nz_out;

	int yes_flag = 0;

	//copy info from grid_info

	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int n_local_size = grid_info.n_local_real_size;
	int nx_local = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;

	//some simple array boundary checks

	if(ixmax>nx)
		ixmax = nx;
	if(iymax>ny)
		iymax = ny;
	if(izmax>nz)
		izmax = nz;

	if(ixmin<0)
		ixmin = 0;
	if(iymin<0)
		iymin = 0;
	if(izmin<0)
		izmin = 0;


	//restrict the output to the data available on each process

	nx_min = ixmin;
	nx_max = ixmax;

	if(nx_min<nx_local_start)
	{
		nx_min  = nx_local_start;
	}

	if((nx_max-nx_local_start)>nx_local)
	{
		nx_max = nx_local + nx_local_start;
	}
	

	//each process should have the complete y and z range

	ny_min = iymin;
	ny_max = iymax;

	nz_min = izmin;
	nz_max = izmax;


	nx_out = nx_max - nx_min;
	ny_out = ny_max - ny_min;
	nz_out = nz_max - nz_min;

	if( (ixmin<nx_local_start+nx_local)&&(ixmax>=nx_local_start) )
		yes_flag = 1;



	//loop over number of processors
	for(int ip=0;ip<numprocs;ip++)
	{
		MPI_Barrier(world);
		//check to see if the output strides over this process

		if( (ixmin<nx_local_start+nx_local)&&(ixmax>=nx_local_start)&&(ip==myid) )
		{

			if(!initial_flag)
			{

				//printf("processor %d initiated for file %s\n",myid,output_fname);
				//fflush(stdout);
				//this processor is the first 
				//to write to the file
	
				initial_flag = 1;

				//open a new file

				if(!(fp = fopen(output_fname,"w")))
				{
					printf("Error opening %s by process %d\n",output_fname,myid);
					fflush(stdout);

					error_flag = 1;
				}else{
					//printf("File %s opened by %d.\n",output_fname,myid);
					//printf("nx %d ny %d nz %d\n",nx,ny,nz);
					//printf("ixmin %d ixmax %d\n",ixmin,ixmax);
					//printf("iymin %d iymax %d\n",iymin,iymax);
					//printf("izmin %d izmax %d\n",izmin,izmax);
					//printf("nxmin %d nxmax %d\n",nx_min,nx_max);
					//printf("nymin %d nymax %d\n",ny_min,ny_max);
					//printf("nzmin %d nzmax %d\n",nz_min,nz_max);
					//printf("nxout %d nyout %d nzyout %d\n",nx_out,ny_out,nz_out);
					//fflush(stdout);
				}


				//the data file has opened correctly, so continue

				if(!error_flag)
				{

					//write the grid dimensions

					fwrite(&nx,1,sizeof(int),fp);
					fwrite(&ny,1,sizeof(int),fp);
					fwrite(&nz,1,sizeof(int),fp);

					//write the restricted grid dimensions

					fwrite(&ixmin,1,sizeof(int),fp);
					fwrite(&ixmax,1,sizeof(int),fp);
					fwrite(&iymin,1,sizeof(int),fp);
					fwrite(&iymax,1,sizeof(int),fp);
					fwrite(&izmin,1,sizeof(int),fp);
					fwrite(&izmax,1,sizeof(int),fp);

					//write this process's data to file


					//allocate the output buffer
					if(!(xout = (double *) malloc(nx_out*ny_out*nz_out*sizeof(double))))
					{
						printf("Error allocating output array xout on process %d (nx_out %d ny_out %d nz_out %d).\n",myid,nx_out,ny_out,nz_out);
						fflush(stdout);
						error_flag = 1;
					}
	
					if(!error_flag)
					{
						for(int i=nx_min;i<nx_max;++i)
							for(int j=ny_min;j<ny_max;++j)
							{
								for(int k=nz_min;k<nz_max;++k)
								{
									ijk_out       = ( (i-nx_min)*ny_out + (j-ny_min) )*nz_out + (k-nz_min);
									if(grid_info.ndim==2)
									{
										ijk           = ( (i-nx_local_start) )*(2*(ny/2+1)) + j;
									}else{
										ijk           = ( (i-nx_local_start)*ny + j )*(2*(nz/2+1)) + k;
									}

									if(ijk_out>=(nx_out*ny_out*nz_out))
									{
										printf("error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk_out,nx_out*ny_out*nz_out);
										fflush(stdout);
									}

									if(ijk>=n_local_size)
									{
										printf("second error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk,nx_local*ny*nz);
										fflush(stdout);
									}
									xout[ijk_out] = data[ijk]; 
								}
							}

						//write the data

						fwrite(xout,nx_out*ny_out*nz_out,sizeof(double),fp);

						//close the data file
	
						fclose(fp);
	
						//free the output buffer

						free(xout);
					}
					
				}

			}else{

				//printf("processor %d continuing for file %s\n",myid,output_fname);
				//fflush(stdout);
				//here, we're not the first processsor to output our data
				//so the file should be appended, not created

				//open the file to append

				if(!(fp = fopen(output_fname,"a")))
				{
					printf("Error opening %s by process %d\n",output_fname,myid);
					fflush(stdout);

					error_flag = 1;
				}else{
					//printf("File %s appended by %d.\n",output_fname,myid);
					//printf("nx %d ny %d nz %d\n",nx,ny,nz);
					//printf("ixmin %d ixmax %d\n",ixmin,ixmax);
					//printf("iymin %d iymax %d\n",iymin,iymax);
					//printf("izmin %d izmax %d\n",izmin,izmax);
					//printf("nxmin %d nxmax %d\n",nx_min,nx_max);
					//printf("nymin %d nymax %d\n",ny_min,ny_max);
					//printf("nzmin %d nzmax %d\n",nz_min,nz_max);
					//printf("nxout %d nyout %d nzyout %d\n",nx_out,ny_out,nz_out);
					//fflush(stdout);
				}

				//the data file has opened correctly, so continue

				if(!error_flag)
				{

					//write this process's data to file

					nx_out = nx_max - nx_min;
					ny_out = ny_max - ny_min;
					nz_out = nz_max - nz_min;

					//allocate the output buffer
					if(!(xout = (double *) malloc(nx_out*ny_out*nz_out*sizeof(double))))
					{
						printf("Error allocating alt output array xout on process %d (nx %d ny %d nz %d).\n",myid,nx_out,ny_out,nz_out);
						fflush(stdout);
						error_flag = 1;
					}
	
					if(!error_flag)
					{
						for(int i=nx_min;i<nx_max;++i)
							for(int j=ny_min;j<ny_max;++j)
								for(int k=nz_min;k<nz_max;++k)
								{
									ijk_out       = ( (i-nx_min)*ny_out + (j-ny_min) )*nz_out + (k-nz_min);
									if(grid_info.ndim==2)
									{
										ijk           = ( (i-nx_local_start) )*(2*(ny/2+1)) + j;
									}else{
										ijk           = ( (i-nx_local_start)*ny + j )*(2*(nz/2+1)) + k;
									}

									if(ijk_out>=(nx_out*ny_out*nz_out))
									{
										printf("alt error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk_out,nx_out*ny_out*nz_out);
										fflush(stdout);
									}

									if(ijk>=n_local_size)
									{
										printf("alt second error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk,nx_local*ny*nz);
										fflush(stdout);
									}
									xout[ijk_out] = data[ijk]; 
								}


						//write the data

						fwrite(xout,nx_out*ny_out*nz_out,sizeof(double),fp);

						//close the data file
	
						fclose(fp);
	
						//free the output buffer

						free(xout);
					}

				}
			}
		}



		//Check to see if previous processors wrote some data
		//and created the output file

		MPI_Allreduce(&initial_flag,&total_initial,1,MPI_INT,MPI_SUM,world);
		initial_flag = total_initial;
		if(initial_flag>0)
		{
			initial_flag  = 1;
			total_initial = 1;
		}



		//Check for errors, and if there is an error abort

		//AllCheckError(error_flag,myid,numprocs,world);
		MPI_Barrier(world);
	}
}

/*
void convolve_complex_fftw_grid(fftw_complex *C_transposed, fftw_complex *A_transposed, fftw_complex *B_transposed, FFTW_Grid_Info grid_info, int local_ny_after_transpose, int nx, int ny, int nzi, int myid, int numprocs, MPI_Comm world)
{
	int ijk;

	//transform normalization
	//is handled in forward_transform_fftw_grid

	double scale = 1.;

	int nz  = nzi;
	int nzl = nz/2+1;
	if(grid_info.ndim==2)
	{
		nz=1;
		nzl=1;
	}
	for(int j=0;j<local_ny_after_transpose;++j)
		for(int i=0;i<nx;++i)
			for(int k=0;k<nzl;++k)
			{
				//ijk = (j*nx + i)*(nz/2+1) + k;

				ijk = grid_transpose_ijk(i,j,k,grid_info);
				C_transposed[ijk][0] = scale*(A_transposed[ijk][0]*B_transposed[ijk][0] - A_transposed[ijk][1]*B_transposed[ijk][1]);
				C_transposed[ijk][1] = scale*(A_transposed[ijk][0]*B_transposed[ijk][1] + A_transposed[ijk][1]*B_transposed[ijk][0]);
			}
}




void output_fft_grid_procs(char *output_fname, double *data, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int nx_local_start, int nx_local, int n_local_size, int myid, int numprocs, MPI_Comm world)
{

	//in this version
	//each process outputs its own data

	FILE *fp;


	int     ijk;
	int     ijk_out;
	double *xout;

	int  initial_flag  = 0;
	int  total_initial = 0;


	int  error_flag  = 0;


	int  nx_max;
	int  nx_min;

	int  ny_max;
	int  ny_min;

	int  nz_max;
	int  nz_min;


	int nx_out;
	int ny_out;
	int nz_out;

	int yes_flag = 0;

	//some simple array boundary checks

	if(ixmax>nx)
		ixmax = nx;
	if(iymax>ny)
		iymax = ny;
	if(izmax>nz)
		izmax = nz;

	if(ixmin<0)
		ixmin = 0;
	if(iymin<0)
		iymin = 0;
	if(izmin<0)
		izmin = 0;


	//restrict the output to the data available on each process

	nx_min = ixmin;
	nx_max = ixmax;

	//if(nx_min>=nx_local_start)
	if(nx_min<nx_local_start)
	{
		nx_min  = nx_local_start;
	}

	if((nx_max-nx_local_start)>nx_local)
	{
		//nx_max = nx_local;
		nx_max = nx_local + nx_local_start;
	}
	

	//each process should have the complete y and z range

	ny_min = iymin;
	ny_max = iymax;

	nz_min = izmin;
	nz_max = izmax;


	nx_out = nx_max - nx_min;
	ny_out = ny_max - ny_min;
	nz_out = nz_max - nz_min;

	if( (ixmin<nx_local_start+nx_local)&&(ixmax>=nx_local_start) )
		yes_flag = 1;

	if(myid==0)
		printf("testing output\n");
	fflush(stdout);

	for(int ip=0;ip<numprocs;ip++)
		if(myid==ip)
			printf("id %d nxmin %d nxmax %d nymin %d nymax %d nzmin %d nzmax %d ixmin %d ixmax %d iymin %d iymax %d izmin %d izmax %d nxout %d nyout %d nzout %d lxs %d lnx %d lxs+lnx %d yf %d\n",ip,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ixmin,ixmax,iymin,iymax,izmin,izmax,nx_out,ny_out,nz_out,nx_local_start,nx_local,nx_local_start+nx_local,yes_flag);


	//loop over number of processors
	for(int ip=0;ip<numprocs;ip++)
	{
		//check to see if the output strides over this process

		if( (ixmin<nx_local_start+nx_local)&&(ixmax>=nx_local_start)&&(ip==myid) )
		{

			if(!initial_flag)
			{

				//this processor is the first 
				//to write to the file
	
				initial_flag = 1;

				//open a new file

				if(!(fp = fopen(output_fname,"w")))
				{
					printf("Error opening %s by process %d\n",output_fname,myid);
					fflush(stdout);

					error_flag = 1;
				}else{
					//printf("File %s opened by %d.\n",output_fname,myid);
					//printf("nx %d ny %d nz %d\n",nx,ny,nz);
					//printf("ixmin %d ixmax %d\n",ixmin,ixmax);
					//printf("iymin %d iymax %d\n",iymin,iymax);
					//printf("izmin %d izmax %d\n",izmin,izmax);
					//printf("nxmin %d nxmax %d\n",nx_min,nx_max);
					//printf("nymin %d nymax %d\n",ny_min,ny_max);
					//printf("nzmin %d nzmax %d\n",nz_min,nz_max);
					//printf("nxout %d nyout %d nzyout %d\n",nx_out,ny_out,nz_out);
					//fflush(stdout);
				}


				//the data file has opened correctly, so continue

				if(!error_flag)
				{

					//write the grid dimensions

					fwrite(&nx,1,sizeof(int),fp);
					fwrite(&ny,1,sizeof(int),fp);
					fwrite(&nz,1,sizeof(int),fp);

					//write the restricted grid dimensions

					fwrite(&ixmin,1,sizeof(int),fp);
					fwrite(&ixmax,1,sizeof(int),fp);
					fwrite(&iymin,1,sizeof(int),fp);
					fwrite(&iymax,1,sizeof(int),fp);
					fwrite(&izmin,1,sizeof(int),fp);
					fwrite(&izmax,1,sizeof(int),fp);

					//write this process's data to file


					//allocate the output buffer
					if(!(xout = (double *) malloc(nx_out*ny_out*nz_out*sizeof(double))))
					{
						printf("Error allocating output array xout on process %d (nx_out %d ny_out %d nz_out %d).\n",myid,nx_out,ny_out,nz_out);
						fflush(stdout);
						error_flag = 1;
					}
	
					if(!error_flag)
					{
						for(int i=nx_min;i<nx_max;++i)
							for(int j=ny_min;j<ny_max;++j)
							{
								for(int k=nz_min;k<nz_max;++k)
								{
									ijk_out       = ( (i-nx_min)*ny_out + (j-ny_min) )*nz_out + (k-nz_min);
									ijk           = ( (i-nx_local_start)*ny + j )*(2*(nz/2+1)) + k;

									if(ijk_out>=(nx_out*ny_out*nz_out))
									{
										printf("error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk_out,nx_out*ny_out*nz_out);
										fflush(stdout);
									}

									if(ijk>=n_local_size)
									{
										printf("second error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk,nx_local*ny*nz);
										fflush(stdout);
									}
									xout[ijk_out] = data[ijk]; 
								}
							}

						//write the data

						fwrite(xout,nx_out*ny_out*nz_out,sizeof(double),fp);

						//close the data file
	
						fclose(fp);
	
						//free the output buffer

						free(xout);
					}
					
				}

			}else{

				//here, we're not the first processsor to output our data
				//so the file should be appended, not created

				//open the file to append

				if(!(fp = fopen(output_fname,"a")))
				{
					printf("Error opening %s by process %d\n",output_fname,myid);
					fflush(stdout);

					error_flag = 1;
				}else{
					//printf("File %s appended by %d.\n",output_fname,myid);
					//printf("nx %d ny %d nz %d\n",nx,ny,nz);
					//printf("ixmin %d ixmax %d\n",ixmin,ixmax);
					//printf("iymin %d iymax %d\n",iymin,iymax);
					//printf("izmin %d izmax %d\n",izmin,izmax);
					//printf("nxmin %d nxmax %d\n",nx_min,nx_max);
					//printf("nymin %d nymax %d\n",ny_min,ny_max);
					//printf("nzmin %d nzmax %d\n",nz_min,nz_max);
					//printf("nxout %d nyout %d nzyout %d\n",nx_out,ny_out,nz_out);
					//fflush(stdout);
				}

				//the data file has opened correctly, so continue

				if(!error_flag)
				{

					//write this process's data to file

					nx_out = nx_max - nx_min;
					ny_out = ny_max - ny_min;
					nz_out = nz_max - nz_min;

					//allocate the output buffer
					if(!(xout = (double *) malloc(nx_out*ny_out*nz_out*sizeof(double))))
					{
						printf("Error allocating alt output array xout on process %d (nx %d ny %d nz %d).\n",myid,nx_out,ny_out,nz_out);
						fflush(stdout);
						error_flag = 1;
					}
	
					if(!error_flag)
					{
						for(int i=nx_min;i<nx_max;++i)
							for(int j=ny_min;j<ny_max;++j)
								for(int k=nz_min;k<nz_max;++k)
								{
									ijk_out       = ( (i-nx_min)*ny_out + (j-ny_min) )*nz_out + (k-nz_min);

									ijk           = ( (i-nx_local_start)*ny + j )*(2*(nz/2+1)) + k;

									if(ijk_out>=(nx_out*ny_out*nz_out))
									{
										printf("alt error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk_out,nx_out*ny_out*nz_out);
										fflush(stdout);
									}

									if(ijk>=n_local_size)
									{
										printf("alt second error here i %d j %d k %d nx_min %d nx_max %d ny_min %d ny_max %d nz_min %d nz_max %d ijk %d max %d\n",i,j,k,nx_min,nx_max,ny_min,ny_max,nz_min,nz_max,ijk,nx_local*ny*nz);
										fflush(stdout);
									}
									xout[ijk_out] = data[ijk]; 
								}


						//write the data

						fwrite(xout,nx_out*ny_out*nz_out,sizeof(double),fp);

						//close the data file
	
						fclose(fp);
	
						//free the output buffer

						free(xout);
					}

				}
			}
		}



		//Check to see if previous processors wrote some data
		//and created the output file

		MPI_Allreduce(&initial_flag,&total_initial,1,MPI_INT,MPI_SUM,world);
		initial_flag = total_initial;



		//Check for errors, and if there is an error abort

		AllCheckError(error_flag,myid,numprocs,world);
	}
}

void output_fft_grid_complex(char *output_fname, fftw_complex *data, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int local_y_start_after_transpose, int local_ny_after_transpose, int n_local_size, int myid, int numprocs, MPI_Comm world)
{


	//output the complex data
	//but remember that the data is transposed
	//and that the slabs are split across the
	//y dimension


	// Also remember that we performed
	// a transform of real data.
	// For real data, F(-k) = F(k)*
	// and the nz/2 element is the
	// same as the -nz/2 element


	// the format of the gridded complex
	// data will be
	//
	// nx 
	// {
	// re(ny,nz)
	// im(ny,nz)
	// }

	// the duplicative information in
	// the z-direction *will be included*


	// and note that for simplicity
	// izmin = 0;
	// izmax = nz;
	// always

	FILE *fp;

	MPI_Status status;

	int     jk;
	int     jk_out;
	int     jtot;

	double *xout_imaginary, *xout_real;

	double *y_block_imaginary, *y_block_real;

	int  initial_flag  = 0;
	int  total_initial = 0;


	int  error_flag  = 0;


	int  nx_max;
	int  nx_min;

	int  ny_max;
	int  ny_min;

	int  nz_max;
	int  nz_min;

	int nx_out;
	int ny_out;
	int nz_out;

	int *all_local_ny, *all_local_y_start;

	int yes_flag = 0;

	//some simple array boundary checks

	if(ixmax>nx)
		ixmax = nx;
	if(iymax>ny)
		iymax = ny;

	if(ixmin<0)
		ixmin = 0;
	if(iymin<0)
		iymin = 0;


	//enforce full z coverage
	izmin = 0;
	izmax = nz;

	//each process should have the complete x and z range

	//we will pretend the whole y range is available

	ny_min = iymin;
	ny_max = iymax;

	nx_min = ixmin;
	nx_max = ixmax;

	nx_out = nx_max - nx_min;
	ny_out = ny_max - ny_min;



	//The data is transposed and is complex, so the output method differs from that for real data:


	//First, the root process opens a new file and outputs the essential 
	//info about the data



	if(myid==0)
	{
		//open a new file

		if(!(fp = fopen(output_fname,"w")))
		{
			printf("Error opening %s by process %d\n",output_fname,myid);
			fflush(stdout);

			error_flag = 1;
		}

		if(!error_flag)
		{

			//the data file has opened correctly, so continue

			//write the grid dimensions

			fwrite(&nx,1,sizeof(int),fp);
			fwrite(&ny,1,sizeof(int),fp);
			fwrite(&nz,1,sizeof(int),fp);

			//write the restricted grid dimensions

			fwrite(&ixmin,1,sizeof(int),fp);
			fwrite(&ixmax,1,sizeof(int),fp);
			fwrite(&iymin,1,sizeof(int),fp);
			fwrite(&iymax,1,sizeof(int),fp);
			fwrite(&izmin,1,sizeof(int),fp);
			fwrite(&izmax,1,sizeof(int),fp);


			fclose(fp);
		}
	}


	//Check for errors, and if there is an error abort

	AllCheckError(error_flag,myid,numprocs,world);


	//The data is transposed, so to output
	//in row-major format with the x-direction 
	//as primary, we're gonna do some funky outputing


	//First, figure out the y-direction extent of each
	//processor and share that with the other processes


	//allocate arrays to hold local y extent info


	if(!(all_local_ny = (int *) malloc(numprocs*sizeof(int))))
	{
		printf("Error allocating all_local_y on process %d.\n",myid);
		error_flag = 1;
	}

	if(!(all_local_y_start = (int *) malloc(numprocs*sizeof(int))))
	{
		printf("Error allocating all_local_y_start on process %d.\n",myid);
		error_flag = 1;
	}

	//Check for errors, and if there is an error abort

	AllCheckError(error_flag,myid,numprocs,world);

	//Share local y extent info


	MPI_Allgather(&local_ny_after_transpose,1,MPI_INT,all_local_ny,1,MPI_INT,world);	

	MPI_Allgather(&local_y_start_after_transpose,1,MPI_INT,all_local_y_start,1,MPI_INT,world);	


	if(myid==0)
	{
		for(int i=0;i<numprocs;i++)
			printf("proc %d lny %d lys %d\n",i,all_local_ny[i],all_local_y_start[i]);

		fflush(stdout);
	}

	//Check for errors, and if there is an error abort

	AllCheckError(error_flag,myid,numprocs,world);



	//Second, begin a loop after the x-direction.

	//The root process will allocate a page of size ny_out*nz_out


	if(myid==0)
	{
		if(!(xout_real = (double *) malloc(ny_out*nz*sizeof(double))))
		{
			printf("Error allocating xout_real page on root process.\n");
			fflush(stdout);

			error_flag = 1;
		}
		if(!(xout_imaginary = (double *) malloc(ny_out*nz*sizeof(double))))
		{
			printf("Error allocating xout_imaginary page on root process.\n");
			fflush(stdout);

			error_flag = 1;
		}
	
	}

	//Check for errors, and if there is an error abort

	AllCheckError(error_flag,myid,numprocs,world);


	//begin the loop over the x-direction
	//for the real part of the complex data

	for(int i=nx_min;i<nx_max;++i)
	{

		jtot = ny_min;



		//begin the loop over the number of processes

		for(int ip=0;ip<numprocs;ip++)
		{

			//Check to see if this processor contributes to the y-direction
			//extent of the output.
			//
			//Otherwise, just skip it

			if( (iymin<all_local_y_start[ip]+all_local_ny[ip])&&(iymax>=all_local_y_start[ip]) )
			{
				//printf("i %d j %d ip %d processor %d entered\n",i,jtot,ip,myid);
				//fflush(stdout);

				//This processor contributes to the y-direction extent of the output

				if(ip==0)
				{

					if(ip==myid)
					{
						//it is time for the root process
						//to copy its y-direction info
						//into the xout page

						for(int j=0;(j<local_ny_after_transpose)&&(jtot<ny_max);++j)
						{
							//DC component

							jk_out            = ( (jtot-ny_min) )*nz;
							jk                = (j*nx + i)*(nz/2+1);
							xout_real[jk_out] = data[jk].re;


							for(int k=1;k<(nz/2+1);++k)
							{
								// do the real part of the positive frequencies first

								jk_out            = ( (jtot-ny_min) )*nz + k;
								jk                = (j*nx + i)*(nz/2+1)  + k;
								xout_real[jk_out] = data[jk].re;

								// do the real part of the negative frequencies second

								jk_out            = ( (jtot-ny_min) )*nz + (nz-k);
								jk                = ( (jtot-ny_min) )*nz + k;
								xout_real[jk_out] = xout_real[jk];

								// do the imaginary part of the positive frequencies third

								jk_out            = ( (jtot-ny_min) )*nz + k;
								jk                = (j*nx + i)*(nz/2+1)  + k;
								//jk                = (j*nx + i)*(nz/2+1)  + (nz-k);
								xout_imaginary[jk_out] = data[jk].im;

								// do the imaginary part of the negative frequencies fourth

								jk_out            = ( (jtot-ny_min) )*nz + (nz-k);
								jk                = ( (jtot-ny_min) )*nz + k;
								xout_imaginary[jk_out] = (-xout_imaginary[jk]);
							}

							if((nz%2)==0)
							{
								//nz is even

								jk_out            = ( (jtot-ny_min) )*nz + (nz/2);
								jk                = (j*nx + i)*(nz/2+1)  + (nz/2);
								xout_real[jk_out] = data[jk].re;
								xout_imaginary[jk_out] = 0;
							}
							jtot++;

						}

					}//end if(ip==myid)

				}else{

					//Some other processor contributes to the y-extent of the output data

					if((myid==0)||(ip==myid))
					{

						//first, we have to allocate a block for the y
						//data on the current process
	
						if(!(y_block_real = (double *) malloc(all_local_ny[ip]*nz*sizeof(double))))
						{
							printf("Error allocating y_block_real on process %d.\n",myid);
							fflush(stdout);	

							error_flag = 1;
						
						}

						if(!(y_block_imaginary = (double *) malloc(all_local_ny[ip]*nz*sizeof(double))))
						{
							printf("Error allocating y_block_imaginary on process %d.\n",myid);
							fflush(stdout);	

							error_flag = 1;
						
						}
					}

					//Check for errors, and if there is an error abort
		
					AllCheckError(error_flag,myid,numprocs,world);
				
					//printf("i %d j %d ip %d proc %d about to exchange data\n",i,jtot,ip,myid);
					//fflush(stdout);

					if(myid==0)
					{
						//printf("i %d j %d ip %d root waiting for exchange data\n",i,jtot,ip);
						//fflush(stdout);

						//Receive the data from the active process
						MPI_Recv(y_block_real,all_local_ny[ip]*nz,MPI_DOUBLE,ip,ip,world,&status);
						MPI_Recv(y_block_imaginary,all_local_ny[ip]*nz,MPI_DOUBLE,ip,ip,world,&status);
						
						//copy the data from processor ip to the output buffers

						for(int j=0;(j<all_local_ny[ip])&&(jtot<ny_max);++j)
						{
							for(int k=0;k<nz;++k)
							{
								// simply copy the blocks

								jk_out            = ( (jtot-ny_min) )*nz + k;
								jk                = ( (jtot-all_local_y_start[ip]) )*nz + k;

								xout_real[jk_out]      = y_block_real[jk];
								xout_imaginary[jk_out] = y_block_imaginary[jk];

							}

							jtot++;

						}

	
						//free y_blocks
						free(y_block_real);
						free(y_block_imaginary);
					}else{

						//Check to see if this is the contributing process

						//printf("i %d j %d ip %d proc %d exchange data\n",i,jtot,ip,myid);
						//fflush(stdout);

						if(ip==myid)
						{
							//copy the data to a buffer to send to the root process for output

							for(int j=0;j<all_local_ny[ip];++j)
							{

								//DC Component	
								jk_out               = ( j*nz );
								jk                   = ( j*nx + i)*(nz/2+1);
								y_block_real[jk_out]      = data[jk].re;
								y_block_imaginary[jk_out] = 0;

								for(int k=1;k<(nz/2+1);++k)
								{
									
									// do the real part of the positive frequencies first

									jk_out               = ( j*nz + k );
									jk                   = (j*nx + i)*(nz/2+1) + k;
									y_block_real[jk_out] = data[jk].re;

									// do the real part of the negative frequencies second
									// note this is the complex conjugate

									jk_out               = ( j*nz + (nz-k) );
									jk                   = (j*nx + i)*(nz/2+1) + k;
									y_block_real[jk_out] = data[jk].re;

									// do the imaginary part of the positive frequencies third
									jk_out               = ( j*nz + k );
									jk                   = (j*nx + i)*(nz/2+1) + k;
									y_block_imaginary[jk_out] = data[jk].im;

									// do the imaginary part of the negative frequencies fourth
									// note this is the complex conjugate

									jk_out               = ( j*nz + (nz-k) );
									jk                   = (j*nx + i)*(nz/2+1) + k;
									y_block_imaginary[jk_out] = -(data[jk].im);
								}
								if((nz%2)==0)
								{
									jk_out               = ( j*nz + nz/2);
									jk                   = ( j*nx + i)*(nz/2+1) + nz/2;
									y_block_real[jk_out]      = data[jk].re;
									y_block_imaginary[jk_out] = data[jk].im;
								}

							}

							//printf("i %d j %d ip %d about to send exchange data\n",i,jtot,ip);
							//fflush(stdout);
				
							//Send the data to the root process
							MPI_Send(y_block_real,all_local_ny[ip]*nz,MPI_DOUBLE,0,ip,world);
							MPI_Send(y_block_imaginary,all_local_ny[ip]*nz,MPI_DOUBLE,0,ip,world);
	
							//free y_blocks
							free(y_block_real);
							free(y_block_imaginary);

						}//end if(ip==myid)

					}//end if(imyid==0)

				}//end if((ip==0)&&(ip==myid))

			}//end yes_flag

			//Check for errors, and if there is an error abort

			AllCheckError(error_flag,myid,numprocs,world);


			//Share jtot
			MPI_Bcast(&jtot,1,MPI_DOUBLE,0,world);

			
		} //end loop over nproc

	
		if(myid==0)
		{	
			//append the real part of the data to the data file

			//printf("i %d outputting data\n");	
			//fflush(stdout);

			if(!(fp = fopen(output_fname,"a")))
			{
				printf("Error opening %s by process %d\n",output_fname,myid);
				fflush(stdout);

				error_flag = 1;
			}

			if(!error_flag)
			{
				fwrite(xout_real,ny_out*nz,sizeof(double),fp);
				fwrite(xout_imaginary,ny_out*nz,sizeof(double),fp);

				fclose(fp);
			}
		}

		//Check for errors, and if there is an error abort
		AllCheckError(error_flag,myid,numprocs,world);

	} //end loop over x



	//free some data

	free(all_local_ny);
	free(all_local_y_start);

	//free the grid pages
	if(myid==0)
	{
		free(xout_real);
		free(xout_imaginary);
	}

	//done!
}


void grid_particle_data(int npart, float *pos, double *data, int nx_local_start, int nx_local, int n_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world)
{
	double BS = BoxSize*1000.0; //in kpc/h

#ifdef   CLOUD_IN_CELL
	grid_particle_data_cloud_in_cell( npart, pos, data, nx_local_start, nx_local, n_local_size, nx, ny, nz, npart_total, BS, myid, numprocs, world);
#else  //CLOUD_IN_CELL
	grid_particle_data_nearest_grid(  npart, pos, data, nx_local_start, nx_local, n_local_size, nx, ny, nz, npart_total, BS, myid, numprocs, world);
#endif //CLOUD_IN_CELL
}

void grid_particle_data_cloud_in_cell(int npart, float *pos, double *data, int nx_local_start, int nx_local, int n_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world)
{
	//grid particle data using the cloud in cell method

	//need to adjust CIC to deal with different grid sizes

	int error_flag  = 0;
	int i,j,k;

	int ic, jc, kc;

	int ijk;

	double xc, yc, zc;

	double xp, yp, zp;

	double dx;
	double dy;
	double dz;

	double tx;
	double ty;
	double tz;

	double value;

	int u_dest   = myid+1;
	int u_source = myid-1;
	int l_dest   = myid-1;
	int l_source = myid+1;

	char variable_name[200];

	//this buffer allows for imperfect CIC
	//assignments for varying grid cells

	int nxb = 10;
	int sbuf_size = nxb*ny*(2*(nz/2+1));

	double *x_u, *x_ur;
	double *x_l, *x_lr;

	double x_u_count = 0;
	double x_l_count = 0;

	MPI_Request requests[4];

	MPI_Status statuses[4];



	//wrap destinations and sources

	if(u_dest>=numprocs)
		u_dest-=numprocs;
	if(u_source<0)
		u_source+=numprocs;

	if(l_dest<0)
		l_dest+=numprocs;
	if(l_source>=numprocs)
		l_source-=numprocs;

	//first, initialize the grid to zero
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
				data[  (i*ny + j) * (2*(nz/2+1)) + k] = 0.0;


	//we need to allocate the x-direction grid buffers

	if(myid==0)
	{
		printf("\n");
		fflush(stdout);
	}

	// grid for densities interpolated on grid slab before this slab

	sprintf(variable_name,"x_l");
	//x_l  = allocate_real_fftw_grid(nxb*ny*(2*(nz/2+1)), "x_l",  myid, numprocs, world,0);
	x_l  = allocate_real_fftw_grid(nxb*ny*(2*(nz/2+1)), variable_name,  myid, numprocs, world,0);
	//x_lr = allocate_real_fftw_grid(nxb*ny*(2*(nz/2+1)), "x_lr", myid, numprocs, world,0);
	sprintf(variable_name,"x_lr");
	x_lr = allocate_real_fftw_grid(nxb*ny*(2*(nz/2+1)), variable_name, myid, numprocs, world,0);

	// grid for densities interpolated on grid slab after this slab

	sprintf(variable_name,"x_u");
	x_u  = allocate_real_fftw_grid(nxb*ny*(2*(nz/2+1)), variable_name,  myid, numprocs, world,0);
	sprintf(variable_name,"x_ur");
	x_ur = allocate_real_fftw_grid(nxb*ny*(2*(nz/2+1)), variable_name, myid, numprocs, world,0);


	for(i=0;i<nxb;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				x_u[   (i*ny + j) * (2*(nz/2+1)) + k] = 0.0;
				x_ur[  (i*ny + j) * (2*(nz/2+1)) + k] = 0.0;
				x_l[   (i*ny + j) * (2*(nz/2+1)) + k] = 0.0;
				x_lr[  (i*ny + j) * (2*(nz/2+1)) + k] = 0.0;
			}

			
	for(int ip=0;ip<npart;ip++)
	{
		xp = ((double) pos[3*ip + 0])*((double) nx)/BoxSize;
		yp = ((double) pos[3*ip + 1])*((double) ny)/BoxSize;
		zp = ((double) pos[3*ip + 2])*((double) nz)/BoxSize;

		i = iifloor(xp);
		j = iifloor(yp);
		k = iifloor(zp);


		//check to see if this particle is actually located
		//in this slab
		
		if( (i>=nx_local_start)&&(i<(nx_local_start+nx_local)) )
		{
			wrap_particle(&i,&j,&k,nx,ny,nz,&xp,&yp,&zp);


			// cell centers

			xc = ((double) i);
			yc = ((double) j);
			zc = ((double) k);

			dx = xp - xc;

			dy = yp - yc;

			dz = zp - zc;

			tx = 1 - dx;

			ty = 1 - dy;

			tz = 1 - dz;

	
			//first do cell containing particle

			ic = i-nx_local_start;
			jc = j;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error A on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value      = tx*ty*tz;

			if(fabs(value)>=3.0)
			{
				printf("Error AA on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d tx %e ty %e tz %e value %e.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size,tx,ty,tz,value);
				fflush(stdout);
				error_flag = 1;
			}

			data[ijk] += value;

	
			//do i,j+1,k

			jc = j+1;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error B on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value      = tx*dy*tz;

			if(fabs(value)>=3.0)
			{
				printf("Error AB on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d tx %e dy %e tz %e value %e.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size,tx,dy,tz,value);
				fflush(stdout);
				error_flag = 1;
			}

			data[ijk] += value;

	
			//do i,j,k+1

			jc = j;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error C on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value      = tx*ty*dz;

			if(fabs(value)>=3.0)
			{
				printf("Error AC on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d tx %e dy %e tz %e value %e.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size,tx,dy,tz,value);
		 		fflush(stdout);
				error_flag = 1;
			}

			data[ijk] += value;


			//do i,j+1,k+1

			jc = j+1;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error D on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value      = tx*dy*dz;

			if(fabs(value)>=3.0)
			{
				printf("Error AD on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d tx %e dy %e dz %e value %e.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size,tx,dy,dz,value);
				fflush(stdout);
				error_flag = 1;
			}

			data[ijk] += value;


			if(i+1<nx_local_start+nx_local)
			{
				//particle is completely on this processor

				//do i+1,j,k

				ic = i+1-nx_local_start;
				jc = j;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error E on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*ty*tz;
				data[ijk] += value;


				//do i+1,j+1,k

				jc = j+1;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error F on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*dy*tz;
				data[ijk] += value;

				//do i+1,j,k+1

				jc = j;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error G on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*ty*dz;
				data[ijk] += value;


				//do i+1,j+1,k+1

				jc = j+1;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error H on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*dy*dz;
				data[ijk] += value;


			}else{
				//particle is shared between processors
				//so add it to the x_u grid

				//do i+1,j,k

				ic = i+1-nx_local_start-nx_local;
				jc = j;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				value     = dx*ty*tz;
				x_u[ijk] += value;


				//do i+1,j+1,k

				jc = j+1;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				value     = dx*dy*tz;
				x_u[ijk] += value;


				//do i+1,j,k+1

				jc = j;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				value     = dx*ty*dz;
				x_u[ijk] += value;
	

				//do i+1,j+1,k+1

				jc = j+1;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				value     = dx*dy*dz;
				x_u[ijk] += value;
			}
	
		}else{


			//OK this particle isn't actually on the slab
			//so we need to place it in one of the grid buffers

			//but, if it's i coordinate is nx_local_start-1, then it will still be partially on
			//this slab.  So check for that case first

			wrap_particle(&i,&j,&k,nx,ny,nz,&xp,&yp,&zp);


			// cell centers

			xc = ((double) i);
			yc = ((double) j);
			zc = ((double) k);

			dx = xp - xc;

			dy = yp - yc;

			dz = zp - zc;

			tx = 1 - dx;

			ty = 1 - dy;

			tz = 1 - dz;

			if(i==(nx_local_start-1))
			{
				//particle is split between the previous slab and this slab

				//do i,j,k, which are on the previous slab

				ic = nxb-1;
				jc = j;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=sbuf_size)
				{
					printf("Error BA on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
					fflush(stdout);
					error_flag = 1;
				}

				value     = tx*ty*tz;
				x_l[ijk] += value;


				//do i,j+1,k

				jc = j+1;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=sbuf_size)
				{
					printf("Error BB on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
					fflush(stdout);
					error_flag = 1;
				}

				value     = tx*dy*tz;
				x_l[ijk] += value;

				//do i,j,k+1

				jc = j;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=sbuf_size)
				{
					printf("Error BC on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
					fflush(stdout);
					error_flag = 1;
				}

				value     = tx*ty*dz;
				x_l[ijk] += value;


				//do i,j+1,k+1

				jc = j+1;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=sbuf_size)
				{
					printf("Error BD on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
					fflush(stdout);
					error_flag = 1;
				}

				value     = tx*dy*dz;
				x_l[ijk] += value;

				//do the i+1,j,k values, which are on this slab


				ic = 0;
				jc = j;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error BE on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*ty*tz;
				data[ijk] += value;

				//do i+1,j+1,k

				jc = j+1;
				kc = k;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error BF on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*dy*tz;
				data[ijk] += value;

				//do i+1,j,k+1

				jc = j;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error BG on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*ty*dz;
				data[ijk] += value;


				//do i+1,j+1,k+1

				jc = j+1;
				kc = k+1;

				wrap_indices(&ic,&jc,&kc,nx,ny,nz);

				ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

				if(ijk>=n_local_size)
				{
					printf("Error BH on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
					fflush(stdout);
					error_flag = 1;
				}

				value      = dx*dy*dz;
				data[ijk] += value;


			}else{
				//particle is completely on another slab 


				//if the particle is at i>=nx_local_start+nx_local	
				//then it belongs in x_u


				if(i>=(nx_local_start+nx_local))	
				{
					//particle belongs in x_u


					//do i,j,k

					ic = i-nx_local-nx_local_start;
					jc = j;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CA on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d BS %e.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size,BoxSize);
						printf("nx %d ny %d nz %d local_x %d nx_local_start %d\n",nx,ny,nz,nx_local, nx_local_start);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*ty*tz;
					x_u[ijk] += value;


					//do i,j+1,k

					jc = j+1;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CB on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*dy*tz;
					x_u[ijk] += value;

					//do i,j,k+1

					jc = j;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CC on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*ty*dz;
					x_u[ijk] += value;


					//do i,j+1,k+1

					jc = j+1;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CD on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*dy*dz;
					x_u[ijk] += value;


					//do i+1,j,k

					ic = i+1-nx_local-nx_local_start;
					jc = j;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CE on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d i %d j %d k %d lxs %d nx %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size,i,j,k,nx_local_start,nx_local);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*ty*tz;
					x_u[ijk] += value;


					//do i,j+1,k

					jc = j+1;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CF on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*dy*tz;
					x_u[ijk] += value;

					//do i,j,k+1

					jc = j;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CG on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*ty*dz;
					x_u[ijk] += value;


					//do i,j+1,k+1

					jc = j+1;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error CH on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*dy*dz;
					x_u[ijk] += value;


				}else{
					//particle belongs in x_l


					//do i,j,k

					ic = nxb+(i-nx_local_start);
					jc = j;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DA on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*ty*tz;
					x_l[ijk] += value;

					//do i,j+1,k

					jc = j+1;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DB on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*dy*tz;
					x_l[ijk] += value;

					//do i,j,k+1

					jc = j;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DC on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*ty*dz;
					x_l[ijk] += value;


					//do i,j+1,k+1

					jc = j+1;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DD on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = tx*dy*dz;
					x_l[ijk] += value;


					//do i+1,j,k

					ic = nxb+1+(i-nx_local_start);
					jc = j;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DE on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*ty*tz;
					x_l[ijk] += value;


					//do i,j+1,k

					jc = j+1;
					kc = k;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DF on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*dy*tz;
					x_l[ijk] += value;

					//do i,j,k+1

					jc = j;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DG on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*ty*dz;
					x_l[ijk] += value;


					//do i,j+1,k+1

					jc = j+1;
					kc = k+1;

					wrap_indices(&ic,&jc,&kc,nx,ny,nz);

					ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

					if(ijk>=sbuf_size)
					{
						printf("Error DH on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nxb,ijk,sbuf_size);
						fflush(stdout);
						error_flag = 1;
					}

					value     = dx*dy*dz;
					x_l[ijk] += value;
				}
			}
		}
	}

	MPI_Isend(x_u,  nxb*ny*(2*(nz/2+1))*sizeof(double), MPI_BYTE, u_dest,   myid,     world, &requests[0]);

	MPI_Irecv(x_ur, nxb*ny*(2*(nz/2+1))*sizeof(double), MPI_BYTE, u_source, u_source, world, &requests[1]);

	MPI_Isend(x_l,  nxb*ny*(2*(nz/2+1))*sizeof(double), MPI_BYTE, l_dest,   myid,     world, &requests[2]);

	MPI_Irecv(x_lr, nxb*ny*(2*(nz/2+1))*sizeof(double), MPI_BYTE, l_source, l_source, world, &requests[3]);

	MPI_Waitall(4, &requests[0], &statuses[0]);

	MPI_Barrier(world);

	if(myid==0)
	{
		printf("Done with grid overlap buffer exchange.\n");
		fflush(stdout);
	}


	//add exchanged particles to grid

	//do x_ur

	for(i=0;i<nxb;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
				data[  (i*ny + j) * (2*(nz/2+1)) + k] += x_ur[  (i*ny + j) * (2*(nz/2+1)) + k];

	//do x_lr

	for(i=0;i<nxb;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
				data[  ((nx_local-1-(nxb-1-i))*ny + j) * (2*(nz/2+1)) + k] += x_lr[  (i*ny + j) * (2*(nz/2+1)) + k];


	//Check for errors

	AllCheckError(error_flag,myid,numprocs,world);

	//done, so free x_l and x_u

	free(x_l);
	free(x_lr);
	free(x_u);
	free(x_ur);
}
double *interpolate_grid_data_cloud_in_cell_conditional(int npart, double *pos, double *data, double *condition, int nx_local_start, int nx_local, int n_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world)
{

//#define SIGMA_CORRECTION

	//grid particle data using the cloud in cell method

	//need to adjust CIC to deal with different grid sizes

	int error_flag  = 0;
	int i,j,k;

	int ic, jc, kc;

	int ijk;

	double xc, yc, zc;

	double xp, yp, zp;

	double dx;
	double dy;
	double dz;

	double tx;
	double ty;
	double tz;

	double value;

	double *answer; //array containing interpolated values
	double *total_answer; //array containing interpolated values


	double *count; //array containing interpolated values
	double *total_count; //array containing interpolated values

	int u_dest   = myid+1;
	int u_source = myid-1;
	int l_dest   = myid-1;
	int l_source = myid+1;

	int yes_flag;

	char variable_name[200];

	//this buffer allows for imperfect CIC
	//assignments for varying grid cells

	int nxb = 10;
	int sbuf_size = nxb*ny*(2*(nz/2+1));

	MPI_Request requests[4];

	MPI_Status statuses[4];

#ifdef SIGMA_CORRECTION
	
	double sigma_correction;
	double threshold = 99.0;

#endif


	//wrap destinations and sources

	if(u_dest>=numprocs)
		u_dest-=numprocs;
	if(u_source<0)
		u_source+=numprocs;

	if(l_dest<0)
		l_dest+=numprocs;
	if(l_source>=numprocs)
		l_source-=numprocs;

	//we need to allocate the x-direction grid buffers

	if(myid==0)
	{
		printf("\n");
		fflush(stdout);
	}

	//allocate interpolated values

	sprintf(variable_name,"answer");
	answer       = allocate_double_array(npart, variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"total_answer");
	total_answer = allocate_double_array(npart, variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"count");
	count = allocate_double_array(npart, variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"total_count");
	total_count = allocate_double_array(npart, variable_name, myid, numprocs, world, 0);


	//initialize interpolated values
	for(i=0;i<npart;++i)
	{
		answer[i] = 0;
		total_answer[i] = 0;
	}


	//loop over the number of particles
	//and interpolate
			
	for(int ip=0;ip<npart;ip++)
	{
		xp = ((double) pos[3*ip + 0])*((double) nx)/BoxSize;
		yp = ((double) pos[3*ip + 1])*((double) ny)/BoxSize;
		zp = ((double) pos[3*ip + 2])*((double) nz)/BoxSize;

		i = iifloor(xp);
		j = iifloor(yp);
		k = iifloor(zp);


		//check to see if the i values of the CIC are
		//in this slab

		wrap_position(&i,&j,&k,nx,ny,nz,&xp,&yp,&zp);

		yes_flag = 0;
		if( (i>=nx_local_start)&&(i<(nx_local_start+nx_local)) )
			yes_flag = 1;


#ifdef   SIGMA_CORRECTION

		sigma_correction = 0;
#endif

		if(yes_flag)
		{
			count[ip]+=1.0;

			// cell centers

			xc = ((double) i);
			yc = ((double) j);
			zc = ((double) k);

			dx = xp - xc;

			dy = yp - yc;

			dz = zp - zc;

			tx = 1 - dx;

			ty = 1 - dy;

			tz = 1 - dz;

	
			//first do cell containing particle

			ic = i-nx_local_start;
			jc = j;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error A on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef   SIGMA_CORRECTION
			value      = tx*ty*tz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += tx*ty*tz;
				value      = tx*ty*tz*data[ijk];
			}

#endif  //SIGMA_CORRECTION

			answer[ip] += value;




	
			//do i,j+1,k

			jc = j+1;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error B on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef SIGMA_CORRECTION
			value      = tx*dy*tz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += tx*dy*tz;
				value      = tx*dy*tz*data[ijk];
			}

#endif  //SIGMA_CORRECTION

			answer[ip] += value;




	
			//do i,j,k+1

			jc = j;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error C on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef SIGMA_CORRECTION
			value      = tx*ty*dz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += tx*ty*dz;
				value      = tx*ty*dz*data[ijk];
			}

#endif  //SIGMA_CORRECTION

			answer[ip] += value;





			//do i,j+1,k+1

			jc = j+1;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error D on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef   SIGMA_CORRECTION
			value      = tx*dy*dz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += tx*dy*dz;
				value      = tx*dy*dz*data[ijk];
			}

#endif  //SIGMA_CORRECTION

			answer[ip] += value;
		}



		//check the i+1 values in the CIC interpolation

		yes_flag = 0;
		if( ((i+1)>=nx_local_start)&&((i+1)<(nx_local_start+nx_local)) )
			yes_flag = 1;
		if((i+1==nx)&&(nx_local_start==0))
			yes_flag = 1;
		if((i+1==nx)&&(nx_local_start!=0))
			yes_flag = 0;

		if(yes_flag)
		{
			count[ip]+=1.0;

			if((i+1)==nx)
			{
				xp -= ((double) nx);
				i  = -1;
			}


			// cell centers

			xc = ((double) i);
			yc = ((double) j);
			zc = ((double) k);

			dx = xp - xc;

			dy = yp - yc;

			dz = zp - zc;

			tx = 1 - dx;

			ty = 1 - dy;

			tz = 1 - dz;

	
			//first do cell containing particle

			ic = (i+1)-nx_local_start;
			jc = j;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BA on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef SIGMA_CORRECTION
			value      = dx*ty*tz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += dx*ty*tz;
				value      = dx*ty*tz*data[ijk];
			}

#endif  //SIGMA_CORRECTION


			answer[ip] += value;




	
			//do i,j+1,k

			jc = j+1;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BB on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef SIGMA_CORRECTION
			value      = dx*dy*tz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += dx*dy*tz;
				value      = dx*dy*tz*data[ijk];
			}

#endif  //SIGMA_CORRECTION


			answer[ip] += value;




	
			//do i,j,k+1

			jc = j;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BC on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef SIGMA_CORRECTION
			value      = dx*ty*dz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += dx*ty*dz;
				value      = dx*ty*dz*data[ijk];
			}

#endif  //SIGMA_CORRECTION


			answer[ip] += value;





			//do i,j+1,k+1

			jc = j+1;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BD on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

#ifndef SIGMA_CORRECTION
			value      = dx*dy*dz*data[ijk];
#else   //SIGMA_CORRECTION

			if(condition[ijk]>=threshold)
			{		
				value      = 0;
			}else{
				sigma_correction += dx*dy*dz;
				value      = dx*dy*dz*data[ijk];
			}

#endif  //SIGMA_CORRECTION


			answer[ip] += value;
		}

#ifdef	SIGMA_CORRECTION
		if(sigma_correction!=0.0)
		{
			answer[ip]/=(sigma_correction);
		}else{
			answer[ip]=0.0;
		}
#endif //SIGMA_CORRECTION

	}//end loop over npart


	//Check for errors

	AllCheckError(error_flag,myid,numprocs,world);


	//now sum up all contributions to each particle

	MPI_Allreduce(answer,total_answer,npart,MPI_DOUBLE,MPI_SUM,world);


	//now sum up all contributions to each particle

	MPI_Allreduce(count,total_count,npart,MPI_DOUBLE,MPI_SUM,world);

	for(int ip=0;ip<npart;ip++)
	{
		xp = ((double) pos[3*ip + 0])*((double) nx)/BoxSize;
		yp = ((double) pos[3*ip + 1])*((double) ny)/BoxSize;
		zp = ((double) pos[3*ip + 2])*((double) nz)/BoxSize;

		i = iifloor(xp);
		j = iifloor(yp);
		k = iifloor(zp);


		if(total_count[ip]>=3)
			if(myid==0)
			{
				printf("Error XX on proc %d in x direction ip %d count %e x %e y %e z %e i %d j %d k %d.\n",myid,ip,total_count[ip],xp,yp,zp,i,j,k);
				fflush(stdout);
				
			}
	}



	//free processor slab data

	free(answer);
	free(total_count);
	free(count);

	return total_answer;
}
#ifdef PARTICLE_FLOAT
double *interpolate_grid_data_cloud_in_cell(int npart, float *pos, double *data, int nx_local_start, int nx_local, int n_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world)
#else
double *interpolate_grid_data_cloud_in_cell(int npart, double *pos, double *data, int nx_local_start, int nx_local, int n_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world)
#endif
{


	//grid particle data using the cloud in cell method

	//need to adjust CIC to deal with different grid sizes

	int error_flag  = 0;
	int i,j,k;

	int ic, jc, kc;

	int ijk;

	double xc, yc, zc;

	double xp, yp, zp;

	double dx;
	double dy;
	double dz;

	double tx;
	double ty;
	double tz;

	double value;

	double *answer; //array containing interpolated values
	double *total_answer; //array containing interpolated values

	int u_dest   = myid+1;
	int u_source = myid-1;
	int l_dest   = myid-1;
	int l_source = myid+1;

	int yes_flag;


	//this buffer allows for imperfect CIC
	//assignments for varying grid cells

	int nxb = 10;
	int sbuf_size = nxb*ny*(2*(nz/2+1));

	char variable_name[200];

	MPI_Request requests[4];

	MPI_Status statuses[4];

	//wrap destinations and sources

	if(u_dest>=numprocs)
		u_dest-=numprocs;
	if(u_source<0)
		u_source+=numprocs;

	if(l_dest<0)
		l_dest+=numprocs;
	if(l_source>=numprocs)
		l_source-=numprocs;

	//we need to allocate the x-direction grid buffers

	if(myid==0)
	{
		printf("\n");
		fflush(stdout);
	}

	//allocate interpolated values

	sprintf(variable_name,"answer");
	answer       = allocate_double_array(npart, variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"total_answer");
	total_answer = allocate_double_array(npart, variable_name, myid, numprocs, world, 0);



	//initialize interpolated values
	for(i=0;i<npart;++i)
	{
		answer[i] = 0;
		total_answer[i] = 0;
	}


	//loop over the number of particles
	//and interpolate
			
	for(int ip=0;ip<npart;ip++)
	{
		xp = ((double) pos[3*ip + 0])*((double) nx)/BoxSize;
		yp = ((double) pos[3*ip + 1])*((double) ny)/BoxSize;
		zp = ((double) pos[3*ip + 2])*((double) nz)/BoxSize;

		i = iifloor(xp);
		j = iifloor(yp);
		k = iifloor(zp);


		//check to see if the i values of the CIC are
		//in this slab

		wrap_position(&i,&j,&k,nx,ny,nz,&xp,&yp,&zp);

		yes_flag = 0;
		if( (i>=nx_local_start)&&(i<(nx_local_start+nx_local)) )
			yes_flag = 1;


		if(yes_flag)
		{
			// cell centers

			xc = ((double) i);
			yc = ((double) j);
			zc = ((double) k);

			dx = xp - xc;

			dy = yp - yc;

			dz = zp - zc;

			tx = 1 - dx;

			ty = 1 - dy;

			tz = 1 - dz;

	
			//first do cell containing particle

			ic = i-nx_local_start;
			jc = j;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error A on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = tx*ty*tz*data[ijk];
			answer[ip] += value;




	
			//do i,j+1,k

			jc = j+1;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error B on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = tx*dy*tz*data[ijk];
			answer[ip] += value;




	
			//do i,j,k+1

			jc = j;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error C on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = tx*ty*dz*data[ijk];
			answer[ip] += value;





			//do i,j+1,k+1

			jc = j+1;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error D on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = tx*dy*dz*data[ijk];
			answer[ip] += value;
		}



		//check the i+1 values in the CIC interpolation

		yes_flag = 0;
		if( ((i+1)>=nx_local_start)&&((i+1)<(nx_local_start+nx_local)) )
			yes_flag = 1;
		if((i+1==nx)&&(nx_local_start==0))
			yes_flag = 1;
		if((i+1==nx)&&(nx_local_start!=0))
			yes_flag = 0;

		if(yes_flag)
		{
			if((i+1)==nx)
			{
				xp -= ((double) nx);
				i  = -1;
			}

			// cell centers

			xc = ((double) i);
			yc = ((double) j);
			zc = ((double) k);

			dx = xp - xc;

			dy = yp - yc;

			dz = zp - zc;

			tx = 1 - dx;

			ty = 1 - dy;

			tz = 1 - dz;

	
			//first do cell containing particle

			ic = (i+1)-nx_local_start;
			jc = j;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BA on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = dx*ty*tz*data[ijk];
			answer[ip] += value;




	
			//do i,j+1,k

			jc = j+1;
			kc = k;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BB on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = dx*dy*tz*data[ijk];
			answer[ip] += value;




	
			//do i,j,k+1

			jc = j;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BC on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = dx*ty*dz*data[ijk];
			answer[ip] += value;





			//do i,j+1,k+1

			jc = j+1;
			kc = k+1;

			wrap_indices(&ic,&jc,&kc,nx,ny,nz);

			ijk = (ic*ny + jc) * (2*(nz/2+1)) + kc;

			if(ijk>=n_local_size)
			{
				printf("Error BD on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d ijk %d tls %d.\n",myid,ip,xp,yp,zp,ic,jc,kc,nx_local,ijk,n_local_size);
				fflush(stdout);
				error_flag = 1;
			}

			value       = dx*dy*dz*data[ijk];
			answer[ip] += value;
		}

	}//end loop over npart


	//Check for errors

	AllCheckError(error_flag,myid,numprocs,world);


	//now sum up all contributions to each particle

	MPI_Allreduce(answer,total_answer,npart,MPI_DOUBLE,MPI_SUM,world);


	//free processor slab data

	free(answer);

	return total_answer;
}
void grid_particle_data_nearest_grid(int npart, float *pos, double *data, int nx_local_start, int nx_local, int n_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world)
{


	//grid particle data using nearest grid point method

	int error_flag  = 0;
	int i,j,k;
	int ijk;

	double x, y, z;

	//first, initialize the grid to zero
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
				data[  (i*ny + j) * (2*(nz/2+1)) + k] = 0.0;
			
	for(int ip=0;ip<npart;ip++)
	{
		x = ((double) pos[3*ip + 0])*((double) nx)/BoxSize;
		y = ((double) pos[3*ip + 1])*((double) ny)/BoxSize;
		z = ((double) pos[3*ip + 2])*((double) nz)/BoxSize;

		i = iifloor(x) - nx_local_start;
		j = iifloor(y);
		k = iifloor(z);

		wrap_indices(&i,&j,&k,nx,ny,nz);

		if((i<0)||(i>=nx_local))
		{
			printf("Error on proc %d in x direction ip %d x %e y %e z %e i %d j %d k %d nxl %d.\n",myid,ip,x,y,z,i,j,k,nx_local);
			fflush(stdout);
			error_flag = 1;
		}

		ijk = (i*ny + j) * (2*(nz/2+1)) + k;
		if(!error_flag)
			if(ijk>n_local_size)
			{
				printf("Error on proc %d ip %d x %e y %e z %e i %d j %d k %d nxl %d.\n",myid,ip,x,y,z,i,j,k,nx_local);
				fflush(stdout);
				error_flag = 1;
			}else{
				data[ijk] += 1.0;
			}
	}

	//Check for errors

	AllCheckError(error_flag,myid,numprocs,world);

	//done
}
void grid_to_overdensity(double *data, int total_npart, int nx, int ny, int nz, int nx_local_start, int nx_local)
{
	double rho_mean = ((double) total_npart)/( ((double) nx) * ((double) ny) * ((double) nz) );

	for(int i=0;i<nx_local;++i)
		for(int j=0;j<ny;++j)
			for(int k=0;k<nz;++k)
				data[ (i*ny + j)*(2*(nz/2+1)) + k] = (data[ (i*ny + j)*(2*(nz/2+1)) + k]/rho_mean) -  1.0;
}

int iifloor(double x)
{
        return ((int) floor(x));
}

void wrap_indices(int *ii, int *jj, int *kk, int nx, int ny, int nz)
{
        //printf("wi ii %d jj %d kk %d N %d\n",*ii,*jj,*kk,N);
        if(*jj<0)
	{
                *jj+=ny;
	}else{
        	if(*jj>=ny)
               		*jj-=ny;
	}
        if(*kk<0)
	{
                *kk+=nz;
	}else{
        	if(*kk>=nz)
		{
                	*kk-=nz;
        		//printf("wi ii %d jj %d kk %d nx %d ny %d nz %d\n",*ii,*jj,*kk,nx,ny,nz);
			//fflush(stdout);
		}
	}
}

void wrap_position(int *ii, int *jj, int *kk, int nx, int ny, int nz, double *xp, double *yp, double *zp)
{
        if(*ii<0)
	{
                *ii+=nx;
		*xp+=((double) nx);
	}else{
        	if(*ii>=nx)
		{
               		*ii-=nx;
			*xp-=((double) nx);
		}
	}
        if(*jj<0)
	{
                *jj+=ny;
		*yp+=((double) ny);
	}else{
        	if(*jj>=ny)
		{
               		*jj-=ny;
			*yp-=((double) ny);
		}
	}
        if(*kk<0)
	{
                *kk+=nz;
		*zp+=((double) nz);
	}else{
        	if(*kk>=nz)
		{
                	*kk-=nz;
			*zp-=((double) nz);
        		//printf("wi ii %d jj %d kk %d nx %d ny %d nz %d\n",*ii,*jj,*kk,nx,ny,nz);
			//fflush(stdout);
		}
	}
}
void wrap_particle(int *ii, int *jj, int *kk, int nx, int ny, int nz, double *xp, double *yp, double *zp)
{
        //printf("wi ii %d jj %d kk %d N %d\n",*ii,*jj,*kk,N);
        if(*jj<0)
	{
                *jj+=ny;
		*yp+=((double) ny);
	}else{
        	if(*jj>=ny)
		{
               		*jj-=ny;
			*yp-=((double) ny);
		}
	}
        if(*kk<0)
	{
                *kk+=nz;
		*zp+=((double) nz);
	}else{
        	if(*kk>=nz)
		{
                	*kk-=nz;
			*zp-=((double) nz);
        		//printf("wi ii %d jj %d kk %d nx %d ny %d nz %d\n",*ii,*jj,*kk,nx,ny,nz);
			//fflush(stdout);
		}
	}
}

float *get_particle_data(char *fname_particle_data, int *npart, int *total_npart, int myid, int numprocs, MPI_Comm world)
{
	FILE  *fp_particle_data;

	float *pos;

	int error_flag = 0;
	int flag_tot   = 0;

	//printf("myid %d particle data %s\n",myid,fname_particle_data);
	//fflush(stdout);

	if(!(fp_particle_data = fopen(fname_particle_data,"r")))
	{
		printf("Error opening %s on process %d.\n",fname_particle_data,myid);
		fflush(stdout);
		error_flag=1;
	}
	AllCheckError(error_flag,myid,numprocs,world);

	fread(npart,sizeof(int),1,fp_particle_data);

	if(!(pos = (float *) malloc(3*(*npart)*sizeof(float))))
	{
		printf("Error allocating pos on process %d.\n",myid);
		error_flag = 1;
	}


	//check for errors

	AllCheckError(error_flag,myid,numprocs,world);

	fread(pos,sizeof(float),3*(*npart),fp_particle_data);
	
	fclose(fp_particle_data);

	MPI_Allreduce(npart,total_npart,1,MPI_INT,MPI_SUM,world);

	return pos;
}

void AllCheckError(int error_flag, int myid, int numprocs, MPI_Comm world)
{
	int total_error = 0;

	MPI_Allreduce(&error_flag,&total_error,1,MPI_INT,MPI_SUM,world);

	if(total_error)
	{
		if(myid==0)
		{
			printf("Aborting...\n");
			fflush(stdout);
		}
		MPI_Abort(world,total_error);
		exit(-1);
	}
}

void check_window_function(char *window_function_fname, double *window_data, fftw_complex *cwindow_data, double *work, double BoxSize, double R, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int nx_local_start, int nx_local, int n_local_size, int local_ny_after_transpose, int myid, int numprocs, MPI_Comm world)
{

	FFTW_Grid_Info grid_info;

	int x, y, z;

	//make the units easy

	BoxSize   = 120.0;	
	R         = 10.0;

	//initialize window function
	for(x=0;x<nx_local;++x)
		for(y=0;y<ny;++y)
			for(z=0;z<nz;++z)
				window_data[(x*ny + y) * (2*(nz/2+1)) + z] = window_function(R,BoxSize,x+nx_local_start,y,z,nx,ny,nz);

	if(myid==0)
	{
		printf("Outputting real window function...\n");
		fflush(stdout);
	}	

	//output the real space window function

	sprintf(window_function_fname,"window_function.real_space.dat");


	grid_info.nx = nx;
	grid_info.ny = ny;
	grid_info.nz = nz;
	grid_info.n_local_size = n_local_size;
	grid_info.nx_local = nx_local;
	grid_info.local_ny_after_transpose = local_ny_after_transpose ;
	grid_info.nx_local_start = nx_local_start;
	//grid_info.local_y_start_after_transpose  = local_y_start_after_transpose ;

	//output_fft_grid(window_function_fname,   window_data,   nx, ny, nz, ixmin, ixmax, iymin, iymax, izmin, izmax, nx_local_start, nx_local, n_local_size, myid, numprocs, world);
	output_fft_grid(window_function_fname,   window_data,   grid_info, ixmin, ixmax, iymin, iymax, izmin, izmax, myid, numprocs, world);


	//Do the forward transform on the window function

	if(myid==0)
	{
		printf("Performing forward transform on window function...\n");
		fflush(stdout);
	}	


	//perform forward transform here
	forward_transform_fftw_grid(window_data, work, cwindow_data, grid_info, myid, numprocs, world);

	//the window function is now complex, so typecast a pointer

	cwindow_data = (fftw_complex *) window_data;


	//output the k-space window function

	if(myid==0)
	{
		printf("Outputting complex window function transform...\n");
		fflush(stdout);
	}	

	sprintf(window_function_fname,"window_function.k_space.complex.dat");

	output_fft_grid_complex(window_function_fname,   cwindow_data,   nx, ny, nz, ixmin, ixmax, iymin, iymax, izmin, izmax, nx_local_start, nx_local, n_local_size, myid, numprocs, world);


	//if we are just checking the window function, we are done.

	if(myid==0)
	{
		printf("Done checking window function; exiting.\n");
		fflush(stdout);
	}	

	//done
}


double *input_fft_grid(char *input_fname, int *nx, int *ny, int *nz, int *ixmin, int *ixmax, int *iymin, int *iymax, int *izmin, int *izmax, int *nx_local_start, int *nx_local, int *local_y_start_after_transpose, int *local_ny_after_transpose, int *n_local_size, int myid, int numprocs, MPI_Comm world)
{

	//read in an fft grid file
		
	FILE *fp;


	FFTW_Grid_Info grid_info;


	int     ijk;
	int     ijk_in;

	double *xbuf;

	int     error_flag  = 0;

	int     itot;	

	int    *nx_local_array;
	int    *nx_local_start_array;

	char variable_name[200];

	int nzl;

	double *data;

	MPI_Status status;

	//open the data file

	if(myid==0)
	{
		if(!(fp = fopen(input_fname,"r")))
		{
			printf("Error opening %s by process %d\n",input_fname,myid);
			fflush(stdout);

			error_flag = 1;
		}
		
	}
	AllCheckError(error_flag,myid,numprocs,world);
	
	if(myid==0)
	{

		//read in the essential info about the file

		//read the grid dimensions

		fread(nx,1,sizeof(int),fp);
		fread(ny,1,sizeof(int),fp);
		fread(nz,1,sizeof(int),fp);

		//read the restricted grid dimensions

		fread(ixmin,1,sizeof(int),fp);
		fread(ixmax,1,sizeof(int),fp);
		fread(iymin,1,sizeof(int),fp);
		fread(iymax,1,sizeof(int),fp);
		fread(izmin,1,sizeof(int),fp);
		fread(izmax,1,sizeof(int),fp);

	}

	//send the file info to all the processors

	MPI_Bcast(nx,1,MPI_INT,0,world);
	MPI_Bcast(ny,1,MPI_INT,0,world);
	MPI_Bcast(nz,1,MPI_INT,0,world);
	MPI_Bcast(ixmin,1,MPI_INT,0,world);
	MPI_Bcast(ixmax,1,MPI_INT,0,world);
	MPI_Bcast(iymin,1,MPI_INT,0,world);
	MPI_Bcast(iymax,1,MPI_INT,0,world);
	MPI_Bcast(izmin,1,MPI_INT,0,world);
	MPI_Bcast(izmax,1,MPI_INT,0,world);




	//initialize fttw grid

	grid_info.nx = *nx;
	grid_info.ny = *ny;
	grid_info.nz = *nz;
	if(grid_info.nz==1)
	{
		grid_info.ndim=2;
		grid_info.nz=0;
		nzl=1;
	}else{
		grid_info.ndim=3;
		nzl=grid_info.nz;
	}
	initialize_fftw_grid(&grid_info, myid, numprocs, world);


	//establish local sizes for the fftw

	initialize_mpi_local_sizes(&grid_info, myid, numprocs, world);

	*nx_local = grid_info.nx_local;	
	*local_ny_after_transpose = grid_info.local_ny_after_transpose;	
	*nx_local_start = grid_info.nx_local_start;	
	*local_y_start_after_transpose = grid_info.local_y_start_after_transpose;	
	*n_local_size = grid_info.n_local_size;	


	//Allocate the grid data for each processor	

	sprintf(variable_name,"data");
	data   = allocate_real_fftw_grid(*n_local_size, variable_name,   myid, numprocs, world, 0);



	//exchange the nx_local and nx_local_start arrays

	sprintf(variable_name,"nx_local_array");	
	nx_local_array      = allocate_int_array(numprocs, variable_name,      myid, numprocs, world, 0);
	sprintf(variable_name,"nx_local_start_array");	
	nx_local_start_array = allocate_int_array(numprocs, variable_name, myid, numprocs, world, 0);


	nx_local_array[myid]      = *nx_local;
	nx_local_start_array[myid] = *nx_local_start;

	MPI_Allgather(nx_local,      1,MPI_INT,nx_local_array,     1,MPI_INT,world);	

	MPI_Allgather(nx_local_start, 1,MPI_INT,nx_local_start_array,1,MPI_INT,world);	


	//read in the data for the root process


	if(myid==0)
	{
		printf("Reading in grid data...\n");
		fflush(stdout);
	}
  	for(int ip=0;ip<numprocs;ip++)
	{
		sprintf(variable_name,"xbuf");
		xbuf = allocate_double_array(nx_local_array[ip]*(*ny)*(*nz), variable_name, myid, numprocs, world, 0);

		if(ip==myid)
		{

			if(myid==0)
			{

				fread(xbuf,nx_local_array[ip]*(*ny)*(*nz),sizeof(double),fp);

			}else{

				MPI_Recv(xbuf,nx_local_array[ip]*(*ny)*(*nz),MPI_DOUBLE,0,ip,world,&status);
			}

			for(int i=0;i<nx_local_array[myid];++i)
				for(int j=0;j<(*ny);++j)	
					for(int k=0;k<(*nz);++k)
					{
						ijk_in        = ( i*(*ny) + j )*(*nz) + k;
						//ijk           = ( i*(*ny) + j )*(2*((*nz)/2+1)) + k;
						ijk = grid_ijk(i,j,k,grid_info);

						data[ijk] = xbuf[ijk_in];
					}

		}else{

			if(myid==0)
			{
				fread(   xbuf,nx_local_array[ip]*(*ny)*(*nz),sizeof(double),fp);
				MPI_Send(xbuf,nx_local_array[ip]*(*ny)*(*nz),MPI_DOUBLE,ip,ip,world);
			}
		}

		free(xbuf);

		MPI_Barrier(world);
	}
	if(myid==0)
	{
		printf("done!\n");
		fflush(stdout);
	}
	if(myid==0)
		fclose(fp);

	free(nx_local_array);
	free(nx_local_start_array);

	return data;
}


double window_function(double Rw, double BoxSize, int x, int y, int z, int nx, int ny, int nz)
{

#ifdef  GAUSSIAN

	//use a gaussian window function

	return gaussian_window(Rw, BoxSize, x, y, z, nx, ny, nz);


#else  //GAUSSIAN
#ifdef  REAL_SPACE_TOPHAT


	//use a real space tophat window function

	return real_space_tophat_window(Rw, BoxSize, x, y, z, nx, ny, nz);


#else  //REAL_SPACE_TOPHAT


	//use the default
	//which is a gaussian window function

	return gaussian_window(Rw, BoxSize, x, y, z, nx, ny, nz);

#endif //REAL_SPACE_TOPHAT
#endif //GAUSSIAN
}
double real_space_tophat_window(double Rw, double BoxSize, int x, int y, int z, int nx, int ny, int nz)
{
	double R;
	double dx;
	double dy;
	double dz;

	double W;

	if(x>nx/2)
	{
		dx = (double) (x-nx);
	}else{
		dx = (double) x;
	}

	if(y>ny/2)
	{
		dy = (double) (y-ny);
	}else{
		dy = (double) y;
	}

	if(z>nz/2)
	{
		dz = (double) (z-nz);
	}else{
		dz = (double) z;
	}

	dx*=BoxSize/((double) nx);
	dy*=BoxSize/((double) ny);
	dz*=BoxSize/((double) nz);


	R = sqrt( dx*dx + dy*dy + dz*dz);

	if(R<=Rw)
	{
#ifdef   TEST_ONE_DIMENSION
		W = 1.0/(2.*Rw);
#else  //TEST_ONE_DIMENSION
		W = 1.0/(4.*M_PI*Rw*Rw*Rw/3.0);
#endif //TEST_ONE_DIMENSION
	}else{
		W = 0.0;
	}	

	return W;
}
double gaussian_window(double Rw, double BoxSize, int x, int y, int z, int nx, int ny, int nz)
{
	double R;
	double dx;
	double dy;
	double dz;

	double W;

	if(x>nx/2)
	{
		dx = (double) (x-nx);
	}else{
		dx = (double) x;
	}

	if(y>ny/2)
	{
		dy = (double) (y-ny);
	}else{
		dy = (double) y;
	}

	if(z>nz/2)
	{
		dz = (double) (z-nz);
	}else{
		dz = (double) z;
	}

	dx*=BoxSize/((double) nx );
	dy*=BoxSize/((double) ny );
	dz*=BoxSize/((double) nz );


	R = sqrt( dx*dx + dy*dy + dz*dz);

#ifdef   TEST_ONE_DIMENSION
	W = exp(-R*R/(2*Rw*Rw))/sqrt(2*C.pi*Rw*Rw);
#else  //TEST_ONE_DIMENSION 
	W = exp(-R*R/(2*Rw*Rw))/pow(2*M_PI*Rw*Rw,1.5);
#endif //TEST_ONE_DIMENSION 


	return W;
}


double vector_magnitude(double *x, int ndim)
{
	double dp = 0.;
	for(int i=0;i<ndim;i++)
		dp += x[i] * x[i];

	return sqrt(dp);
}

double vector_dot_product(double *x, double *y, int ndim)
{
	double dp = 0.;
	for(int i=0;i<ndim;i++)
		dp += x[i] * y[i];

	return dp;
}

double *vector_cross_product(double *x, double *y, int ndim)
{
	double *cp;

	if(ndim==2)
	{
		cp = (double *) calloc(1,sizeof(double));
		cp[0] = x[0]*y[1] -  x[1]*y[0];
	}else{
		cp = (double *) calloc(3,sizeof(double));
		cp[0] = x[1]*y[2] -  x[2]*y[1];
		cp[1] = x[2]*y[0] -  x[0]*y[2];
		cp[2] = x[0]*y[1] -  x[1]*y[0];
	}


	return cp;
}
*/
