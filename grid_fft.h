/*! \file grid_fft.h
 *  \brief Routines for calculating FFTs on a grid
 *	   using the MPI-capable FFTW library.  */

#include<mpi.h>
#include<fftw3-mpi.h>
#ifndef BRANT_GRID_FFT
#define BRANT_GRID_FFT
#define PARTICLE_FLOAT


/*! \struct FFTW_Grid_Info
 *  \brief  Structure to contain
 *          information for use with
 *          3-d dimensional FFTW grids
 *	    parallelized with MPI.
 */
struct FFTW_Grid_Info
{
	/*! \var ptrdiff_t nx
	 *  \brief Number of grid points in x-direction. */
	ptrdiff_t nx;
	/*! \var ptrdiff_t ny
	 *  \brief Number of grid points in y-direction. */
	ptrdiff_t ny;

	/*! \var ptrdiff_t nz
	 *  \brief Number of grid points in z-direction. */
	ptrdiff_t nz;

	/*! \var ptrdiff_t nz_complex
	 *  \brief Number of grid points in z-direction for complex data. */
	ptrdiff_t nz_complex;

	/*! \var ptrdiff_t nx_local
	 *  \brief Local number of grid points in x-direction */
	ptrdiff_t nx_local;

	/*! \var ptrdiff_t nx_local_start
	 *  \brief First grid point in x-direction */
	ptrdiff_t nx_local_start;

	/*! \var int *n_local_real_size
	 *  \brief Total size of real grids on local process.*/
	ptrdiff_t n_local_real_size;

	/*! \var int *n_local_complex_size
	 *  \brief Total size of complex grids on local process.*/
	ptrdiff_t n_local_complex_size;

	/*! \var double BoxSize
	 *  \brief 1-d length of the grid. */
	double BoxSize;

	/*! \var double dx 
	 *  \brief Length of one grid cell in x direction. */
	double dx;

	/*! \var double dy 
	 *  \brief Length of one grid cell in y direction. */
	double dy;

	/*! \var double dz 
	 *  \brief Length of one grid cell in z direction. */
	double dz;

	/*! \var double dV 
	 *  \brief Volume of one grid cell. */
	double dV;

	/*! \var double dVk
	 *  \brief Volume of one grid cell in k-space. */
	double dVk;

	/*! \var int ndim
	 *  \brief Number of dimensions. */
	int ndim;

	/*! \var fftw_plan plan
	 *  \brief Forward plan for FFTs */
	fftw_plan plan;

	/*! \var fftw_plan iplan
	 *  \brief Inverse plan for FFTs */
	fftw_plan iplan;
};


/*! \fn void Copy_FFTW_Grid_Info(FFTW_Grid_Info source, FFTW_Grid_Info *dest)
 *  \brief Copy an FFTW_Grid_info struct. */
void Copy_FFTW_Grid_Info(FFTW_Grid_Info source, FFTW_Grid_Info *dest);

/*! \fn int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for fftw grid based on coordinates i,j,k. */
int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info);

/*! \fn int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info)
 *  \brief Given a position, return the grid index. */
int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info);

/*! \fn int grid_complex_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for complex fftw grid based on coordinates i,j,k. */
int grid_complex_ijk(int i, int j, int k, FFTW_Grid_Info grid_info);

/*! \fn int grid_complex_from_real_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for complex fftw grid created from r2c based on coordinates i,j,k. */
int grid_complex_from_real_ijk(int i, int j, int k, FFTW_Grid_Info grid_info);

/*! \fn void initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, MPI_Comm world);
 *  \brief Function to determine local grid sizes for parallel FFT. */
void initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, MPI_Comm world);

/*! \fn double *allocate_real_fftw_grid_sized(int n_size);
 *  \brief Allocates a pre-sized 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid_sized(int n_size);

/*! \fn double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info);

/*! \fn fftw_complex *allocate_complex_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a 3-d complex grid for use with fftw.*/
fftw_complex *allocate_complex_fftw_grid(FFTW_Grid_Info grid_info);

/*! \fn double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info)
 *  \brief Allocates a field[ndim][total_local_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info);

/*! \fn void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info); 
 *  \brief De-allocates a field[ndim][total_local_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info);

/*! \fn double ***allocate_tensor_fftw_grid(FFTW_Grid_Info grid_info);
 *  \brief Allocates a tensor[ndim][ndim][total_local_size] (of dimension ndim*ndim) of  3-d real grids for use with fftw.*/
double ***allocate_tensor_fftw_grid(FFTW_Grid_Info grid_info);

/*! \fn void deallocate_tensor_fftw_grid(double **tensor, FFTW_Grid_Info grid_info);
 *  \brief De-allocates a tensor[ndim][ndim][total_local_size] (of dimension ndim*ndim) of  3-d real grids for use with fftw.*/
void deallocate_tensor_fftw_grid(double ***tensor, FFTW_Grid_Info grid_info);


/*! \fn void inverse_transform_fftw_grid(double *work, fftw_complex *cdata, FFTW_Grid_Info grid_info)
 *  \brief Takes a forward transform, already re-normalized by 1./(nx*ny*nz), and returns a properly normalized inverse transform */
void inverse_transform_fftw_grid(double *work, fftw_complex *cwork, FFTW_Grid_Info grid_info);

/*! \fn void forward_transform_fftw_grid(double *work, fftw_complex *cwork, FFTW_Grid_Info grid_info)
 *  \brief Produces a forward transform normalized by 1./(nx*ny*nz). */
void forward_transform_fftw_grid(double *work, fftw_complex *cwork, FFTW_Grid_Info grid_info);


/*! \fn void output_fft_grid(char *output_fname, double *data, FFTW_Grid_Info grid_info, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int myid, int numprocs, MPI_Comm world);
 *  \brief Output a grid to file. */
void output_fft_grid(char *output_fname, double *data, FFTW_Grid_Info grid_info, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int myid, int numprocs, MPI_Comm world);

/*

void convolve_complex_fftw_grid(fftw_complex *C_transposed, fftw_complex *A_transposed, FFTW_Grid_Info grid_info, fftw_complex *B_transposed, int local_ny_after_transpose, int nx, int ny, int nz, int myid, int numprocs, MPI_Comm world);

void output_fft_grid_procs(char *output_fname, double *data, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int local_x_start, int nx_local, int total_local_size, int myid, int numprocs, MPI_Comm world);
void output_fft_grid_root(char *output_fname, double *data, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int local_x_start, int nx_local, int total_local_size, int myid, int numprocs, MPI_Comm world);

double *input_fft_grid(char *input_fname, int *nx, int *ny, int *nz, int *ixmin, int *ixmax, int *iymin, int *iymax, int *izmin, int *izmax, int *local_x_start, int *nx_local, int *local_y_after_transpose, int *local_ny_after_transpose, int *total_local_size, int myid, int numprocs, MPI_Comm world);

void output_fft_grid_complex(char *output_fname, fftw_complex *data, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int local_y_start_after_transpose, int local_ny_after_transpose, int total_local_size, int myid, int numprocs, MPI_Comm world);

void grid_particle_data(int npart, float *pos, double *data, int total_local_size, int local_x_start, int nx_local, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int nprocs, MPI_Comm world);

void grid_particle_data_cloud_in_cell(int npart, float *pos, double *data, int local_x_start, int nx_local, int total_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int nprocs, MPI_Comm world);

void grid_particle_data_nearest_grid(int npart, float *pos, double *data, int local_x_start, int nx_local, int total_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int nprocs, MPI_Comm world);

int iifloor(double x);

void wrap_indices(int *ii, int *jj, int *kk, int nx, int ny, int nz);

void wrap_particle(int *ii, int *jj, int *kk, int nx, int ny, int nz, double *xp, double *yp, double *zp);

void wrap_position(int *ii, int *jj, int *kk, int nx, int ny, int nz, double *xp, double *yp, double *zp);

float *get_particle_data(char *fname_particle_data, int *npart, int *total_npart, int myid, int numprocs, MPI_Comm world);

void grid_to_overdensity(double *data, int total_part, int nx, int ny, int nz, int local_x_start, int nx_local);

void check_window_function(char *window_function_fname, double *window_data, fftw_complex *cwindow_data, double *work, double BoxSize, double R, int nx, int ny, int nz, int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax, int local_x_start, int nx_local, int total_local_size, int local_ny_after_transpose, int myid, int nprocs, MPI_Comm world);

void AllCheckError(int error_flag, int myid, int numprocs, MPI_Comm world);

double window_function(double R, double BoxSize, int x, int y, int z, int nx, int ny, int nz);


#ifdef PARTICLE_FLOAT
double *interpolate_grid_data_cloud_in_cell(int npart, float *pos, double *data, int local_x_start, int nx_local, int total_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world);
#else
double *interpolate_grid_data_cloud_in_cell(int npart, double *pos, double *data, int local_x_start, int nx_local, int total_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int numprocs, MPI_Comm world);
#endif
double *interpolate_grid_data_cloud_in_cell_conditional(int npart, double *pos, double *data, double *condition, int local_x_start, int nx_local, int total_local_size, int nx, int ny, int nz, int npart_total, double BoxSize, int myid, int nprocs, MPI_Comm world);


//void generate_field_from_power_spectrum( double *data, int total_local_size, int local_x_start, int nx_local, int nx, int ny, int nz, double BoxSize, int myid, int nprocs, MPI_Comm world);



double window_function(double Rw, double BoxSize, int x, int y, int z, int nx, int ny, int nz);
double real_space_tophat_window(double Rw, double BoxSize, int x, int y, int z, int nx, int ny, int nz);
double gaussian_window(double Rw, double BoxSize, int x, int y, int z, int nx, int ny, int nz);


//get the field component from the processor ip
void recover_real_grid(double *u_in, double *u_out, int ip);

*/

#endif //BRANT_GRID_FFT
