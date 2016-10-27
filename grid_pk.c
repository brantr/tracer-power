#include <math.h>
#include <vector>
#include "grid_pk.h"
#include "read_athena_tracers.hpp"

//#define CONTINUOUS

double *grid_ngp_alt(double *x, double *y, double *z, double *m, int N, FFTW_Grid_Info grid_info)
{
  //create a new grid
  //with an NGP assignment
  //of the particles

  //grid sizes
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;

  //grid indices
  double dx, dy, dz;
  int ix, iy, iz;
  int ijk;

  //create the grid
  double *u = allocate_real_fftw_grid(grid_info);

  for(ix=0;ix<nx;ix++)
    for(iy=0;iy<ny;iy++)
      for(iz=0;iz<nz;iz++)
      {
        ijk = grid_ijk(ix,iy,iz,grid_info);
        u[ijk] = 0.0;
      }

  //loop over the particles and assign them
  //to the grid using NGP
  for(int i=0;i<N;i++)
  {
    dx = nx*x[i];
    if(fmod(dx,1.) >= 0.5)
    {
      ix = floor(dx)+1;
    }else{
      ix = floor(dx);
    }
    if(ix>=nx)
      ix-=nx;
    if(ix<0)
      ix+=nx;

    dy = ny*y[i];
    if(fmod(dy,1.) >= 0.5)
    {
      iy = floor(dy)+1;
    }else{
      iy = floor(dy);
    }
    if(iy>=ny)
      iy-=ny;
    if(iy<0)
      iy+=ny;

    dz = nz*z[i];
    if(fmod(dz,1.) >= 0.5)
    {
      iz = floor(dz)+1;
    }else{
      iz = floor(dz);
    }
    if(iz>=nz)
      iz-=nz;
    if(iz<0)
      iz+=nz;

    //get index on the grid
    ijk = grid_ijk(ix,iy,iz,grid_info);

    //add the particle to the grid
    u[ijk] += m[i];
  }

  //return the grid
  return u;
}
double *grid_ngp(double *x, double *y, double *z, double *m, int N, FFTW_Grid_Info grid_info)
{
  //create a new grid
  //with an NGP assignment
  //of the particles

  //grid sizes
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;

  //grid indices
  double dx, dy, dz;
  int ix, iy, iz;
  int ijk;

  //create the grid
  double *u = allocate_real_fftw_grid(grid_info);

  for(ix=0;ix<nx;ix++)
    for(iy=0;iy<ny;iy++)
      for(iz=0;iz<nz;iz++)
      {
        ijk = grid_ijk(ix,iy,iz,grid_info);
        u[ijk] = 0.0;
      }

  //loop over the particles and assign them
  //to the grid using NGP
  for(int i=0;i<N;i++)
  {
    dx = nx*x[i];
    ix = floor(dx);
    if(ix>=nx)
      ix-=nx;
    if(ix<0)
      ix+=nx;

    dy = ny*y[i];
    iy = floor(dy);
    if(iy>=ny)
      iy-=ny;
    if(iy<0)
      iy+=ny;

    dz = nz*z[i];
    iz = floor(dz);
    if(iz>=nz)
      iz-=nz;
    if(iz<0)
      iz+=nz;

    //get index on the grid
    ijk = grid_ijk(ix,iy,iz,grid_info);

    //add the particle to the grid
    u[ijk] += m[i];
  }

  //return the grid
  return u;
}
double *grid_cic(double *x, double *y, double *z, double *m, int N, FFTW_Grid_Info grid_info)
{
  //create a new grid
  //with an CIC assignment
  //of the particles

  //grid sizes
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;

  //grid indices
  double dcx = 1./((double) nx);
  double dcy = 1./((double) ny);
  double dcz = 1./((double) nz);

  double xijk, yijk, zijk;

  double dx, dy, dz;
  int ix, iy, iz;
  int ijk;
  int i, j, k;

  //create the grid
  double *u = allocate_real_fftw_grid(grid_info);

  for(ix=0;ix<nx;ix++)
    for(iy=0;iy<ny;iy++)
      for(iz=0;iz<nz;iz++)
      {
        ijk = grid_ijk(ix,iy,iz,grid_info);
        u[ijk] = 0.0;
      }

  //loop over the particles and assign them
  //to the grid using NGP
  for(int n=0;n<N;n++)
  {
    dx = nx*x[n];
    if(fmod(dx,1.) >= 0.5)
    {
      ix = floor(dx)+1;
    }else{
      ix = floor(dx);
    }
    dy = ny*y[n];
    if(fmod(dy,1.) >= 0.5)
    {
      iy = floor(dy)+1;
    }else{
      iy = floor(dy);
    }
    dz = nz*z[n];
    if(fmod(dz,1.) >= 0.5)
    {
      iz = floor(dz)+1;
    }else{
      iz = floor(dz);
    }
    //printf("ix %d iy %d iz %d\n",ix,iy,iz);

    for(int ii=ix-1;ii<ix+1;ii++)
    {
      i = ii;
      if(ii<0)
        i += nx;
      if(ii>=nx)
        i -= nx;

      for(int jj=iy-1;jj<iy+1;jj++)
      {
        j = jj;
        if(jj<0)
          j += ny;
        if(jj>=ny)
          j -= ny;
        for(int kk=iz-1;kk<iz+1;kk++)
        {
          k = kk;
          if(kk<0)
            k += nz;
          if(kk>=nz)
            k -= nz;

          //ijk is the index of the cell
          //xijk is the cell x position
          //yijk is the cell y position
          //zijk is the cell z position
          ijk = grid_ijk(i,j,k,grid_info);
          xijk = (((double) ii)+0.5)*dcx;
          yijk = (((double) jj)+0.5)*dcy;
          zijk = (((double) kk)+0.5)*dcz;

          //printf("i %d j %d k %d\n",i,j,k);

          u[ijk] += m[n]*(1.0 - fabs(x[n]-xijk)/dcx )*(1.0 - fabs(y[n]-yijk)/dcy )*(1.0 - fabs(z[n]-zijk)/dcz );

        }
      }
    }
  }

  //return the grid
  return u;
}
double *grid_tsc(vector<tracer> t, FFTW_Grid_Info grid_info)
{
  //create a new grid
  //with an TSC assignment
  //of the particles

  //grid sizes
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;

  //number of particles
  long N = t.size();

  //grid indices
  double dcx = 1./((double) nx);
  double dcy = 1./((double) ny);
  double dcz = 1./((double) nz);

  double xijk, yijk, zijk;

  double dx, dy, dz;
  int ix, iy, iz;
  int ijk;
  int i, j, k;
  double wx, wy, wz;
  double dxx, dyy, dzz;

  //create the grid
  double *u = allocate_real_fftw_grid(grid_info);


  for(ix=0;ix<nx;ix++)
    for(iy=0;iy<ny;iy++)
      for(iz=0;iz<nz;iz++)
      {
        ijk = grid_ijk(ix,iy,iz,grid_info);
        u[ijk] = 0.0;
      }
      


  //loop over the particles and assign them
  //to the grid using NGP
  for(long n=0;n<N;n++)
  {
    dx = nx*t[n].x[0];
    ix = floor(dx);

    dy = ny*t[n].x[1];
    iy = floor(dy);

    dz = nz*t[n].x[2];
    iz = floor(dz);

    double ucheck = 0;

    for(int ii=ix-1;ii<ix+2;ii++)
    {
      i = ii;
      if(ii<0)
        i += nx;
      if(ii>=nx)
        i -= nx;

      for(int jj=iy-1;jj<iy+2;jj++)
      {
        j = jj;
        if(jj<0)
          j += ny;
        if(jj>=ny)
          j -= ny;
        for(int kk=iz-1;kk<iz+2;kk++)
        {
          k = kk;
          if(kk<0)
            k += nz;
          if(kk>=nz)
            k -= nz;

          //ijk is the index of the cell
          //xijk is the cell x position
          //yijk is the cell y position
          //zijk is the cell z position
          ijk = grid_ijk(i,j,k,grid_info);
          xijk = (((double) ii)+0.5)*dcx;
          yijk = (((double) jj)+0.5)*dcy;
          zijk = (((double) kk)+0.5)*dcz;

          dxx = (t[n].x[0]-xijk)/dcx;
          dyy = (t[n].x[1]-yijk)/dcy;
          dzz = (t[n].x[2]-zijk)/dcz;

          if(fabs(dxx)<0.5)
          {
            wx = 0.75-dxx*dxx;
          }else if (fabs(dxx)<1.5){
            wx = 0.5*pow(1.5-fabs(dxx),2);
          }else{
            wx = 0;
          }
          if(fabs(dyy)<0.5)
          {
            wy = 0.75-dyy*dyy;
          }else if (fabs(dyy)<1.5){
            wy = 0.5*pow(1.5-fabs(dyy),2);
          }else{
            wy = 0;
          }
          if(fabs(dzz)<0.5)
          {
            wz = 0.75-dzz*dzz;
          }else if (fabs(dzz)<1.5){
            wz = 0.5*pow(1.5-fabs(dzz),2);
          }else{
            wz = 0;
          }
          u[ijk] += t[n].m*wx*wy*wz;
        }
      }
    }
  }
  //return the grid
  return u;
}
double w_p(int p, double kx, double ky, double kz, FFTW_Grid_Info grid_info)
{
  int nx = grid_info.nx;
  double kN = M_PI*((double) nx)/grid_info.BoxSize;
  double k1 = 0.5*M_PI*kx/kN;
  double k2 = 0.5*M_PI*ky/kN;
  double k3 = 0.5*M_PI*kz/kN;
  return pow( sin(k1)*sin(k2)*sin(k3)/(k1*k2*k3), p);
}

double *grid_dfk(int N, double *u, FFTW_Grid_Info grid_info, MPI_Comm world)
{
  int nx       = grid_info.nx;
  int ny       = grid_info.ny;
  int nz       = grid_info.nz;

  //double N = 0; //number of objects

  //normalization
  double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

  fftw_complex *uk;
  fftw_plan plan;

  int ijk, ijkc;

  //real power spectrum
  double *dfk = allocate_real_fftw_grid(grid_info);


  //allocate work and transform
  uk    = allocate_complex_fftw_grid(grid_info);

  //create the fftw plans
  //plan  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uk, uk, world, FFTW_FORWARD,  FFTW_ESTIMATE);
  plan  = fftw_mpi_plan_dft_r2c_3d(grid_info.nx, grid_info.ny, grid_info.nz, u, uk, world, FFTW_ESTIMATE);

  //get complex version of A
  //grid_copy_real_to_complex_in_place(u, uk, grid_info);

  //perform the forward transform on the components of u
  fftw_execute(plan);

  //compute dfk
#ifndef CONTINUOUS

  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      for(int k=0;k<(nz+1)/2;++k)
      {
        ijk  = grid_ijk(i,j,k,grid_info);
        ijkc = grid_complex_ijk(i,j,k,grid_info);
        uk[ijkc][0]/=((double) N);
        uk[ijkc][1]/=((double) N);
      }
  if(nz%2==0)
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++)
        for(int k=nz/2;k<nz/2+1;k++)
        {
          ijk  = grid_ijk(i,j,k,grid_info);
          ijkc = grid_complex_ijk(i,j,k,grid_info);
          uk[ijkc][0]/=((double) N);
          uk[ijkc][1]/=((double) N);
        }
  uk[0][0] -= 1.0;
#endif // CONTINUOUS
  printf("DC = %e %e\n",uk[0][0],uk[0][1]);

  double real_tot=0, img_tot=0;

  //find delta^f_k;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      for(int k=0;k<(nz+1)/2;++k)
      {
        ijk  = grid_ijk(i,j,k,grid_info);
        ijkc = grid_complex_ijk(i,j,k,grid_info);
        //printf("i %d j %d k %d ijk %d ijkc %d\n",i,j,k,ijk,ijkc);
        real_tot += uk[ijkc][0];
        img_tot  += uk[ijkc][1];

        dfk[ijk] = uk[ijkc][0]*uk[ijkc][0] + uk[ijkc][1]*uk[ijkc][1];
        //dfk[ijk] *= (scale*scale); // one for each factor of u
#ifdef CONTINUOUS
        dfk[ijk] = uk[ijkc][0];
        dfk[ijk] *= (scale); // one for each factor of u
#endif //CONTINUOUS
      }

  //printf("check %e %e\n",uk[0][0]*scale,sqrt(dfk[0]));
  if(nz%2==0)
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++)
        for(int k=nz/2;k<nz/2+1;k++)
        {
          ijk  = grid_ijk(i,j,k,grid_info);
          ijkc = grid_complex_ijk(i,j,k,grid_info);
          //printf("i %d j %d k %d ijk %d ijkc %d\n",i,j,k,ijk,ijkc);
          real_tot += uk[ijkc][0];
          img_tot  += uk[ijkc][1];

          //get |d^f(k)|^2
          dfk[ijk] = uk[ijkc][0]*uk[ijkc][0] + uk[ijkc][1]*uk[ijkc][1];
#ifdef CONTINUOUS
          dfk[ijk] = uk[ijkc][0];
          dfk[ijk] *= (scale); // one for each factor of u
#endif //CONTINUOUS
        }

  //printf("real_tot %e img_tot %e N %d\n",real_tot,img_tot,N);


  //free memory
  fftw_free(uk);
  fftw_destroy_plan(plan);

  //return the result
  return dfk;
}

/*! \fn double *grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place into the real elements of a complex grid. */
void grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
{
        int i, j, k;

        int nx_local = grid_info.nx_local;
        int nx       = grid_info.nx;
        int ny       = grid_info.ny;
        int nz       = grid_info.nz;

        int ijk;
        int ijkc;

        //Copy source into copy.
        for(i=0;i<nx_local;++i)
                for(j=0;j<ny;++j)
                        for(k=0;k<nz;++k)
                        {
                                //real grid index
                                ijk = grid_ijk(i,j,k,grid_info);

                                //complex grid index
                                ijkc = grid_complex_ijk(i,j,k,grid_info);

                                //copy
                                copy[ijkc][0] = source[ijk];
                                copy[ijkc][1] = 0;
                        }
}
