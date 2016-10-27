#include <stdio.h>
#include <stdlib.h>
#include "read_athena_header.hpp"
#include "read_athena_tracers.hpp"


using namespace std;

AthenaHeaderH *read_athena_tracers(char fname[], vector<tracer> *t)
{
  FILE *fp;
  long n_tracers;       /*number of tracers in the file*/
  float *d;             /*tracer densities*/
  float *x;             /*tracer x positions*/
  float *y;             /*tracer y positions*/
  float *z;             /*tracer z positions*/
  float *vx;            /*tracer x velocities*/
  float *vy;            /*tracer y velocities*/
  float *vz;            /*tracer z velocities*/
  long  *l;             /*tracer ids*/
  long ntd = 0;         /*number of tracers above the density threshold*/
  AthenaHeaderH *h;


  //buffer for storing tracers into the tracer vector *t
  tracer tin;

  /*open tracer file*/
  if(!(fp = fopen(fname,"r")))
  {
    printf("Error opening %s in load tracers.\n",fname);
    fflush(stdout); 
  }

  /* Read header */
  h = ReadAthenaHeaderFunc(fp);

  //ShowAthenaHeaderFunc(h);


  /* read the number of tracer in this file */ 
  fread(&n_tracers,1,sizeof(long),fp);

  printf("Read tracers n_tracers = %ld\n",n_tracers);
  fflush(stdout);

  /* Allocate buffer */
  if(!(d = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property d (%ld).\n",n_tracers);
    fflush(stdout);
    exit(-1);
  }

  /* Allocate buffer */
  if(!(x = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property x (%ld).\n",n_tracers);
    fflush(stdout);
    exit(-1);
  }

  /* Allocate buffer */
  if(!(y = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property y (%ld).\n",n_tracers);
    fflush(stdout);
    exit(-1);
  }

  /* Allocate buffer */
  if(!(z = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property z (%ld).\n",n_tracers);
    fflush(stdout);
    exit(-1);
  }

  /* Allocate buffer */
  if(!(vx = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property z.\n");
    fflush(stdout);
    exit(-1);
  }

    /* Allocate buffer */
    if(!(vy = (float *)malloc(n_tracers*sizeof(float))))
    {
      printf("Error allocating tracer property z.\n");
      fflush(stdout);
      exit(-1);
    }


    /* Allocate buffer */
    if(!(vz = (float *)malloc(n_tracers*sizeof(float))))
    {
      printf("Error allocating tracer property z.\n");
      fflush(stdout);
      exit(-1);
    }


    /* read density */
    fread(d,n_tracers,sizeof(float),fp);

    double dmin = 1.0e9;
    double dmax = -1.0e9;
    for(long tt=0;tt<n_tracers;tt++)
    {
      if(d[tt]<dmin)
        dmin = d[tt];
      if(d[tt]>dmax)
        dmax = d[tt];
    }
    printf("density extrema %e %e\n",dmin,dmax);
    

    /* read M1 */
    fread(x,n_tracers,sizeof(float),fp);
    for(long tt=0;tt<n_tracers;tt++)
    {
      if(d[tt]>0)
      {
        vx[tt] = x[tt]/d[tt];
      }else{
        vx[tt] = x[tt];
      }
    }
  
    /* read M2 */
    fread(x,n_tracers,sizeof(float),fp);
    for(long tt=0;tt<n_tracers;tt++)
    {
      if(d[tt]>0)
      {
        vy[tt] = x[tt]/d[tt];
      }else{
        vy[tt] = x[tt];
      }
    }

    /* read M3 */
    fread(x,n_tracers,sizeof(float),fp);
    for(long tt=0;tt<n_tracers;tt++)
    {
      if(d[tt]>0)
      {
        vz[tt] = x[tt]/d[tt];
      }else{
        vz[tt] = x[tt];
      }
    }

/*
#ifndef BAROTROPIC
    // read E 
    fread(x,n_tracers,sizeof(float),fp);
#endif // BAROTROPIC 

#ifdef MHD
    // read B1c 
    fread(x,n_tracers,sizeof(float),fp);

    // read B2c 
    fread(x,n_tracers,sizeof(float),fp);

    // read B3c 
    fread(x,n_tracers,sizeof(float),fp);
#endif //MHD

#if (NSCALARS > 0)
    for(k=0;k<NSCALARS;k++)
      fread(x,n_tracers,sizeof(float),fp);
#endif
*/
    /* read x1 */
    fread(x,n_tracers,sizeof(float),fp);

    printf("Read x position\n");
  
    /* read x2 */
    fread(y,n_tracers,sizeof(float),fp);
    printf("Read y position\n");

    /* read x3 */
    fread(z,n_tracers,sizeof(float),fp);  
    printf("Read z position\n");

    /* Allocate buffer */
    if(!(l = (long *)malloc(n_tracers*sizeof(long))))
    {
      printf("Error allocating tracer property buf.\n");
      fflush(stdout);
    }

    /* read id */
    fread(l,n_tracers,sizeof(long),fp);

    printf("Read ids\n");

    /*close tracer file*/
    fclose(fp);


    /*keep only particles with densities above or = threshold*/
    ntd = 0;
    for(long tt=0;tt<n_tracers;tt++)
    {
      tin.id = l[tt];
      tin.d = d[tt];
      tin.x[0] = x[tt];
      tin.x[1] = y[tt];
      tin.x[2] = z[tt];
      tin.v[0] = vx[tt];
      tin.v[1] = vy[tt];
      tin.v[2] = vz[tt];

      //if(tt<100)
        //printf("Reading tt %ld x %e %e %e vx %e %e %e d %e l %ld\n",tt,x[tt],y[tt],z[tt],vx[tt],vy[tt],vz[tt],d[tt],l[tt]);


      //add to tracer list
      (*t).push_back(tin);

      //remember that we've kept a particle
      ntd++;
    }

    printf("stored tracers\n");

    /*free buffer memory*/
    free(d);
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(l);


  /*free header*/
  //free(h);

  printf("Freed memory\n");

  //return number of tracers > density
  return h;

}



void write_athena_tracers(char fname[], AthenaHeaderH *h, vector<tracer> t)
{
  FILE *fp;
  long n_tracers;       /*number of tracers in the file*/
  float *d;             /*tracer densities*/
  float *x;             /*tracer x positions*/
  float *y;             /*tracer y positions*/
  float *z;             /*tracer z positions*/
  float *vx;            /*tracer x velocities*/
  float *vy;            /*tracer y velocities*/
  float *vz;            /*tracer z velocities*/
  long  *l;             /*tracer ids*/
  long ntd = 0;         /*number of tracers above the density threshold*/


  /*open tracer file*/
  if(!(fp = fopen(fname,"w")))
  {
    printf("Error opening %s in write_athena_tracers.\n",fname);
    fflush(stdout); 
  }
  

  /* Read header */
  WriteAthenaHeaderFunc(fp,h);

  //ShowAthenaHeader(h);

  //printf("t.size() = %ld\n",t.size());


  /* read the number of tracer in this file */
  n_tracers = t.size();
  fwrite(&n_tracers,1,sizeof(long),fp);

  if(!(x = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property buf.\n");
    fflush(stdout);
    exit(-1);
  }

  double dmin = 1.0e9;
  double dmax = -1.0e9;
  for(long tt = 0;tt<t.size();tt++)
  {
    x[tt] = t[tt].d;
    if(x[tt]<dmin)
      dmin = x[tt];
    if(x[tt]>dmax)
      dmax = x[tt];
  }
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp);

  printf("Output density extrema %e %e\n",dmin,dmax);

  for(long tt = 0;tt<t.size();tt++)
  {
    if(t[tt].d>0)
    {
      x[tt] = t[tt].d*t[tt].v[0];
    }else{
      x[tt] = t[tt].v[0];      
    }

    //if(tt<100)
      //printf("tt %ld x %e d %e vx %e\n",tt,x[tt],t[tt].d,t[tt].v[0]);
  }
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp); 

  for(long tt = 0;tt<t.size();tt++)
    if(t[tt].d>0)
    {
      x[tt] = t[tt].d*t[tt].v[1];
    }else{
      x[tt] = t[tt].v[1];      
    }
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp);

  for(long tt = 0;tt<t.size();tt++)
    if(t[tt].d>0)
    {
      x[tt] = t[tt].d*t[tt].v[2];
    }else{
      x[tt] = t[tt].v[2];      
    }
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp); 

  for(long tt = 0;tt<t.size();tt++)
    x[tt] = t[tt].x[0];
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp); 
    for(long tt = 0;tt<t.size();tt++)
    x[tt] = t[tt].x[1];
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp); 
    for(long tt = 0;tt<t.size();tt++)
    x[tt] = t[tt].x[2];
  // write x1 
  fwrite(x,n_tracers,sizeof(float),fp); 


  free(x);
  if(!(l = (long *)malloc(n_tracers*sizeof(long))))
    {
      printf("Error allocating tracer property l.\n");
      fflush(stdout);
      exit(-1);
    }
  for(long tt = 0;tt<t.size();tt++)
    l[tt] = t[tt].id;
  // write id 
  fwrite(l,n_tracers,sizeof(long),fp);
  
/*
  if(!(d = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property buf.\n");
    fflush(stdout);
    exit(-1);
  }


  if(!(x = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property buf.\n");
    fflush(stdout);
    exit(-1);
  }

  if(!(y = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property y.\n");
    fflush(stdout);
    exit(-1);
  }

  if(!(z = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property z.\n");
    fflush(stdout);
    exit(-1);
  }

  if(!(vx = (float *)malloc(n_tracers*sizeof(float))))
  {
    printf("Error allocating tracer property z.\n");
    fflush(stdout);
    exit(-1);
  }

    if(!(vy = (float *)malloc(n_tracers*sizeof(float))))
    {
      printf("Error allocating tracer property z.\n");
      fflush(stdout);
      exit(-1);
    }


    if(!(vz = (float *)malloc(n_tracers*sizeof(float))))
    {
      printf("Error allocating tracer property z.\n");
      fflush(stdout);
      exit(-1);
    }

    if(!(l = (long *)malloc(n_tracers*sizeof(long))))
    {
      printf("Error allocating tracer property l.\n");
      fflush(stdout);
      exit(-1);
    }*/
    
    /*
    for(long tt=0;tt<n_tracers;tt++)
    {
      d[tt] = t[tt].d;
      x[tt] = t[tt].x[0];
      y[tt] = t[tt].x[1];
      z[tt] = t[tt].x[2];
      vx[tt] = t[tt].d*t[tt].v[0];
      vy[tt] = t[tt].d*t[tt].v[1];
      vz[tt] = t[tt].d*t[tt].v[2];
      l[tt]  = t[tt].id;
    }
*/
/*
    // write density 
    fwrite(d,n_tracers,sizeof(float),fp);

    // write M1 
    fwrite(vx,n_tracers,sizeof(float),fp);

    // write M2 
    fwrite(vy,n_tracers,sizeof(float),fp);

    // write M3 
    fwrite(vz,n_tracers,sizeof(float),fp);

    // write x1 
    fwrite(x,n_tracers,sizeof(float),fp);
  
    // write x2 
    fwrite(y,n_tracers,sizeof(float),fp);

    // write x3 
    fwrite(z,n_tracers,sizeof(float),fp);  

    // write id 
    fwrite(l,n_tracers,sizeof(long),fp);
    */

    /*close tracer file*/
    fclose(fp);




    /*free buffer memory*/
    /*free(d);
    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    free(l);*/


}

