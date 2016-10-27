#include <stdio.h>
#include <stdlib.h>
#include "read_athena_header.hpp"


/*! \fn AthenaHeader *ReadAthenaHeader(char *fname);
 *  \brief Function to read an Athena binary file header */
AthenaHeaderH *ReadAthenaHeaderFunc(FILE *fp)
{
  AthenaHeaderH *h;

  /* allocate the header */
  if(!(h = (AthenaHeaderH *) malloc(sizeof(AthenaHeaderH))))
  {
    printf("Error allocating Athena header.\n");
    fflush(stdout);
    return NULL;
  }

  /* read the header from the file */
  fread(h,1,sizeof(AthenaHeaderH),fp);

  /* return the header */
  return h;
}

/*! \fn void WriteAthenaHeader(FILE *fp, AthenaHeader *h);
 *  \brief Function to read an Athena binary file header */
void WriteAthenaHeaderFunc(FILE *fp, AthenaHeaderH *h)
{
  /* write the header to the file */
  fwrite(h,1,sizeof(AthenaHeaderH),fp);
}



/*  \fn void ShowAthenaHeader(AthenaHeader *h)
 *  \brief Function to print an Athena Header to screen */
void ShowAthenaHeaderFunc(AthenaHeaderH *h)
{
#ifdef ATHENA4
  printf("CoordSys = %d\n",h->CoordinateSystem);
#endif /*ATHENA4*/
  printf("nx       = %d\n",h->nx);
  printf("ny       = %d\n",h->ny);
  printf("nz       = %d\n",h->nz);
  printf("nvar     = %d\n",h->nvar);
  printf("nscalars = %d\n",h->nscalars);
  printf("ngrav    = %d\n",h->ngrav);

#ifdef ATHENA4
  printf("flag_tracers    = %d\n",h->flag_tracers);
#endif /*ATHENA4*/

  printf("gamma_minus_1   = %e\n",h->gamma_minus_1);
  printf("c_s_iso         = %e\n",h->c_s_iso);
  printf("t               = %e\n",h->t);
  printf("dt              = %e\n",h->dt);
}
