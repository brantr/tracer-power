#include <vector>
#ifndef  READ_ATHENA_TRACERS_H
#define  READ_ATHENA_TRACERS_H
#include "read_athena_header.hpp"

using namespace std;

struct tracer
{
  long id;
  float d;
  float m;
  float x[3];
  float v[3];
};

AthenaHeaderH *read_athena_tracers(char fname[], vector<tracer> *t);
void write_athena_tracers(char fname[], AthenaHeaderH *h, vector<tracer> t);


#endif /*READ_ATHENA_TRACERS_H*/
