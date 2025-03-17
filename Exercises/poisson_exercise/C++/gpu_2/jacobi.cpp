#include "jacobi.h"

/**
 * Makes a step of the jacobi interaction. Compute the new field, given the old field and the known densitiy field.
 */
void compute_jacobi(field_t * phi_new, field_t * phi_old, field_t * rho, int nFields,grid_t * g)
{
  int size_var = g->get_size(); 
  for(int iField=0 ; iField < nFields ; iField++ )
  {
    auto field_phi_new=phi_new[iField].get_data();
    auto field_phi_old=phi_old[iField].get_data();
    auto field_rho=rho[iField].get_data();
    #pragma omp target teams distribute parallel for
    //map(to:*g,g->n[0:2],g->dx[0:2],g->start[0:2],g->end[0:2],\
    //field_phi_old[0:size_var], \
    //field_rho[0:size_var])\
    //map(tofrom:field_phi_new[0:size_var])
    for( int j=0; j<g->n[1] ; j++){
      for(int i=0; i<g->n[0] ; i++) {
        auto k = g->get_index(i,j);

        auto nx = g->n[0];
        auto ny = g->n[1];

        auto dx = g->dx[0];
        auto dy = g->dx[1];
        double aspect2 = (dy/dx)*(dy/dx);

        field_phi_new[ k ] = 0.5*( (field_phi_old[k - 1] +  field_phi_old[k + 1 ])/(1 + 1/aspect2) + 
        (field_phi_old[k - (ny+2)] +  field_phi_old[(k + ny+2) ]) /(1 + aspect2) -
        field_rho[k] * dx*dx/( 1 + aspect2) );
      }
    } 
  }
};
