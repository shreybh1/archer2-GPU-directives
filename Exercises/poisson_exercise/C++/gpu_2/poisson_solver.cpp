

/// Contains geometrical and discretisation information

#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include "grid.h"
#include "timer.h"
#include "field.h"
#include "bc.h"
#include "jacobi.h"
#include "tools.h"


double get_distance( const field_t & field1 , const  field_t & field2, grid_t * g)
{
  auto field_phi_new=field1.get_data();
  auto field_phi_old=field2.get_data();

  double distance=0;

  for(int i=0; i<g->n[0] ; i++)
    for( int j=0; j<g->n[1] ; j++)
    {
      auto k = g->get_index(i,j);
      distance+= (field_phi_new[ k ] - field_phi_old[ k ]) * (field_phi_new[ k ] - field_phi_old[ k ]);
    }

  return distance;

}

// custom data mapper 
struct poisson_struct {
  field_t * phi_old;
  field_t * phi_new;
  field_t * rho; 
  int nFields, nIterationsOutput, nIterations; 
  grid_t current_grid; 
}; 

int main(int argc, char ** argv)
{
  /**
   * Definitions
   */
  poisson_struct poissonStruct; 
  //int nFields= 1; // number of equations to solve
  //int nIterations = 10000; // Total number of iterations
  int nOutputBlocks = 10; // Approximate number of times to perform output during the simulation
  poissonStruct.nFields= 1; // number of equations to solve
  poissonStruct.nIterations = 10000; // Total number of iterations

  size_t shape[2] = { 500 , 500 }; // Grid shape
  double left_box[2]= {-1,-1}; // Coordinate of the bottom left corner
  double right_box[2]= {1,1}; // Cooridinate of the top right corner
  
  /**
  * Pragma custom mapper
  */ 
  #pragma omp declare mapper(poisson_struct poissonStruct)  \
  map(poissonStruct, poissonStruct.current_grid,\
  poissonStruct.current_grid.n[0:2],poissonStruct.current_grid.dx[0:2],\
  poissonStruct.current_grid.start[0:2],poissonStruct.current_grid.end[0:2],\
  poissonStruct.phi_old, poissonStruct.phi_old->data[0:poissonStruct.current_grid.get_size()],\
  poissonStruct.rho, poissonStruct.rho->data[0:poissonStruct.current_grid.get_size()],\
  poissonStruct.nFields, poissonStruct.nIterationsOutput, poissonStruct.nIterations, \
  poissonStruct.phi_new, poissonStruct.phi_new->data[0:poissonStruct.current_grid.get_size()]) 

  /**
   * Initialization
   */

  //int nIterationsOutput = poissonStruct.nIterations/poissonStruct.nOutputBlocks; // Number of iterations between successive outputs
  poissonStruct.nIterationsOutput = poissonStruct.nIterations/nOutputBlocks; // Number of iterations between successive outputs
  std::cout << "Initialise" << std::endl;

  // auto current_grid = make_grid(left_box,right_box,shape);
  poissonStruct.current_grid = make_grid(left_box,right_box,shape);

  //field_t * rho = new field_t[poissonStruct.nFields];
  poissonStruct.rho = new field_t[poissonStruct.nFields];
  field_t * phi1 = new field_t[poissonStruct.nFields];
  field_t * phi2 = new field_t[poissonStruct.nFields];

  for (int iField=0 ; iField < poissonStruct.nFields; iField++ )
  {

    poissonStruct.rho[iField].init( &poissonStruct.current_grid);
    phi1[iField].init( &poissonStruct.current_grid);
    phi2[iField].init(&poissonStruct.current_grid);

    init_laplacian_gaussian( 10.0 ,poissonStruct.rho[iField].get_data(), &poissonStruct.current_grid );
    init_gaussian( 30 ,phi1[iField].get_data(),&poissonStruct.current_grid, 0.1 );
    init_gaussian( 30 ,phi2[iField].get_data(),&poissonStruct.current_grid, 0.1 );

    apply_drichlet_bc(poissonStruct.rho[iField].get_data(), &poissonStruct.current_grid, 0);
    apply_drichlet_bc(phi1[iField].get_data(), &poissonStruct.current_grid, 0);
    apply_drichlet_bc(phi2[iField].get_data(), &poissonStruct.current_grid, 0);

    print_to_file( poissonStruct.rho[iField].get_data(), &poissonStruct.current_grid, "rho" + std::to_string(iField) + ".dat" );
    print_to_file( phi1[iField].get_data(), &poissonStruct.current_grid, "phi" + std::to_string(iField) + "_" + std::to_string(0) + ".dat" );

  }

  //field_t * phi_old;
  //field_t * phi_new;

  poissonStruct.phi_new = phi1;
  poissonStruct.phi_old = phi2;

  timer compute_jacobi_timer("compute_jacobi");
  timer total_time_timer("total_time");

  total_time_timer.start();

  std::cout << "Start " << poissonStruct.nIterations << " iterations" << std::endl;
  int i=0;

  if (poissonStruct.nIterationsOutput < 0) 
  {
    std::cout << "No iterations per block."<<std::endl;
    exit(1);
  }

  while( i<poissonStruct.nIterations )
  {

    /**
     * Calculations 
     */

    // <------- OpenMP directives go here
    #pragma omp target data \
    map(to: poissonStruct)
    //map(to:current_grid,current_grid.n[0:2],current_grid.dx[0:2],\
    //current_grid.start[0:2],current_grid.end[0:2],\
    //phi_old, phi_old->data[0:current_grid.get_size()],\
    //rho, rho->data[0:current_grid.get_size()],\
    //nFields, nIterationsOutput, nIterations, \
    //phi_new, phi_new->data[0:current_grid.get_size()]) 
    {
    for (int iBlock=0;iBlock<poissonStruct.nIterationsOutput;iBlock++)
    {
      std::swap(poissonStruct.phi_old,poissonStruct.phi_new);

      compute_jacobi_timer.start();

      compute_jacobi(poissonStruct.phi_new,poissonStruct.phi_old,poissonStruct.rho,poissonStruct.nFields,&poissonStruct.current_grid);

      compute_jacobi_timer.stop();
      i++;
    }
    } 
    #pragma omp target data map(from:poissonStruct) 
    //#pragma omp target data map(from:phi_new, \
    //phi_new->data[0:current_grid.get_size()]) 
    // <------- OpenMP directives go here

    /**
     * Output 
     */
    for(int iField=0;iField<poissonStruct.nFields;iField++)
    {

      print_to_file(poissonStruct.phi_new[iField].get_data(),&poissonStruct.current_grid,"phi" + std::to_string(iField) + "_" + std::to_string(i) + ".dat" );
      std::cout << "Distance old - new field " << iField << " = " << get_distance( poissonStruct.phi_new[iField] , poissonStruct.phi_old[iField] ,&poissonStruct.current_grid) << std::endl;

    }


  }



  total_time_timer.stop();

  std::cout << "Finalize" << std::endl;

  std::cout << total_time_timer << std::endl;
  std::cout << compute_jacobi_timer << std::endl;


}
