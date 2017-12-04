# GP_LWPR_Quadrotor

This package is used to compare different GP based method and LWPR in the application of quadrotor control.
Gaussian Process (GP), 
LWPR, 
Sparse Spectrum GP, 
SSGP under uncertainty.

## Dependencies

Python packages 'GPy' and 'LWPR' are required to run the code. 'ssgpy.so' needs to be generated and placed in to the directory '/SSGP_Uncertainty'.

## Usage

Simulation data is generated in '/Sim_data/genSimData'. For example:
$ python WaypointsGroundTruth03.py

Experimental data is stored in '/Quadrotor_data'.

To run comparison on Simulation data:

    $ python SimCompare.py

To run comparison on Experimental data:

    $ python ExpCompare.py

