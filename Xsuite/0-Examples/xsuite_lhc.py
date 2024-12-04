###################################################################################
## Import libraries ##
###################################################################################

import numpy as np			## numerical computing library for Python.
import xobjects as xo		## memory management and compile/execute code.
import xpart as xp			## generate and manipulate particle ensembles.
import xtrack as xt			## create/import lattices, 1-particle tracking.
import xfields as xf		## compute EMfields with PIC/analytical dists.
import xdeps as xd			## part-matter sims and interface FLUKA/Geant4.

## Link to Xsuite   : https://xsuite.readthedocs.io/en/latest/index.html
## Link to Numpy    : https://numpy.org/

###################################################################################
## Lattice description ##
###################################################################################

line = xt.Line(
    elements=[xt.Drift(length=2.),
				xt.Multipole(knl=[0, 0.5], ksl=[0,0]),
				xt.Drift(length=1.),
				xt.Multipole(knl=[0, -0.5], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

###################################################################################
## Reference particle ##
###################################################################################

line.particle_ref = xt.Particles(p0c=6500e9,
                                q0=1, mass0=xt.PROTON_MASS_EV)

###################################################################################
## Create context ##
###################################################################################

context = xo.ContextCpu()         # For normal CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

###################################################################################
## Build tracker ##
###################################################################################

line.build_tracker(_context=context)

###################################################################################
## Twiss params ##
###################################################################################

tw = line.twiss(method='4d')
tw.cols['s betx bety'].show()

###################################################################################
## Generate particles ##
###################################################################################

n_part = 200
particles = line.build_particles(
                        x=np.random.uniform(-1e-3, 1e-3, n_part),
                        px=np.random.uniform(-1e-5, 1e-5, n_part),
                        y=np.random.uniform(-2e-3, 2e-3, n_part),
                        py=np.random.uniform(-3e-5, 3e-5, n_part),
                        zeta=np.random.uniform(-1e-2, 1e-2, n_part),
                        delta=np.random.uniform(-1e-4, 1e-4, n_part))

###################################################################################
## Track particles ##
###################################################################################

n_turns = 100
line.track(particles, num_turns=n_turns,
            turn_by_turn_monitor=True)

###################################################################################
## Extra readouts ##
###################################################################################

## Turn-by-turn data is available at:
line.record_last_track.x
line.record_last_track.px

###################################################################################
###################################################################################