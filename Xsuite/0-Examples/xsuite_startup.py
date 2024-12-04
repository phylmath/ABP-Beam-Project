###################################################################################
## Import libraries ##
# Step 0: Bring in the necessary python and xsuite libraries for the simulation.
###################################################################################

import numpy as np
import xobjects as xo
import xtrack as xt

###################################################################################
## Lattice description ##
# Step 1: Define a lattice/line within Xsuite or import the lattice from MAD-X.
# Lattice elements can be manipulated later in the code 'line['quad_0'].knl[1] = 2'
###################################################################################

line = xt.Line(
    elements=[xt.Drift(length=2.),
				xt.Multipole(knl=[0, 0.5], ksl=[0,0]),
				xt.Drift(length=1.),
				xt.Multipole(knl=[0, -0.5], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

###################################################################################
## Reference particle ##
# Step 2: Set the mass, charge, energy of the reference particle in the simulation.
# These values will be automatically used for all other generated particles later.
###################################################################################

line.particle_ref = xt.Particles(p0c=6500e9,
                                q0=1, mass0=xt.PROTON_MASS_EV)

###################################################################################
## Create context ##
# Step 3: Choose the hardware to run the accelerator simulation.
###################################################################################

context = xo.ContextCpu()         # For normal CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

###################################################################################
## Build tracker ##
# Step 4: Associate a tracker object with the selected lattice and context.
###################################################################################

line.build_tracker(_context=context)

###################################################################################
## Twiss params ##
# Step 5: Define how twiss parameters need to be computed.
# 4d method means the RF cavities are off or the longitudinal position is frozen.
###################################################################################

tw = line.twiss(method='4d')
tw.cols['s betx bety'].show()

###################################################################################
## Generate particles ##
# Step 6: Define the values of the particles moving through the created lattice.
# The coordinates of each defined particle can be accessed via 'particles.x[20]'
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
## Step 7: Follow the evolution of particle parameters over a defined no. of turns.
###################################################################################

n_turns = 100
line.track(particles, num_turns=n_turns,
            turn_by_turn_monitor=True)

###################################################################################
## Extra readouts ##
## Step 8: Pull out any additional information needed from the simulation.
###################################################################################

## Turn-by-turn data is available at:
line.record_last_track.x
line.record_last_track.px

###################################################################################
###################################################################################