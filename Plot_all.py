#!/usr/bin/env python
# encoding: utf-8

r""" Run the suite of tests for the 1d two-layer equations with rarefaction"""

from clawpack.riemann import layered_shallow_water_1D
import clawpack.clawutil.runclaw as runclaw
import clawpack.pyclaw.plot as plot


import os

import numpy as np

# Plot customization
import matplotlib

# Markers and line widths
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 6
matplotlib.rcParams['lines.markersize'] = 8

# Font Sizes
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# DPI of output images
matplotlib.rcParams['savefig.dpi'] = 100

# Need to do this after the above
import matplotlib.pyplot as plt
from copy import deepcopy
from copy import copy

from clawpack.pyclaw.solution import Solution

from multilayer.aux import bathy_index, kappa_index, wind_index
import multilayer.plot as plot



# ==========================================================
# ================ Compute the solution ====================
# ==========================================================

def dry_state(num_cells,eigen_method,entropy_fix,velocity,**kargs):
    r"""Run and plot a multi-layer dry state problem"""

    # Construct output and plot directory paths
    name = 'multilayer/dry_state_rarefaction'
    prefix = 'ml_e%s_m%s_fix' % (eigen_method, num_cells)

    if entropy_fix:
        prefix = "".join((prefix, "T"))
    else:
        prefix = "".join((prefix, "F"))
    outdir,plotdir,log_path = runclaw.create_output_paths(name, prefix, **kargs)

    # Redirect loggers
    # This is not working for all cases, see comments in runclaw.py
    for logger_name in ['pyclaw.io', 'pyclaw.solution', 'plot', 'pyclaw.solver',
                        'f2py','data']:
        runclaw.replace_stream_handlers(logger_name,log_path,log_file_append=False)

    # Load in appropriate PyClaw version
    if kargs.get('use_petsc',False):
        import clawpack.petclaw as pyclaw
    else:
        import clawpack.pyclaw as pyclaw


    # =================
    # = Create Solver =
    # =================
    if kargs.get('solver_type', 'classic') == 'classic':
        solver = pyclaw.ClawSolver1D(riemann_solver=layered_shallow_water_1D)
    else:
        raise NotImplementedError('Classic is currently the only supported solver.')

    # Solver method parameters
    solver.cfl_desired = 0.9
    solver.cfl_max = 1.0
    solver.max_steps = 5000
    solver.fwave = True
    solver.kernel_language = 'Fortran'
    solver.limiters = 3
    solver.source_split = 1

    # Boundary conditions
    solver.bc_lower[0] = 1
    solver.bc_upper[0] = 1
    solver.aux_bc_lower[0] = 1
    solver.aux_bc_upper[0] = 1

    # Set the before step function
    solver.before_step = lambda solver, solution:ml.step.before_step(
                                                               solver, solution)

    # Use simple friction source term
    solver.step_source = ml.step.friction_source

    # ============================
    # = Create Initial Condition =
    # ============================
    num_layers = 2

    x = pyclaw.Dimension(0.0, 1.0, num_cells)
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain, 2 * num_layers, 3 + num_layers)
    state.aux[ml.aux.kappa_index,:] = 0.0

    # Set physics data
    state.problem_data['g'] = 9.8
    state.problem_data['manning'] = 0.0
    state.problem_data['rho_air'] = 1.15e-3
    state.problem_data['rho'] = [0.95, 1.0]
    state.problem_data['r'] =   \
                     state.problem_data['rho'][0] / state.problem_data['rho'][1]
    state.problem_data['one_minus_r'] = 1.0 - state.problem_data['r']
    state.problem_data['num_layers'] = num_layers

    # Set method parameters, this ensures it gets to the Fortran routines
    state.problem_data['eigen_method'] = eigen_method
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['inundation_method'] = 2
    state.problem_data['entropy_fix'] = entropy_fix

    solution = pyclaw.Solution(state, domain)
    solution.t = 0.0

    # Set aux arrays including bathymetry, wind field and linearized depths
    ml.aux.set_jump_bathymetry(solution.state, 0.5, [-1.0, -1.0])
    ml.aux.set_no_wind(solution.state)
    ml.aux.set_h_hat(solution.state, 0.5, [0.0,-0.5], [0.0,-1.0])

    # Set sea at rest initial condition
    q_left = [0.5 * state.problem_data['rho'][0], -velocity*0.5 * state.problem_data['rho'][0],
              0.5 * state.problem_data['rho'][1], -velocity*0.5 * state.problem_data['rho'][1]]
    q_right = [0.5 * state.problem_data['rho'][0], velocity*0.5 * state.problem_data['rho'][0],
               0.5 * state.problem_data['rho'][1], velocity*0.5 * state.problem_data['rho'][1]]
    ml.qinit.set_riemann_init_condition(solution.state, 0.5, q_left, q_right)


    # ================================
    # = Create simulation controller =
    # ================================
    controller = pyclaw.Controller()
    controller.solution = solution
    controller.solver = solver

    # Output parameters
    controller.output_style = 3
    controller.nstepout = 1
    controller.num_output_times = 100
    controller.write_aux_init = True
    controller.outdir = outdir
    controller.write_aux = True


    # ==================
    # = Run Simulation =
    # ==================
    state = controller.run()

# ==========================================================
# ============ Compute the elements to plot ================
# ==========================================================

# Load bathymetery
b = Solution(0, path=plotdata.outdir,read_aux=True).state.aux[bathy_index,:]

# Load gravitation
g = Solution(0, path=plotdata.outdir,read_aux=True).state.problem_data['g']

# Load one minus r
one_minus_r=Solution(0, path=plotdata.outdir,read_aux=True).state.problem_data['one_minus_r']


def bathy(cd):
    return b

def kappa(cd):
    return Solution(cd.frameno,path=plotdata.outdir,read_aux=True).state.aux[kappa_index,:]

def wind(cd):
    return Solution(cd.frameno,path=plotdata.outdir,read_aux=True).state.aux[wind_index,:]

def h_1(cd):
    return cd.q[0,:] / rho[0]

def h_2(cd):
    return cd.q[2,:] / rho[1]

def eta_2(cd):
    return h_2(cd) + bathy(cd)

def eta_1(cd):
    return h_1(cd) + eta_2(cd)

def u_1(cd):
    index = np.nonzero(h_1(cd) > dry_tolerance)
    u_1 = np.zeros(h_1(cd).shape)
    u_1[index] = cd.q[1,index] / cd.q[0,index]
    return u_1

def u_2(cd):
    index = np.nonzero(h_2(cd) > dry_tolerance)
    u_2 = np.zeros(h_2(cd).shape)
    u_2[index] = cd.q[3,index] / cd.q[2,index]
    return u_2

def froude_number(u,h):
    Fr=abs(u)/((g*h)**(1/2))
    return Fr

def Richardson_number(cd):
    index=np.nonzero(h_1(cd)+h_2(cd)>0)
    Ri=np.zeros(h_1(cd).shape)
    Ri[index]=(u_1(cd)[index]-u_2(cd)[index])**2/(g* one_minus_r *(h_1(cd)[index]+h_2(cd)[index]))
    return(Ri)


def eigenspace_velocity(cd):
    #Problem for left and right
    total_depth_l = h_1(cd)+h_2(cd)
    total_depth_r = h_1(cd)+h_2(cd)
    mult_depth_l = h_1(cd)*h_2(cd)
    mult_depth_r = h_1(cd)*h_2(cd)
    s = np.zeros((4, len(h_1(cd))))
    s[0,:]=(h_1(cd)[:]*u_1(cd)[:] + h_2(cd)[:]*u_2(cd)[:]) / total_depth_l - np.sqrt(g*total_depth_l)
    s[1,:]=(h_2(cd)[:]*u_1(cd)[:] + h_1(cd)[:]*u_2(cd)[:]) / total_depth_l - np.sqrt(g*one_minus_r*mult_depth_l/total_depth_l * (1-(u_1(cd)[:]-u_2(cd)[:])**2/(g*one_minus_r*total_depth_l)))
    s[2,:]=(h_2(cd)[:]*u_1(cd)[:] + h_1(cd)[:]*u_2(cd)[:]) / total_depth_l + np.sqrt(g*one_minus_r*mult_depth_l/total_depth_l * (1-(u_1(cd)[:]-u_2(cd)[:])**2/(g*one_minus_r*total_depth_l)))
    s[3,:]=(h_1(cd)[:]*u_1(cd)[:] + h_2(cd)[:]*u_2(cd)[:]) / total_depth_l - np.sqrt(g*total_depth_l)
    if isinstance(s[1,:], complex) or isinstance(s[2,:], complex):
        print("Hyperbolicity lost for the speed at %s", cd.frameno)

    alpha=np.zeros((4,len(h_1(cd))))
    alpha[0:1,:]=((s[0:1,:]-u_1(cd)[:])**2 - g*h_1(cd)[:])/(g*h_1(cd)[:])
    alpha[2:3,:]=((s[2:3,:]-u_1(cd)[:])**2 - g*h_1(cd)[:])/(g*h_1(cd)[:])

    eig_vec = np.zeros((4,4,len(h_1(cd))))
    eig_vec[0,:,:] = 1.0
    eig_vec[1,:,:] = s[:,:]
    eig_vec[2,:,:] = alpha[:,:]
    eig_vec[3,:,:] = s[:,:]*alpha[:,:]
    return(eig_vec)

def eigenspace_velocity_3(cd):
    total_depth_l = h_1(cd)+h_2(cd)
    total_depth_r = h_1(cd)+h_2(cd)
    mult_depth_l = h_1(cd)*h_2(cd)
    mult_depth_r = h_1(cd)*h_2(cd)

    s = np.zeros(h_1(cd).shape)
    s = (h_2(cd)*u_1(cd) + h_1(cd)[:]*u_2(cd) / total_depth_l) + np.sqrt(g*one_minus_r*mult_depth_l/total_depth_l * (1-(u_1(cd)-u_2(cd))**2/(g*one_minus_r*total_depth_l)))
    alpha=np.zeros(h_1(cd).shape)
    alpha=((s-u_1(cd))**2 - g*h_1(cd))/(g*h_1(cd))

    eig_vec=alpha*s
    return(eig_vec)

def eigenspace_velocity_4(cd):
    total_depth_l = h_1(cd)+h_2(cd)
    total_depth_r = h_1(cd)+h_2(cd)
    mult_depth_l = h_1(cd)*h_2(cd)
    mult_depth_r = h_1(cd)*h_2(cd)

    s = np.zeros(h_1(cd).shape)
    s=(h_1(cd)*u_1(cd) + h_2(cd)*u_2(cd)) / total_depth_l - np.sqrt(g*total_depth_l)

    alpha=np.zeros(h_1(cd).shape)
    alpha=((s-u_1(cd))**2 - g*h_1(cd))/(g*h_1(cd))

    eig_vec=s*alpha
    return(eig_vec)

def eigenvalues(cd):
    index = np.nonzero(np.all([h_1(cd) > dry_tolerance, h_2(cd)>dry_tolerance], axis=0))
    eigenvalues1 = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    eigenvalues2 = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    eigenvalues3 = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    eigenvalues4 = np.zeros(min(h_1(cd).shape, h_2(cd).shape))

    frac = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    sqrt1 = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    sqrt2 = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    frac[index] = (h_1(cd)[index]*u_2(cd)[index] + h_2(cd)[index]*u_1(cd)[index])/(h_1(cd)[index] + h_2(cd)[index])
    sqrt1[index] = np.sqrt(g*(h_1(cd)[index] + h_2(cd)[index]))
    sqrt2[index] = np.sqrt(one_minus_r*g*h_1(cd)[index]*h_2(cd)[index]/(h_1(cd)[index] + h_2(cd)[index])*(1-(u_1(cd)[index]-u_2(cd)[index])**2/(one_minus_r*g*(h_1(cd)[index] + h_2(cd)[index]))))
    eigenvalues1[index] = frac[index]-sqrt1[index]
    eigenvalues2[index] = frac[index]-sqrt2[index]
    eigenvalues3[index] = frac[index]+sqrt2[index]
    eigenvalues4[index] = frac[index]+sqrt1[index]
    return([eigenvalues1, eigenvalues2, eigenvalues3, eigenvalues4])


def entropy(cd):
    index = np.nonzero(np.all([h_1(cd) > dry_tolerance, h_2(cd)>dry_tolerance], axis=0))
    entropy = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    h_1i=cd.q[0,index] / rho[0]
    h_2i=cd.q[2,index] / rho[1]
    u_1i=cd.q[1,index] / cd.q[0,index]
    u_2i=cd.q[3,index] / cd.q[2,index]
    entropy[index] = rho[0]*1/2*(h_1i*(u_1i)**2+g*(h_1i)**2) + rho[1]*1/2*(h_2i*(u_2i)**2+g*(h_2i)**2) + rho[0]*g*h_1i*h_2i + g*b[index]*(rho[0]*h_1i+rho[1]*h_2i)
    return entropy



def entropy_flux(cd):
    index = np.nonzero(np.all([h_1(cd)>dry_tolerance, h_2(cd)>dry_tolerance], axis=0))
    entropy_flux = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    h_1i=cd.q[0,index] / rho[0]
    h_2i=cd.q[2,index] / rho[1]
    u_1i=cd.q[1,index] / cd.q[0,index]
    u_2i=cd.q[3,index] / cd.q[2,index]
    entropy_flux[index] = rho[0]*(h_1i*(u_1i**2)/2+g*(h_1i**2))*u_1i + rho[1]*(h_2i*(u_2i**2)/2+g*(h_2i**2))*u_2i + rho[0]*g*h_1i*h_2i*(u_1i+u_2i) + g*b[index]*(rho[0]*h_1i*u_1i
     + rho[1]*h_2i*u_2i)
    return entropy_flux


def entropy_condition_is_valid(cd):

    index_t = int(cd.frameno)
    if index_t>0 :
        #entropy at t=0 doesn't exist
        (x,) = np.nonzero(np.all([h_1(cd)>dry_tolerance, h_2(cd)>dry_tolerance],axis=0))
        len_x = len(x)
        delta_t = Solution(index_t, path=plotdata.outdir,read_aux=True).t - Solution(index_t-1, path=plotdata.outdir,read_aux=True).t
        delta_x = cd.dx
        entropy_cond = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
        for index_x in range(len_x-1):
            index_x_next = index_x + 1
            entropy_flux_actual = entropy_flux(cd)[index_x]
            entropy_flux_prev = entropy_flux(cd)[index_x_next]
            entropy_next=entropy(cd)[index_x]
            entropy_actual=entropy(Solution(index_t-1, path=plotdata.outdir,read_aux=True))[index_x]

            entropy_cond[index_x]= entropy_next-entropy_actual + (delta_t/delta_x)*(entropy_flux_actual-entropy_flux_prev)


        #print(eigenspace_velocity(cd))

        return entropy_cond
    else :
        return([0]*500)

def froude_number_1(cd):
    index=np.nonzero(h_1(cd) > dry_tolerance)
    Fr=np.zeros(h_1(cd).shape)
    Fr[index] = froude_number(u_1(cd)[index],h_1(cd)[index])
    #print(Fr)
    return(Fr)

def froude_number_2(cd):
    index=np.nonzero(h_2(cd) > dry_tolerance)
    Fr=np.zeros(h_2(cd).shape)
    Fr[index] = froude_number(u_2(cd)[index],h_2(cd)[index])
    #print(Fr)
    return(Fr)

def composite_Froude_nb(cd):
    index = np.nonzero(np.all([h_1(cd)>dry_tolerance, h_2(cd)>dry_tolerance], axis=0))
    Fr = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    Fr1_carre = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    Fr2_carre = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
    Fr1_carre[index] = (u_1(cd)[index])**2/(one_minus_r * g * h_1(cd)[index])
    Fr2_carre[index] = (u_2(cd)[index])**2/(one_minus_r * g * h_2(cd)[index])
    Fr[index] = np.sqrt( Fr1_carre[index] + Fr2_carre[index] )
    #Fr[index] = np.sqrt(Fr1_carre[index] + Fr2_carre[index]  - one_minus_r * np.sqrt(Fr1_carre) * np.sqrt(Fr2_carre))
    return(Fr)

def dry_tolerance_(cd):
    return ([dry_tolerance]*(len(cd.q[1])) )

def limit_entropy_condition(cd):
    return ([0]*(len(cd.q[1])))

def flow_type(cd):
    return ([1]*len(cd.q[1]))

def charac(cd):
    values1=[0]*500
    values2=[0]*500

    (eigenvalues1, eigenvalues2, eigenvalues3, eigenvalues4) = eigenvalues(cd)
    values1[0:249]=[(k-500)*eigenvalues1[249]/1000 for k in range(0,500,2)]
    values2[0:249]=[(k-500)*eigenvalues2[249]/1000 for k in range(0,500,2)]
    values2[250:501]=[(k-500)*eigenvalues3[250]/1000 for k in range(500,1000,2)]
    values1[250:501]=[(k-500)*eigenvalues4[250]/1000 for k in range(500,1000,2)]
    return([values1, values2])


# ==========================================================
# ============ Compute the elements to plot ================
# ==========================================================

def plot_all(values_to_plot,nb_frames, plot_dir):

    plt.clearfigures()

    nb_test = len(values_to_plot)
    values_Froude_number=[]
    values_Richardson_number=[]
    values_eigenvalues=[]
    values_eigenspaces=[]
    values_momentum=[]
    values_depth=[]
    values_velocity=[]
    values_entropy=[]
    values_entropy_flux=[]
    values_entropy_condition=[]

    for i in range(nb_test):
        dry_state(500,2,False,values_to_plot[i],htmlplot=True)
        Froude_number=[]
        Richardson_number=[]
        eigenvalues=[]
        eigenspaces=[]
        momentum=[]
        depth=[]
        velocity=[]
        entropy=[]
        entropy_flux=[]
        entropy_condition=[]
        for t in range(nb_frames):
            cd=Solution(t,path=plotdata.outdir,read_aux=True)
            Froude_number += [composite_Froude_nb(cd)]
            Richardson_number += []
