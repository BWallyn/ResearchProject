#!/usr/bin/env python
"""
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.

"""

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
import matplotlib.pyplot as mpl
from copy import deepcopy
from copy import copy

from clawpack.pyclaw.solution import Solution

from multilayer.aux import bathy_index, kappa_index, wind_index
import multilayer.plot as plot

# matplotlib.rcParams['figure.figsize'] = [6.0,10.0]


#--------------------------
def setplot(plotdata,rho,dry_tolerance):
#--------------------------

    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of pyclaw.plotters.data.ClawPlotData.
    Output: a modified version of plotdata.

    """

    def jump_afteraxes(current_data):
        # Plot position of jump on plot
        mpl.hold(True)
        mpl.plot([0.5,0.5],[-10.0,10.0],'k--')
        mpl.plot([0.0,1.0],[0.0,0.0],'k--')
        mpl.hold(False)
        mpl.title('Layer Velocities')

    # Load bathymetery
    b = Solution(0, path=plotdata.outdir,read_aux=True).state.aux[bathy_index,:]

	# Load gravitation
    g = Solution(0, path=plotdata.outdir,read_aux=True).state.problem_data['g']

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
        index_max = np.where(u_1 > 0.5)
        u_1[index_max] = 0.5
        return u_1

    def u_2(cd):
        index = np.nonzero(h_2(cd) > dry_tolerance)
        u_2 = np.zeros(h_2(cd).shape)
        u_2[index] = cd.q[3,index] / cd.q[2,index]

        #limit the velocity
        index_max = np.where(u_2 > 0.5)
        u_2[index_max] = 0.5
        return u_2


    def entropy_at_x(q, index):
        h_1i=q[0,index] / rho[0]
        h_2i=q[2,index] / rho[1]
        u_1i=q[1,index] / q[0,index]
        u_2i=q[3,index] / q[2,index]
        entropy_at_x = rho[0]*1/2*(h_1i*(u_1i)**2+g*(h_1i)**2) + rho[1]*1/2*(h_2i*(u_2i)**2+g*(h_2i)**2) + rho[0]*g*h_1i*h_2i + g*b[index]*(rho[0]*h_1i+rho[1]*h_2i)
        return entropy_at_x

    def entropy(cd):
        index = np.nonzero(np.all([h_1(cd) > dry_tolerance, h_2(cd)>dry_tolerance], axis=0))
        entropy = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
        entropy[index] = entropy_at_x(cd.q, index)
        return entropy

    def entropy_flux_at_x(q, index):
        h_1i=q[0,index] / rho[0]
        h_2i=q[2,index] / rho[1]
        u_1i=q[1,index] / q[0,index]
        u_2i=q[3,index] / q[2,index]
        print(u_2i)
        entropy_flux_at_x = rho[0]*(h_1i*(u_1i**2)/2+g*(h_1i**2))*u_1i + rho[1]*(h_2i*(u_2i**2)/2+g*(h_2i**2))*u_2i + rho[0]*g*h_1i*h_2i*(u_1i+u_2i) + g*b[index]*(rho[0]*h_1i*u_1i
         + rho[1]*h_2i*u_2i)
        #print(entropy_flux_at_x)
        return entropy_flux_at_x

    def entropy_flux(cd):
        index = np.nonzero(np.all([h_1(cd)>dry_tolerance, h_2(cd)>dry_tolerance], axis=0))
        entropy_flux = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
        entropy_flux[index] = entropy_flux_at_x(cd.q,index)

        return entropy_flux


    def entropy_condition_is_valid(cd):

        index_t = int(cd.frameno)
        if index_t>0 :
            #entropy at t=0 doesn't exist
            delta_t = Solution(index_t, path=plotdata.outdir,read_aux=True).t - Solution(index_t-1, path=plotdata.outdir,read_aux=True).t
            delta_x = cd.dx
            entropy_cond = np.zeros(min(h_1(cd).shape, h_2(cd).shape))
            index_x = np.nonzero(np.all([h_1(cd)>dry_tolerance, h_2(cd)>dry_tolerance],axis=0))
            index_x_array = index_x[0]
            index_x_next=(deepcopy(index_x_array[1:]),)
            index_x_array = index_x_array[:len(index_x_array)-1]
            index_x=(index_x_array,)
            entropy_flux_actual = entropy_flux_at_x(cd.q, index_x)
            entropy=entropy_at_x(cd.q,index_x)
            entropy_prev=entropy_at_x(Solution(index_t-1, path=plotdata.outdir,read_aux=True).q,index_x)
            entropy_flux_prev = entropy_flux_at_x(cd.q, index_x_next)
            entropy_cond[index_x]= entropy-entropy_prev + (delta_t/delta_x)*(entropy_flux_actual-entropy_flux_prev)
            # if index_t == 67 or index_t==68 or index_t==69 or index_t==70 :
            #     print('=======================================')
            #     print(entropy_flux_actual)
            #     print(entropy_flux_prev)
            #     print(delta_t/delta_x)
            #     print(entropy)
            #     print(entropy_prev)
            #     print(entropy_cond)
            return entropy_cond
        else :
            return([0]*500)

    def dry_tolerance_(cd):
        return ([dry_tolerance]*(len(cd.q[1])) )

    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Window Settings
    xlimits = [0.0,1.0]
    xlimits_zoomed = [0.45,0.55]
    ylimits_momentum = [-0.004,0.004]
    ylimits_depth = [-1.0,0.2]
    ylimits_depth_zoomed = ylimits_depth
    ylimits_velocities = [-0.75,0.75]
    ylimits_velocities_zoomed = ylimits_velocities
    y_limits_entropy = [-5.0 , 0.5]
    y_limits_entropy_flux = [-0.023 , 0.003 ]
    y_limits_entropy_condition = y_limits_entropy_flux
    y_limits_entropy_shared =y_limits_entropy_flux


    # ========================================================================
    #  Depth and Momentum Plot
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name='Depth and Momentum')
    plotfigure.show = True

    def twin_axes(cd,xlimits):
        fig = mpl.gcf()
        fig.clf()

        # Get x coordinate values
        x = cd.patch.dimensions[0].centers

        # Create axes for each plot, sharing x axis
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212,sharex=ax1)     # the velocity scale

        # Bottom layer
        ax1.fill_between(x,bathy(cd),eta_1(cd),color=plot.bottom_color)
        # Top Layer
        ax1.fill_between(x,eta_1(cd),eta_2(cd),color=plot.top_color)
        # Plot bathy
        ax1.plot(x,bathy(cd),'k',linestyle=plot.bathy_linestyle)
        # Plot internal layer
        ax1.plot(x,eta_2(cd),'k',linestyle=plot.internal_linestyle)
        # Plot surface
        ax1.plot(x,eta_1(cd),'k',linestyle=plot.surface_linestyle)

        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.2f' % cd.t)
        ax1.set_xlim(xlimits)
        ax1.set_ylim(ylimits_depth)
        # ax1.set_xlabel('x')
        ax1.set_ylabel('Depth (m)')

        # Bottom layer velocity
        bottom_layer = ax2.plot(x,u_2(cd),'k',linestyle=plot.internal_linestyle,label="Bottom Layer Velocity")
        # Top Layer velocity
        top_layer = ax2.plot(x,u_1(cd),'b',linestyle=plot.surface_linestyle,label="Top Layer velocity")


        # Add legend
        ax2.legend(loc=4)
        ax2.set_title('')
        # ax1.set_title('Layer Velocities')
        ax2.set_ylabel('Velocities (m/s)')
        ax2.set_xlabel('x (m)')
        ax2.set_xlim(xlimits)
        ax2.set_ylim(ylimits_velocities)

        # This does not work on all versions of matplotlib
        try:
            mpl.subplots_adjust(hspace=0.1)
        except:
            pass

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:twin_axes(cd,xlimits)

    # ========================================================================
    #  Fill plot zoom
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name='full_zoom')

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:twin_axes(cd,xlimits_zoomed)

    # ========================================================================
    #  Momentum
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name="momentum")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Momentum"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = ylimits_momentum

    # Top layer
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 1
    plotitem.plotstyle = 'b-'
    plotitem.show = True

    # Bottom layer
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 3
    plotitem.plotstyle = 'k--'
    plotitem.show = True

    # ========================================================================
    #  h-valuesplot
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name='depths')
    plotfigure.show = False

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(2,1,1)'
    plotaxes.title = 'Depths'
    plotaxes.xlimits = xlimits
    plotaxes.afteraxes = jump_afteraxes
    # plotaxes.ylimits = [-1.0,0.5]
    # plotaxes.xlimits = [0.45,0.55]
    # plotaxes.xlimits = [0.0,2000.0]
    # plotaxes.ylimits = [-2000.0,100.0]

    # Top layer
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-'
    plotitem.color = (0.2,0.8,1.0)
    plotitem.show = True

    # Bottom layer
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 2
    plotitem.color = 'b'
    plotitem.plotstyle = '-'
    plotitem.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(2,1,2)'
    plotaxes.title = 'Depths Zoomed'
    plotaxes.afteraxes = jump_afteraxes
    # plotaxes.xlimits = [0.0,1.0]
    # plotaxes.ylimits = [-1.0,0.5]
    plotaxes.xlimits = [0.45,0.55]
    # plotaxes.xlimits = [0.0,2000.0]
    # plotaxes.ylimits = [-2000.0,100.0]

    # Top layer
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0
    plotitem.plotstyle = 'x'
    plotitem.color = (0.2,0.8,1.0)
    plotitem.show = True

    # Bottom layer
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 2
    plotitem.color = 'b'
    plotitem.plotstyle = '+'
    plotitem.show = True

    # ========================================================================
    #  Plot Layer Velocities
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name='velocities')
    plotfigure.show = False

    plotaxes = plotfigure.new_plotaxes()
    #plotaxes.axescmd = 'subplot(1,2,1)'
    plotaxes.title = "Layer Velocities"
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'

    # Top layer
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = u_1
    plotitem.color = 'b'
    plotitem.show = True

    # Bottom layer
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.color = (0.2,0.8,1.0)
    plotitem.plot_var = u_2
    plotitem.show = True

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    # ========================================================================
    #  h-values
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name = "Depths and dry tolerance")
    plotfigure.show = True

    def depths_same_plot(cd,xlimits):
        fig = mpl.gcf()
        fig.clf()

        # Get x coordinate values
        x = cd.patch.dimensions[0].centers

        # Create axes for each plot, sharing x axis
        ax1 = fig.add_subplot(111)

        # Bottom layer
        ax1.fill_between(x,bathy(cd),eta_1(cd),color=plot.bottom_color)
        # Top Layer
        ax1.fill_between(x,eta_1(cd),eta_2(cd),color=plot.top_color)
        # Plot bathy
        ax1.plot(x,bathy(cd),'k',linestyle=plot.bathy_linestyle)
        # Plot internal layer
        ax1.plot(x,eta_2(cd),'k',linestyle=plot.internal_linestyle)
        # Plot surface
        ax1.plot(x,eta_1(cd),'k',linestyle=plot.surface_linestyle)
        # Plot dry tolerance
        ax1.plot(x,dry_tolerance_(cd),'k',linestyle=':', color = 'red')

        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.2f' % cd.t)
        ax1.set_xlim(xlimits)
        ax1.set_ylim(ylimits_depth)
        # ax1.set_xlabel('x')
        ax1.set_ylabel('Depth (m)')


        # # This does not work on all versions of matplotlib
        # try:
        #     mpl.subplots_adjust(hspace=0.1)
        # except:
        #     pass

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:depths_same_plot(cd,xlimits)

    # ====================================================
    # Plot Entropy
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="entropy")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Entropy"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = 'auto'

    # Entropy
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = entropy
    plotitem.color = 'b'
    plotitem.show = True



	# Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?


    # ====================================================
    # Plot Entropy flux
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="entropy flux")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Entropy flux"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = y_limits_entropy_flux

    # Entropy
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = entropy_flux
    plotitem.color = 'b'
    plotitem.show = True



	# Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    # ====================================================
    # Plot Entropy Condition
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="entropy condition")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Entropy Condition"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = y_limits_entropy_condition

    # Entropy
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = entropy_condition_is_valid
    plotitem.color = 'b'
    plotitem.show = True



	# Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    return plotdata
