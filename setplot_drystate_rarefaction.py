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


    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Window Settings
    xlimits = [0.0,1.0]
    xlimits_zoomed = [0.45,0.55]
    ylimits_momentum = [-5.0,5.0]
    ylimits_depth = [-1.0,0.5]
    ylimits_depth_zoomed = ylimits_depth
    ylimits_velocities = [-8.75,8.75]
    ylimits_velocities_zoomed = ylimits_velocities

    y_limits_depth_only=[0.,5.0]
    y_limits_entropy = [-5.0 , 0.5]
    y_limits_entropy_flux = [-0.023 , 0.003 ]
    y_limits_entropy_condition = y_limits_entropy_flux
    y_limits_entropy_shared =y_limits_entropy_flux
    y_limits_richardson = [-0.1,17.0]
    y_limits_Froude=[-1.0,10.0]
    y_limits_eigenspace=[-12.0,12.0]
    y_limits_eigenspace_4=[-30.0,15.0]
    y_limits_charac=[-5.0,10.0]



    # ========================================================================
    #  Depth and Momentum Plot
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name='Depth and Momentum')
    plotfigure.show = False

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
        ax1.set_title('Solution at t = %3.5f' % cd.t)
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

    # plotdata.printfigs = True                # print figures
    # plotdata.print_format = 'png'            # file format
    # plotdata.print_framenos = 'all'          # list of frames to print
    # plotdata.print_fignos = 'all'            # list of figures to print
    # plotdata.html = True                     # create html files of plots?
    # plotdata.html_homelink = '../README.html'   # pointer for top of index
    # plotdata.latex = True                    # create latex file of plots?
    # plotdata.latex_figsperline = 2           # layout of plots
    # plotdata.latex_framesperline = 1         # layout of plots
    # plotdata.latex_makepdf = False           # also run pdflatex?

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
        #plot depth 1
        #ax1.plot(x,h_1(cd),'k',color='green')
        #plot_depth 2
        #ax1.plot(x,h_2(cd),'k',color='orange')
        # Plot dry tolerance
        ax1.plot(x,dry_tolerance_(cd),'k',linestyle=':', color = 'red')

        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.5f' % cd.t)
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

    plotfigure = plotdata.new_plotfigure(name="Entropy")
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



	# # Parameters used only when creating html and/or latex hardcopy
    # # e.g., via pyclaw.plotters.frametools.printframes:
    #
    # plotdata.printfigs = True                # print figures
    # plotdata.print_format = 'png'            # file format
    # plotdata.print_framenos = 'all'          # list of frames to print
    # plotdata.print_fignos = 'all'            # list of figures to print
    # plotdata.html = True                     # create html files of plots?
    # plotdata.html_homelink = '../README.html'   # pointer for top of index
    # plotdata.latex = True                    # create latex file of plots?
    # plotdata.latex_figsperline = 2           # layout of plots
    # plotdata.latex_framesperline = 1         # layout of plots
    # plotdata.latex_makepdf = False           # also run pdflatex?


    # ====================================================
    # Plot Entropy flux
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="Entropy flux")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Entropy flux"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = 'auto'

    # Entropy
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = entropy_flux
    plotitem.color = 'b'
    plotitem.show = True



	# Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    # plotdata.printfigs = True                # print figures
    # plotdata.print_format = 'png'            # file format
    # plotdata.print_framenos = 'all'          # list of frames to print
    # plotdata.print_fignos = 'all'            # list of figures to print
    # plotdata.html = True                     # create html files of plots?
    # plotdata.html_homelink = '../README.html'   # pointer for top of index
    # plotdata.latex = True                    # create latex file of plots?
    # plotdata.latex_figsperline = 2           # layout of plots
    # plotdata.latex_framesperline = 1         # layout of plots
    # plotdata.latex_makepdf = False           # also run pdflatex?

    # ====================================================
    # Plot Entropy Condition
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="Entropy condition")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Entropy Condition"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = 'auto'

    # Entropy
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = entropy_condition_is_valid
    plotitem.color = 'b'
    plotitem.show = True

    # ====================================================
    # Plot Richardson Number
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="Kappa")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Richardson number"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = y_limits_richardson

    # Richardson
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = Richardson_number
    plotitem.color = 'b'
    plotitem.show = True

    # ========================================================================
    #  Froude number
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name = "Froude number")
    plotfigure.show = False

    def froude_same_plot(cd,xlimits):
        fig = mpl.gcf()
        fig.clf()

        # Get x coordinate values
        x = cd.patch.dimensions[0].centers

        # Create axes for each plot, sharing x axis
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212,sharex=ax1)

        # Froude number 1
        ax1.plot(x,froude_number_1(cd),'k',color = 'blue')
        # Plot limit flow type
        ax1.plot(x,flow_type(cd),'k',linestyle=':', color = 'red')


        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.5f' % cd.t)
        ax1.set_xlim(xlimits)
        ax1.set_ylim(y_limits_Froude)
        # ax1.set_xlabel('x')
        ax1.set_ylabel('Froude number 1')


        # froude_number_2
        ax2.plot(x,froude_number_2(cd),'k',color='green')
        # Plot limit flow type
        ax2.plot(x,flow_type(cd),'k',linestyle=':', color = 'red')


        # Add legend
        ax2.legend(loc=4)
        ax2.set_title('')
        # ax1.set_title('Layer Velocities')
        ax2.set_ylabel('Froude number 2')
        ax2.set_xlabel('x (m)')
        ax2.set_xlim(xlimits)
        ax2.set_ylim(y_limits_Froude)

        # This does not work on all versions of matplotlib
        try:
            mpl.subplots_adjust(hspace=0.1)
        except:
            pass

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:froude_same_plot(cd,xlimits)

    # ====================================================
    # Plot Richardson Number
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="Composite Froude")
    plotfigure.show = True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Composite Froude number"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = 'auto'

    # Composite Froude number
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = composite_Froude_nb
    plotitem.color = 'b'
    plotitem.show = True

    # ========================================================================
    #  Eigenspaces
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name = "Eigenspace")
    plotfigure.show = True

    def eigenspace_same_plot(cd,xlimits):
        fig = mpl.gcf()
        fig.clf()

        # Get x coordinate values
        x = cd.patch.dimensions[0].centers

        # Create axes for each plot, sharing x axis
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(223,sharex=ax1)
        ax3 = fig.add_subplot(222)
        ax4 = fig.add_subplot(224,sharex=ax3)

        # Eigenspace 1
        ax1.plot(x,eigenspace_velocity(cd)[0,0,:],'k',color = 'blue')
        ax1.plot(x,eigenspace_velocity(cd)[0,1,:],'k',color = 'red')
        ax1.plot(x,eigenspace_velocity(cd)[0,2,:],'k',color = 'green')
        ax1.plot(x,eigenspace_velocity(cd)[0,3,:],'k',color = 'grey')

        #Eigenspace 2
        ax2.plot(x,eigenspace_velocity(cd)[1,0,:],'k',color = 'blue')
        ax2.plot(x,eigenspace_velocity(cd)[1,1,:],'k',color = 'red')
        ax2.plot(x,eigenspace_velocity(cd)[1,2,:],'k',color = 'green')
        ax2.plot(x,eigenspace_velocity(cd)[1,3,:],'k',color = 'grey')

        #Eigenspace 3
        ax3.plot(x,eigenspace_velocity(cd)[2,0,:],'k',color = 'blue')
        ax3.plot(x,eigenspace_velocity(cd)[2,1,:],'k',color = 'red')
        ax3.plot(x,eigenspace_velocity(cd)[2,2,:],'k',color = 'green')
        ax3.plot(x,eigenspace_velocity(cd)[2,3,:],'k',color = 'grey')

        #Eigenspace 4
        ax4.plot(x,eigenspace_velocity(cd)[3,0,:],'k',color = 'blue')
        ax4.plot(x,eigenspace_velocity(cd)[3,1,:],'k',color = 'red')
        ax4.plot(x,eigenspace_velocity(cd)[3,2,:],'k',color = 'green')
        ax4.plot(x,eigenspace_velocity(cd)[3,3,:],'k',color = 'grey')

        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.5f' % cd.t)
        ax1.set_xlim(xlimits)
        ax1.set_ylim(y_limits_eigenspace)
        ax1.set_ylabel('Eigenspace 1')

        # Add legend
        ax2.legend(loc=4)
        ax2.set_title('')
        ax2.set_ylabel('Eigenspace 2')
        ax2.set_xlabel('x (m)')
        ax2.set_xlim(xlimits)
        ax2.set_ylim(y_limits_eigenspace)


        ax3.set_xlim(xlimits)
        ax3.set_ylim(y_limits_eigenspace)
        #ax3.set_ylabel('Eigenspace 3')

        # Add legend
        ax4.legend(loc=4)
        ax4.set_title('')
        #ax4.set_ylabel('Eigenspace 4')
        ax4.set_xlabel('x (m)')
        ax4.set_xlim(xlimits)
        ax4.set_ylim(y_limits_eigenspace)



        # This does not work on all versions of matplotlib
        try:
            mpl.subplots_adjust(hspace=0.1)
        except:
            pass

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:eigenspace_same_plot(cd,xlimits)

    # ========================================================================
    #  Zoom eigenspaces
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name = "Eigenspaces 4")
    plotfigure.show = True

    def eigenspace_same_plot_zoom(cd,xlimits):
        fig = mpl.gcf()
        fig.clf()

        # Get x coordinate values
        x = cd.patch.dimensions[0].centers

        # Create axes for each plot, sharing x axis
        ax1 = fig.add_subplot(111)


        #Eigenspace 4
        ax1.plot(x,eigenspace_velocity(cd)[3,0,:],'k',color = 'blue')
        ax1.plot(x,eigenspace_velocity(cd)[3,1,:],'k',color = 'red')
        ax1.plot(x,eigenspace_velocity(cd)[3,2,:],'k',color = 'green')
        ax1.plot(x,eigenspace_velocity(cd)[3,3,:],'k',color = 'grey')

        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.5f' % cd.t)
        ax1.set_xlim(xlimits)
        ax1.set_ylim(y_limits_eigenspace_4)
        ax1.set_ylabel('Eigenspace 4')



        # This does not work on all versions of matplotlib
        try:
            mpl.subplots_adjust(hspace=0.1)
        except:
            pass

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:eigenspace_same_plot_zoom(cd,xlimits)

    # ====================================================
    # Plot Eigenspace 3
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="Eigenspace 3")
    plotfigure.show = False

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Eigenspace 3"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = 'auto'

    # Richardson
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eigenspace_velocity_3
    plotitem.color = 'b'
    plotitem.show = True



    # ====================================================
    # Plot Eigenspace 4
    # ====================================================

    plotfigure = plotdata.new_plotfigure(name="Eigenspace 4")
    plotfigure.show = False

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Eigenspace 4"
    plotaxes.xlimits = xlimits
    plotaxes.ylimits = 'auto'

    # Richardson
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = eigenspace_velocity_4
    plotitem.color = 'b'
    plotitem.show = True

    # ========================================================================
    #  Zoom eigenspaces
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name = "Characteristic")
    plotfigure.show = True

    def charac_same_plot(cd,xlimits):
        fig = mpl.gcf()
        fig.clf()

        # Get x coordinate values
        x = cd.patch.dimensions[0].centers

        # Create axes for each plot, sharing x axis
        ax1 = fig.add_subplot(111)


        #Eigenspace 4
        ax1.plot(x,charac(cd)[0],'k',color = 'blue')
        ax1.plot(x,charac(cd)[1],'k',color = 'green')
        #ax1.plot(x,charac(cd)[2],'k',color = 'green')
        #ax1.plot(x,charac(cd)[3],'k',color = 'grey')
        ax1.plot(x,limit_entropy_condition(cd),'k',linestyle=':',color = 'red')

        # Remove ticks from top plot
        locs,labels = mpl.xticks()
        labels = ['' for i in xrange(len(locs))]
        mpl.xticks(locs,labels)

        # ax1.set_title('')
        ax1.set_title('Solution at t = %3.5f' % cd.t)
        ax1.set_xlim(xlimits)
        ax1.set_ylim(y_limits_charac)
        ax1.set_ylabel('t')



        # This does not work on all versions of matplotlib
        try:
            mpl.subplots_adjust(hspace=0.1)
        except:
            pass

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.afteraxes = lambda cd:charac_same_plot(cd,xlimits)


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
