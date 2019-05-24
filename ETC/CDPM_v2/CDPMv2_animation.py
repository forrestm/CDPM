#!/usr/bin/python
###############################################################################
# Filename    : CDPMv2_animation.py 
# Created     : May 19, 2016
# Author      : Forrest
'''
Description   :
    This file will animate the response of the CDPM.
    '''
# Modified    :
###############################################################################



def animate(y, a, b, D, seconds, filename):
    '''
    Creates an animation of the CDPM's motion

           y : An array of the response in this order [x,y,beta,e]
     seconds : A integer of the length in seconds the animation will be
    filename : A string of the filename
           a : An integer of the plate width
           b : An integer of the plate height
           D : An integer of the rod length

    Returns: The animation in the directory the file is in.
    '''

    import numpy as np
    x_resp = y[:,0]
    y_resp = y[:,1]
    beta_resp = y[:,2]
    e_resp = y[:,3]
    plate_width = a
    plate_height = b
    rod_length = D
    # For the cables and top of rectangle
    left_point_x = (x_resp - (plate_width/2) * np.cos(beta_resp) +
                            (plate_height/2) * np.sin(beta_resp))
    left_point_y = (y_resp - (plate_width/2) * np.sin(beta_resp) -
                            (plate_height/2) * np.cos(beta_resp))

    right_point_x = (x_resp + (plate_width/2) * np.cos(beta_resp) +
                             (plate_height/2) * np.sin(beta_resp))
    right_point_y = (y_resp + (plate_width/2) * np.sin(beta_resp) -
                             (plate_height/2) * np.cos(beta_resp))
    # For the Rod
    bottom_x = (-(rod_length/2 - e_resp)*np.sin(beta_resp) + x_resp)
    bottom_y = ((rod_length/2 - e_resp)*np.cos(beta_resp) + y_resp)

    top_x = (-(-rod_length/2 - e_resp)*np.sin(beta_resp) + x_resp)
    top_y = ((-rod_length/2 - e_resp)*np.cos(beta_resp) + y_resp)

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    # Change some plot properties to make the video work and look better
    import matplotlib as mpl
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    mpl.rcParams['savefig.dpi'] = 160
    mpl.rcParams['savefig.bbox'] = 'standard'
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(111, aspect='equal')
    plt.ylim(20,0)
    plt.xlim(0,20)
    plt.xlabel('Horizontal Motion', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel('Vertical Motion', fontsize=22, weight='bold', labelpad=10)
    # plt.axes().set_aspect('equal')

    leftcable, = plt.plot([],[], linewidth=2, linestyle = '-',
                          label='leftcable', color='b')
    rightcable, = plt.plot([],[], linewidth=2, linestyle = '-',
                           label='rightcable', color='b')
    barLine, = plt.plot([],[], linewidth=2, linestyle = '-', label='Bar')
    patch = patches.Rectangle((0, 0), 0, 0, angle=0)

    centerG, = plt.plot([],[], 'ro', label='Center of Gravity')
    rod,    = plt.plot([],[], linewidth=6, linestyle = '-', label='rod',
                       color='r')

    def init():
        """ Initialize the lines in the plot """
        leftcable.set_data([], [])
        rightcable.set_data([], [])
        barLine.set_data([],[])
        centerG.set_data([],[])
        ax.add_patch(patch)
        rod.set_data([],[])

        return barLine, leftcable, rightcable, centerG, patch, rod,

    def animate_un(i):
        """ Update the plot for frame i """
        if not (i % 30): # print notice every 30th frame
            print('Processing frame {}'.format(i))

        rightcable.set_data([0, left_point_x[i]], [0, left_point_y[i]])
        leftcable.set_data([20, right_point_x[i]], [0, right_point_y[i]])
        barLine.set_data([left_point_x[i], right_point_x[i]], [left_point_y[i],
                          right_point_y[i]])
        centerG.set_data([x_resp[i]],[y_resp[i]])
        patch.set_width(plate_width)
        patch.set_height(plate_height)
        patch.set_xy([left_point_x[i], left_point_y[i]])
        patch._angle = np.rad2deg(beta_resp[i])
        rod.set_data([bottom_x[i], top_x[i]],[bottom_y[i],top_y[i]])

        return barLine, leftcable, rightcable, centerG, patch, rod,

    ani_un = animation.FuncAnimation(fig, animate_un, interval = 30,
            frames = 30*seconds, blit = True, init_func = init)

    ani_un.save('{}.mp4'.format(filename), bitrate = 2500, fps = 30)
