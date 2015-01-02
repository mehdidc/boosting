#!/usr/bin/env python

import sys
import pylab as plt
import os.path

HEADER =                []
VERBOSE =               0
TRAIN_PLOT_STYLE =      {'linestyle': 'dashed'}
TEST_PLOT_STYLE =       {'linestyle': 'solid'}
GUESS_TYPE =            True
INTERACTIVE =           False
USER_SELECTED_HEADER =  None
PLOT_FUNC = plt.plot

HEADER_PROVIDED = None

def _guess_file_label(file_name, type):

        if type == "suffix":
            # get rid of the extension
            if file_name[-4] == '.' : file_name = file_name[:-4]
            #if '_' in file_name: 
            label = '_'.join(file_name.split('_')[1:]) + ' '
            #_log('Label: ' + label, 2)
        elif type == "dir":
            path_split = file_name.strip('/').split('/')
            label = ' '.join(path_split[:-1]) + ' '
        elif type == "none":
            label = ''
        else:
        # auto
            if len(file_name.strip('/').split('/')) > 1:
                label = _guess_file_label(file_name, "dir")
            elif len(file_name.split('_')) > 1:
                label = _guess_file_label(file_name, "suffix")
            else:
                label = _guess_file_label(file_name, "none")
        return label 

def plotResults(filename):

    global HEADER
    global USER_SELECTED_HEADER
    global PLOT_FUNC
    global HEADER_PROVIDED

    with open(filename) as f:

        if HEADER_PROVIDED:
            headerFileName = HEADER_PROVIDED
        else:
            headerFileName = filename + '.header'
        if os.path.exists(headerFileName) :
            with open(headerFileName) as fh:
                header = fh.readline().split()
        else :
            header = f.readline().split()

        # check if the header is not a number
        try: 
            float(header[0])
            print "Error: Problem with the header..."
            print header
            sys.exit(1)
        except:
            pass

        column_indices = range(len(header))

        # drop the iteration column
        if header[0] == 't': 
            del header[0]
            del column_indices[0]

        # drop the Time column
        if header[-1].lower() == 'time': 
            del header[-1]
            del column_indices[-1]

        # interactive selection of the plots
        if INTERACTIVE and USER_SELECTED_HEADER == None:
            print USER_SELECTED_HEADER
            print "[+] Available columns:" 
            print ' --> ' + '\n --> '.join(set(header))
            USER_SELECTED_HEADER = raw_input("[+] Choose what to plot: ").split()
            while (not set(USER_SELECTED_HEADER).issubset(set(header))): 
                 USER_SELECTED_HEADER = raw_input("[+] Choose what to plot *among the given propositions* : ").split()


        if USER_SELECTED_HEADER != None:

            # dirty "if" (for MDDAG)
            if 'ep' in header and 'ep' not in USER_SELECTED_HEADER: USER_SELECTED_HEADER.append('ep')

            column_indices = [column_indices[i] for i in range(len(column_indices)) if header[i] in USER_SELECTED_HEADER]
            header = [h for h in header if h in USER_SELECTED_HEADER]


        # check if the headers of the different files to plot are all the same
        if HEADER: assert header == HEADER
        else: HEADER = header

        # get the results
        results = plt.loadtxt(f, usecols=column_indices)
        
        # add the file label to the legend
        file_label = _guess_file_label(filename, GUESS_TYPE)

        # whether it contains both train and test results
        num_datasets = 1
        if len(header) % 2 == 0:
            if header[0: len(header)/2] == header[len(header)/2: len(header)]:
                num_datasets = 2


        # dirty if 
        # mddag plot
        if header[0] == 'ep':
            # del header[0]
        
            X = results[::2,0]
            results = results[:,1:]
            num_plots = results.shape[1]
            
            for i in range(num_plots):
                p, = PLOT_FUNC(X,results[::2,i], label=header[i+1] + ' (%strain)'%file_label, **TRAIN_PLOT_STYLE)
                PLOT_FUNC(X,results[1::2,i], label=header[i+1] + ' (%stest)'%file_label, color=p.get_color(), **TEST_PLOT_STYLE)


        else:
            num_plots = results.shape[1] / num_datasets
            for i in range(num_plots):
                suffix = '' if num_datasets != 2 else ' (%strain)'%file_label
                p, = PLOT_FUNC(results[:,i], label=header[i] + suffix, **TRAIN_PLOT_STYLE)
                if num_datasets == 2:
                    suffix = ' (%stest)'%file_label
                    PLOT_FUNC(results[:,i + num_plots], color=p.get_color(), label=header[i] + suffix, **TEST_PLOT_STYLE)
        



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='*', help='list of results files to plot')
    parser.add_argument('-i', dest='interactive', action='store_true', help='interactively select the columns to plot', default=False)
    parser.add_argument('-s', metavar='column', dest='header', nargs='+', default=None, help='select the columns to plot')
    parser.add_argument('-o', metavar='filename', dest='save', help='the output name file (the figure to save)')
    parser.add_argument('-g', '--guess-name', dest='guess', help='Specify how to label the different plots. (suffix, dir(ectory) or none', default='auto')
    parser.add_argument('-t', '--plot-type', dest='plottype', help='The type of the plot: loglog, semilogx, semilogy or plot', default='plot')
    parser.add_argument('--header', metavar='headerfile', dest='headerfile', help='Specify the header file name.')



    args = parser.parse_args()
    INTERACTIVE = args.interactive
    USER_SELECTED_HEADER = args.header
    PLOT_FUNC = getattr(plt, args.plottype)

    GUESS_TYPE = args.guess
    HEADER_PROVIDED = args.headerfile

    if len(args.files) == 0: args.files = ['results.dta']

    # small font 
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')

    map(plotResults, args.files)
    plt.figure(1).set_facecolor('white')

    # legend outside
    ax = plt.subplot(111)
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # plt.figure(1).set_figheight(box.width)
    # plt.figure(1).set_figwidth(box.width)

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='best', prop = fontP)
    plt.xlabel('$t$')
    #plt.ylabel('$err$', rotation='horizontal')
    plt.title('Learning curve')
    plt.grid(axis='both')

    if args.save: 
        plt.savefig(args.save)
        print "[+] Figure saved as: " + args.save
    else:
        plt.show()