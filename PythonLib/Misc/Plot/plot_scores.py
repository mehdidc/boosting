#! /usr/bin/env python
# -*- coding: iso-8859-15 -*

import pylab as plt
from collections import defaultdict

plt.set_printoptions(linewidth=1000000, threshold='nan')

###################################

def read_adaboost_scores(filename):
    """
    Deprecated
    """

    with open(filename) as f:
        scores = f.readlines()
        assert len(scores) != 0

        # if the file contains only the last iteration scores
        if len(scores[0].split(',')) != 1:
            data = [[map(float, i.split(',')) for i in scores]]
        # or it contains iteration-wise  scores
        else:
            data = []
            for l in scores:
                if len(l.split(',')) == 1:
                    # currentIteration = int(l.split(',')[0])
                    data.append([])
                else:
                    data[-1].append(map(float, l.split(',')))
    return data
    

###################################

def read_adaboost_scores_opt(filename, iteration):
    """
    This version reads one line at a time
    """
    with open(filename) as f:

        line = f.readline()
        f.seek(0)
        data = []
        if len(line.split(',')) != 1:
            for line in f:
                data.append(map(float, line.split(',')))
        else:

            if iteration == -1:
                # get the last iteration number
                for line in f:
                    if len(line.split(',')) == 1: iteration = int(line.strip().split(',')[0])
                f.seek(0)

            for line in f:
                try:
                    if len(line.split(',')) == 1 and int(line.strip().split(',')[0]) == iteration:
                        break 
                except ValueError:
                    print "Something was wrong with the line:"
                    print '<', line, '>'
                    continue

            for line in f:
                if len(line.split(',')) == 1:
                    assert int(line.strip().split(',')[0]) == iteration + 1 # just for fun
                    break
                data.append(map(float, line.split(',')))

    return data
    

###################################

def read_mddag_scores(filename, num_classes):
    scores = []
    with open(filename) as f:
        for line in f:
            s = map(float, line.split()[1: num_classes + 1])
            if num_classes == 2 : 
                s[0] = s[1]
                s[1] = -s[0]
            scores.append(s)
            
    return scores

###################################

def read_labels_from_file(filename):
    with open(filename) as f:
        return f.read().split()

###################################

def get_classes_from_arff(filename):
    with open(filename) as f:
        for l in f:
            if '@attribute class' in l.lower() : 
                return [l.strip() for l in l.split('{')[1].split('}')[0].split(',')]

###################################

def plot_pairwise_scores(data, labels, class_list, nbins=25, stacked=False, normed=False):

    class_set = set(class_list)
    stash = set()

    # plt.figure(1).set_facecolor('white')

    for l1 in class_list:
        stash.add(l1)
        for l2 in class_set - stash:
            plt.figure(facecolor='white')

            delta = data[:, class_list.index(l1)] - data[:, class_list.index(l2)]

            d1 = delta[labels == l1]
            d2 = delta[labels == l2]

            if stacked:
                plt.hist([d1, d2], bins=nbins, label=[str(l1), str(l2)], color=['#4D0587', '#F26500'], alpha=0.6, stacked=stacked, normed=normed)
            else:
                plt.hist(d1, bins=nbins, label=str(l1), color='#4D0587', alpha=0.6, normed=normed) # normed=True, cumulative=1
                plt.hist(d2, bins=nbins, label=str(l2), color='#F26500', alpha=0.6, normed=normed) # cumulative=-1


            plt.title('$\Delta$(%s, %s)' % (str(l1), str(l2)))
            plt.axvline(0, color='grey')
            plt.legend(loc='best')

            # plt.savefig('/Users/djabbz/mnist_scores/delta_score_' + str(l2) + '-' + str(l1) + '.pdf', dpi=200)

###################################

def plot_best_scores(data, labels, class_list, nbins=25, stacked=False, normed=False):

    maxz = data.max(axis=1)
    # argmaxz = data.argmax(axis=1)

    plt.figure(facecolor='white')
    d = []
    plot_labels = []
    for l in class_list:
        d.append(maxz[labels == l]) #[argmaxz == class_list.index(l)]
        plot_labels.append(str(l))

    plt.hist(d, bins=nbins, label=plot_labels, alpha=0.6, stacked=True, normed=normed) # normed=True, cumulative=1


    plt.axvline(0, color='grey')
    plt.legend(loc='best')


###################################

def plot_classwise_scores(data, labels, class_list, nbins=25, stacked=False, normed=False, cumulative=0):

    plt.figure(facecolor='white', figsize=(20, 8))
    maxz = data.max(axis=1)
    # argmaxz = data.argmax(axis=1)

    d = []
    plot_labels = []
    for l in class_list:
        delta_max = maxz - data[:, class_list.index(l)] 
        d.append( delta_max[labels == l] )
        plot_labels.append(str(l))

    plt.hist(d, bins=nbins, label=plot_labels, alpha=0.6, stacked=False, normed=normed, cumulative=cumulative) # normed=True, cumulative=1

    plt.axvline(0, color='grey')
    plt.legend(loc='best')
    plt.title('$\max{F} - f_i$')

###################################

def plot_oaa_scores(data, labels, class_list, nbins=25, stacked=False, normed=False):

    for l in class_list:
        plt.figure(facecolor='white')
        d1 = data[labels == l]
        d2 = data[labels != l]

        if stacked:
            plt.hist([d1[:, class_list.index(l)], d2[:, class_list.index(l)]], bins=nbins,  stacked=stacked, normed=normed, label=[str(l), "non " + str(l)], alpha=0.6, color=['#4D0587', '#F26500'])
        else:    
            plt.hist(d1[:, class_list.index(l)], bins=nbins, label=str(l), alpha=0.6, normed=normed)
            plt.hist(d2[:, class_list.index(l)], bins=nbins, label="non " + str(l), alpha=0.6, normed=normed)


        plt.axvline(0, color='grey')
        plt.legend(loc='best')

###################################

def read_mddag_paths(filename, num_classes):

    with open(filename) as f:    
        paths = defaultdict(int)
        lengths = []
        forecast = []
        for l in f:
            path = l.split()[num_classes + 1:]
            forecast.append(int(l.split()[0]))
            lengths.append(len(path))
            path = ','.join(path)
            paths[path] += 1
            # print path
            # raw_input()

    return paths, lengths, forecast

###################################

def get_labels_from_arff(filename):

    return_labels = []
    with open(filename) as arffFile:
        # read the header
        while 1:
            line = arffFile.readline()
            if len(line.split()) == 0: continue
            if line[:5].lower() == "@data":
                break

        for l in arffFile :
            if len(l.split()) == 0: continue
            line = l.strip().split(',')
            label = line[-1].strip()

            # sparse or weighted representation in the arff
            if '{' in label:
                weight_label = label[1:-1].split()
                labels = []
                for (i, j) in zip(weight_label[::2], weight_label[1::2]):
                    if float(j) > 0 :
                        labels.append(i)

                assert len(labels) == 1
                label = labels[0]

            return_labels.append(label)
    return return_labels
                        
###################################
###################################

if __name__ == '__main__':

    import argparse, sys
    from time import time

    t = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('scoresfile', metavar='scores file name', help='The scores (posteriors) file name.')
    parser.add_argument('datafile', metavar='data file name', help='The data (in arff format file name or a list of labels if --labels is activated.')
    parser.add_argument('-s', '--save', metavar='file name', dest='save', help='The output name file (the figure to save)')
    parser.add_argument('-n', '--nbins', type=int, dest='nbins', metavar='integer', help='The number of bins used in the histogram.', default=25)
    parser.add_argument('-a', '--algo', type=str, dest='algo', metavar='algorithm', help='The algorithm used to produce the scores (adaboost or mddag)', default='adaboost')
    parser.add_argument('-i', '--iteration', type=int, dest='iteration', metavar='integer', 
        help='The iteration number (or -1 for the last iteration) for which the scores are plotted (only for adaboost).', default=-1)
    parser.add_argument('-m', '--mode', type=str, dest='mode', metavar='integer', 
        help='The mode of plotting the scores (pw (pairwise) or oaa (one against all) or best (for the distribution of the max) or cw (class-wise difference from the max) or eval{1,2} for mddag whyp evaluations, \
            "1" to distinguish classes, "2" for correct/incorrect classification.', default='cw', choices=['pw', 'oaa', 'best', 'cw', 'eval1', 'eval2'])
    parser.add_argument('--stacked', dest='stacked', action='store_true', help='Stacked histograms', default=False)
    parser.add_argument('--normed', dest='normed', action='store_true', help='Normalized histograms ( n/(len(x)*dbin) )', default=False)
    parser.add_argument('--cumulative', dest='cumulative', action='store_true', help='CDF (only for cw mode)', default=False)
    parser.add_argument('-l', '--labels', nargs='+', dest='labels', help='Provide the classes inline and indicate that the "datafile" positional argument is the labels file.', default=False)

    parser.add_argument('--bias', nargs='+', type=float, dest='bias', help='Experimental and rather useless, thus mandatory.')

    args = parser.parse_args()

    filename = args.scoresfile
    datafile = args.datafile

    # classes and labels
    if args.labels :
        labels = plt.array(read_labels_from_file(datafile))
        assert set(labels) == set(args.labels)
        class_list = args.labels # to keep the right order
    else :
        labels = plt.array(get_labels_from_arff(datafile))
        class_list = get_classes_from_arff(datafile)

    print 'Classes ->', class_list

    print '[+] Initz done (', time() - t,'sec ).'
    t = time()

    # what to plot
    if args.algo == 'adaboost':
        data = plt.array(read_adaboost_scores_opt(filename, args.iteration))
    elif args.algo == 'mddag':
        data = plt.array(read_mddag_scores(filename, len(class_list)))
    else:
        print '[!] Error, wrong algo:', args.algo
        sys.exit(1)

    assert len(labels) == len(data)

    print '[+] Data in memory (', time() - t,'sec ).'
    t = time()

    if args.bias:
        assert len(args.bias) % 2 == 0
        bias = args.bias
        for i,j in zip(map(int,bias[::2]), bias[1::2]):
            data[:,i] += j

    print '[+] Now plotting...'

    # how to plot it
    if args.mode == 'pw':
        plot_pairwise_scores(data, labels, class_list, args.nbins, args.stacked, args.normed)
    elif args.mode == 'best':
        plot_best_scores(data, labels, class_list, args.nbins, args.stacked, args.normed)
    elif args.mode == 'cw':
        plot_classwise_scores(data, labels, class_list, args.nbins, args.stacked, args.normed, -int(args.cumulative))
    elif args.mode == 'oaa':
        plot_oaa_scores(data, labels, class_list, args.nbins, args.stacked, args.normed)
    elif args.mode[:4] == 'eval':
        assert args.algo == 'mddag'
        paths, lengths, forecast = read_mddag_paths(filename, len(class_list))
        lengths = plt.array(lengths)
        forecast = plt.array(forecast)

        logs_class = []
        colors = ['#4D0587', '#F26500']
        if args.mode[4] == '1':
            for l in class_list:
                logs_class.append(lengths[labels == l])
            plot_labels = class_list
        elif args.mode[4] == '2':
            for l in (0,1):
                logs_class.append(lengths[forecast == l])
            plot_labels = ['error', 'correct']
        else:
            print '[!] Error, wrong mode:', args.mode
            sys.exit(1)

        plt.figure(facecolor='white')
        # if args.stacked:
        plt.hist(logs_class, label=plot_labels, bins=args.nbins, stacked=args.stacked, alpha=0.6, color=colors, normed=args.normed)
        # else:
        #     for i in range(len(logs_class)):
        #         plt.hist(logs_class[i], label=plot_labels[i], bins=args.nbins, alpha=0.6, color=colors[i], normed=args.normed)    
        plt.legend()

    else:
        print '[!] Error, wrong mode:', args.mode
        sys.exit(1)


    print '[+] Done (', time() - t,'sec ).'
    if args.save: 
        for fignum in plt.get_fignums():
            savefile = args.save
            savefile = '.'.join(savefile.split('.')[:-1]) + '_' + str(fignum) + '.' + savefile.split('.')[-1]
            plt.figure(fignum).savefig(savefile)
            print "[+] Figure saved as: " + args.save
    else:
        plt.show()