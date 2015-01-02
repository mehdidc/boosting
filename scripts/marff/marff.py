def strlist_to_num(l):
    if l[-1]=="\n":
        l=l[0:-1] 
    a = l.split(",")
    a =  map(str_to_num, a)
    try:
        a[-1] = int(a[-1])
    except:
        pass
    return a

def str_to_num(s):
    try:
        a=float(s)
        return a
    except:
        return s

def read_arff(filename):
    data = (open(filename, "r").readlines())
    for i in xrange(len(data)):
        if data[i].startswith("@DATA"):
            break
    return data[0:i], map(strlist_to_num, data[i + 1:])

end_of_header = None
def read_arff_lazy(filename):
    header = True
    for line in open(filename, "r"):
        if line.startswith("@DATA"):
            header = False
            yield end_of_header
        else:
            if header == True:
                yield line
            else:
                yield strlist_to_num(line)

def write_arff(header, data, filename):
    print len(data)
    f = open(filename, "w")
    for h in header:
        f.write(h)
    f.write("@DATA\n")
    for d in data:
        f.write(",".join(map(str,d)) + "\n")
    f.close()

def export2arff(inputs, targets, targets_names, filename, relationName, output_folder):
        #print '%s/%s' % (output_folder, filename)
        print targets_names
        file = open('%s/%s' % (output_folder, filename), 'w')
        s = '@RELATION %s\n\n' % relationName
        for i in xrange(inputs.shape[1]):
            s += '@ATTRIBUTE value_%d NUMERIC\n' % i
        s += '@ATTRIBUTE class {'
        for name in targets_names:
            s += name + ','
        if s[-1] == ',' : s = s[:-1]
        s += '}\n\n'
        s += '@DATA\n'
        for i in xrange(inputs.shape[0]):
            for j in xrange(inputs.shape[1]):
                s += str(inputs[i,j]) + ','
            s += (targets_names[targets[i]] if type(targets[i]) == int else targets[i]) + '\n'
        if s[-1] == '\n' : s = s[:-1]
        file.write(s)
        file.close()
