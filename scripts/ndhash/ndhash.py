from md5 import md5
def ndhash(ndarray):
    return md5("".join(map(str, ndarray.flatten()))).hexdigest()


