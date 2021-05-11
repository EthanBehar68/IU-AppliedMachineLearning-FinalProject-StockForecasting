import pickle


def pkl(filename, data):
    # do stuff
    f = open(filename, 'ab')
    pickle.dump(data, f)
    f.close()



def unpkl(filename):
    # do stuff
    f = open(filename, 'rb')
    return pickle.load(f)
