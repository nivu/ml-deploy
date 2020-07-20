import pickle

filename = 'dogs'

infile = open(filename, 'rb')
new_dict = pickle.load(infile)
infile.close()

print(new_dict)
print(type(new_dict))
