# this code snippet was used to create dataset list files for training and validation

HDF5Folder = "/media/local_hdd/Data/HDF5/"
definitions = HDF5Folder + "definition/"

for o in range(2, 3):
    for v in range(0, 3):
        test = 115+o*20+v
        filename = "test_{:d}k_order1.txt".format(test)
        with open(definitions + filename, "w") as file:
            file.write(HDF5Folder + "02_00{:d}000-00{:d}000_1_o1.h5\n".format(test, test+1))
        d = []
        for s in range(0,3):
            if s == v:
                continue
            d.append(100+o*20+s*5)
        filename = "train_{:d}k_{:d}k_order1.txt".format(d[0], d[1])
        with open(definitions + filename, "w") as file:
            for d_ in d:
                file.write(HDF5Folder + "20_00{:d}000-00{:d}000_1_o1.h5\n".format(d_, d_+5))