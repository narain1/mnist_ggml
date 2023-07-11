import os
import numpy as np
import struct

fname_img = os.path.join('models', 't10k-images.idx3-ubyte')


fimg = open(fname_img, 'rb')
magic_rn, size, rows, cols = struct.unpack('>IIII', fimg.read(16))
#img = np.array("B", fimg.read())
fimg.close()

print(size, rows, cols)
