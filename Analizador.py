import os
import numpy as np
import  rasterio as rt
import matplotlib
import  matplotlib.pyplot as plt
import folium
from pyrsgis.convert import changeDimension
from rasterio.plot import show, show_hist


os.chdir("../Imagenes")

imagen_1 = "op_sf_09_04_2008_bandas.tif"

dataset1 = rt.open(imagen_1)
dtgt1 = dataset1.transform
nbandas = dataset1.count
filas, columnas = dataset1.shape

print('Number of bands in image: {n}\n'.format(n=nbandas))
print('Image geo-transform:\n{dtgt1}\n'.format(dtgt1=dtgt1))

#show(dataset1.read(5), dtgt1, cmap = 'viridis')
bandas = []
b=1
for n in range(nbandas-1):
    bandas.append(dataset1.read(b))
    b += 1


landsat_img = np.array(bandas, dtype=bandas[0].dtype)
print(landsat_img.shape)
show(landsat_img[[0,1,2],:,:], dtgt1)



#rt.plot.show_hist(landsat_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

