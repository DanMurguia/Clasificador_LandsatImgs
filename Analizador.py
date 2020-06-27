import os
import numpy as np
import  rasterio as rt
import matplotlib
import  matplotlib.pyplot as plt
import folium
from pyrsgis.convert import changeDimension
from rasterio.plot import show, show_hist, reshape_as_raster, reshape_as_image


os.chdir("../Imagenes")

imagen_1 = "op_sf_09_04_2008_bandas.tif"

dataset1 = rt.open(imagen_1)
img_dtst = dataset1.read()
dtgt1 = dataset1.transform
nbandas = dataset1.count
filas, columnas = dataset1.shape

print('Number of bands in image: {n}\n'.format(n=nbandas))
print('Image geo-transform:\n{dtgt1}\n'.format(dtgt1=dtgt1))

blue = dataset1.read(1)
green = dataset1.read(2)
red = dataset1.read(3)
NIR = dataset1.read(4)
SWIR = dataset1.read(5)
'''####2D plotting###def plot_2D():
print('Array shape before: {shp} (size is {sz})'.format(shp=red.shape, sz=red.size))

red_flat = np.ndarray.flatten(red) 
nir_flat = np.ndarray.flatten(NIR) 

print('Array shape after: {shp} (size is {sz})'.format(shp=red_flat.shape, sz=red_flat.size))

fig,(ax, ax2) = plt.subplots(1,2, figsize=((10,3)))
#ax.scatter(red_flat, nir_flat, color='r', marker='o')
#ax.set_xlabel('Red Reflectance')
#ax.set_ylabel('NIR label')
max_ref = np.amax([np.amax(NIR),np.amax(red)])

img_nir = ax.imshow(NIR, cmap=plt.cm.Greys)
ax.set_title("Banda NIR")
img_nir.set_clim(vmin=0, vmax=max_ref)
fig.colorbar(img_nir, ax = ax)

img_red = ax2.imshow(red, cmap=plt.cm.Greys)
ax2.set_title("Banda Roja")
img_nir.set_clim(vmin=0, vmax=max_ref)
fig.colorbar(img_red, ax = ax2)

'''######3D plotting########
fig,axs = plt.subplots(1,2, figsize=((10,5)))
index = np.array([4,3,2])
nir_band = img_dtst[3,:,:]
###Se ajustan los colores a los valores dentro de RGB###
colors = img_dtst[index,:,:].astype(np.float64)
max_val = 5000
min_val = 0

colors[colors[:,:,:] > max_val] = max_val
colors[colors[:,:,:] < min_val] = min_val

for b in range(colors.shape[0]):
    colors[b, :, :] = colors[b, :, :] * 1 / (max_val - min_val)
    
colors_reshaped = reshape_as_image(colors)
print (colors_reshaped.shape)
'''#######Calcular NDVI
red_band = img_dtst[2,:,:]
np.seterr(divide='ignore', invalid="ignore")
ndvi = (nir_band.astype(float)-red_band.astype(float))/(nir_band.astype(float)+red_band.astype(float))
###Calcular NDWI###
green_band = img_dtst[1,:,:]
np.seterr(divide='ignore', invalid="ignore")
ndwi = (green_band.astype(float)-nir_band.astype(float))/(green_band.astype(float)+nir_band.astype(float))'''
##Calcular NDBI##
swir_band = img_dtst[4,:,:]
np.seterr(divide='ignore', invalid="ignore")
ndvi= (swir_band.astype(float)-nir_band.astype(float))/(swir_band.astype(float)+nir_band.astype(float))
#mostrar imagen a color
axs[0].imshow(colors_reshaped)
axs[0].set_title('Color Image')

# Mostrar Indice
axs[1].imshow(ndwi, cmap='RdYlGn')
axs[1].set_title('NDWI')