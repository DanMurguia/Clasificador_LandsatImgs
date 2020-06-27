import os
import numpy as np
import  rasterio as rt
import matplotlib
import  matplotlib.pyplot as plt
import folium
from pyrsgis.convert import changeDimension
from rasterio.plot import show, show_hist, reshape_as_raster, reshape_as_image


os.chdir("/home/pixt3/Imágenes/")

'''
####BANDAS SEPARADAS####
landsat_bands =["/home/pixt3/Documentos/Señales/Proyecto/imagenes/2018-10-13, Sentinel-2B L1C, B02.tiff",#Blue
                "/home/pixt3/Documentos/Señales/Proyecto/imagenes/2018-10-13, Sentinel-2B L1C, B03.tiff",#Green
                "/home/pixt3/Documentos/Señales/Proyecto/imagenes/2018-10-13, Sentinel-2B L1C, B04.tiff",#Red
                "/home/pixt3/Documentos/Señales/Proyecto/imagenes/2018-10-13, Sentinel-2B L1C, B05.tiff",#NIR_BAND
                "/home/pixt3/Documentos/Señales/Proyecto/imagenes/2018-10-13, Sentinel-2B L1C, B06.tiff",#SWIR 1
                "/home/pixt3/Documentos/Señales/Proyecto/imagenes/2018-10-13, Sentinel-2B L1C, B07.tiff",]#SWIR 2

dataset1 = []

for band in landsat_bands:
    with rt.open(band) as f:
        dataset1.append(f.read(1))

img_dtst = np.array(dataset1, dtype=dataset1[0].dtype)
'''

###IMAGEN COMPLETA####
dataset = rt.open("/home/pixt3/Imágenes/LE70220491999322EDC01_stack.gtif")
img_dtst = dataset.read()


print(img_dtst.shape)

'''
dtgt1 = dataset1.transform
nbandas = dataset1.count
filas, columnas = dataset1.shape
print('Number of bands in image: {n}\n'.format(n=nbandas))
print('Image geo-transform:\n{dtgt1}\n'.format(dtgt1=dtgt1))
'''
#show(img_dtst[[2,1,0], :, :])

BLUE_BAND = img_dtst[0, :, :]
GREEN_BAND = img_dtst[1, :, :]
RED_BAND = img_dtst[2, :, :]
NIR_BAND = img_dtst[3, :, :]
SWIR1_BAND = img_dtst[4, :, :]
SWIR2_BAND = img_dtst[5, :, :]

'''
show(BLUE_BAND)
show(GREEN_BAND)
show(RED_BAND)
show(NIR_BAND)
show(SWIR1_BAND)
show(SWIR2_BAND)
'''

print('Array shape before: {shp} (size is {sz})'.format(shp=RED_BAND.shape, sz=RED_BAND.size))
red_flat = np.ndarray.flatten(RED_BAND) 
nir_flat = np.ndarray.flatten(NIR_BAND) 
print('Array shape after: {shp} (size is {sz})'.format(shp=red_flat.shape, sz=red_flat.size))

'''
####2D plotting###def plot_2D():
fig,(ax, ax2) = plt.subplots(1,2, figsize=((10,3)))
max_ref = np.amax([np.amax(NIR_BAND),np.amax(RED_BAND)])

img_nir = ax.imshow(NIR_BAND, cmap='Greys')
ax.set_title("Banda NIR")
img_nir.set_clim(vmin=0, vmax=max_ref)
fig.colorbar(img_nir, ax = ax)

img_red = ax2.imshow(RED_BAND, cmap='Greys')
ax2.set_title("Banda Roja")
img_nir.set_clim(vmin=0, vmax=max_ref)
fig.colorbar(img_red, ax = ax2)
'''

######3D plotting#######
index = np.array([4,3,2])
##Se ajustan los colores a los valores dentro de RGB###
colors = img_dtst[index,:,:].astype(np.float64)
max_val = 5000
min_val = 0

colors[colors[:,:,:] > max_val] = max_val
colors[colors[:,:,:] < min_val] = min_val

for b in range(colors.shape[0]):
    colors[b, :, :] = colors[b, :, :] * 1 / (max_val - min_val)

print(colors.shape)
colors_reshaped = reshape_as_image(colors)
print (colors_reshaped.shape)

'''
#####Calcular NDVI
np.seterr(divide='ignore', invalid="ignore")
ndvi = (NIR_BAND.astype(float)-RED_BAND.astype(float))/(NIR_BAND.astype(float)+RED_BAND.astype(float))
fig,axs = plt.subplots(1,2, figsize=((10,5)))
###Mostrar NDVI
axs[1].imshow(ndvi, cmap='RdYlGn')
axs[1].set_title('NDVI')
'''

'''
###Calcular NDWI###
np.seterr(divide='ignore', invalid="ignore")
ndwi = (GREEN_BAND.astype(float)-NIR_BAND.astype(float))/(GREEN_BAND.astype(float)+NIR_BAND.astype(float))
fig,axs = plt.subplots(1,2, figsize=((10,5)))
###Mostrar NWVI
axs[1].imshow(ndwi, cmap='RdYlGn')
axs[1].set_title('NDWI')
'''

##Calcular NDBI##
np.seterr(divide='ignore', invalid="ignore")
ndbi= (SWIR1_BAND.astype(float)-NIR_BAND.astype(float))/(SWIR1_BAND.astype(float)+NIR_BAND.astype(float))
fig,axs = plt.subplots(1,2, figsize=((10,5)))
###Mostrar NDBI
axs[1].imshow(ndbi, cmap='RdYlGn')
axs[1].set_title('NDBI')


###Imagen a color###
axs[0].imshow(colors_reshaped)
axs[0].set_title('Color image')

'''
#Mostrar Indice
axs[1].imshow(ndbi, cmap='RdYlGn')##se debe cambiar el indice a graficar
axs[1].set_title('NDWI')
'''
plt.show()