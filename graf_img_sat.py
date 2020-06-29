import os
import numpy as np
import  rasterio as rt
import matplotlib
import  matplotlib.pyplot as plt
import folium
from pyrsgis.convert import changeDimension
from rasterio.plot import show, show_hist, reshape_as_raster, reshape_as_image, adjust_band


def plot_2D(Band1,Band2):
    fig,(ax, ax2) = plt.subplots(1,2, figsize=((10,3)))
    max_ref = np.amax([np.amax(Band1),np.amax(Band2)])
    
    img_nir = ax.imshow(Band1, cmap='Greys')
    ax.set_title("Banda 1")
    img_nir.set_clim(vmin=0, vmax=max_ref)
    fig.colorbar(img_nir, ax = ax)
    
    img_red = ax2.imshow(Band2, cmap='Greys')
    ax2.set_title("Banda 2")
    img_nir.set_clim(vmin=0, vmax=max_ref)
    fig.colorbar(img_red, ax = ax2)


def plot_3D(Imagen, Indice, nombre_indice):
    index = np.array([4,3,2])
    ##Se ajustan los colores a los valores dentro de las bandas###
    colors = Imagen[index,:,:].astype(np.float64)
    max_val = 5000
    min_val = 0
    
    colors[colors[:,:,:] > max_val] = max_val
    colors[colors[:,:,:] < min_val] = min_val
    
    for b in range(colors.shape[0]):
        colors[b, :, :] = colors[b, :, :] * 1 / (max_val - min_val)
    
    print(colors.shape)
    colors_reshaped = reshape_as_image(colors)
    print (colors_reshaped.shape)
    fig,axs = plt.subplots(1,2, figsize=((10,5)))
    ###Imagen a color###
    axs[0].imshow(colors_reshaped)
    axs[0].set_title('Color image')
    axs[1].imshow(Indice, cmap='RdYlGn')
    axs[1].set_title(nombre_indice)

def calculo_NDVI(NIR_BAND,RED_BAND):
    np.seterr(divide='ignore', invalid="ignore")
    ndvi = (NIR_BAND.astype(float)-RED_BAND.astype(float))/(NIR_BAND.astype(float)+RED_BAND.astype(float))
    return ndvi


def calculo_NDWI(NIR_BAND,GREEN_BAND):
    np.seterr(divide='ignore', invalid="ignore")
    ndwi = (GREEN_BAND.astype(float)-NIR_BAND.astype(float))/(GREEN_BAND.astype(float)+NIR_BAND.astype(float))
    return ndwi



def calculo_NDBI(NIR_BAND,SWIR1_BAND):
    np.seterr(divide='ignore', invalid="ignore")
    ndbi= (SWIR1_BAND.astype(float)-NIR_BAND.astype(float))/(SWIR1_BAND.astype(float)+NIR_BAND.astype(float))
    return ndbi

def mostrar_rgb(imagen):
    rgb = imagen[0:3] # Lee las bandas RGB
    rgb_norm = adjust_band(rgb) # normaliza los valores entre 1 y 0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    show(rgb_norm, ax=ax)
    ax.set_title("RGB ")

def mostrar_histograma(dataset):
    fig, ax = plt.subplots(figsize=(10,5))
    show_hist(dataset, ax=ax, bins=50, lw=0.0, stacked=False, alpha=0.3, 
               histtype='stepfilled', title="Histograma sobrelapado")
##############################################################################
if __name__ == '__main__':
    os.chdir("./imagenes")
    
    ###IMAGEN COMPLETA####
    dataset = rt.open("LE70220491999322EDC01_stack.gtif")
    img_dtst = dataset.read()
    BLUE_BAND = img_dtst[0,:,:]
    GREEN_BAND = img_dtst[1,:,:]
    RED_BAND = img_dtst[2,:,:]
    NIR_BAND = img_dtst[3,:,:]
    SWIR1_BAND = img_dtst[4,:,:]
    
    '''dtgt1 = dataset.transform## matriz geotransformada 
    nbandas = dataset.count
    filas, columnas = dataset.shape'''
    
    

