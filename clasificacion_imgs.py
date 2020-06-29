import rasterio
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import mapping
from sklearn.naive_bayes import GaussianNB

def unir_bandas(images_fp, img_dir, img_name):
    
    # Encuentra todas las imagenes en el directorio
    band_paths = [os.path.join(images_fp, f) for f in os.listdir(images_fp) if os.path.isfile(os.path.join(images_fp, f))]
    band_paths.sort()
    
    # Checa si el directorio existe
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # path para la imagen reconstruida
    img_fp = img_dir + img_name
    
    # Lee la metadat de la primer banda
    with rasterio.open(band_paths[0]) as src0:
        meta = src0.meta
    
    # Actualiza la metadata para mostrar el numero total de bandas
    meta.update(count = len(band_paths))
    
    # Lee cada banda y se apila
    with rasterio.open(img_fp, 'w', **meta) as dst:
        for id, layer in enumerate(band_paths, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))
    print(img_fp)
    return img_fp
                
####################paths de imagen#################                
images_fp = "./imagenes/BandasSeparadas/"
img_dir = "./imagenes/BandasJuntas/"
img_name= 'sentinel_bandas_juntas.tif'
img_fp = "./imagenes/BandasJuntas/sentinel_bandas_juntas.tif"   

##########obtención de la geometria del mapa#######  
dataset = rasterio.open(img_fp)
'''clipped_img = dataset.read([4,3,2])[:, 150:600, 250:1400]
print(clipped_img.shape)
fig, ax = plt.subplots(figsize=(10,7))
#se agrega la geotransformada para desplegarlo en coordenadasde longitud y latitud
show(clipped_img[:, :, :], ax=ax, transform=dataset.transform)'''

shapefile = gpd.read_file('./data/rcr/rcr_landcover.shp')
shapefile = shapefile.to_crs(dataset.crs)
# genera una lista de figuras geometricas de shapely
geoms = shapefile.geometry.values
geometry = geoms[0] 
# transforma las figuraas a formato geoJSON
caracteristicas = [mapping(geometry)]
#print(type(feature))
#print(feature)
dataset.close()

##############Creación de los datos de entrenamiento##############
x_train = np.array([], dtype=np.int8).reshape(0,8) # pixels de entrenamiento
y_train = np.array([], dtype=np.string_) # clases de entrenamiento

#Se extraen los valores de raster del poligono 
with rasterio.open(img_fp) as src:
    band_count = src.count
    for index, geom in enumerate(geoms):
        feature = [mapping(geom)]

        # la función mask regresa un arreglo de pixeles que se encuentran en las caracteristicas
        imagen_salida, transformada_salida = mask(src, caracteristicas, crop=True) 
        # Se eliminan los pixeles con valor igual 0 en todas las bandas
        imagen_salida_recortada = imagen_salida[:,~np.all(imagen_salida == 0, axis=0)]
        # Se eliminan los pixeles con valores de 255
        imagen_salida_recortada = imagen_salida_recortada [:,~np.all(imagen_salida_recortada == 255, axis=0)]
        # Reorganiza el arreglo en [conteo de pixeles, bandas]
        imagen_salida_remodelada = imagen_salida_recortada .reshape(-1, band_count)
        # Se adjuntan las etiquetas de clase al arreglo de y
        y_train = np.append(y_train,[shapefile["Classname"][index]] * imagen_salida_remodelada .shape[0]) 
        # stack the pizels onto the pixel array
        x_train = np.vstack((x_train,imagen_salida_remodelada))


labels = np.unique(shapefile["Classname"])
print('Los datos de entrenamiento incluyen {n} classes: {classes}\n'.format(n=labels.size,                                                                 classes=labels))
# La matriz X contien las caracteristicas y las Y las etiquetas
print('Tamaño de matriz x: {sz}'.format(sz=x_train.shape))
print('Tamaño de matriz y: {sz}'.format(sz=y_train.shape))

###############Mostrar valores espectrales############################
fig, ax = plt.subplots(1,3, figsize=[20,8])

# numeros 1-8
band_count = np.arange(1,9)

classes = np.unique(y_train)
for class_type in classes:
    band_intensity = np.mean(x_train[y_train==class_type, :], axis=0)
    ax[0].plot(band_count, band_intensity, label=class_type)
    ax[1].plot(band_count, band_intensity, label=class_type)
    ax[2].plot(band_count, band_intensity, label=class_type)
# Se imprimen las clases como lineas

# Add some axis labels
ax[0].set_xlabel('Band #')
ax[0].set_ylabel('Reflectance Value')
ax[1].set_ylabel('Reflectance Value')
ax[1].set_xlabel('Band #')
ax[2].set_ylabel('Reflectance Value')
ax[2].set_xlabel('Band #')
#ax[0].set_ylim(32,38)
ax[1].set_ylim(32,38)
ax[2].set_ylim(70,140)
#ax.set
ax[1].legend(loc="upper right")
# Add a title
ax[0].set_title('Band Intensities Full Overview')
ax[1].set_title('Band Intensities Lower Ref Subset')
ax[2].set_title('Band Intensities Higher Ref Subset')

###################3Entrenar modelo#################################
gnb = GaussianNB()
gnb.fit(x_train, y_train)