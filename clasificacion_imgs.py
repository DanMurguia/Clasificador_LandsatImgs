import rasterio
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.plot import show,  reshape_as_raster, reshape_as_image
from rasterio.windows import Window
from shapely.geometry import mapping
from sklearn.naive_bayes import GaussianNB
from graf_img_sat import calculo_NDBI, calculo_NDVI, calculo_NDWI

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
                
##########obtención de la geometria del mapa#######
def obtener_geometria(img_path, shp_path):
    dataset = rasterio.open(img_path)
    '''clipped_img = dataset.read([4,3,2])[:, 150:600, 250:1400]
    #print(clipped_img.shape)
    fig, ax = plt.subplots(figsize=(10,7))
    #se agrega la geotransformada para desplegarlo en coordenadasde longitud y latitud
    show(clipped_img[:, :, :], ax=ax, transform=dataset.transform)'''
    
    shapefile = gpd.read_file(shp_path)
    shapefile = shapefile.to_crs(dataset.crs)
    # genera una lista de figuras geometricas de shapely
    geoms = shapefile.geometry.values
    '''geometry = geoms[0] 
    print(type(geometry))
    print(geometry)
    # transforma las figuraas a formato geoJSON
    caracteristicas = [mapping(geometry)]
    print(type(caracteristicas))
    print(caracteristicas)
    imagen_salida, transformada_salida = mask(dataset, caracteristicas, crop=True)
    print("out image shape:", imagen_salida.shape)'''
    dataset.close()
    return geoms


##############Creación de los datos de entrenamiento##############
def matrices_entrenamiento(img_path, geoms, shp_path):
    X_train = np.array([], dtype=np.int8).reshape(0,8) # pixeles para entrenamiento
    y_train = np.array([], dtype=np.string_) # etiquetas de entrenamiento
    shapefile = gpd.read_file(shp_path)
    
    #Se extraen los valores de raster del poligono 
    with rasterio.open(img_path) as src:
        band_count = src.count
        for index, geom in enumerate(geoms):
            caracteristica = [mapping(geom)]
    
            # La funcion mask regresa un arreglo de pixeles de raster contenidos en esta caracteristica de la img
            imagen_salida, transformada_salida = mask(src, caracteristica, crop=True) 
            # se eliminana los pixeles con valor de 0 para todas las bandas
            imagen_salida_recortada = imagen_salida[:,~np.all(imagen_salida == 0, axis=0)]
            # se elimijnan los pixeles con valores >= 255 para todas las bandas
            imagen_salida_recortada = imagen_salida_recortada[:,~np.all(imagen_salida_recortada == 255, axis=0)]
            # se reacomoda el arreglo en orden [conteo de pixeles, bandas]
            imagen_salida_reshaped = imagen_salida_recortada.reshape(-1, band_count)
            # adjuntan las etiquetas de clases al arreglo y
            y_train = np.append(y_train,[shapefile["Classname"][index]] * imagen_salida_reshaped.shape[0]) 
            # se apilan los pixeles en el arreglo de pixeles
            X_train = np.vstack((X_train,imagen_salida_reshaped))
    
    
    # Muestra las etiquetas de clasificacion
    labels = np.unique(shapefile["Classname"])
    print('Los datos de entrenamiento incluyen {n} clases: {classes}\n'.format(n=labels.size, classes=labels))
    
    # La matriz "X" contiene las caracteristicas de nuestra imagen y la matriz "y" contiene las etiquetas de clases
    #print('tamaño de la matriz X de entrenamiento: {sz}'.format(sz=X_train.shape))
    #print('tamaño de la matriz y de entrenamiento {sz}'.format(sz=y_train.shape))
    return X_train, y_train

###################Entrenar modelo#################################
def entrenar_clasificador(X_train, y_train) :   
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

###############Mostrar valores espectrales############################
def val_espectral(X_train, y_train):
    fig, ax = plt.subplots(1,3, figsize=[20,8])
    
    # numeros 1-8
    band_count = np.arange(1,9)
    
    classes = np.unique(y_train)
    for class_type in classes:
        band_intensity = np.mean(X_train[y_train==class_type, :], axis=0)
        ax[0].plot(band_count, band_intensity, label=class_type)
        ax[1].plot(band_count, band_intensity, label=class_type)
        ax[2].plot(band_count, band_intensity, label=class_type)
    # Se imprimen las clases como lineas
    
    # Add some axis labels
    ax[0].set_xlabel('Banda #')
    ax[0].set_ylabel('Valor de Reflectancia')
    ax[1].set_ylabel('Valor de Reflectancia')
    ax[1].set_xlabel('Banda #')
    ax[2].set_ylabel('Valor de Reflectancia')
    ax[2].set_xlabel('Banda #')
    #ax[0].set_ylim(32,38)
    ax[1].set_ylim(32,38)
    ax[2].set_ylim(70,140)
    #ax.set
    ax[1].legend(loc="upper right")
    # Add a title
    ax[0].set_title('Vista general de Intensidades de BAnda')
    ax[1].set_title('Subset de Intensisdades de Banda más Bajas')
    ax[2].set_title('Subset de Intensisdades de Banda más Altas')

################Función de ayuda para interpretar clases ########################
def str_class_to_int(class_array):
    class_array[class_array == 'Subtidal Haline'] = 0
    class_array[class_array == 'WetSand'] = 1
    class_array[class_array == 'Emergent Wetland'] = 2
    class_array[class_array == 'Sand'] = 3
    class_array[class_array == 'Herbaceous'] = 4
    class_array[class_array == 'Forested Wetland'] = 5
    return class_array.astype(int)

########################prediccion de clases en imagen#########################
def predecir_clases(clasificador, path_imagen):    
    with rasterio.open(path_imagen) as src: 
        img = src.read()

        
        
    print(img.shape)
    reshaped_img = reshape_as_image(img)
    print(reshaped_img.shape)
    class_prediction = gnb.predict(reshaped_img.reshape(-1, 8))

# Reestructura el mapa con la clasificación de pixeles en un arreglo de 2 dimensiones
    class_prediction = class_prediction.reshape(reshaped_img[:, :, 0].shape)
    class_prediction = str_class_to_int(class_prediction)
    return class_prediction, reshaped_img


def color_stretch(image, index):
    colors = image[:, :, index].astype(np.float64)
    for b in range(colors.shape[2]):
        colors[:, :, b] = rasterio.plot.adjust_band(colors[:, :, b])
    return colors
    return(class_array.astype(int))


##### Crea un mapa de colores para mostrar en matplotlib#####
def color_map(class_prediction):
    # Encuentra el mayor valor de los pixeles en la clasificacion
    n = int(np.max(class_prediction))
    colors = dict((
        (0, (48, 156, 214, 255)),   # Azul- clase Water
        (1, (139,69,19, 255)),      # Café - clase WetSand
        (2, (96, 19, 134, 255)),    # Morado - clase Emergent Wetland
        (3, (244, 164, 96, 255)),   # Bronceado - clase Sand
        (4, (206, 224, 196, 255)),  # Verde lima - clase Herbaceous
        (5, (34, 139, 34, 255)),    # verde obscuro - clase Forest 
    ))
    
    # Cambia valores 0 - 255 a float 0 - 1
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
        
    index_colors = [colors[key] if key in colors else 
                    (255, 255, 255, 0) for key in range(0, n+1)]
    
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n+1)
    return cmap

def mostrar_prediccion(path_imagen, class_prediction, reshaped_img):
    
    with rasterio.open(path_imagen) as src: 
        green_band = src.read(3)
        red_band = src.read(4)
        nir_band = src.read(8)
    
    cmap = color_map(class_prediction)
    fig, axs = plt.subplots(2,1,figsize=(10,7))
    
    img_stretched = color_stretch(reshaped_img, [4, 3, 2])
    axs[0].imshow(img_stretched)
    
    axs[1].imshow(class_prediction, cmap=cmap, interpolation='none')
    
    fig.show()

    ndwi = calculo_NDWI(nir_band, green_band)
    ndvi = calculo_NDVI(nir_band, red_band)
   
    fig, axs = plt.subplots(2,2,figsize=(15,7))

    img_stretched = color_stretch(reshaped_img, [3, 2, 1])
    axs[0,0].imshow(img_stretched)
    
    axs[0,1].imshow(class_prediction, cmap=cmap, interpolation='none')
    
    nwdi_plot = axs[1,0].imshow(ndwi, cmap="RdYlGn")
    axs[1,0].set_title("NDWI")
    fig.colorbar(nwdi_plot, ax=axs[1,0])
    
    ndvi_plot = axs[1,1].imshow(ndvi, cmap="RdYlGn")
    axs[1,1].set_title("NDVI")
    fig.colorbar(ndvi_plot, ax=axs[1,1])
    
    plt.show() 
   


if __name__ == '__main__':
####################paths de imagen#################                
    images_fp = "./imagenes/BandasSeparadas/"
    img_dir = "./imagenes/BandasJuntas/"
    img_name= 'sentinel_bandas_juntas.tif'
    img_fp = "./imagenes/BandasJuntas/sentinel_bandas_juntas.tif" 
    shp_path = "./data/rcr/rcr_landcover.shp" 
#####################################################
    geoms = obtener_geometria(img_fp, shp_path)
    X_train, y_train = matrices_entrenamiento(img_fp, geoms, shp_path)
    gnb = entrenar_clasificador(X_train, y_train)
    val_espectral(X_train, y_train)
    prediccion_clases, image_reshaped = predecir_clases(gnb, img_fp)
    mostrar_prediccion(img_fp, prediccion_clases, image_reshaped)
    
    
    

    
    
    