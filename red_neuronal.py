import random
import math
import itertools
import matplotlib.pyplot as plt
import rasterio.features
import rasterio.warp
import rasterio.mask
import numpy as np
import seaborn as sns
import pandas as pd
from rasterio.plot import adjust_band
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.plot import show
from rasterio.windows import Window
from pyproj import Proj, transform
from tqdm import tqdm
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from graf_img_sat import calculo_NDVI, calculo_NDWI
from graf_img_sat import mostrar_histograma, mostrar_junto
from clasificacion_imgs import color_map, valores_espectrales
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from time import time
from sklearn.manifold import TSNE
from sklearn import neighbors, datasets

###Diccionario global de clases NLCD################################
class_names = dict((
(0,  'Background'),
(1, 'Unclassified'),
(2, 'High Intensity Developed'),
(3, 'Medium Intensity Developed'),
(4, 'Low Intensity Developed'),
(5, 'Open Space Developed'),
(6, 'Cultivated Land'),
(7, 'Pasture/Hay'),
(8, 'Grassland'),
(9, 'Deciduous Forest'),
(10, 'Evergreen Forest'),
(11, 'Mixed Forest'),
(12, 'Scrub/Shrub'),
(13, 'Palustrine Forested Wetland'),
(14, 'Palustrine Scrub/Shrub Wetland'),
(15, 'Palustrine Emergent Wetland'),
(16, 'Estuarine Forested Wetland'),
(17, 'Estuarine Scrub/Shrub Wetland'),
(18, 'Estuarine Emergent Wetland'),
(19, 'Unconsolidated Shore'),
(20, 'Bare Land'),
(21, 'Water'),
(22, 'Palustrine Aquatic Bed'),
(23, 'Estuarine Aquatic Bed'),
(24, 'Tundra'),
(25, 'Snow/Ice')
))

#########Diccionario global de colores para clases#####################
colors = dict((
(0, (245,245,245, 255)), # Background
(1, (0,0,0)), # Unclassified (Cloud, Shadow, etc)
(2, (255,0,0)), # High Intensity Developed
(3, (255, 110, 51)), # Medium Intensity Developed
(4, (255, 162, 51)), # Low Intensity Developed
(5, (255, 162, 51)), # Open Space Developed
(6, (162, 89, 0)), # Cultivated Land
(7, (229, 221, 50)), # Pasture/Hay
(8, (185, 251, 96)), # Grassland
(9, (83, 144, 0)), # Deciduous Forest
(10, (13, 118, 0  )), # Evergreen Forest
(11, (62, 178, 49)), # Mixed Forest
(12, (100, 241, 125)), # Scrub/Shrub
(13, (68, 160, 85)), # Palustrine Forested Wetland
(14, (118, 192, 131)), # Palustrine Scrub/Shrub Wetland
(15, (188, 0, 211)), # Palustrine Emergent Wetland
(16, (188, 0, 211)), # Estuarine Forested Wetland
(17, (0, 0, 0)), # Estuarine Scrub/Shrub Wetland
(18, (172, 0, 191)), # Estuarine Emergent Wetland
(19, (159, 251, 255)), # Unconsolidated Shore 
(20, (172, 177, 68)), # Bare Land
(21, (29, 0, 189)), # Water
(22, (40, 40, 40)), # Pal Bed
))
#######################################################################

def merge_classes(y):
    ###Esta función combina clases manejadas por la 
    ###NLCD para obtener un subset de clases fácil de manejar
    ###reclasifica 255 a 0
    y[y == 255] = 0
    # Intensidades medias y altas
    y[y == 3] = 2
    # intensidades bajas y altas
    y[y == 4] = 2
    # se toman las clases NLCD y se convinan 
    
    # open space developed, cultivated land, and pasture hay
    y[y == 5] = 6
    y[y == 7] = 6

    # decidious -> mixed
    y[y == 9] = 11
    # evergreen -> mixed
    y[y == 10] = 11
    # shrub -> mixed
    y[y == 12] = 11
    # wetland forest -> mixed
    y[y == 13] = 11
    # pal wetland -> pal scrub shrub
    y[y == 14] = 18
    y[y == 15] = 18
    y[y == 16] = 18
    y[y == 17] = 18
    
    # pal bed -> water
    y[y == 22] = 21
    # unconsol shore -> water
    y[y == 19] = 21
    
    return(y)


def gen_balanced_pixel_locations(image_datasets, train_count, label_dataset, merge=False):
    ### esta función extrae un número train_count + val_count de píxeles aleatorios de una lista de datasets ráster
    ### y devuelve una lista de ubicaciones de píxeles de entrenamiento e índices de imágenes
    ### y una lista de ubicaciones e índices de píxeles de validación
    label_proj = Proj(label_dataset.crs)  
    train_pixels = []
    labels_image = label_dataset.read()
    
    train_count_per_dataset = math.ceil(train_count / len(image_datasets))
    for index, image_dataset in enumerate(tqdm(image_datasets)):
        # Numero de puntos por cada clase
        points_per_class = train_count_per_dataset // len(np.unique(merge_classes(labels_image)))
        
        # obtiene límites de Landsat en esta imagen
        # crea aproximadamente una máscara de conjunto de datos en coordenadas geográficas
        # este fcn asigna ubicaciones de píxeles en (fila, col) coordenadas a (x, y) posiciones espaciales
        raster_points = image_dataset.transform * (0, 0), image_dataset.transform * (image_dataset.width, 0), image_dataset.transform * (image_dataset.width, image_dataset.height), image_dataset.transform * (0, image_dataset.height)
        l8_proj = Proj(image_dataset.crs)
        new_raster_points = []
        # Convierte los límites de trama de Landsat en etiquetas crs
        for x,y in raster_points:
            x,y = transform(l8_proj,label_proj,x,y)
            # convierte de crs a fila, columna en las etiquetas de coordenadas de la imagen
            row, col = label_dataset.index(x, y)
            # se invierte el orden col,fila para empatar con las coordenadas xy
            new_raster_points.append((col, row))
        # Vuelve el conjunto de puntos un polígono
        raster_poly = Polygon(new_raster_points)
        # Window.from_slices((inicio de fila, final de fila), (inicio de columna, final de columna))
        masked_label_image = label_dataset.read(window=Window.from_slices((int(raster_poly.bounds[1]), int(raster_poly.bounds[3])), (int(raster_poly.bounds[0]), int(raster_poly.bounds[2])), height=1, width=1, boundless=False))
        if merge:
            masked_label_image = merge_classes(masked_label_image)
        # repite para cada clase
        all_points_per_image = []
        for cls in np.unique(merge_classes(labels_image)):
            cls = int(cls)
            # enmascarar la imagen del subconjunto de etiquetas a cada clase
            # saca las indicaciones donde la máscara es verdadera
            rows,cols = np.where(masked_label_image[0] == cls)
            all_locations = list(zip(rows,cols))
       
            # barajea todas las localizaciones en la imagen
            random.shuffle(all_locations)
            # ahora convertir a Landsat imagen crs
            l8_points = []
            if len(all_locations)!=0:
                for r,c in all_locations[:points_per_class]:
                # convertir la etiqueta de fila y col en espacio geográfico de etiqueta
                    x,y = label_dataset.xy(r+raster_poly.bounds[1],c+raster_poly.bounds[0])
                # pasar de la proyección de etiquetas a la proyección de Landsat
                    x,y = transform(label_proj, l8_proj,x,y)
                # convertir del espacio geográfico del Landsat a fila, col
                    r,c = image_dataset.index(x,y)
                    l8_points.append((r,c))
                all_points_per_image += l8_points

        dataset_index_list = [index] * len(all_points_per_image)

        dataset_pixels = list(zip(all_points_per_image, dataset_index_list))
        train_pixels += dataset_pixels
    random.shuffle(train_pixels)
    return (train_pixels)

def tile_generator(l8_image_datasets, label_dataset, tile_height, tile_width, pixel_locations, batch_size, merge=False):
    ### Este es un generador de datos compatible con Keras que genera mosaicos de datos y etiquetas sobre la marcha
    ### de un conjunto de ubicaciones de píxeles, una lista de conjuntos de datos de imágenes y un conjunto de datos de etiquetas
     

    c = r = 0
    i = 0
    
    label_proj = Proj(label_dataset.crs)

    # Se asume que todas las imagenes tienen el mismo numero de bandas
    l8_band_count = l8_image_datasets[0].count  
    band_count = l8_band_count
    class_count = len(class_names)
    buffer = math.ceil(tile_height / 2)
  
    while True:
        image_batch = np.zeros((batch_size, tile_height, tile_width, band_count-1)) # Se quita una banda porque no usamos la banda QA
        label_batch = np.zeros((batch_size,class_count))
        b = 0
        while b < batch_size:
            # Si nos encontramos al final de los datos, se reinicia
            if i >= len(pixel_locations):
                i=0
            r, c = pixel_locations[i][0]
            dataset_index = pixel_locations[i][1]
            i += 1
            tile = l8_image_datasets[dataset_index].read(list(np.arange(1, l8_band_count+1)), window=Window(c-buffer, r-buffer, tile_width, tile_height))
            if tile.size == 0:
                pass
            elif np.amax(tile) == 0: # No se incluye si es una parte de la imagen sin pixeles
                pass
            elif np.isnan(tile).any() == True or -9999 in tile: 
                # no queremos mosaicos que contengan nan o -999 esto proviene de bordes
                # esto lleva un tiempo y es ineficiente
                pass
            elif tile.shape != (l8_band_count, tile_width, tile_height):
                #print('forma erronea')
                #print(tile.shape)
                #de alguna manera estamos obteniendo mosaicos al azar sin las dimensiones correctas
                pass
            elif np.isin(tile[7,:,:], [352, 368, 392, 416, 432, 480, 840, 864, 880, 904, 928, 944, 1352]).any() == True:
                # Se asegúra de que el píxel no contenga nubes
                # esto probablemente sea bastante ineficiente, pero solo verifica el ancho x la altura de cada mosaico
                #print('Nube encontrada.')
                #print(tile[7,:,:])
                pass
            else:                
                # Se remueve la banda QA
                tile = tile[0:7]
                #cambniar del formato ráster al formato de imagen y estandarizar de acuerdo con las estadísticas de imagen amplia
                reshaped_tile = (reshape_as_image(tile)  - 982.5) / 1076.5

                ### obtener datos de etiqueta
                # encontrar gps de ese píxel dentro de la imagen
                (x, y) = l8_image_datasets[dataset_index].xy(r, c)

                # convertir el punto del que estamos tomando muestras a la misma proyección que el conjunto de datos de la etiqueta si es necesario
                if l8_proj != label_proj:
                    x,y = transform(l8_proj,label_proj,x,y)

                # GPS de referencia en label_image
                row, col = label_dataset.index(x,y)

                # encontrar etiqueta
                # la imagen de la etiqueta podría ser enorme, por lo que necesitamos esto para obtener una sola posición
                window = ((row, row+1), (col, col+1))
                data = merge_classes(label_dataset.read(1, window=window, masked=False, boundless=True))
                label = data[0,0]
                # si esta etiqueta es parte del área no clasificada, lo ignora
                if label == 0 or np.isnan(label).any() == True:
                    pass
                else:                   
                    # agrega la etiqueta al lote en un estilo de codificación dinámico
                    label_batch[b][label] = 1
                    image_batch[b] = reshaped_tile
                    b += 1
        yield (image_batch, label_batch)
        



def plot_confusion_matrix(y_true, y_pred, classes, class_dict,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y traza la matriz de confusión.
    La normalización se puede aplicar estableciendo `normalize = True`.
    """
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión sin normalizar'

    # Computa la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    # Se utilizan las etiquetas que aparecen en los datos
    classes = classes[unique_labels(y_true, y_pred)]
    # convierte el id de clase a id de nombre usando el diccionario
    cover_names = []
    for cover_class in classes:
        cover_names.append(class_dict[cover_class])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    #print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Tomamos todos los "ticks"
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... y se etiquetan con su respectiva entrada en una lista
           xticklabels=cover_names, yticklabels=cover_names,
           title=title,
           ylabel='Etiqueta verdadera',
           xlabel='Etiqueta predecida')

    # Se rotan las etiquetas en un angulo de 45°
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Recorre las dimensiones de los  datos y crea las anotaciones de texto.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


landsat_dataset = rasterio.open('./data/Train/landsat_image.tif')
landsat_image = landsat_dataset.read()
bandNIR = landsat_image[4, :, :]
bandRed = landsat_image[3, :, :]
ndvi = np.clip(calculo_NDVI(bandNIR,bandRed),-1,1)
'''print('\nMax NDVI: {m}'.format(m=ndvi.max()))
print('Mean NDVI: {m}'.format(m=ndvi.mean()))
print('Median NDVI: {m}'.format(m=np.median(ndvi)))
print('Min NDVI: {m}'.format(m=ndvi.min()))
mostrar_histograma(ndvi, "NDVI")
mostrar_histograma(landsat_image, "imagen Landsat")
mostrar_junto(landsat_image,ndvi)

#mostramos la banda de calidad (QA)
qa_band = landsat_image[7, :, :]
qa_band[qa_band == -9999] = 0

print(np.unique(qa_band))

fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(qa_band, cmap='gray')'''

######Leemos la imagen de entrenamiento predecida############

labels_dataset = rasterio.open('./data/Train/labels_image.tif')
labels_image = merge_classes(labels_dataset.read())#se convinan clases para reducir el número
print(labels_image.shape)
#Se cuentan cuantos pixeles hay de cada clase
unique, counts = np.unique(labels_image, return_counts=True)
print(list(zip(unique, counts)))
'''#Muestra la imagen de entrenamiento con clases convinadas
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(labels_image[0])'''
cmap = color_map(labels_image,colors)
'''#Muestra la imagen con el mapa de colores asignado
ig, axs = plt.subplots(figsize=(10,10))
axs.imshow(labels_image[0,:, :], cmap=cmap, interpolation='none')'''

#####Genera un conjunto de ubicaciones de píxeles balanceadas##########
train_pixels = gen_balanced_pixel_locations([landsat_dataset], train_count=8000,
                                            label_dataset=labels_dataset, merge=True)
landsat_datasets = [landsat_dataset]
# genera la ubicación de los pixeles de entrenamiento y de validación
all_labels = []
label_locations = []
for pixel in train_pixels:
    # ubicacion fila, col de landsat
    r,c = pixel[0]
    ds_index = pixel[1]
    l8_proj = Proj(landsat_datasets[ds_index].crs)
    label_proj = Proj(labels_dataset.crs)
    
    '''# ubicacion geografica de landsat
    x,y = landsat_datasets[ds_index].xy(r,c)
    # pasar de la proyección de etiquetas a la proyección de Landsat
    x,y = transform(l8_proj, label_proj ,x,y)
    # obtiene la ubicacion de fila y col de la etiqueta
    r,c = labels_dataset.index(x,y)
    
    label_locations.append([r,c])
    
    # Se da el formato (bandas, altura, anchura)
    window = ((r, r+1), (c, c+1))
    data = merge_classes(labels_dataset.read(1, window=window, masked=False, boundless=True))
    all_labels.append(data[0,0])
    
label_locations = np.array(label_locations)

unique, counts = np.unique(np.array(all_labels), return_counts=True)
print(dict(zip(unique, counts)))
fig, ax = plt.subplots(figsize=[15,15])
#Muestra los pixeles de entrenamiento sobre la imagen de entrenamiento recoloreada#
ax.imshow(labels_image[0,:, :], cmap=cmap, interpolation='none')

plt.scatter(label_locations[:, 1], label_locations[:, 0], c='r')'''

####################Genera zonas recortadas de la imagen de manera aleatoria###############

'''im_batch = None

count = 0
for (im, label) in tile_generator(landsat_datasets, labels_dataset, 128, 128, train_pixels, 10):
    if count > 3:
        break
    count += 1
    im_batch = im

fig, axs = plt.subplots(2, 3, figsize=(18, 10)) 

axs[0,0].imshow(im_batch[0,:,:,3:6])
axs[0,1].imshow(im_batch[1,:,:,3:6])
axs[0,2].imshow(im_batch[2,:,:,3:6])
axs[1,0].imshow(im_batch[3,:,:,3:6])
axs[1,1].imshow(im_batch[4,:,:,3:6])
axs[1,2].imshow(im_batch[5,:,:,3:6])'''
#Genera un dataset  con una zona para que scikit-learn pueda visualizar los datos
'''im_batch = None
label_batch = None

sample_size = 500


count = 0
for (im, label) in tile_generator(landsat_datasets, labels_dataset, 1, 1, train_pixels, sample_size):
    if count > 0:
        break
    count += 1
    im_batch = im
    label_batch = label
    
#se reestructuran los datos para que quede con formato (muestras, bandas)
im_batch_reshaped = im_batch.reshape(sample_size,7)
#valores_espectrales(label_batch, im_batch_reshaped, class_names)

###########Ejecuta el análisis de componentes principales para visualizar puntos#########
pca = PCA(n_components=3)
pca_result = pca.fit_transform(im_batch_reshaped)

print('Variacion aproximada por componente principal: {}'.format(pca.explained_variance_ratio_))

df = pd.DataFrame({'pca-1':pca_result[:,0],'pca-2':pca_result[:,1],'pca-3':pca_result[:,2], 'y' : np.argmax(label_batch, axis=1)})
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df["pca-1"], 
    ys=df["pca-1"], 
    zs=df["pca-3"], 
    c=df["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-1')
ax.set_ylabel('pca-2')
ax.set_zlabel('pca-3')
plt.show()'''

#################Se utiliza el método de incrustación de vecinos estocásticos distribuidos en T para visualizar en 2D##############
'''time_start = time()
tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)
tsne_results = tsne.fit_transform(im_batch_reshaped)
print('TSNE completado! Tiempo transcurrido: {} seconds'.format(time()-time_start))
df_subset = df.copy()
df_subset['tsne-2d-1'] = tsne_results[:,0]
df_subset['tsne-2d-2'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-1", y="tsne-2d-2",
    hue="y",
    palette=sns.color_palette("hls", len(np.unique(np.argmax(label_batch, axis=1)))),
    data=df_subset,
    legend="full",
    alpha=0.3
)'''
#####################Se obtiene la Matriz de confusión con el algoritmo K Nearest Neighbors
im_batch = None
label_batch = None

sample_size = 2000
train_count = 1500
val_count = 500

count = 0
for (im, label) in tile_generator(landsat_datasets, labels_dataset, 1, 1, train_pixels, sample_size):
    if count > 0:
        break
    count += 1
    im_batch = im
    label_batch = label

im_batch_reshaped = im_batch.reshape(sample_size,7)

X_train = im_batch_reshaped[:train_count]
X_val = im_batch_reshaped[train_count:]
y_train = np.argmax(label_batch, axis=1)[:train_count]
y_val = np.argmax(label_batch, axis=1)[train_count:]
n_neighbors = 50

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X_train, y_train)
clf.score(X_val, y_val)
pred_index = clf.predict(X_val)

# Imprime la matriz de confusion no normalizada
plot_confusion_matrix(y_val, pred_index, classes=np.array(list(class_names)),
                      class_dict=class_names)

# Imprime la matriz de confusion normalizada
plot_confusion_matrix(y_val, pred_index, classes=np.array(list(class_names)),
                      class_dict=class_names,
                      normalize=True)