# DeepVision - Analizando el Amazonas desde el espacio

## Introducción

En este trabajo, se utilizarán imágenes satelitales del Amazonas para entrenar un modelo de Deep Learning que sea capaz de dar información acerca del contenido de las mismas.

### Imágenes
Se trabajará con más de 40.000 imágenes satelitales en RGB donde cada píxel representa un área de 3.7 metros. Los datos provienen de los satélites Flock 2 de la compañía Planet, recolectados entre el 1 de Enero de 2016 y el 1 de Febrero de 2017. Todas las escenas provienen de la cuenca del Amazonas que incluye Brasil, Perú, Uruguay, Colombia, Venezuela, Guyana, Bolivia y Ecuador.

### Etiquetas
Las clases a predecir para cada imágen representan un conjunto de fenómenos de interés que se desean analizar en el Amazonas. Estas etiquetas pueden distribuirse en tres grupos:

- Condiciones atmosféricas
- Fenómenos comunes de cobertura/uso de suelo
- Fenómenos raros de cobertura/uso de suelo

Cada imágen tiene al menos una, y potencialmente, más de una etiqueta del tipo de condiciones atmosféricas. Así como también cero o varias etiquetas del tipo de fenómenos comunes o raros de cobertura/uso
de suelo. Aquellas imágenes etiquetadas como "_cloudy_" (nublado), no deberían tener ninguna otra anotación, aunque pueden haber errores de anotación.

Puede ocurrir que en muchas imágenes no se hayan etiquetados todos los fenómenos observables, como así también puede que haya etiquetas incorrectas.

Trabajaremos con un total de 17 etiquetas posibles, descritas a continuación:

**Cloudy**: Imagen completamente nublada que no permite obtener observaciones adicionales.

**Partly Cloudy**: Imagen parcialmente nublada.

**Clear**: Imagen sin nubes.

**Haze**: Imagen con neblina.

**Primary (rainforest)**: Presencia de bosque "virgen" o áreas con alta densidad de vegetación.

**Water**: Presencia de ríos, embalses o lagos.

**Habitation**: Presencia de casas o edificios.

**Agriculture**: Porciones de tierra sin árboles siendo utilizadas para la agricultura.

**Road**: Caminos o rutas.

**Cultivation**: Es un subconjunto de Agriculture que se refiere a pequeñas granjas siendo explotadas por una persona o familia como medio de supervivencia.

**Bare Ground**: Áreas sin presencia de árboles por causas naturales y no como resultado de la actividad humana.

**Slash and Burn**: Áreas que han sido incendiadas recientemente.

**Selective Logging**: Zonas de tala selectiva de especies de árboles de alto valor.

**Blooming**: Es un fenómeno natural donde ciertas especies de árboles, eclosionan, dan frutos y florecen para aumentar las chances de polinización cruzada.

**Conventional Mining**: Zona con presencia de minería legal a larga escala.

**"Artisinal" Mining**: Operaciones de minería a baja escala, usualmente ilegales. 

**Blow Down**: Fenómeno natural que ocurre cuando el aire frío y seco de los Andes se asienta sobre el aire cálido y húmedo en la selva. Estos fuertes vientos derriban los árboles más grandes de la selva tropical, y las áreas abiertas resultantes son visibles desde el espacio.

## Modelos

Se consideran 2 modelos, uno tomando como base la red ResNet50 y otro considerando la red EfficientB0. En cada caso se ignoran las capas superiores y se agregan al modelo final 2 capas densas a la salida
de cada una de las cuales se incluye una capa de *dropout*. La siguientes figuras muestran la comparación entre estos modelos. Los modelos *ResNet* y *Efficient 1* son entrenados considerando los
siguientes **data augmentation**: 
- horizontal flip
- vertical flip
- rotation
- width shift
- height shift

Mientras que el modelo *Efficient 2* solo se entrenó utilizando *flip* vertical y horizontal.

![Accuracy](https://github.com/masgro/DeepVision/blob/master//images/image01.png "Accuracy")
![Loss](https://github.com/masgro/DeepVision/blob/master//images/image02.png "Loss")
![Recall](https://github.com/masgro/DeepVision/blob/master//images/image03.png "Recall")

### Train vs Validation ResNet

![Accuracy ResNet50](https://github.com/masgro/DeepVision/blob/master//images/image10.png "Accuracy ResNet50")
![Loss ResNet50](https://github.com/masgro/DeepVision/blob/master//images/image12.png "Loss ResNet50")
![Recall ResNet50](https://github.com/masgro/DeepVision/blob/master//images/image11.png "Recall ResNet50")

### Train vs Validation Efficient 1

![Accuracy EfficientB0](https://github.com/masgro/DeepVision/blob/master//images/image04.png "Accuracy EfficientB0")
![Loss EfficientB0](https://github.com/masgro/DeepVision/blob/master//images/image05.png "Loss EfficientB0")
![Recall EfficientB0](https://github.com/masgro/DeepVision/blob/master//images/image06.png "Recall EfficientB0")

### Train vs Validation Efficient 2

En este caso solo se utilizó _flip_ vertical y horizontal para hacer el _data augmentation_

![Accuracy EfficientB0](https://github.com/masgro/DeepVision/blob/master//images/image07.png "Accuracy EfficientB0")
![Loss EfficientB0](https://github.com/masgro/DeepVision/blob/master//images/image09.png "Loss EfficientB0")
![Recall EfficientB0](https://github.com/masgro/DeepVision/blob/master//images/image08.png "Recall EfficientB0")

## Conclusión

Las siguientes tablas resumen los resultados obtenidos

### Training

| Modelo | Accuracy | Recall | Loss |
| ----------- | ----------- | ----------- | ----------- |
| ResNet | 0.9498 | 0.8085 | 0.1537 |
| Efficient 1 | 0.9628 | 0.8567 | 0.1070 | 
| Efficient 2 | 0.9785 | 0.9196 | 0.0594 |

### Validation

| Modelo | Accuracy | Recall | Loss |
| ----------- | ----------- | ----------- | ----------- |
| ResNet | 0.9043 | 0.5652 | 1.474 |
| Efficient 1 | 0.9634 | 0.8677 | 0.104 | 
| Efficient 2 | 0.9624 | 0.877 | 0.1249 |

