# Clasificación de emociones faciales con una CNN (PyTorch)

Red neuronal convolucional entrenada desde cero en PyTorch para clasificar imágenes de rostros en 7 emociones: enojo, disgusto, miedo, felicidad, neutral, tristeza y sorpresa.

## Resumen de hallazgos

- La CNN alcanza **59 % de accuracy** en un conjunto de prueba de 7,177 imágenes (F1 macro de 0.51 y F1 ponderado de 0.57). Es un problema de 7 clases con etiquetas ruidosas, así que el azar estaría alrededor de 14 % y predecir siempre la clase mayoritaria (felicidad) daría alrededor de 24 %.
- **Las clases fáciles son felicidad (F1 0.84) y sorpresa (F1 0.73).** La felicidad es la clase más grande y sus rasgos (sonrisa) son muy distintivos.
- **Las clases difíciles son miedo (F1 0.23) y disgusto (F1 0.21).** Su recall es de solo 0.16 y 0.13: el modelo casi nunca las predice, aun cuando disgusto recibe un peso de 2.44, unas 4 veces el de felicidad. Disgusto tiene apenas 436 imágenes de entrenamiento (1.5 % del total) y miedo también queda por debajo del resto en cantidad de ejemplos bien clasificados.
- **No hay sobreajuste, pero sí margen de mejora.** Al final del entrenamiento la accuracy de entrenamiento (≈54 %) es menor que la de validación (≈59 %), porque el aumento de datos y el dropout solo se aplican al entrenar. Que ambas sean bajas indica que el modelo es todavía limitado (subajuste), no que memorice.
- La pérdida de validación bajó de 1.79 en la primera época a ≈1.12 en la última, y el *scheduler* redujo la tasa de aprendizaje a 1.25 × 10⁻⁴ hacia el final.
- Como demostración cualitativa, el modelo clasificó una foto propia como *happy* con 84.7 % de probabilidad. Es una sola imagen y no constituye una evaluación.

### Métricas por clase en el conjunto de prueba

| Clase | Precision | Recall | F1 | Imágenes |
|---|---|---|---|---|
| Happy | 0.82 | 0.86 | 0.84 | 1,755 |
| Surprise | 0.68 | 0.79 | 0.73 | 819 |
| Neutral | 0.58 | 0.56 | 0.57 | 1,237 |
| Sad | 0.42 | 0.59 | 0.49 | 1,271 |
| Angry | 0.46 | 0.47 | 0.47 | 951 |
| Fear | 0.40 | 0.16 | 0.23 | 1,020 |
| Disgust | 0.55 | 0.13 | 0.21 | 124 |
| **Accuracy** | | | **0.59** | **7,177** |

## Datos

Dataset de reconocimiento de emociones faciales en formato FER-2013: imágenes en escala de grises de 48 × 48 píxeles, organizadas en carpetas `train/` y `test/` con una subcarpeta por emoción (se carga desde un `archive.zip`).

- Se **unieron** los conjuntos originales de entrenamiento y prueba (35,887 imágenes) y se **volvieron a dividir** con semilla 42: **25,839 de entrenamiento (72 %), 2,871 de validación (8 %) y 7,177 de prueba (20 %)**.
- Las clases están muy desbalanceadas. En el conjunto original de entrenamiento: felicidad 7,215; neutral 4,965; tristeza 4,830; miedo 4,097; enojo 3,995; sorpresa 3,171 y disgusto 436.

## Método

**Preprocesamiento.** Escala de grises, redimensionado a 48 × 48 y normalización a media 0.5 y desviación 0.5. Solo en entrenamiento se aplica aumento de datos: recorte aleatorio con relleno de 4 píxeles, volteo horizontal (p = 0.5), rotación de ±15° y variación leve de brillo y contraste.

**Arquitectura** (≈1.57 millones de parámetros):

```
4 bloques [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2]  con 32, 64, 128 y 256 filtros
Flatten (256 × 3 × 3) → Linear 512 → ReLU → Dropout 0.5 → Linear 7
```

**Entrenamiento.** Adam con tasa de aprendizaje de 10⁻³, lotes de 64 imágenes y 25 épocas. Como función de pérdida, entropía cruzada con **pesos por clase** calculados como el inverso de la raíz de la frecuencia y normalizados (suaviza el balanceo: disgusto 2.44, felicidad 0.60). Se usa `ReduceLROnPlateau` (factor 0.5, paciencia 1) sobre la pérdida de validación.

**Evaluación.** Matriz de confusión y reporte de clasificación sobre el conjunto de prueba, que no se usó para ajustar nada. Al final se prueba el modelo con una imagen nueva.

## Estructura

```
.
├── CNN_Emociones.ipynb   # preparación de datos, modelo, entrenamiento y evaluación
└── README.md
```

## Cómo ejecutarlo

El notebook está pensado para **Google Colab con GPU**. Sube el `archive.zip` del dataset cuando la celda de carga lo pida (se extrae en `/content/archive`).

Para correrlo en local, instala las dependencias y ajusta las rutas del dataset:

```bash
pip install torch torchvision torchinfo scikit-learn seaborn matplotlib opencv-python pillow tqdm
```

## Limitaciones y siguientes pasos

- El conjunto de validación y el de prueba usan un recorte aleatorio en la transformación (`RandomCrop` con relleno), lo que añade algo de variación a las métricas. Lo ideal es que la evaluación sea determinista.
- Las clases minoritarias (disgusto y miedo) siguen mal resueltas. Opciones para mejorarlas: sobremuestreo o pesos más agresivos, una arquitectura más profunda o *transfer learning* con una red preentrenada, y entrenar más épocas con *early stopping*.
- Los rostros de FER-2013 tienen etiquetas ruidosas y baja resolución, lo que limita la accuracy alcanzable.
- La demostración con una foto personal no sustituye una evaluación con rostros externos al dataset.

## Autor

Emilio Páez de la Mora, A01801224. Tecnológico de Monterrey.
