# SMS Spam Classification — Logistic Regression

Clasificación de mensajes SMS como legítimos o spam mediante TF-IDF y regresión logística.

Proyecto desarrollado como parte de mi Roadmap V3 hacia AI/NLP/LLM Engineering.

## Objetivo

Construir y evaluar un clasificador de spam, priorizando evitar que mensajes legítimos se envíen incorrectamente a la carpeta de spam.

Se estableció un objetivo educativo de **Precision de spam ≥ 98 %**, buscando conservar el mayor Recall posible entre las configuraciones evaluadas que cumplieran esa condición.

## Dataset

Se utilizó **SMS Spam Collection**, disponible en UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Etiquetas:

- `0 — ham`: mensaje legítimo.
- `1 — spam`: mensaje no deseado.

Antes de dividir los datos se comprobaron etiquetas contradictorias para textos idénticos y se eliminaron duplicados exactos.

El conjunto resultante contiene **5,171 mensajes**:

| Conjunto | Mensajes |
|---|---:|
| Train | 4,136 |
| Test | 1,035 |

La separación se realizó con `test_size=0.20`, `stratify=y` y `random_state=42`.

Test permaneció aislado durante el análisis y la selección de la configuración.

## Distribución de clases en Train

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| Legítimo | 3,614 | 87.38 % |
| Spam | 522 | 12.62 % |

El desbalance hace que Accuracy sea insuficiente para evaluar el problema: predecir siempre “legítimo” produce una Accuracy elevada, pero no detecta spam.

## Metodología

1. Carga y comprobación de integridad.
2. Eliminación de duplicados exactos.
3. Separación estratificada de Train y Test.
4. Análisis del desbalance utilizando únicamente Train.
5. Evaluación de un baseline con Stratified 5-Fold Cross-Validation.
6. Construcción de un pipeline TF-IDF + regresión logística.
7. Comparación de intensidades de regularización.
8. Obtención de probabilidades out-of-fold para estudiar umbrales.
9. Análisis de errores OOF.
10. Entrenamiento con todo Train e interpretación de coeficientes.
11. Evaluación final en Test con la configuración fijada.

El notebook conserva el recorrido de aprendizaje, incluidos ejemplos didácticos y comparaciones intermedias.

## Prevención de data leakage

TF-IDF se incluyó dentro del pipeline. En cada fold, el vocabulario y los valores IDF se aprendieron exclusivamente con la parte de entrenamiento.

Las predicciones out-of-fold se generaron con modelos que no utilizaron para entrenarse la observación que estaban prediciendo.

Los resultados de Cross-Validation y OOF se utilizaron para desarrollo y selección; no constituyen una evaluación independiente de todo el proceso de selección.

No se utilizó Test para seleccionar `C`, modificar el vocabulario o ajustar el umbral.

## Baseline

Se utilizó `DummyClassifier(strategy="most_frequent")`.

Resultados medios de cinco folds:

| Métrica | Baseline |
|---|---:|
| Accuracy | 0.8738 |
| Precision de spam | 0.0000 |
| Recall de spam | 0.0000 |
| F1 de spam | 0.0000 |

El baseline siempre predice la clase legítima. Precision se reportó como cero por convención cuando no hubo predicciones positivas.

## Representación y modelo

Los mensajes se representaron mediante `TfidfVectorizer` con su configuración predeterminada.

TF-IDF genera características numéricas a partir de los términos del texto. La regresión logística aprende sus coeficientes para estimar la probabilidad de spam.

Se utilizó `LogisticRegression` con regularización L2 y `max_iter=1000`.

### Comparación de C

`C` controla inversamente la intensidad de regularización: un valor menor implica mayor regularización.

Resultados medios de Cross-Validation con la decisión estándar:

| C | Accuracy | Precision spam | Recall spam | F1 spam |
|---:|---:|---:|---:|---:|
| 0.01 | 0.8738 | 0.0000 | 0.0000 | 0.0000 |
| 0.10 | 0.8738 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.9647 | 0.9950 | 0.7240 | 0.8370 |
| 10 | 0.9819 | 0.9809 | 0.8734 | 0.9235 |
| 100 | 0.9850 | 0.9735 | 0.9060 | 0.9382 |

Se seleccionó **C=10** porque ofreció el mayor Recall entre los candidatos que alcanzaron una Precision media de al menos 98 % con la decisión estándar.

Esta búsqueda fue secuencial y limitada: no se optimizaron conjuntamente todos los valores de `C` y todos los umbrales posibles.

## Selección del umbral

Para `C=10` se generaron probabilidades mediante `cross_val_predict(method="predict_proba")`.

Resultados sobre las predicciones OOF agrupadas:

| Umbral | Precision spam | Recall spam | F1 spam | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.9213 | 0.9425 | 0.9318 | 42 | 30 |
| 0.30 | 0.9584 | 0.9272 | 0.9426 | 21 | 38 |
| 0.40 | 0.9752 | 0.9042 | 0.9384 | 12 | 50 |
| 0.50 | 0.9806 | 0.8736 | 0.9240 | 9 | 66 |
| 0.60 | 0.9866 | 0.8448 | 0.9102 | 6 | 81 |
| 0.70 | 0.9906 | 0.8103 | 0.8915 | 4 | 99 |
| 0.80 | 0.9949 | 0.7490 | 0.8546 | 2 | 131 |

- **FP:** mensajes legítimos clasificados como spam.
- **FN:** spam clasificado como legítimo.

Se seleccionó **umbral=0.50**, que obtuvo el mayor Recall entre los umbrales evaluados que cumplieron Precision ≥ 98 %.

La configuración final quedó fijada en:

- **C:** 10.
- **Umbral:** 0.50.

## Análisis de errores e interpretación

Se revisaron falsos positivos y falsos negativos OOF, empezando por los errores con mayor confianza.

Se observaron mensajes legítimos con vocabulario parecido al promocional y mensajes etiquetados como spam con apariencia conversacional.

En el modelo entrenado con todo Train:

- Términos como `txt`, `text`, `reply`, `call`, `www` y `free` tuvieron coeficientes positivos.
- Términos como `me`, `my`, `can` y `ok` tuvieron coeficientes negativos.

Un coeficiente positivo favorece spam y uno negativo favorece legítimo. La contribución de un término al puntaje depende del producto entre su coeficiente y su valor TF-IDF.

Estas asociaciones no son reglas universales ni demuestran causalidad. Tampoco explican por sí solas las predicciones de los modelos OOF, cuyos parámetros pueden diferir.

## Resultados finales en Test

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9787 |
| Precision de spam | 0.9739 |
| Recall de spam | 0.8550 |
| F1 de spam | 0.9106 |
| ROC-AUC | 0.9900 |
| Average Precision | 0.9673 |
| Log Loss | 0.0700 |

### Matriz de confusión

Filas: clase real. Columnas: clase predicha.

| | Predicho legítimo | Predicho spam |
|---|---:|---:|
| Real legítimo | 901 | 3 |
| Real spam | 19 | 112 |

El modelo:

- Detectó **112 de los 131 mensajes spam**.
- Dejó pasar **19 mensajes spam**.
- Clasificó incorrectamente **3 de los 904 mensajes legítimos** como spam.

### Cumplimiento del objetivo

**No se alcanzó el objetivo de Precision ≥ 98 % en Test.**

La Precision final fue **97.39 %**. Se conservó este resultado sin reajustar el modelo ni el umbral usando Test.

El ROC-AUC elevado indica una buena capacidad para ordenar spam por encima de mensajes legítimos; no garantiza alcanzar una Precision determinada con un umbral concreto.

## Supuestos y limitaciones

- El modelo utiliza únicamente el texto; no conoce al remitente ni si el destinatario solicitó el mensaje.
- La evaluación supone que la separación aleatoria aproxima el escenario de uso.
- Solo se eliminaron duplicados exactos. Mensajes similares o de una misma campaña podrían aparecer en conjuntos distintos.
- No se evaluó generalización temporal ni transferencia a mensajes actuales o en español.
- TF-IDF con términos individuales no representa explícitamente el orden de las palabras ni su contexto completo.
- Algunas etiquetas pueden ser difíciles de interpretar sin información adicional; no se modificaron basándose en las predicciones.
- La selección se realizó con los mismos folds de desarrollo, por lo que sus resultados pueden ser optimistas.
- No se calcularon intervalos de confianza para el objetivo de Precision.
- Una Log Loss baja no demuestra por sí sola que las probabilidades estén bien calibradas.
- Este proyecto es educativo y no constituye una validación para despliegue en producción.

## Aprendizajes

- Construcción de un pipeline de clasificación de texto.
- Regularización y efecto del parámetro `C`.
- Diferencia entre probabilidades, etiquetas y umbrales.
- Evaluación con clases desbalanceadas.
- Uso de predicciones out-of-fold.
- Diferencia entre ROC-AUC, Precision, Recall, F1 y Log Loss.
- Interpretación de coeficientes y revisión de errores.
- Separación entre selección del modelo y evaluación final.
- Documentación de objetivos no alcanzados sin ajustar repetidamente sobre Test.

## Ejecución

Dependencias principales:

```bash
pip install numpy pandas scikit-learn jupyterlab
```

Abrir JupyterLab:

```bash
jupyter lab
```

Ejecutar el notebook de principio a fin. La descarga inicial del dataset requiere conexión a Internet.

Las versiones de las bibliotecas pueden introducir pequeñas variaciones numéricas.

## Conclusión

TF-IDF y regresión logística permitieron detectar el 85.50 % del spam en Test, con una Precision de 97.39 %.

El modelo mejoró la detección de spam respecto al baseline, aunque no alcanzó el objetivo educativo de Precision del 98 %. La evaluación final se mantuvo independiente de los ajustes.

**Proyecto evidenciable en GitHub.**

---

Built as part of my journey toward AI/NLP/LLM Engineering. — SM