# Localización 3D en Interiores con Deep Learning y Redes WiFi | Animado con Manim (Python)

Presentación oficial del Proyecto 3 para el curso de Machine Learning de la Universidad de Ingeniería y Tecnología (UTEC). 

El presente proyecto expone una solución integral al problema de localización tridimensional en interiores utilizando el dataset estándar UJIIndoorLoc. A partir de la intensidad de señal de 520 puntos de acceso WiFi (WAPs), el objetivo central es predecir con exactitud milimétrica el edificio y el piso en el que se encuentra un dispositivo, abarcando un total de 13 zonas espaciales.

### 🔬 Metodología y Análisis:
A lo largo de este análisis visual, documentamos la transición entre técnicas clásicas y redes profundas:

* **Reducción de Dimensionalidad:** Aplicación de PCA para optimizar la carga computacional preservando la varianza de los datos.
* **Aprendizaje No Supervisado:** Evaluación de K-Means, DBSCAN y Agglomerative Clustering. Se demuestra cómo el clustering clásico logra agrupar los 3 edificios principales (alcanzando un ARI de 0.213 con K-Means), pero fracasa al intentar separar las alturas de los pisos (Eje Z).
* **Deep Learning:** Implementación de una arquitectura compuesta por un Autoencoder, diseñado para extraer características esenciales y filtrar el ruido del espectro WiFi mediante un cuello de botella de 32 dimensiones, conectado a un Perceptrón Multicapa (MLP). Usando la activación no lineal ReLU, el modelo final logró trazar fronteras de decisión complejas superando el 95% de precisión.

### 💻 Desarrollo Audiovisual:
Este proyecto documental no es una presentación tradicional. Todas las animaciones, transformaciones de matrices y representaciones de redes neuronales fueron programadas matemáticamente desde cero en Python utilizando el motor gráfico Manim, con soporte de LaTeX para la notación científica y postproducción en Audacity.

### 👥 Equipo de Trabajo:
* **Autor:** Angel Oriundo, Carlos Enrique
* **Docente:** Luciano A. Romero Calla
