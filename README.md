# Herramienta para la Inferencia de Información Privada en Redes Sociales

## Descripción del Proyecto

Este proyecto desarrolla un laboratorio reproducible para estudiar hasta qué punto la estructura de una red social permite inferir atributos privados que los usuarios han ocultado de forma explícita. El caso de estudio se basa en el dataset de Facebook de Stanford, una colección clásica de ego-redes que permite analizar la relación entre topología, homofilia y filtración involuntaria de información sensible.

El objetivo científico del trabajo es demostrar que, incluso cuando el perfil del ego no se observa directamente, la combinación de la vecindad estructural, los círculos sociales, la conectividad a dos saltos, la similitud topológica, la modularidad comunitaria, la correlación estadística y la centralidad de intermediación puede reconstruir parte de los atributos privados con un nivel de precisión significativo.

La solución implementa siete estrategias de inferencia basadas en teoría de grafos y las integra con un motor de políticas semánticas adaptativas. Este motor ajusta el comportamiento de selección de atributos mediante umbrales de confianza y un mecanismo de top-k elástico que varía según la categoría semántica inferida. Sobre ese proceso actúa un sistema de inteligencia artificial explicable selectiva que conserva únicamente la trazabilidad de los amigos que contribuyen a los atributos finalmente aceptados, lo que permite auditar cada predicción desde una perspectiva topológica y semántica.

## Estructura del Repositorio

```text
TFM/
├── datos/
│   └── facebook/
├── resultados/
├── lib/
├── src/
│   ├── estrategias/
│   │   ├── base.py
│   │   ├── vecinos.py
│   │   ├── circulos.py
│   │   ├── transitividad.py
│   │   ├── similitud.py
│   │   ├── comunidades.py
│   │   ├── correlacion.py
│   │   └── puentes.py
│   └── nucleo/
│       ├── analisis_grafo.py
│       ├── cargador_datos.py
│       ├── evaluacion.py
│       ├── mapeo_atributos.py
│       └── politicas_semanticas.py
├── .gitignore
├── requirements.txt
├── main.py
└── dashboard.py
```

## Estrategias de Inferencia Implementadas

El núcleo metodológico de la solución está compuesto por siete estrategias algorítmicas complementarias que abordan la propagación de información desde distintas perspectivas topológicas, comunitarias y estadísticas.

### 1. Homofilia Local (Vecinos)
* **Fundamento:** Votación mayoritaria uniforme y homofilia directa de primer lazo.
* **Mecanismo:** Estima la probabilidad de un atributo a partir de la frecuencia media observada en los amigos directos válidos del ego. Constituye la línea base (*baseline*) de comparación del sistema.

### 2. Contexto de Grupos Íntimos (Círculos)
* **Fundamento:** Filtros de confianza basados en estructuras de confianza explícitas (`.circles`).
* **Mecanismo:** Pondera a los vecinos de acuerdo con su pertenencia a los círculos sociales del ego. Aplica una penalización elástica inversa basada en el tamaño del grupo, otorgando mayor peso específico a los microentornos íntimos (círculos pequeños).

### 3. Transitividad Estructural (Tránsitos)
* **Fundamento:** Propagación lineal y saltos profundos en la red.
* **Mecanismo:** Amplía el espectro analítico a dos saltos combinando el impacto de la vecindad inmediata con la de contactos indirectos (amigos de amigos). Aplica una atenuación basada en la raíz del grado de intermediación para mitigar el ruido en la red lejana.

### 4. Similitud Estructural (Similitud)
* **Fundamento:** Equivalencia estructural y vecindarios homólogos.
* **Mecanismo:** Utiliza una variante adaptativa del índice de Adamic-Adar para descubrir y explotar perfiles de actores que, sin estar necesariamente conectados de forma directa, comparten una topología análoga en la red, priorizando los enlaces hacia vecinos raros o exclusivos.

### 5. Detección de Comunidades (Comunidades)
* **Fundamento:** Segmentación modular por particiones latentes.
* **Mecanismo:** Descompone la topología de la ego-red mediante el algoritmo de optimización de modularidad de Louvain. Combina la fuerza de la homofilia global con el sesgo de la comunidad densa dominante a la que pertenece el ego.

### 6. Coocurrencia Estadística (Correlación)
* **Fundamento:** Interdependencia e inferencia cruzada de atributos.
* **Mecanismo:** Construye una matriz de correlación cruzada a partir de los perfiles de toda la red. Aplica un factor de empuje (*boost*) sobre aquellos atributos que estadísticamente tienden a manifestarse juntos, capturando dependencias que escapan al análisis puramente topológico.

### 7. Conectores Críticos (Puentes)
* **Fundamento:** Flujos de información y teoría de intermediación de Burt.
* **Mecanismo:** Evalúa la centralidad de intermediación (*betweenness centrality*) del grafo para identificar nodos puente estratégicos. Combina este peso métrico con el recuento de círculos compartidos para priorizar a los actores que controlan la comunicación entre regiones aisladas de la red.

## Instalación y Configuración

Clona el repositorio en tu máquina local y sitúate en la raíz del proyecto.

A continuación, coloca los datos crudos del dataset de Facebook de Stanford dentro de la carpeta [datos/facebook/](datos/facebook). Deben permanecer allí los ficheros originales de cada ego-red, incluyendo `.edges`, `.circles`, `.egofeat`, `.feat` y `.featnames`, porque el pipeline los consume directamente para reconstruir el grafo y calcular la inferencia.

Después instala las dependencias del proyecto con el gestor estándar de Python:

```bash
pip install -r requirements.txt
```

Si el entorno se encuentra correctamente configurado, el proyecto queda listo tanto para ejecutar la simulación masiva como para lanzar la interfaz interactiva.

## Modo de Uso

### Pipeline de Simulación Masiva

Ejecuta `main.py` para procesar en lote los distintos Ego IDs disponibles en `datos/facebook/`. Este flujo reconstruye cada ego-red, aplica de forma aislada todas las estrategias de inferencia en memoria y calcula las métricas frente al vector real del ego. El resultado se exporta de forma incremental a la carpeta [resultados/](resultados), donde se generan las tablas analíticas consolidadas, las predicciones por atributo y los mapas de explicabilidad XAI asociados a cada estrategia.

```bash
python main.py
```

### Plataforma Visual Interactiva

Lanza `dashboard.py` para abrir la interfaz web construida con Streamlit. Esta plataforma permite comparar visualmente el rendimiento de las estrategias, explorar los resultados históricos exportados por el pipeline y auditar sobre grafos interactivos de PyVis qué amigos concretos influyeron en cada predicción semántica. La vista se actualiza a partir de los CSV almacenados en [resultados/](resultados) y ofrece una inspección guiada del perfil real frente a la inferencia obtenida.

```bash
streamlit run dashboard.py
```