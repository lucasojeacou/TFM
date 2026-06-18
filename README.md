# Inferencia privada en redes sociales

Trabajo Fin de Máster sobre **privacidad relacional** en redes sociales: estudia hasta qué punto se pueden inferir atributos personales de un usuario a partir de la información de sus contactos, aunque ese usuario no los haya revelado directamente.

El proyecto trabaja con ego-redes de Facebook (dataset SNAP), compara ocho enfoques de inferencia y ofrece un **dashboard interactivo** para analizar resultados, simular el proceso paso a paso y explicar qué nodos influyen en cada predicción.

## Qué incluye

- **Pipeline batch** que evalúa todas las ego-redes disponibles y genera métricas comparativas.
- **Ocho estrategias** basadas en la estructura social (vecinos, círculos, comunidades, similitud, etc.).
- **Dashboard** con tres vistas: resultados globales, simulador en cuatro fases y auditoría con grafo explicativo (XAI).
- **Métricas** de precisión, sensibilidad, F1 y exactitud por ego y estrategia.

## Instalación

**Requisitos:** Python 3.11+, dataset SNAP de Facebook ego networks, entorno virtual recomendado.

```bash
git clone <url-de-tu-repositorio>
cd TFM
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Coloca los ficheros del dataset en `datos/facebook/`. Por cada usuario (ego ID) se necesitan, como mínimo, los archivos de enlaces, atributos del ego, atributos de amigos y nombres de atributos (formato estándar SNAP: `.edges`, `.egofeat`, `.feat`, `.featnames`; opcionalmente `.circles`).

El sistema detecta automáticamente qué ego-redes están completas y pueden procesarse.

## Uso

**Evaluación completa** (genera los CSV en `resultados/`):

```bash
python main.py
```

Si alguna ego-red falla al cargarse, se registra el error y se continúa con las demás.

**Dashboard interactivo:**

```bash
streamlit run dashboard.py
```

## Resultados

Tras ejecutar `main.py` se obtienen tres informes en `resultados/`:


| Salida                  | Descripción                                     |
| ----------------------- | ----------------------------------------------- |
| Métricas por estrategia | F1, precisión, sensibilidad y exactitud por ego |
| Explicabilidad          | Qué amigos contribuyen a cada atributo inferido |
| Predicciones            | Atributos activados por ego y estrategia        |


## Dashboard


| Vista                     | Para qué sirve                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Resultados**            | Compara el rendimiento de las estrategias con gráficos y tablas                                        |
| **Simulador**             | Recorre el proceso de inferencia en cuatro fases sobre un ego y una categoría                          |
| **Auditoría y grafo XAI** | Contrasta valor real e inferido; al elegir una categoría, muestra el grafo y los nodos más influyentes |


El simulador vuelve al paso inicial si cambias ego, estrategia o categoría. La inferencia se ejecuta en cada interacción de la pestaña; el avance por fases solo cambia la capa visual del grafo.

## Estrategias comparadas


| Estrategia               | Idea                                      |
| ------------------------ | ----------------------------------------- |
| Vecinos directos         | Homofilia entre amigos inmediatos         |
| Círculos sociales        | Grupos declarados en el perfil SNAP       |
| Transitividad            | Señal de contactos de segundo salto       |
| Similitud estructural    | Cercanía en la topología de la red        |
| Comunidades locales      | Agrupaciones detectadas en la vecindad    |
| Correlación de atributos | Refuerzo por coocurrencia entre atributos |
| Puentes de la red        | Nodos que conectan grupos distintos       |
| Prestigio social         | Peso según centralidad de los contactos   |


## Cómo se toma la decisión

El flujo es deliberadamente simple de explicar:

1. Cada estrategia calcula **qué tan probable** es cada atributo a partir de la red y los perfiles de los amigos.
2. Un módulo común convierte esas puntuaciones en predicciones finales **categoría por categoría**, aplicando umbrales adaptativos y reglas según el tipo de atributo (por ejemplo, categorías ruidosas como cumpleaños no se infieren).

Así, todas las estrategias comparten la misma lógica de corte y la comparación entre ellas es justa.

## Estructura general

```text
TFM/
├── main.py           # Evaluación batch
├── dashboard.py      # Interfaz web
├── datos/            # Dataset SNAP
├── resultados/       # CSV generados
├── src/
│   ├── registro.py   # Registro único de estrategias
│   ├── nucleo/       # Carga de datos, políticas y métricas
│   ├── estrategias/  # Algoritmos de inferencia y sus parámetros
│   └── interfaz/     # Dashboard, grafo y visualización
└── tests/            # Pruebas automáticas
```

## Tests

```bash
pytest -q
```

La suite incluye **17 pruebas** que cubren el pipeline de las nueve estrategias del registro, la explicabilidad de transitividad y similitud, y componentes del núcleo (métricas, umbrales y exportación). No requiere el dataset SNAP real: usa un grafo sintético de prueba.