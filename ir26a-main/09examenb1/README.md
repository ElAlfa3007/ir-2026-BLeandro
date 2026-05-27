# Sistema de Recuperación de Información - Examen 1Bim

Este repositorio contiene la implementación de un Sistema de Recuperación de Información (SRI) basado en representaciones vectoriales densas (embeddings) y similitud coseno. Este proyecto fue desarrollado como parte de la evaluación del primer bimestre de la asignatura **ICCD753 Recuperación de Información** (2026-A) en la **Escuela Politécnica Nacional (EPN - FIS)**.

## Autor
**Leandro Omar Bravo Orellana**

## Objetivo del Proyecto
Desarrollar un SRI capaz de indexar un corpus de documentos textuales mediante embeddings y recuperar los documentos más relevantes para una consulta de texto libre utilizando la medida de similitud coseno.

El corpus utilizado es el conjunto de datos **Rotten Tomatoes Movies and Critic Reviews Dataset**, descargado automáticamente a través de Kaggle.

## Características Implementadas
1. **Preprocesamiento de Texto:** Limpieza de reseñas mediante eliminación de caracteres especiales, lowercases y puntuación  utilizando herramientas personalizadas (`prepro_func.py`) que aseguran la consistencia semántica.
2. **Generación de Embeddings:** Uso del modelo `SentenceTransformer` para generar representaciones vectoriales densas para el texto completo de las reseñas de películas.
3. **Mecanismo de Recuperación:** Implementación matemática de la similitud coseno (`scikit-learn`) para devolver el Top-K de documentos relevantes.
4. **Desafío de Excelencia:** Visualización en 2D del espacio semántico del corpus de documentos frente a las consultas, empleando Análisis de Componentes Principales (PCA) mediante `matplotlib`.

## Estructura del Proyecto

Para la correcta ejecución del sistema, se requiere la siguiente estructura de directorios:

```text
├── BravoLeandro_ex1bim_ir26a.ipynb  # Jupyter Notebook principal con la ejecución del examen
├── requirements.txt                 # Archivo de dependencias del proyecto
└── README.md                        # Documentación del repositorio
```

## Instrucciones de Instalación y Ejecución

Para replicar el entorno de trabajo y ejecutar el notebook, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

### 2. Configurar Entorno Virtual (Recomendado)
Para Windows:
```bash
python -m venv env
.\env\Scripts\activate
```
Para Linux/macOS:
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Instalar Dependencias
Asegúrate de instalar los paquetes requeridos especificados en el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Nota: El archivo principal importa librerías como `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `matplotlib`, `gensim` y `kagglehub`).*

### 4. Ejecutar el Notebook
Inicia el servidor de Jupyter:
```bash
jupyter notebook
```
Abre el archivo `BravoLeandro_ex1bim_ir26a.ipynb` y ejecuta las celdas en orden. Asegúrate de tener conexión a Internet activa en la primera ejecución, ya que el sistema descargará el corpus directamente desde Kaggle y el modelo de embeddings desde Hugging Face.

## Consultas de Prueba
El notebook evalúa el sistema sobre 8 consultas predefinidas enfocadas en diversos géneros y descripciones cinematográficas, generando para cada una:
* Una tabla detallada con el **Top-10** de resultados, mostrando el Ranking, ID del documento, Título, Fragmento y nivel de Similitud.
* Una **Tabla Resumen General** consolidando el mejor documento recuperado (Top-1) para cada consulta.
