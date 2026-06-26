# cross-encoder-reranker

Modulo reordenador (Reranker) de alta precision basado en la arquitectura Cross-Encoder de redes Transformer.

Este proyecto actua como la segunda etapa en sistemas de recuperacion y RAG. Toma los documentos candidatos recuperados de forma rapida por un indice vectorial o léxico y los reordena exhaustivamente utilizando atencion cruzada total para seleccionar los mejores resultados antes de enviarlos a la ventana de contexto del LLM.

## Arquitectura y Fundamentos Teoricos

En la arquitectura de sistemas de busqueda semantica modernos (como RAG), la recuperacion se divide en dos fases:

### 1. Primera Fase: Recuperación Rápida (Bi-Encoder)
Los documentos y las consultas se vectorizan de forma independiente mediante un codificador (como `contrastive-embedding-trainer`). La similitud se calcula mediante similitud de coseno en un espacio vectorial.
*   **Ventaja:** Permite indexar y buscar sobre millones de documentos en milisegundos usando indices HNSW (como `nano-vector-db`).
*   **Desventaja:** No hay atencion cruzada entre los tokens de la consulta y del documento durante la codificacion, lo que puede omitir sutilezas gramaticales.

### 2. Segunda Fase: Reordenamiento Preciso (Cross-Encoder)
Toma los mejores candidatos (ej. los mejores 20 o 50) y los procesa en parejas `[Consulta, Documento]` unificadas a traves de un clasificador Transformer de secuencias.
*   **Ventaja:** Atencion cruzada total (Full Cross-Attention). Cada token de la consulta puede interactuar con cada token del documento en todas las capas del Transformer. Esto permite entender negaciones, contextos complejos y sinonimias con la maxima precision.
*   **Desventaja:** Es computacionalmente costoso y lento de evaluar sobre todo el corpus. Por ello, solo se aplica como filtro de reordenamiento sobre los mejores resultados de la primera fase.

## Conexion con el Ecosistema

Este modulo se integra en el flujo RAG de `ai-core-infra`:
*   **hybrid-search-retrieval-pipeline:** El pipeline de recuperacion hibrida devuelve una lista de documentos candidatos. Este modulo toma esa lista y la optimiza mediante reordenamiento, filtrando los 3 documentos con mayor peso semantico real para la generacion final.

## Estructura del Proyecto

*   **reranker.py:** Clase principal `CrossEncoderReranker` que inicializa el modelo de clasificacion de secuencias de Hugging Face y gestiona el fallback lexico de coincidencia local en entornos offline.
*   **test_reranker.py:** Suite de pruebas unitarias que comprueba el algoritmo de solapamiento lexico y valida la inferencia en tensores mockeando a PyTorch.
*   **example.py:** Demostracion interactiva de reordenamiento de candidatos multitopicos midiendo el impacto y los cambios en los rankings.

## Instalacion y Requisitos

1. Crea e inicia un entorno virtual dentro de la carpeta del proyecto:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Instrucciones de Uso

### Ejecutar Pruebas Unitarias
Para validar la logica de ordenamiento y el flujo del tensor clasificador:
```bash
python -m unittest test_reranker.py
```

### Ejecutar Demostración
Para realizar un ciclo completo de reordenamiento sobre candidatos simulados:
```bash
python example.py
```
El script evaluara la relevancia de la consulta frente a los candidatos en el dispositivo de computo mas veloz (CUDA, MPS o CPU) y mostrara por consola la puntuacion semantica del clasificador y el nuevo ranking resultante.
