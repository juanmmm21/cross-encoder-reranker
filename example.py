import time
from typing import List, Dict, Any

from reranker import CrossEncoderReranker

def main() -> None:
    print("==================================================")
    print("      Demostracion de Cross-Encoder Reranker      ")
    print("==================================================")
    
    # 1. Consulta y candidatos devueltos por un pipeline de recuperacion inicial (ej. BM25 o Vector)
    # Algunos candidatos son semanticamente muy cercanos, otros solo comparten palabras clave sin relevancia real.
    query = "Como programar algoritmos con codigo limpio en Python"
    
    candidates = [
        {
            "id": 1,
            "text": "La exploracion espacial por satelites aporta datos clave del sistema solar.",
            "source": "astronomia",
            "initial_rank": 1
        },
        {
            "id": 2,
            "text": "Aprender a programar en Python abre muchas puertas en desarrollo web y ciencia de datos.",
            "source": "programacion",
            "initial_rank": 2
        },
        {
            "id": 3,
            "text": "El codigo limpio y refactorizado reduce la deuda tecnica de los proyectos de software.",
            "source": "programacion",
            "initial_rank": 3
        },
        {
            "id": 4,
            "text": "Para freir patatas perfectas se recomienda usar aceite de oliva a fuego medio en la cocina.",
            "source": "cocina",
            "initial_rank": 4
        },
        {
            "id": 5,
            "text": "Guia de optimizacion y buenas practicas para programar algoritmos estructurados usando Python limpio.",
            "source": "programacion",
            "initial_rank": 5
        }
    ]
    
    print(f"\nConsulta: '{query}'")
    print("\nDocumentos antes de Reordenacion (Orden Inicial):")
    for doc in candidates:
        print(f"  [{doc['initial_rank']}] ID: {doc['id']} | Categoria: {doc['source']} | '{doc['text'][:65]}...'")

    # 2. Inicializacion del Reranker
    # Por defecto intenta cargar el modelo de Hugging Face.
    # Si no hay conexion de red o PyTorch no esta configurado, caera de forma automatica
    # y transparente al algoritmo de solapamiento lexico offline.
    print("\nInicializando reordenador...")
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 3. Ejecucion de la reordenacion (Reranking)
    start_time = time.perf_counter()
    ranked_results = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=3,  # Nos quedamos con los 3 mejores resultados para inyectar en la ventana del LLM
        text_key="text"
    )
    exec_time = (time.perf_counter() - start_time) * 1000
    
    print("\n" + "="*50)
    print(f" Documentos despues de Reordenacion (Top 3 en {exec_time:.2f} ms) ")
    print("="*50)
    for i, doc in enumerate(ranked_results, start=1):
        print(f"  • Rango {i} | ID: {doc['id']} | Score Cross-Encoder: {doc['rerank_score']:.4f}")
        print(f"    Texto: '{doc['text']}'")
        print(f"    Origen: {doc['source']} | Rango inicial: {doc['initial_rank']}\n")


if __name__ == "__main__":
    main()
