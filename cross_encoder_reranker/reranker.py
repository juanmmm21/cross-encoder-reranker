import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Intentamos importar librerias de Deep Learning para ejecutar el Cross-Encoder real
DL_AVAILABLE = False
torch = None
AutoTokenizer = None
AutoModelForSequenceClassification = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    DL_AVAILABLE = True
except ImportError:
    pass


class CrossEncoderReranker:
    """
    Reordenador (Reranker) de alta precision basado en arquitectura Cross-Encoder.
    
    A diferencia de un Bi-Encoder que calcula embeddings de forma independiente,
    el Cross-Encoder procesa simultaneamente la consulta y el documento a traves
    del Transformer, permitiendo atencion cruzada total (Full Cross-Attention)
    para una evaluacion semantica mucho mas precisa.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ) -> None:
        """
        Args:
            model_name: Nombre del modelo pre-entrenado en Hugging Face.
            device: Dispositivo de computo ('cpu', 'cuda', 'mps'). Autodetectado si es None.
        """
        self.model_name = model_name
        self.is_online = DL_AVAILABLE
        
        # Stop-words basicas en español e ingles para reducir ruido en modo offline
        self.stopwords = {
            "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "no",
            "de", "del", "a", "al", "en", "con", "por", "para", "como", "que", "es", "son",
            "cuales", "cual", "como", "cuando", "donde", "quien", "quienes", "que", "porque", "por que",
            "algun", "alguna", "algunos", "algunas", "otro", "otra", "otros", "otras", "este", "esta",
            "estos", "estas", "un", "una", "todo", "todos", "toda", "todas",
            "the", "a", "an", "and", "or", "but", "if", "not", "of", "to", "in", "with", "for", "is", "are"
        }
        
        if self.is_online:
            # Seleccion inteligente del dispositivo
            if device is None:
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                elif torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
                
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Cross-Encoder inicializado ONLINE en '{self.device}' con: {model_name}")
            except Exception as e:
                logger.warning(f"Error al cargar modelo Hugging Face. Fallback a modo OFFLINE. Detalle: {str(e)}")
                self.is_online = False
        else:
            logger.info("Cross-Encoder inicializado en modo OFFLINE (Algoritmo de solapamiento lexico).")

    def _tokenize_clean(self, text: str) -> List[str]:
        """
        Tokeniza y limpia un string basico, normalizando acentos y extrayendo stems.
        """
        accents = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
        text_norm = text.lower()
        for a, b in accents.items():
            text_norm = text_norm.replace(a, b)
        words = re.findall(r'\b\w+\b', text_norm)
        tokens = []
        for w in words:
            if w in self.stopwords:
                continue
            tokens.append(w)
            if len(w) > 3:
                tokens.append(w[:3])
            if len(w) > 4:
                tokens.append(w[:4])
        return tokens

    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Calcula la puntuacion de relevancia semantica de cada documento frente a la consulta.
        """
        if not documents:
            return []
            
        if self.is_online:
            return self._compute_scores_online(query, documents)
        else:
            return self._compute_scores_offline(query, documents)

    def _compute_scores_online(self, query: str, documents: List[str]) -> List[float]:
        """
        Ejecuta la inferencia real de clasificacion de secuencias emparejadas con PyTorch.
        """
        # Formateamos pares [[Query, Doc1], [Query, Doc2], ...]
        pairs = [[query, doc] for doc in documents]
        
        try:
            # Tokenizamos de forma agrupada aplicando padding y truncamiento automatico
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            # Transferimos los tensores al dispositivo seleccionado de forma compatible con diccionarios
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Obtenemos los logits brutos del clasificador (normalmente de dimension 1 para regresion de relevancia)
                logits = outputs.logits.squeeze(-1)
                
                # Si es un solo documento, squeeze(-1) colapsa a tensor escalar. Controlamos eso:
                if len(documents) == 1:
                    return [float(logits.item())]
                    
                return logits.cpu().tolist()
        except Exception as e:
            logger.error(f"Error en inferencia online del Cross-Encoder: {str(e)}. Fallback a offline.")
            return self._compute_scores_offline(query, documents)

    def _compute_scores_offline(self, query: str, documents: List[str]) -> List[float]:
        """
        Calcula un score heuristico basado en la interseccion lexica normalizada
        y proximidad basica de terminos de la consulta dentro del documento.
        """
        query_words = set(self._tokenize_clean(query))
        if not query_words:
            return [0.0] * len(documents)
            
        scores = []
        for doc in documents:
            doc_words = self._tokenize_clean(doc)
            if not doc_words:
                scores.append(0.0)
                continue
                
            # Interseccion de palabras
            overlap = query_words.intersection(doc_words)
            
            # Puntuacion basica: proporcion de palabras de la query encontradas
            # Añadimos un pequeño bono por densidad (longitud mas corta del doc es mas denso)
            ratio = len(overlap) / len(query_words)
            density_bonus = len(overlap) / len(doc_words) if doc_words else 0.0
            
            score = float(ratio * 5.0 + density_bonus)
            
            # Penalizacion de paginas de bibliografia, indices o anexos (sin importar acentos)
            doc_lower = doc.lower()
            accents = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
            doc_norm = doc_lower
            for a, b in accents.items():
                doc_norm = doc_norm.replace(a, b)
                
            penalty_keywords = ["bibliografia", "webgrafia", "indice de", "anexo", "referencias"]
            if any(kw in doc_norm[:120] for kw in penalty_keywords):
                score -= 3.0
                
            scores.append(score)
            
        return scores

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3,
        text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Ordena una lista de diccionarios de documentos segun su relevancia frente a la consulta.
        
        Args:
            query: Consulta en texto plano.
            documents: Lista de diccionarios que contienen los documentos (ej. del pipeline hibrido).
            top_k: Cuantos documentos ordenar y retornar.
            text_key: La llave del diccionario donde se encuentra el contenido de texto plano.
            
        Returns:
            Lista de los mejores top_k documentos ordenados de mayor a menor relevancia.
        """
        if not documents:
            return []
            
        # Extraemos solo el texto de cada diccionario
        texts = [doc.get(text_key, "") for doc in documents]
        
        # Calculamos los scores de relevancia de forma paralela/lote
        scores = self.compute_scores(query, texts)
        
        # Inyectamos el score en cada documento para auditoria y ordenacion
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
            
        # Ordenamos descendente por el score asignado
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_docs[:top_k]
