import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Intentamos importar torch si esta disponible para la prueba mockeada de PyTorch
try:
    import torch
except ImportError:
    torch = None

from reranker import CrossEncoderReranker


class TestCrossEncoderReranker(unittest.TestCase):
    """
    Suite de pruebas unitarias para validar el reordenador Cross-Encoder
    tanto en modo offline (heuristico) como online (mockeado).
    """

    def test_offline_scoring(self) -> None:
        """
        Verifica el calculo heuristico de coincidencia de terminos offline.
        """
        reranker = CrossEncoderReranker(device="cpu")
        # Forzamos modo offline para la prueba
        reranker.is_online = False
        
        query = "Python programming tutorial"
        documents = [
            "This is a tutorial on Python programming for beginners.",  # Coincide completo
            "Welcome to a cooking recipe for baking bread.",            # Cero coincidencia
            "Basic tutorial on Java programming syntax."                # Coincidencia parcial (tutorial, programming)
        ]
        
        scores = reranker.compute_scores(query, documents)
        
        self.assertEqual(len(scores), 3)
        # El primer documento debe tener el score mas alto
        self.assertTrue(scores[0] > scores[2])
        # El segundo documento debe tener score 0.0
        self.assertEqual(scores[1], 0.0)

    def test_rerank_dictionaries(self) -> None:
        """
        Verifica que el metodo rerank ordene correctamente listas de diccionarios.
        """
        reranker = CrossEncoderReranker(device="cpu")
        reranker.is_online = False
        
        query = "receta paella"
        docs = [
            {"id": 1, "text": "Aprender programacion en Java y desarrollo backend."},
            {"id": 2, "text": "Receta de paella valenciana con marisco y arroz."},
            {"id": 3, "text": "Como freir patatas usando aceite de oliva caliente."}
        ]
        
        # Solicitamos los 2 mejores resultados
        ranked = reranker.rerank(query, docs, top_k=2, text_key="text")
        
        self.assertEqual(len(ranked), 2)
        # El primer documento devuelto debe ser la paella (ID 2)
        self.assertEqual(ranked[0]["id"], 2)
        self.assertIn("rerank_score", ranked[0])
        # El segundo debe ser el de las patatas (ID 3, coincide parcial 'aceite'/'paella'? No, pero Java tiene 0)
        self.assertTrue(ranked[0]["rerank_score"] > ranked[1]["rerank_score"])

    @unittest.skipIf(torch is None, "PyTorch no esta instalado en el sistema global")
    @patch("reranker.AutoModelForSequenceClassification")
    @patch("reranker.AutoTokenizer")
    def test_online_inference_mock(self, mock_tokenizer_class, mock_model_class) -> None:
        """
        Simula la llamada al modelo Transformer de Hugging Face y comprueba
        el desempaquetado de los tensores de logits.
        """
        # Configurar Mock del Tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Configurar Mock del Modelo
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # Logits simulados para 2 documentos
        mock_outputs.logits = torch.tensor([[4.5], [-1.2]])
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Inicializamos el reranker
        reranker = CrossEncoderReranker(device="cpu")
        # Forzamos flags para simular el modo online
        reranker.is_online = True
        reranker.tokenizer = mock_tokenizer
        reranker.model = mock_model
        
        scores = reranker.compute_scores("query", ["doc1", "doc2"])
        
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 4.5, places=5)
        self.assertAlmostEqual(scores[1], -1.2, places=5)


if __name__ == "__main__":
    unittest.main()
