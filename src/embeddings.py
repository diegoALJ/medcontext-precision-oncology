from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np


class FrozenBioEmbedder:
    """
    Generate frozen biomedical sentence embeddings using BioLORD.

    The model is used only for inference:
    context text -> dense semantic vector
    """

    def __init__(
        self,
        model_name: str = "FremyCompany/BioLORD-2023",
        device: str | None = None,
    ):
        self.model_name = model_name

        if device is None:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )

    def encode(
        self,
        texts: str | list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert one text or a list of texts into dense embeddings.

        Parameters
        ----------
        texts:
            A single context string or a list of context strings.
        normalize:
            If True, L2-normalize the embeddings.

        Returns
        -------
        np.ndarray
            Shape:
            - (embedding_dim,) for one string
            - (n_texts, embedding_dim) for a list
        """
        single_input = isinstance(texts, str)

        if single_input:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        if single_input:
            return embeddings[0]

        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """
        Return the number of dimensions produced by the model.
        """
        return self.model.get_sentence_embedding_dimension()
