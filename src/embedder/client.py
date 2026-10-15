import os
from typing import List, Dict, Optional, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVectorParams

from src.settings import settings
from src.customlogger import setup_logger

logger = setup_logger(__name__)

class QdrantService:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=int(settings.QDRANT_PORT)
        )

    async def create_collection(self, collection_name: str, vector_size: int = 3072):
        """Cria uma nova collection caso ela não exista."""
        if not await self.client.collection_exists(collection_name):
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={'text-dense': VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={'text-sparse': SparseVectorParams(index=models.SparseIndexParams())},
            )
            return True
        return False

    async def upsert_vectors(self, collection_name: str, points: List[PointStruct]):
        """Insere ou atualiza (upsert) vetores em uma collection."""
        if not points:
            logger.warning(f"Upsert chamado com lista de pontos vazia para a collection '{collection_name}'")
            return

        # Extrai o nome do arquivo para contexto de log, se disponível
        first_payload = points[0].payload
        filename = "unknown_file"
        if first_payload:
            if "metadata" in first_payload:
                filename = first_payload["metadata"].get("nome_arquivo", "unknown_file")
            elif "filename" in first_payload:
                filename = first_payload.get("filename", "unknown_file")
        
        logger.info(f"Inserindo {len(points)} vetores do arquivo '{filename}' na collection '{collection_name}'")

        try:
            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"{len(points)} vetores inseridos com sucesso para '{filename}'")
        except Exception as e:
            logger.error(f"Falha ao inserir vetores para '{filename}': {e}")
            raise e


    async def list_unique_documents(self, collection_name: str) -> List[str]:
        """
        Lista os nomes de arquivo únicos de uma collection usando a API de scroll.
        Obs.: pode ser ineficiente para collections muito grandes se não for otimizado.
        Para melhor desempenho, usaríamos um índice de payload ou um conjunto separado de documentos ativos.
        """
        
        unique_filenames = set()
        next_page_offset = None
        
        while True:
            records, next_page_offset = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_page_offset
            )
            
            for record in records:
                if record.payload and "metadata" in record.payload:
                    pdf_name = record.payload["metadata"].get("nome_arquivo")
                    if pdf_name:
                        unique_filenames.add(pdf_name)
            
            if next_page_offset is None:
                break
                
        return list(unique_filenames)

    async def document_exists(self, collection_name: str, filename: str) -> bool:
        """Verifica se um documento com o nome de arquivo informado já existe na collection."""
        try:
            count_result = await self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.nome_arquivo",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                exact=True
            )
            return count_result.count > 0
        except Exception as e:
            logger.error(f"Falha ao verificar se o documento '{filename}' existe em '{collection_name}': {e}")
            return False

    async def delete_document(self, collection_name: str, filename: str):
        """Remove todos os pontos correspondentes ao nome do arquivo."""
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.nome_arquivo",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
        )

    async def collection_exists(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name)

    async def delete_collection(self, collection_name: str):
        """Remove uma collection por completo."""
        await self.client.delete_collection(collection_name=collection_name)
