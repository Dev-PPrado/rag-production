from typing import List, Tuple
import io
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from qdrant_client.http.models import PointStruct
import uuid
from typing import Literal

from src.settings import settings
from src.chat.llm_models import gpt_4_1_mini, embedding
from src.models import DocumentMetadataExtraction
from src.customlogger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Inicializa os embeddings
        self.embeddings = embedding

    def extract_text_from_file(self, file_content: bytes, filename: str) -> List[Tuple[str, int]]:
        """Extrai o texto de PDF ou TXT, retornando trechos com o número da página."""
        pages_content = []
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_content))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_content.append((text, i + 1))
        else:
            # Assume TXT
            text = file_content.decode("utf-8")
            pages_content.append((text, 1))
        return pages_content

    def get_text_splitter(
        self,
        chunk_size: int,
        overlap_chunk: int,
        splitter_type: Literal["recursive", "character"],
        use_tiktoken: bool
    ):
        """Retorna o text splitter apropriado."""
        kwargs = {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap_chunk
        }
        
        if use_tiktoken:
            if splitter_type == "recursive":
                return RecursiveCharacterTextSplitter.from_tiktoken_encoder(**kwargs)
            else:
                return CharacterTextSplitter.from_tiktoken_encoder(**kwargs)
        else:
            if splitter_type == "recursive":
                return RecursiveCharacterTextSplitter(**kwargs)
            else:
                return CharacterTextSplitter(separator="\n\n", **kwargs)

    def extract_metadata_from_first_page(self, first_page_text: str) -> DocumentMetadataExtraction:
        """Usa o LLM para extrair classificação e descrição a partir da primeira página."""
        logger.info("Extraindo metadados com o LLM...")
        try:
            structured_llm = gpt_4_1_mini.with_structured_output(DocumentMetadataExtraction)
            
            prompt = (
                "Analise a seguinte primeira página de um documento e extraia:\n"
                "1. Uma classificação do documento em no máximo 3 palavras.\n"
                "2. Uma descrição do documento em no máximo 2 frases.\n\n"
                f"Texto da página:\n{first_page_text}"
            )
            
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"Falha ao extrair metadados: {e}")
            return DocumentMetadataExtraction(
                classificacao="Desconhecido",
                descricao="Não foi possível extrair a descrição."
            )

    def process_document(
        self, 
        file_content: bytes, 
        filename: str,
        chunk_size: int = 1000,
        overlap_chunk: int = 100,
        splitter_type: Literal["recursive", "character"] = "recursive",
        use_tiktoken: bool = False
    ) -> List[PointStruct]:
        """
        Extrai, divide em chunks, gera embeddings e cria os pontos do Qdrant com configurações dinâmicas.
        """
        pages = self.extract_text_from_file(file_content, filename)
        points = []
        
        if not pages:
            return points
            
        # Extrai os metadados da primeira página
        first_page_text = pages[0][0]
        extracted_meta = self.extract_metadata_from_first_page(first_page_text)
        
        text_splitter = self.get_text_splitter(
            chunk_size=chunk_size, 
            overlap_chunk=overlap_chunk, 
            splitter_type=splitter_type, 
            use_tiktoken=use_tiktoken
        )
        
        chunk_index = 0
        for page_text, page_num in pages:
            chunks = text_splitter.split_text(page_text)
            
            if not chunks:
                continue

            vectors = self.embeddings.embed_documents(chunks)
            
            for chunk, vector in zip(chunks, vectors):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector={"text-dense": vector},
                    payload={
                        "content": chunk,
                        "metadata": {
                            "chunk_index": chunk_index,
                            "nome_arquivo": filename,
                            "pagina": page_num,
                            "classificacao": extracted_meta.classificacao,
                            "descricao": extracted_meta.descricao
                        }
                    }
                ))
                chunk_index += 1
            
        return points
