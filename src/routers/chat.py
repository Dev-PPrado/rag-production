from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
from src.settings import settings
from fastapi import Form, UploadFile, File
import json
from src.pdf_utils import load_pdf_from_bytes
from src.chat.chat import chat
from src.customlogger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str

@router.post("/chat/ask", response_model=ChatResponse, tags=["Chat"])
async def ask_chat(
    query: str = Form(...),
    collection_name: str = Form("teste"),
    file: Optional[UploadFile] = File(None),
    history: Optional[str] = Form(None),
    clear_history: bool = Form(False)
):
    """
    Endpoint de consulta ao chat.
    Aceita um arquivo para ser usado como contexto.
    """
    try:
        # 1. Processa o histórico
        chat_history = []
        
        # Se clear_history for True, ignoramos qualquer histórico enviado, 
        # iniciando uma nova sessão do ponto de vista do backend para esta requisição.
        if clear_history:
             logger.info("Flag de limpeza de histórico recebida. Iniciando com histórico vazio.")
        elif history:
            try:
                history_list = json.loads(history)
                for msg in history_list:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "user":
                        chat_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        chat_history.append(AIMessage(content=content))
            except json.JSONDecodeError:
                logger.warning("Falha ao decodificar o JSON do histórico")

        # 2. Processa o arquivo
        file_text_content = None
        if file:
            logger.info(f"Processando arquivo: {file.filename}")
            if file.filename.lower().endswith('.pdf'):
                content_bytes = await file.read()
                # load_pdf_from_bytes retorna (texto, num_paginas)
                text, _ = load_pdf_from_bytes(content_bytes)
                file_text_content = text
            else:
                # Fallback para texto simples
                content_bytes = await file.read()
                file_text_content = content_bytes.decode('utf-8', errors='ignore')

        logger.info(f"Consulta recebida no chat: {query}")
        
        
        response_message = await chat.gerar_resposta(
            consulta=query,
            collection_name=collection_name,
            chat_history=chat_history,
            file_content=file_text_content,
        )
        
        return ChatResponse(response=response_message.content)

    except Exception as e:
        logger.error(f"Erro no Chat Fiscap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
