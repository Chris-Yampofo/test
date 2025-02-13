"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""
import os
import logging
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        OPENAI_API_KEY: str = ""
        OPENAI_LLM_MODEL: str = "gpt-3.5-turbo"
        OPENAI_LLM_TEMPERATURE: float = 0
        OPENAI_LLM_MAX_TOKENS: int = 2000
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    def __init__(self):
        try:
            self.documents = None
            self.index = None
            self.name = "RAG Pipeline LlamaIndex v1.0"

            self.valves = self.Valves(
                pipelines=["*"],
                OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
            )

            assert self.valves.OPENAI_API_KEY, "OPENAI_API_KEY must be set!"

            logging.info(f"Pipeline initialized with name: {self.name}")
            import asyncio
        except Exception as e:
            logging.error(f"Error initializing Pipeline: {e}")

    async def on_startup(self):
        logging.info("Running on_startup() method...")
        try:
            from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

            data_path = "/mnt/c/Users/chris/Documents/Home/si/ideas/cascade/Youtube/ScriptWrite/api/RAG/script_db/scripts"
            if not os.path.exists(data_path):
                logging.error(f"Directory does not exist: {data_path}")
                return

            self.documents = SimpleDirectoryReader(data_path).load_data()
            if not self.documents:
                logging.error("No documents loaded. Index cannot be created.")
                return

            self.index = VectorStoreIndex.from_documents(self.documents)
            logging.info("Pipeline startup successful, index created.")
        except Exception as e:
            logging.error(f"Error loading documents or creating index: {e}")

    async def on_shutdown(self):
        logging.info("Shutting down the pipeline...")

    async def on_valves_updated(self):
        logging.info("Valves are updated")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        if not self.index:
            logging.error("Index failed to initialize.")
            return "Error: Index is not available."

    logging.info(f"Processing query: {user_message}")
    query_engine = self.index.as_query_engine(streaming=True)
    response = query_engine.query(user_message)