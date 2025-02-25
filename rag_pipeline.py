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
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_API_BASE_URL: str = "http://localhost:11434"
        OLLAMA_API_KEY: str = "if you are hosting ollama, put api key here"
        Pipelines: list[str] = []
        priority: int = 0

    def __init__(self):
        self.documents = None
        self.index = None

        if valves:
            self.valves = self.Valves(**valves)  # Load values from WebUI if provided
        else:
            self.valves = self.Valves(
                OLLAMA_API_BASE_URL=os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434"),
                OLLAMA_API_KEY=os.getenv("OLLAMA_API_KEY", ""),
                Pipelines=[],
                priority=0,
            )  

    async def on_startup(self):
        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        self.documents = SimpleDirectoryReader("./data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen