#grag.py
import asyncio
from gradio_interface import GradioInterface
from vsh import VectorStoreHandler
from selenium_handler2 import EnhancedStrategyTester
from logger_all import logger_all
from langchain_huggingface import HuggingFaceEmbeddings  # Import for embeddings

logger = logger_all.logger

async def async_main():

    vector_store = VectorStoreHandler()
    logger.info("grag.py: initalizing stores")
    await vector_store.initialize_stores()  # Async initialization of vector stores

    strategy_tester = EnhancedStrategyTester("https://www.tradingview.com/chart/79UNq2Rh/?symbol=PHEMEX%3ALUNCUSDT", max_workers=4)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Create embedding model
    interface = GradioInterface(vector_store, strategy_tester, embedding_model)  # Pass it here

    demo = interface.create_interface()
    demo.launch(server_name="0.0.0.0", server_port=8000)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()