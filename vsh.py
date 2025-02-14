#vsh.py
import os
import time
import asyncio
import aiohttp
import aiofiles
from dataclasses import dataclass
from typing import List, Set, Optional

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prometheus_client import Counter, Histogram

from logger_all import logger_all
from selenium_handler2 import EnhancedStrategyTester, TimeframeMetrics

logger = logger_all.logger

# Metrics
url_fetch_duration = Histogram('url_fetch_duration_seconds', 'Time spent fetching URLs')
docs_processed = Counter('docs_processed_total', 'Number of documents processed')
vectorstore_ops = Counter('vectorstore_operations_total', 'Vector store operations', ['operation_type'])

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

@dataclass
class ProcessingMetrics:
    urls_processed: int
    chunks_created: int
    embedding_time: float

class CustomWebBaseLoader:
    def __init__(self, url: str, headers: Optional[dict] = None):
        self.url = url
        self.headers = headers or {"User-Agent": USER_AGENT}

    async def async_load(self) -> List[Document]:
        try:
            logger.info(f"Fetching URL: {self.url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, headers=self.headers) as response:
                    text = await response.text()
                    soup = BeautifulSoup(text, "html.parser")
                    logger.info(f"Successfully fetched and parsed URL: {self.url}")
                    return [Document(page_content=soup.get_text(), metadata={"source": self.url})]
        except Exception as e:
            logger.error(f"Error fetching {self.url}: {e}")
            return []

class VectorStoreHandler:
    def __init__(self):
        self.doc_vectorstore = None
        self.code_vectorstore = None
        self.ingested_urls: Set[str] = set()
        # Create a shared embedding model instance
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._stores_initialized = asyncio.Event()
    
    async def initialize_stores(self):
        try:
            self.doc_vectorstore = await self.initialize_doc_vectorstore()
            logger.info("doc_vectorstore successfully initalzied")
            self.code_vectorstore = await self.initialize_code_vectorstore()
            logger.info("code_vectorestore successfully intialized")
            if self.doc_vectorstore and self.code_vectorstore:
                self._stores_initialized.set()  # Set only if BOTH are successful
            else:
                logger.error("One or both vector stores failed to initialize.")  # Log the problem
                self._stores_initialized.set()  # Set even if there's a failure
        except Exception as e:
            logger.error(f"Error initializing stores: {e}", exc_info=True)
            self._stores_initialized.set()  # ALWAYS set the event
    
    async def initialize_code_vectorstore(self):
        if os.path.exists("code_vectorstore"):
            try:
                code_vectorstore = Chroma(
                    persist_directory="code_vectorstore",
                    embedding_function=self.embedding_model,
                    collection_name="generated-code"
                )
                logger.info("Successfully loaded existing code vector store")
                vectorstore_ops.labels(operation_type='load_existing_code').inc()
                return code_vectorstore
            except Exception as e:
                logger.error(f"Error loading code vectorstore: {e}")
        logger.info("Creating new code vector store")
        code_vectorstore = Chroma(
            persist_directory="code_vectorstore",
            embedding_function=self.embedding_model,
            collection_name="generated-code"
        )
        vectorstore_ops.labels(operation_type='create_new_code').inc()
        logger.info("Created new code vector store")
        return code_vectorstore

    async def initialize_doc_vectorstore(self):
        if os.path.exists("doc_vectorstore"):
            try:
                vectorstore = Chroma(
                    persist_directory="doc_vectorstore",
                    embedding_function=self.embedding_model,
                    collection_name="rag-chroma"
                )
                logger.info("Successfully loaded existing documentation vector store")
                vectorstore_ops.labels(operation_type='load_existing_doc').inc()
                return vectorstore
            except Exception as e:
                logger.error(f"Error loading existing vectorstore: {e}")
        logger.info("Creating new vector store")
        try:
            urls = await self.load_urls_from_file("urls.txt")
            logger.info(f"Total URLs loaded from file: {len(urls)}")
            
            start_time = time.time()
            docs_list = await self.fetch_all_documents(urls)
            fetch_elapsed = time.time() - start_time
            logger.info(f"Time taken to fetch documents: {fetch_elapsed:.2f} seconds")
            logger.info(f"Successfully fetched {len(docs_list)} documents")
            url_fetch_duration.observe(fetch_elapsed)
            if not docs_list:
                logger.info("No documents loaded from URLs")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            doc_splits = text_splitter.split_documents(docs_list)
            
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=self.embedding_model,
                persist_directory="doc_vectorstore"
            )
            docs_processed.inc(len(docs_list))
            vectorstore_ops.labels(operation_type='create_new_doc').inc()
            
            metrics = ProcessingMetrics(
                urls_processed=len(urls),
                chunks_created=len(doc_splits),
                embedding_time=time.time()  # Consider timing only embedding if needed
            )
            logger.info(f"Vector store metrics: {metrics}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating new vectorstore: {e}")
            return None

    async def fetch_all_documents(self, urls: List[str]) -> List[Document]:
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            logger.info(f"Starting to fetch {len(urls)} URLs")
            tasks = [self.fetch_single_document(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_docs = []
        failed_urls = []
        empty_responses = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                failed_urls.append((url, str(result)))
                logger.error(f"Failed to fetch {url}: {result}")
            elif not result:  # Empty response
                empty_responses.append(url)
                logger.error(f"Empty response from {url}")
            else:
                valid_docs.extend(result)
                logger.info(f"Successfully processed {url}")
        
        logger.info("Processing Summary:")
        logger.info(f"Total URLs: {len(urls)}; Successful: {len(valid_docs)}; Failed: {len(failed_urls)}; Empty: {len(empty_responses)}")
        if failed_urls:
            logger.info("Failed URLs and reasons:")
            for url, reason in failed_urls:
                logger.info(f"- {url}: {reason}")
        if empty_responses:
            logger.info("URLs with empty responses:")
            for url in empty_responses:
                logger.info(f"- {url}")
        return valid_docs

    async def fetch_single_document(self, session: aiohttp.ClientSession, url: str) -> List[Document]:
        try:
            async with session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30) as response:
                logger.info(f"Fetching {url} - Status: {response.status}")
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for {url}")
                    return []
                text = await response.text()
                if not text.strip():
                    logger.error(f"Empty content from {url}")
                    return []
                soup = BeautifulSoup(text, "html.parser")
                extracted_text = soup.get_text()  # Get text content

                content = ""  # Initialize an empty string
                if isinstance(extracted_text, list):  # Check if it's a list
                    content = "\n".join(extracted_text)  # Join list elements
                elif isinstance(extracted_text, str):  # Check if it's a string
                    content = extracted_text  # Assign string directly
                else:
                    logger.warning(f"Unexpected text type from BeautifulSoup: {type(extracted_text)}")
                    return []  # Return empty if it's an unexpected type

                if not content.strip():  # Check for empty content *after* joining
                    logger.error(f"No text content after parsing {url}")
                    return []

                return [Document(page_content=content, metadata={"source": url})]  # Use the joined string

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return []
        except Exception as e:
            logger.error(f"Error fetching {url}: {type(e).__name__} - {str(e)}")
            return []
    async def load_urls_from_file(self, filename='urls.txt') -> List[str]:
        try:
            async with aiofiles.open(filename, 'r') as file:
                content = await file.read()
                all_lines = content.splitlines()
                logger.info(f"Raw lines in file: {len(all_lines)}")
                urls = [line.strip() for line in all_lines if line.strip()]
                logger.info(f"Valid URLs after filtering: {len(urls)}")
                self.ingested_urls.update(urls)  # Update with the 'urls' list, not 'loaded_urls'
                return urls
        except FileNotFoundError:
            logger.info(f"URLs file {filename} not found. Creating empty file.")
            async with aiofiles.open(filename, 'w') as file:
                await file.write('')
            return []
        except Exception as e:
            logger.error(f"Error loading URLs from file: {e}")
            return []


    async def save_ingested_urls(self, filename="ingested_urls.txt"):
        try:
            async with aiofiles.open(filename, "w") as f:
                await f.write('\n'.join(self.ingested_urls))
            logger.info(f"Saved {len(self.ingested_urls)} ingested URLs")
        except Exception as e:
            logger.error(f"Error saving ingested URLs: {e}")

    async def add_urls_to_vectorstore(self, new_urls_str: str) -> str:
        new_urls = [url.strip() for url in new_urls_str.strip().splitlines() if url.strip()]
        # Remove already ingested URLs
        new_urls = [url for url in new_urls if url not in self.ingested_urls]
        if not new_urls:
            return "No new URLs to add."
        logger.info(f"Adding {len(new_urls)} new URLs to the vector store.")
        # Load documents concurrently from new URLs
        docs_tasks = [CustomWebBaseLoader(url).async_load() for url in new_urls]
        docs_results = await asyncio.gather(*docs_tasks, return_exceptions=True)
        # Flatten list and filter out failures
        docs_list = [doc for result in docs_results if not isinstance(result, Exception) and result for doc in result]
        if not docs_list:
            return "Failed to add documents: None were fetched successfully."
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(docs_list)
        logger.info(f"Split documents into {len(doc_splits)} chunks.")
        # Ensure vectorstore is available
        if self.doc_vectorstore is None:
            logger.info("Doc vector store not initialized. Initializing now.")
            self.doc_vectorstore = await self.initialize_doc_vectorstore()
            if self.doc_vectorstore is None:
                logger.error("Vector store failed to initialize.")
                return "Failed to add documents: Vector store failed to initialize."
        
        # Add documents to the vector store
        self.doc_vectorstore.add_documents(documents=doc_splits)
        vectorstore_ops.labels(operation_type='add_new_doc').inc()
        logger.info("Documents added to the vector store.")
        # Update ingested URLs and persist the update
        self.ingested_urls.update(new_urls)
        await self.save_ingested_urls()
        logger.info(f"Updated ingested URLs: {len(self.ingested_urls)} total URLs.")
        return f"Successfully added {len(new_urls)} URLs to the vector store."
    
    async def monitor_urls_file(self, interval=60):
        while True:
            try:
                logger.info("Checking for updates to urls.txt...")
                current_urls = set(await self.load_urls_from_file("urls.txt"))
                new_urls = current_urls - self.ingested_urls
                if new_urls:
                    logger.info(f"Found {len(new_urls)} new URLs in urls.txt.")
                    await self.add_urls_to_vectorstore("\n".join(new_urls))
                else:
                    logger.info("No new URLs found in urls.txt.")
            except Exception as e:
                logger.error(f"Error monitoring urls.txt: {e}")
            await asyncio.sleep(interval)

    async def ensure_stores_initialized(self):
        if not self.are_stores_initialized():
            await self.initialize_stores()
        try:
            await asyncio.wait_for(self._stores_initialized.wait(), timeout=60.0)  # 60-second timeout
        except asyncio.TimeoutError:
            logger.error("Vector store initialization timed out.")
            raise  
    def are_stores_initialized(self):
        return self.doc_vectorstore is not None and self.code_vectorstore is not None

    async def add_code_to_vectorstore(self, code: str, success: bool, results=None):
        await self.ensure_stores_initialized() # Ensure before using

        try:
            metadata = {"status": "success" if success else "failure", "timestamp": time.time()}

            if success:
                formatted_performance_summary = self.format_results(results)
                log_message = f"Strategy Performance Summary:\n{formatted_performance_summary}"
                content = f"Code:\n{code}\n\nPerformance Summary:\n{formatted_performance_summary}"
            else:
                failure_message = f"Strategy testing failed with errors:\n{code}\n{results}"
                log_message = failure_message
                content = f"Code:\n{code}\n\nFailure Message:\n{failure_message}"

            document = Document(page_content=content, metadata=metadata)

            if self.code_vectorstore:
                self.code_vectorstore.add_documents([document])
                vectorstore_ops.labels(operation_type='add_new_code').inc()
                logger.info(f"Added code and results to vector store with status: {metadata['status']}")
            else:
                logger.error("Code vector store is not initialized.")
                return "Code vector store not initialized."

            return log_message

        except Exception as e:
            logger.error(f"Error adding code to vectorstore: {e}", exc_info=True)
            return f"Failed to add code to vector store: {e}"

    def format_results(self,results) -> str:
        formatted = []
        for result in results:
            if isinstance(result, dict):
                formatted.append("\n".join(f"{key}: {value}" for key, value in result.items()))
            else:
                formatted.append(str(result))
        return "\n\n".join(formatted)            
# Example usage:
# async def main():
#     vs_handler = VectorStoreHandler()
#     await vs_handler.initialize_stores()
#     # Optionally run the monitor
#     # asyncio.create_task(vs_handler.monitor_urls_file())
#
# if __name__ == "__main__":
#     asyncio.run(main())
