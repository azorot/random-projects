# gradio_interface.py
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from logger_all import logger_all
from vsh import VectorStoreHandler
from selenium_handler2 import EnhancedStrategyTester
import asyncio
from rl_agent import RLAgent, plot_rewards, plot_losses  # Import RLAgent class and plotting functions
import os # Import os for file operations

logger = logger_all.logger
class GradioInterface:
    def __init__(self, vector_store, strategy_tester, embedding_model):  # Removed rl_agent from args
        self.vector_store = vector_store
        self.strategy_tester = strategy_tester
        self.rl_agent = None  # Initialize to None, will be created later
        self.generated_code = None
        self.embedding_model = embedding_model # store the embedding model
        self.episode_rewards_history = None # Store reward history
        self.actor_losses_history = None   # Store actor loss history
        self.critic_losses_history = None  # Store critic loss history

    async def initialize_rl_agent(self, ollama_model, state_dim, action_dim, learning_rate, batch_number): # new method to initialize the rl_agent
        self.rl_agent = RLAgent(self.vector_store, self.strategy_tester, self.embedding_model, ollama_model, state_dim=state_dim, action_dim=action_dim, learning_rate=learning_rate, batch_number=batch_number) # batch_number is already ensured to be int in create_interface


    async def process_input(self, question, chat_history):
        logger.info(f"process_input - Input question type: {type(question)}")

        await self.vector_store.ensure_stores_initialized()

        if not self.vector_store.are_stores_initialized():
            return chat_history, ["Vector stores could not be initialized."]

        doc_retriever = (
            self.vector_store.doc_vectorstore.as_retriever()
            if self.vector_store.doc_vectorstore
            else None
        )

        if doc_retriever is None:
            logger.warning("doc_vectorstore is None. Cannot retrieve documents.")
            context = ""  # Or some default value
            docs = [] # Initialize docs to an empty list
        else:
            docs = await doc_retriever.aget_relevant_documents(question, k=5)
            context = "\n".join(doc.page_content for doc in docs) if docs else "" # handle empty docs

        try:
            generated_code = await self.rl_agent.generate_code(question)  # Use RL agent
            self.generated_code = generated_code  # Store generated code
            logger.info(f"Generated Code (RL):\n{generated_code}")
        except Exception as e: # Catch any errors during generation
            logger.error(f"Error generating code with RL agent: {e}", exc_info=True)
            generated_code = "Error generating code. Please check logs."  # Or a default message
            self.generated_code = None # No code was generated

        chat_history.append((question, generated_code)) # Append generated_code
        # Extract URLs from the retrieved documents
        retrieved_urls = [
            doc.metadata.get("source") for doc in docs if doc.metadata.get("source")
        ]  # safer way to check if source exists
        retrieved_urls = list(set(retrieved_urls))  # removes duplicate urls
        # Format retrieved URLs for display
        formatted_urls = "\n".join(retrieved_urls)
        return chat_history, formatted_urls


    def test_strategy(self):
        """Run the strategy test and store results."""
        if hasattr(self, 'generated_code') and self.generated_code:
            try:
                success, results = asyncio.run(self.strategy_tester.test_strategy_all_timeframes(self.generated_code))

                # Call add_code_to_vectorstore from VectorStoreHandler
                return asyncio.run(self.vector_store.add_code_to_vectorstore(
                    self.generated_code, success, results))

            except Exception as e:
                logger.error(f"Error during strategy testing: {e}")
                error_message = f"Strategy testing encountered an error: {e}"
                return asyncio.run(self.vector_store.add_code_to_vectorstore(
                    self.generated_code, False, error_message))

        return "No code generated to test."

    async def train_rl_agent(self, num_batches, episodes_per_batch, batch_number): # Modified to accept num_batches and episodes_per_batch
        if self.rl_agent:
            self.episode_rewards_history, self.actor_losses_history, self.critic_losses_history = await self.rl_agent.train_loop(num_batches=int(num_batches), episodes_per_batch=int(episodes_per_batch)) # Call train_loop, ensure num_batches and episodes_per_batch are int
            return f"RL Agent trained for {num_batches} batches, {episodes_per_batch} episodes per batch." # Updated message
        else:
            return "RL Agent not initialized."

    def show_plots(self):
        if self.episode_rewards_history and self.actor_losses_history and self.critic_losses_history:
            reward_plot_path = "training_plots/all_rewards_batch_latest.png" # Define file paths for plots, point to training_plots and use "latest" for gradio display
            loss_plot_path = "training_plots/all_losses_batch_latest.png"

            # Save plots with "latest" filenames for gradio to always show latest plots
            plot_rewards(self.episode_rewards_history, save_path="training_plots/all_rewards_batch_latest.png") # Save plots to files with "latest" in filename
            plot_losses(self.actor_losses_history, self.critic_losses_history, save_path="training_plots/all_losses_batch_latest.png")

            return reward_plot_path, loss_plot_path # Return image paths to display in Gradio
        else:
            return None, None # Return None if no data to plot

    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Document Query with Ollama RL Agent")
            gr.Markdown("Query documents and train RL agent to generate Pine Script strategies.")
            ollama_model = ChatOllama(base_url="http://127.0.0.1:11434", model='qwen2.5-coder:0.5b') # Initialize Ollama model
            state_dim = 384 # embedding size + context + retrieved codes info
            action_dim = 2 # number of different prompt templates
            learning_rate = 0.001
            batch_number_input = gr.Number(value=1, label="Starting Batch Number", precision=0) # Changed label
            num_batches_input = gr.Number(value=5, label="Number of Batches to Train", precision=0) # New input for num_batches
            episodes_per_batch_input = gr.Number(value=2, label="Episodes Per Batch", precision=0) # New input for episodes_per_batch

            # Get the value of batch_number_input *outside* the lambda
            initial_batch_number = batch_number_input.value
            print(f"Type of initial_batch_number: {type(initial_batch_number)}, Value: {initial_batch_number}") # Debug print

            demo.load(
                lambda batch_number=initial_batch_number: asyncio.run(self.initialize_rl_agent(ollama_model, state_dim, action_dim, learning_rate, batch_number=int(batch_number))) # Pass batch_number value as int
            )

            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(label="Chat History")
                    question = gr.Textbox(label="Question", placeholder="Type your question here...")
                    submit_button = gr.Button("Submit")
                    stored_urls = gr.Textbox(label="Stored URLs", lines=5)
                    code_display = gr.Code(label="Generated/Stored Code", lines=10)

                with gr.Column():
                    new_urls = gr.Textbox(label="Add URLs (one per line)", lines=5)
                    add_urls_button = gr.Button("Add URLs to Vector Store")
                    add_urls_output = gr.Textbox(label="Add URLs Output", lines=2)
                    load_urls_button = gr.Button("Load URLs from File")
                    load_urls_output = gr.Textbox(label="Load URLs Output", lines=2)

                    # Button to trigger strategy testing
                    test_strategy_button = gr.Button("Test Strategy")
                    strategy_output = gr.Textbox(label="Strategy Test Output", lines=5)

                    # RL Training controls - Modified inputs
                    # num_episodes_input = gr.Number(value=100, label="Total Episodes to Train (per Batch - now irrelevant)") # Changed label, but input is now episodes_per_batch
                    train_button = gr.Button("Train RL Agent")
                    train_output = gr.Textbox(label="RL Training Output", lines=2)

                    # Button to view plots
                    plot_button = gr.Button("View Plots")
                    reward_plot_image = gr.Image(label="Reward Plot") # Image components to display plots
                    loss_plot_image = gr.Image(label="Loss Plot")



            def sync_process_input(q, history):
                logger.info(f"sync_process_input - Input q type: {type(q)}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.process_input(q, history))
                finally:
                    loop.close()

            def sync_add_urls(new_urls_text):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.vector_store.add_urls_to_vectorstore(new_urls_text))

            def sync_load_urls():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.vector_store.load_urls_from_file())

            def sync_test_strategy():
                return self.test_strategy()  # Trigger strategy test

            def sync_train_rl_agent(num_batches, episodes_per_batch, batch_number): # Modified to accept num_batches and episodes_per_batch
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.train_rl_agent(int(num_batches), int(episodes_per_batch), int(batch_number))) # Pass num_batches and episodes_per_batch, convert to int - ENSURE ALL ARE INT

            def sync_show_plots():
                reward_plot_path, loss_plot_path = self.show_plots()
                return reward_plot_path, loss_plot_path # Return paths for image display


            submit_button.click(
                fn=sync_process_input,
                inputs=[question, chatbot],
                outputs=[chatbot, stored_urls],
            ).then( # Added .then to update code_display
                lambda: self.generated_code if hasattr(self, 'generated_code') else "",
                outputs=code_display # Update code_display
            )

            add_urls_button.click(
                fn=sync_add_urls,
                inputs=new_urls,
                outputs=add_urls_output,
            )

            load_urls_button.click(
                fn=sync_load_urls,
                outputs=load_urls_output,
            )

            # Button to test strategy after it's generated
            test_strategy_button.click(
                fn=sync_test_strategy,
                outputs=strategy_output,
            )

            # Button to train RL Agent - Modified inputs to use num_batches_input and episodes_per_batch_input
            train_button.click(
                fn=sync_train_rl_agent,
                inputs=[num_batches_input, episodes_per_batch_input, batch_number_input], # Pass num_batches_input and episodes_per_batch_input
                outputs=train_output
            )

            # Button to view plots
            plot_button.click(
                fn=sync_show_plots,
                outputs=[reward_plot_image, loss_plot_image] # Output image components
            )


        return demo

def extract_code(response: str) -> str:
    # Assuming you have a function to extract code from the response
    start = response.find("```pinescript")
    if start == -1:
        return ""
    start += len("```pinescript")
    end = response.find("```", start)
    return response[start:end].strip()