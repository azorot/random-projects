# rl_agent.py
import asyncio
import time
from typing import List, Dict, Tuple, Any

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from logger_all import logger_all
from vsh import VectorStoreHandler  # Assuming this contains your VectorStoreHandler and other setup
from selenium_handler2 import EnhancedStrategyTester  # Assuming this contains your strategy tester
import torch as T
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt  # Import for plotting
import datetime  # Import for timestamping filenames
import os  # Import os for path operations
import time
logger = logger_all.logger


class PPO:
    def __init__(self, state_dim, learning_rate, action_dim, device):  # Add device here!
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.LogSoftmax(dim=-1)
        ).to(device)  # Move to device immediately
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)  # Move to device immediately
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 2
        self.epochs = 10
        self.device = device  # Store the device


    def choose_action(self, state):
        """Choose an action based on the current state."""
        logger.info(f"choose_action - Input state type: {type(state)}")  # Debug log: type of state
        if isinstance(state, dict):
            logger.info(f"choose_action - State keys: {state.keys()}")  # Debug log: keys in state dict
            if "query_embedding" in state:
                logger.info(f"choose_action - Type of query_embedding in state: {type(state['query_embedding'])}")  # Debug log: type of query_embedding
                if isinstance(state['query_embedding'], list) and len(state['query_embedding']) > 0:
                    logger.info(f"choose_action - Type of first element in query_embedding: {type(state['query_embedding'][0])}")  # Deeper type check

        state_tensor = torch.tensor(np.array(state['query_embedding']), dtype=torch.float32).unsqueeze(0)
        if not self.actor:
            logger.warning("Actor model is not initialized, using exploration action.")
            return self.exploration_action(state), None

        self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        logger.debug(f"Chosen action: {action.item()}, Log probability: {log_prob.item()}")  # Log action and log_prob
        # Optional: Log action probabilities (if action space is small)
        # action_probs_list = torch.exp(action_probs).tolist()[0] # Convert to probabilities
        # logger.debug(f"Action probabilities: {action_probs_list}")
        return action.item(), log_prob
    # def update_policy(self, states, actions, rewards, next_states, old_action_probs):
    def update_policy(self, states, actions, rewards, next_states, old_log_probs, device):
        # Convert to tensors *ONCE* outside the loop and put on device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(self.device)

        actor_losses = []
        critic_losses = []

        for epoch in range(self.epochs):
            # Calculate advantage *INSIDE* the epoch loop
            values = self.critic(states).squeeze(1)
            next_values = self.critic(next_states).squeeze(1)
            advantage = rewards + self.gamma * next_values - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i + self.batch_size]  # No need to detach here
                batch_actions = actions[i:i + self.batch_size]
                batch_advantage = advantage[i:i + self.batch_size]
                batch_old_log_probs = old_log_probs[i:i + self.batch_size]

                # Actor update
                new_action_probs = self.actor(batch_states)
                action_dist = Categorical(new_action_probs)
                log_probs = action_dist.log_prob(batch_actions)
                ratios = torch.exp(log_probs - batch_old_log_probs[torch.arange(len(batch_actions)), batch_actions])
                clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                actor_loss = -(batch_advantage * clipped_ratios).mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Critic update
                critic_loss = batch_advantage.pow(2).mean() # critic update on the advantage
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return actor_losses, critic_losses
class RLAgent:
    def __init__(
        self,
        vector_store: VectorStoreHandler,
        strategy_tester: EnhancedStrategyTester,
        embedding_model: HuggingFaceEmbeddings,
        ollama_model: ChatOllama,  # Add Ollama model
        k_nearest: int = 5,  # Number of nearest neighbors to retrieve
        state_dim: int = None,
        action_dim: int = None,
        learning_rate: float = None,
        actor_model_path: str = None,  # Add actor_model_path
        critic_model_path: str = None,  # Add critic_model_path
        batch_number: int = None # ADD BATCH NUMBER HERE
    ):
        self.vector_store = vector_store
        self.strategy_tester = strategy_tester
        self.embedding_model = embedding_model
        self.ollama_model = ollama_model  # Initialize Ollama model
        self.k_nearest = k_nearest
        self.after_rag_template = """

        You are an expert Pine Script v6 assistant. Generate original, functional code based on:
        1. Official documentation context
        2. Successful strategy patterns
        3. Performance across multiple timeframes

        Documentation Context: {context}
        Previous Successful Strategies: {code_context}

        Generate a robust Pine Script v6 strategy that:
        1. Uses proper risk management (e.g., stop-loss, take-profit)
        2. Implements clear entry/exit rules based on technical indicators (e.g., RSI, MACD, moving averages)
        3. Includes dynamic position sizing based on account balance or risk tolerance
        4. Handles different market conditions (e.g., trending, ranging, volatile)
        5. Works across multiple timeframes (specify which timeframes, e.g., 15m, 1h, 4h, 1D)

        Query: {question} (Question: {question} - Carefully analyze the provided documentation and code examples to generate the most relevant and effective Pine Script strategy.)
        """
        self.after_rag_prompt = ChatPromptTemplate.from_template(self.after_rag_template)
        self.after_rag_chain = self.after_rag_prompt | self.ollama_model | StrOutputParser()
        self.rl_algorithm = PPO(state_dim, learning_rate, action_dim,device='cpu')  # Initialize PPO

        # --- Model Loading (Commented out initially) ---
        # if actor_model_path and critic_model_path:
        #     self.load_models(actor_model_path, critic_model_path)
        # else:
        #     logger.info("Starting with randomly initialized models.")


        self.buffer = []  # buffer to store the training data
        self.action_dim = action_dim  # store action dim
        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2
        self.K_epochs = 80
        self.batch_size = 2
        self.episode_rewards_history = []  # List to store rewards per episode
        self.actor_losses_history = []
        self.critic_losses_history = []
        self.current_batch_folder = None # Track current batch folder
        self.batch_count = batch_number if batch_number is not None else 0

    def embed_query(self, text: str) -> List[float]:

        if isinstance(text, list):
            logger.warning(f"Expected a string, but got a list: {text}")

            # Check if the list is a list of lists
            if isinstance(text[0], list):
                logger.warning(f"Found a list of lists: {text}")

        return self.embedding_model.embed_query(text)  # Corrected embedding - removed embed_documents

    async def get_state(self, query: str) -> Dict[str, Any]:

        if isinstance(query, list):
            logger.warning(f"Expected a string, but received a list: {query}")
            query = " ".join(query)
        query_embedding = self.embed_query(query)  # Keep embedding the query for state representation

        retrieved_codes_with_scores = (
            self.vector_store.code_vectorstore.similarity_search_with_score(  # Reverted to similarity_search_with_score
                query,  # Pass the QUERY STRING here, not the embedding
                k=self.k_nearest
            )
            if self.vector_store.code_vectorstore
            else []
        )

        retrieved_codes = []
        embedded_codes = []  # List to store embedded code contexts

        for doc, score in retrieved_codes_with_scores:
            code_data = {
                "code": doc.page_content,
                "success": doc.metadata.get("status") == "success",
                "results": doc.metadata.get("results"),
                "score": score,
            }
            retrieved_codes.append(code_data)  # Keep the original code data

            # Embed the code content for use in the prompt
            code_embedding = self.embed_query(doc.page_content)  # Embed each code
            embedded_codes.append(code_embedding)

        # Retrieve relevant documentation
        doc_retriever = (
            self.vector_store.doc_vectorstore.as_retriever()
            if self.vector_store.doc_vectorstore
            else None
        )
        docs = await doc_retriever.aget_relevant_documents(query, k=5) if doc_retriever else []
        context = "\n".join(doc.page_content for doc in docs) if docs else ""

        state = {
            "query_embedding": query_embedding,  # Keep query_embedding in state
            "retrieved_codes": retrieved_codes,  # Keep original code data
            "embedded_codes": embedded_codes,  # Add embedded code contexts
            "context": context,
        }
        logger.debug(f"Query for state: {query}")  # Log the query
        logger.debug(f"Retrieved code snippets (first snippet): {retrieved_codes[0]['code'][:200] if retrieved_codes else 'No code retrieved'}")  # Log first retrieved code snippet
        return state

    async def train(self, num_episodes: int, batch_number: int):
        episode_rewards = []
        actor_losses_history = []
        critic_losses_history = []
        self.batch_count = batch_number  # Increment batch count at the start of training
        self.current_batch_folder = f"batch_{self.batch_count}"  # Create batch folder name
        os.makedirs(self.current_batch_folder, exist_ok=True)  # Create folder if it doesn't exist
        time.sleep(1)
        for episode in range(num_episodes):
            query = self.get_user_query()
            state = await self.get_state(query)

            action, log_prob = self.rl_algorithm.choose_action(state)
            code = await self.generate_code_from_action(action, state, query, episode + 1, episode_rewards=episode_rewards)  # generate the code from the action, pass episode number and episode_rewards
            # Save the generated code in the batch folder after generation
            if code:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                reward_value = "N/A"
                if episode_rewards:
                    reward_value = episode_rewards[-1]
                filename = f"episode_{episode + 1}_reward_{reward_value}_{timestamp}.pinescript"
                code_save_path = os.path.join(self.current_batch_folder, filename)

                try:
                    with open(code_save_path, "w") as f:
                        f.write(code)
                    logger.info(f"Generated code saved to: {code_save_path}")  # Log the file path
                except Exception as e:
                    logger.error(f"Failed to save generated code: {e}")

            success, results = await self.test_strategy_async(code)
            reward = self.reward_function(success, results)
            episode_rewards.append(reward)  # Store reward

            next_state = await self.get_state(query)

            # Convert to tensors *here* when adding to buffer
            state_tensor = torch.tensor(state["query_embedding"], dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state["query_embedding"], dtype=torch.float32)

            # ***CRITICAL FIX: Convert reward to float BEFORE adding to buffer***
            reward_tensor = torch.tensor([reward], dtype=torch.float32)  # Ensure float tensor

            self.buffer.append((state_tensor, action, reward_tensor, next_state_tensor, log_prob))  # Store float tensor

            if len(self.buffer) >= self.rl_algorithm.batch_size:
                states, actions, rewards, next_states, old_log_probs = zip(*self.buffer)

                # Convert lists of tensors to tensors
                states = torch.stack(states)
                next_states = torch.stack(next_states)
                actions = torch.tensor(actions, dtype=torch.long)
                old_log_probs = torch.stack(old_log_probs)
                rewards = torch.stack(rewards)

                # Debugging: Print shapes and types HERE
                print("Before update_policy:")
                print(f"states shape: {states.shape}")
                print(f"actions shape: {actions.shape}")
                print(f"rewards shape: {rewards.shape}")
                print(f"next_states shape: {next_states.shape}")
                print(f"old_log_probs shape: {old_log_probs.shape}")

                actor_losses, critic_losses = self.rl_algorithm.update_policy(
                    states, actions, rewards, next_states, old_log_probs, device = 'cpu'
                )
                actor_losses_history.extend(actor_losses)
                critic_losses_history.extend(critic_losses)
                self.buffer = []  # clear the buffer

                # --- Model Saving Logic (Commented out initially) ---
                # if episode % 10 == 0:  # Save every 10 episodes (adjust frequency as needed)
                #     actor_save_path = os.path.join(self.current_batch_folder, f"actor_episode_{episode}.pth")
                #     critic_save_path = os.path.join(self.current_batch_folder, f"critic_episode_{episode}.pth")
                #     torch.save(self.rl_algorithm.actor.state_dict(), actor_save_path)
                #     torch.save(self.rl_algorithm.critic.state_dict(), critic_save_path)
                #     logger.info(f"Saved actor and critic models after episode {episode} to: {self.current_batch_folder}")

        # Save plots after training loop
        reward_plot_filename = f"reward_plot_batch_{self.batch_count}_reward_{np.mean(episode_rewards):.2f}.png"  # Include batch and reward in filename
        loss_plot_filename = f"loss_plot_batch_{self.batch_count}.png"  # Include batch in filename, loss is already tracked over batches

        plot_rewards_path = os.path.join(self.current_batch_folder, reward_plot_filename)  # Save in batch folder
        plot_losses_path = os.path.join(self.current_batch_folder, loss_plot_filename)  # Save in batch folder


        plot_rewards(episode_rewards, save_path=plot_rewards_path)  # Save plots with paths
        plot_losses(actor_losses_history, critic_losses_history, save_path=plot_losses_path)


        logger.info(f"Episode {num_episodes}: Reward = {reward}") # Log last episode reward - corrected episode number

        return episode_rewards, actor_losses_history, critic_losses_history  # Return losses and rewards for plotting


    async def test_strategy_async(self, code: str) -> Tuple[bool, Any]:
        try:
            success, results = await self.strategy_tester.test_strategy_all_timeframes(code)
            return success, results
        except Exception as e:
            logger.error(f"Error during strategy testing: {e}")
            return False, str(e)

    def get_user_query(self) -> str:
        try:
            with open("queries.txt", "r") as f:
                queries = f.readlines()
                return np.random.choice(queries).strip()
        except FileNotFoundError:
            return "Create a basic Pine Script v6 strategy."

    async def generate_code(self, query: str) -> str:
        # logger.info(f"generate_code - Input query type: {type(query)}")
        state = await self.get_state(query)
        action, _ = self.rl_algorithm.choose_action(state)  # choose an action
        return await self.generate_code_from_action(action, state, query, episode_num=0)  # Pass query here, episode_num=0 for generate_code

    async def generate_code_from_action(self, action: int, state: Dict[str, Any], query: str, episode_num: int, episode_rewards: list = None) -> str:
        """Generates code using the LLM based on the given action and state."""
        retrieved_codes = state["retrieved_codes"]
        code_context = "\n".join(c["code"] for c in retrieved_codes)
        prompt_template = self.get_prompt_template_from_action(action)
        chain_input = {
            "context": state["context"],
            "code_context": code_context,
            "question": query,  # Now use the 'query' parameter passed to this function
        }
        prompt = prompt_template.format(**chain_input)
        logger.info(f"Prompt sent to LLM:\n{prompt[-50:]}")
        response = await asyncio.to_thread(self.after_rag_chain.invoke, chain_input)
        logger.info(f"LLM Response:\n{response[-100:]}...")

        code = ""
        start = response.find("```pinescript")
        if start != -1:
            start += len("```pinescript")
            end = response.find("```", start)
            code = response[start:end].strip()

        return code


    def get_prompt_template_from_action(self, action: int):
        # Define your prompt templates here, mapped to actions
        # Example:
        if action == 0:
            return ChatPromptTemplate.from_template("""
            You are an expert Pine Script v6 assistant. Generate a basic Pine Script v6 strategy.
            Documentation Context: {context}
            Previous Successful Strategies: {code_context}
            Query: {question}
            """)
        elif action == 1:
            return ChatPromptTemplate.from_template("""
            You are an expert Pine Script v6 assistant. Generate a Pine Script v6 strategy with risk management.
            Documentation Context: {context}
            Previous Successful Strategies: {code_context}
            Query: {question}
            """)
        # ... more actions and templates
        elif action == self.action_dim - 1:  # default case
            return ChatPromptTemplate.from_template("""
            You are an expert Pine Script v6 assistant. Generate a Pine Script v6 strategy.
            Documentation Context: {context}
            Previous Successful Strategies: {code_context}
            Query: {question}
            """)

    def reward_function(self, success: bool, results: Any) -> float:
        if success:
            # You can customize the reward based on results (e.g., profit, drawdown)
            return 1  # Basic reward for success
        else:
            return -1  # Penalty for failure

    # --- Model Loading Function (Commented out initially) ---
    # def load_models(self, actor_path, critic_path):
    #     """Loads actor and critic models from the specified paths."""
    #     if os.path.exists(actor_path) and os.path.exists(critic_path):
    #         try:
    #             self.rl_algorithm.actor.load_state_dict(torch.load(actor_path))
    #             self.rl_algorithm.critic.load_state_dict(torch.load(critic_path))
    #             logger.info(f"Loaded actor model from: {actor_path}")
    #             logger.info(f"Loaded critic model from: {critic_path}")
    #         except Exception as e:
    #             logger.error(f"Error loading models: {e}. Starting with randomly initialized models.", exc_info=True)
    #     else:
    #         logger.warning("Model files not found. Starting with randomly initialized models.")


# Plotting functions (outside the RLAgent class)
def plot_rewards(episode_rewards, save_path="reward_plot.png"):  # Added save_path
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to free memory - important for Gradio

def plot_losses(actor_losses, critic_losses, save_path="loss_plot.png"):  # Added save_path
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Update Step")  # or "Batch Iteration"
    plt.ylabel("Loss")
    plt.title("Actor and Critic Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to free memory


