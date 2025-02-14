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
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import LR scheduler
from selenium_handler2 import CompilationError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logger_all.logger

# --- Reorganized and Best Practices PPO Class ---
class PPOMemory:
    def __init__(self, batch_size):
        logging.info("PPOMemory.__init__ called")
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.advantages = []
        self.batch_size = batch_size

    def generate_batches(self):
        logging.info("PPOMemory.generate_batches called")
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.advantages), batches

    def store_transition(self, state, action, probs, vals, reward):
        logging.info("PPOMemory.store_transition called")
        self.states.append(state)
        self.actions.append(action)  # Corrected: Should be action, not state
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)

    def clear(self):
        logging.info("PPOMemory.clear called")
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.advantages = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):  # Increased layer size and added checkpoint directory
        logging.info("ActorNetwork.__init__ called")
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)  # Use LogSoftmax for numerical stability # Correction: Should be Softmax
        )

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')  # or 'cuda:0' if available
        self.to(self.device)

    def forward(self, state):
        logging.info("ActorNetwork.forward called")
        # Pass state through the actor network
        probs = self.actor(state)
        dist = Categorical(probs)  # CORRECTED: Wrap tensor in Categorical distribution
        return dist

    def save_checkpoint(self):
        logging.info("ActorNetwork.save_checkpoint called")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info("ActorNetwork.load_checkpoint called")
        T.load(self.state_dict(), self.checkpoint_file)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):  # Increased layer size and added checkpoint directory
        logging.info("CriticNetwork.__init__ called")
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')  # or 'cuda:0' if available
        self.to(self.device)

    def forward(self, state):
        logging.info("CriticNetwork.forward called")
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        logging.info("CriticNetwork.save_checkpoint called")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info("CriticNetwork.load_checkpoint called")
        self.load_state_dict(T.load(self.checkpoint_file))

class PPOAlgorithm:  # Refactored PPO Class - Renamed from Agent to PPOAlgorithm to avoid conflict
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=10):  # Added gae_lambda and epochs, increased batch size, decreased LR
        logging.info("PPOAlgorithm.__init__ called")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.mini_batch_size = batch_size  # Renamed to mini_batch_size for clarity
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size=self.mini_batch_size)  # Use mini_batch_size here
        self.device = T.device('cpu')  # or 'cuda:0' if available

    def remember(self, state, action, probs, vals, reward):
        logging.info("PPOAlgorithm.remember called")
        self.memory.store_transition(state, action, probs, vals, reward)

    def save_models(self):
        logging.info("PPOAlgorithm.save_models called")
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        logging.info("PPOAlgorithm.load_models called")
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, return_distribution=False):
        logging.info("PPOAlgorithm.choose_action called")
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)  # dist is now a distribution object
        value = self.critic(state)
        action = dist.sample()  # Now sample() will work as dist is Categorical
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        if return_distribution:
            return dist, value, action, probs  # return distribution as well
        return action, probs, value

    def learn(self):
        logging.info("PPOAlgorithm.learn called")
        actor_losses = []
        critic_losses = []
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, batches = self.memory.generate_batches()  # removed dones_arr
            values = vals_arr
            if not reward_arr.size:
                logger.warning("Reward array is empty, skipping learn step for this batch.")
                continue
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] - values[k])  # removed dones_arr condition
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                critic_loss = (advantage[batch] - critic_value)**2  # critic loss now on advantage
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            actor_losses.append(actor_loss.item())  # collect losses
            critic_losses.append(critic_loss.item())  # collect losses
        self.memory.clear()  # use .clear() method instead of .clear_memory()
        return actor_losses, critic_losses  # return losses

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
            batch_number: int = None  # ADD BATCH NUMBER HERE
    ):
        logging.info("RLAgent.__init__ called")
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
        self.rl_algorithm = PPOAlgorithm(action_dim, [state_dim], alpha=learning_rate, batch_size=2)  # Initialize PPO Algorithm - corrected init and input dims
        # --- Model Loading (Commented out initially) ---
        # if actor_model_path and critic_model_path:
        # self.load_models(actor_model_path, critic_model_path)
        # else:
        # logger.info("Starting with randomly initialized models.")
        self.action_dim = action_dim  # store action dim
        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2
        self.K_epochs = 80
        self.batch_size = 2
        self.episode_rewards_history = []  # List to store rewards per episode
        self.actor_losses_history = []
        self.critic_losses_history = []
        self.training_output_folder = "training_output"  # Single folder for all training outputs
        os.makedirs(self.training_output_folder, exist_ok=True)  # Ensure it exists at init
        self.batch_count = batch_number if batch_number is not None else 0

    def embed_query(self, text: str) -> List[float]:
        logging.info("RLAgent.embed_query called")
        if isinstance(text, list):
            logger.warning(f"Expected a string, but got a list: {text}")
            # Check if the list is a list of lists
            if isinstance(text[0], list):
                logger.warning(f"Found a list of lists: {text}")
            text = " ".join(
                [item for sublist in text for item in sublist] if isinstance(text[0], list) else text
            )  # Flatten list of lists

        return self.embedding_model.embed_query(text)  # Corrected embedding - removed embed_documents

    async def get_state(self, query: str) -> Dict[str, Any]:
        logging.info("RLAgent.get_state called")
        logger.info(f"get_state - START: Query: {query}")
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
        retrieved_codes = [doc.page_content for doc, _ in retrieved_codes_with_scores]
        code_context = "\n".join(retrieved_codes)
        num_retrieved_codes = len(retrieved_codes)
        state = {
            "query": query,
            "query_embedding": query_embedding,
            "code_context": code_context,
            "num_retrieved_codes": num_retrieved_codes,
        }
        logger.info(f"get_state - END: State: {state}")
        return state

    async def generate_code(self, question: str):
        logging.info("RLAgent.generate_code called")
        state = await self.get_state(question)
        action, prob, value = self.choose_action(state["query_embedding"])  # Action based on query embedding
        logger.info(f"Chosen action: {action}, Probability: {prob}, Value: {value}")
        # Retrieve context and code context
        context = await self.vector_store.get_relevant_documents(question)
        code_context = state["code_context"]
        # Prepare input for the after_rag_chain
        input_data = {"context": context, "code_context": code_context, "question": question}
        logger.info(f"generate_code - Input data before prompt chain: {input_data}")
        generated_code = await self.after_rag_chain.ainvoke(input_data)
        logger.info(f"generate_code - Generated Code: {generated_code}")
        reward = await self.evaluate_code(generated_code)  # Evaluate generated code
        logger.info(f"generate_code - Reward: {reward}")
        self.remember(state["query_embedding"], action, prob, value, reward)  # Store experience
        return generated_code

    def choose_action(self, state):
        logging.info("RLAgent.choose_action called")
        return self.rl_algorithm.choose_action(state)

    async def evaluate_code(self, code: str) -> float:
        logging.info("RLAgent.evaluate_code called")
        try:
            success, results = await self.strategy_tester.test_strategy_all_timeframes(code)
        except CompilationError as e:
            logger.error(f"Compilation Error: {e}")
            return -1.0  # Penalize compilation errors
        if success:
            average_profit = sum(result["profit"] for result in results) / len(results)
            logger.info(f"Code executed successfully. Average profit: {average_profit}")
            return average_profit  # Reward based on average profit
        else:
            logger.warning("Code failed to execute successfully.")
            return -0.5  # Penalize unsuccessful execution

    async def train_loop(self, num_batches, episodes_per_batch):
        logging.info("RLAgent.train_loop called")
        all_rewards = []
        all_actor_losses = []
        all_critic_losses = []
        start_time = time.time()  # Start time for timing the training loop
        for batch in range(num_batches):  # Loop over the number of batches
            episode_rewards = []  # Store rewards per episode for the current batch
            for episode in range(episodes_per_batch):  # Loop over the number of episodes per batch
                try:
                    # Generate a question/query for the episode (modify as needed)
                    question = f"Generate a Pine Script strategy for batch {batch + 1}, episode {episode + 1}"
                    # Generate code based on the question
                    generated_code = await self.generate_code(question)
                    # Evaluate the generated code to get the reward
                    reward = await self.evaluate_code(generated_code)
                    # Append the reward to the episode_rewards list
                    episode_rewards.append(reward)
                except Exception as e:
                    logger.error(f"Error during training episode: {e}", exc_info=True)
                    reward = -1.0  # Assign a penalty reward for the failed episode
                    episode_rewards.append(reward)
                all_rewards.extend(episode_rewards)
                # Learning step after each episode
                actor_losses, critic_losses = self.rl_algorithm.learn()  # Perform one learning step
                all_actor_losses.extend(actor_losses)
                all_critic_losses.extend(critic_losses)
                self.batch_count += 1  # Increment batch_count - correct placement
                # Print episode results
                logger.info(
                    f"Batch {batch + 1}/{num_batches}, Episode {episode + 1}/{episodes_per_batch}, "
                    f"Reward: {reward:.2f}, Actor Loss: {sum(actor_losses) / len(actor_losses):.3f}, "
                    f"Critic Loss: {sum(critic_losses) / len(critic_losses):.3f}"
                )
            self.save_training_data(all_rewards, all_actor_losses, all_critic_losses,
                                     batch)  # Save at the end of each batch
        end_time = time.time()  # End time for timing the training loop
        training_time = end_time - start_time  # Calculate the training time
        logger.info(f"Training completed in {training_time:.2f} seconds.")
        return all_rewards, all_actor_losses, all_critic_losses

    def save_training_data(self, all_rewards, all_actor_losses, all_critic_losses, batch):
        logging.info("RLAgent.save_training_data called")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for filenames
        # Prepare directory names with batch number
        batch_rewards_dir = os.path.join(self.training_output_folder, f"batch_{batch + 1}")
        batch_losses_dir = os.path.join(self.training_output_folder, f"batch_{batch + 1}")
        # Create the directories, including parents, and do not raise an error if they exist
        os.makedirs(batch_rewards_dir, exist_ok=True)
        os.makedirs(batch_losses_dir, exist_ok=True)
        # Save rewards data
        rewards_filename = os.path.join(batch_rewards_dir, f"rewards_{timestamp}.txt")
        with open(rewards_filename, "w") as f:
            for reward in all_rewards:
                f.write(str(reward) + "\n")
        logger.info(f"Rewards saved to {rewards_filename}")
        # Save actor and critic losses data
        actor_losses_filename = os.path.join(batch_losses_dir, f"actor_losses_{timestamp}.txt")
        critic_losses_filename = os.path.join(batch_losses_dir, f"critic_losses_{timestamp}.txt")
        with open(actor_losses_filename, "w") as f_actor, open(critic_losses_filename, "w") as f_critic:
            for actor_loss, critic_loss in zip(all_actor_losses, all_critic_losses):
                f_actor.write(str(actor_loss) + "\n")
                f_critic.write(str(critic_loss) + "\n")
        logger.info(f"Actor losses saved to {actor_losses_filename}")
        logger.info(f"Critic losses saved to {critic_losses_filename}")
        # Always save all rewards and losses for all batches combined
        all_rewards_filename = os.path.join(self.training_output_folder, "all_rewards.txt")
        all_actor_losses_filename = os.path.join(self.training_output_folder, "all_actor_losses.txt")
        all_critic_losses_filename = os.path.join(self.training_output_folder, "all_critic_losses.txt")
        # Append to all_rewards.txt
        with open(all_rewards_filename, "a") as f:
            for reward in all_rewards:
                f.write(str(reward) + "\n")
        # Append to all_actor_losses.txt and all_critic_losses.txt
        with open(all_actor_losses_filename, "a") as f_actor, open(all_critic_losses_filename, "a") as f_critic:
            for actor_loss, critic_loss in zip(all_actor_losses, all_critic_losses):
                f_actor.write(str(actor_loss) + "\n")
                f_critic.write(str(critic_loss) + "\n")

    def load_models(self, actor_model_path, critic_model_path):
        logging.info("RLAgent.load_models called")
        self.rl_algorithm.actor.load_state_dict(T.load(actor_model_path))
        self.rl_algorithm.critic.load_state_dict(T.load(critic_model_path))

    def save_models(self, actor_model_path, critic_model_path):
        logging.info("RLAgent.save_models called")
        T.save(self.rl_algorithm.actor.state_dict(), actor_model_path)
        T.save(self.rl_algorithm.critic.state_dict(), critic_model_path)

# --- Plotting Functions ---
def plot_rewards(episode_rewards_history, save_path="training_plots/rewards.png"):
    logging.info("plot_rewards called")
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards_history, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_losses(actor_losses_history, critic_losses_history, save_path="training_plots/losses.png"):
    logging.info("plot_losses called")
    plt.figure(figsize=(12, 6))
    plt.plot(actor_losses_history, label='Actor Loss')
    plt.plot(critic_losses_history, label='Critic Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Losses Over Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
