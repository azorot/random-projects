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
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import LR scheduler
from selenium_handler2 import CompilationError
logger = logger_all.logger

# --- Reorganized and Best Practices PPO Class ---
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.advantages = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.advantages), batches

    def store_transition(self, state, action, probs, vals, reward):
        self.states.append(state)
        self.actions.append(action) # Corrected: Should be action, not state
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.advantages = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'): # Increased layer size and added checkpoint directory
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1) # Use LogSoftmax for numerical stability # Correction: Should be Softmax
        )

        self.optimizer = T.optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu') # or 'cuda:0' if available
        self.to(self.device)

    def forward(self, state):
        # Pass state through the actor network
        probs = self.actor(state)
        dist = Categorical(probs) # CORRECTED: Wrap tensor in Categorical distribution
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load(self.state_dict(), self.checkpoint_file)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'): # Increased layer size and added checkpoint directory
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
        self.device = T.device('cpu') # or 'cuda:0' if available
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAlgorithm: # Refactored PPO Class - Renamed from Agent to PPOAlgorithm to avoid conflict
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=10): # Added gae_lambda and epochs, increased batch size, decreased LR
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.mini_batch_size = batch_size # Renamed to mini_batch_size for clarity

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size=self.mini_batch_size) # Use mini_batch_size here
        self.device = T.device('cpu') # or 'cuda:0' if available

    def remember(self, state, action, probs, vals, reward):
        self.memory.store_transition(state, action, probs, vals, reward)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, return_distribution=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state) # dist is now a distribution object
        value = self.critic(state)
        action = dist.sample() # Now sample() will work as dist is Categorical
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        if return_distribution:
            return dist, value, action, probs # return distribution as well
        return action, probs, value

    def learn(self):
        actor_losses = []
        critic_losses = []
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, batches = self.memory.generate_batches() # removed dones_arr

            values = vals_arr
            if not reward_arr.size:
                logger.warning("Reward array is empty, skipping learn step for this batch.")
                continue
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] - values[k]) # removed dones_arr condition
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
                critic_loss = (advantage[batch] - critic_value)**2 # critic loss now on advantage
                critic_loss = critic_loss.mean()


                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                actor_losses.append(actor_loss.item()) # collect losses
                critic_losses.append(critic_loss.item()) # collect losses

        self.memory.clear() # use .clear() method instead of .clear_memory()
        return actor_losses, critic_losses # return losses


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
        self.rl_algorithm = PPOAlgorithm(action_dim, [state_dim], alpha = learning_rate, batch_size = 2)  # Initialize PPO Algorithm - corrected init and input dims

        # --- Model Loading (Commented out initially) ---
        # if actor_model_path and critic_model_path:
        #     self.load_models(actor_model_path, critic_model_path)
        # else:
        #     logger.info("Starting with randomly initialized models.")


        self.action_dim = action_dim  # store action dim
        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2
        self.K_epochs = 80
        self.batch_size = 2
        self.episode_rewards_history = []  # List to store rewards per episode
        self.actor_losses_history = []
        self.critic_losses_history = []
        self.training_output_folder = "training_output" # Single folder for all training outputs
        os.makedirs(self.training_output_folder, exist_ok=True) # Ensure it exists at init
        self.batch_count = batch_number if batch_number is not None else 0

    def embed_query(self, text: str) -> List[float]:

        if isinstance(text, list):
            logger.warning(f"Expected a string, but got a list: {text}")

            # Check if the list is a list of lists
            if isinstance(text[0], list):
                logger.warning(f"Found a list of lists: {text}")

        return self.embedding_model.embed_query(text)  # Corrected embedding - removed embed_documents

    async def get_state(self, query: str) -> Dict[str, Any]:
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
        logger.info(f"get_state - END: Query: {query}")
        return state # return numpy array for state

    async def train_loop(self, num_batches: int, episodes_per_batch: int): # added train_loop function
        all_episode_rewards = [] # lists to store all rewards and losses across batches
        all_actor_losses = []
        all_critic_losses = []

        for batch_num in range(self.batch_count, self.batch_count + num_batches): # start from current batch count
            logger.info(f"Starting training batch {batch_num + 1}") # use batch_num + 1 for user-friendly batch numbers
            episode_rewards, actor_losses_history, critic_losses_history = await self.train(episodes_per_batch, batch_num + 1) # train for episodes_per_batch, pass batch number

            all_episode_rewards.extend(episode_rewards) # extend lists with returned data
            all_actor_losses.extend(actor_losses_history)
            all_critic_losses.extend(critic_losses_history)

            avg_reward_batch = np.mean(episode_rewards) # calculate average reward for the batch
            logger.info(f"Finished batch {batch_num + 1}, Average Reward: {avg_reward_batch:.2f}") # log average reward

            # Save models after each batch in the training_output folder
            actor_save_path = os.path.join(self.training_output_folder, f"actor_batch_{batch_num + 1}.pth") # No batch subfolders
            critic_save_path = os.path.join(self.training_output_folder, f"critic_batch_{batch_num + 1}.pth") # No batch subfolders
            T.save(self.rl_algorithm.actor.state_dict(), actor_save_path)
            T.save(self.rl_algorithm.critic.state_dict(), critic_save_path)
            logger.info(f"Saved actor and critic models after batch {batch_num + 1} to: {self.training_output_folder}") # updated log message

    # Plotting combined rewards and losses after each batch
            all_reward_plot_filename = "all_rewards_combined_latest.png" # Consistent filename for combined plot
            all_loss_plot_filename = "all_losses_combined_latest.png" # Consistent filename for combined plot
            all_policy_plot_filename = "policy_plot_latest" # Consistent filename for policy plot - directory

            all_plot_rewards_path = os.path.join(self.training_output_folder, all_reward_plot_filename) # Save combined plots in training_output folder
            all_plot_losses_path = os.path.join(self.training_output_folder, all_loss_plot_filename) # Save combined plots in training_output folder
            all_plot_policy_path = os.path.join(self.training_output_folder, all_policy_plot_filename) # Save policy plots in training_output folder
            os.makedirs(all_plot_policy_path, exist_ok=True) # Ensure policy plot folder exists

            plot_rewards(all_episode_rewards, save_path=all_plot_rewards_path, clear_plot=False) # Append to reward plot
            plot_losses(all_actor_losses, all_critic_losses, save_path=all_plot_losses_path, clear_plot=False) # Append to loss plot
            self.plot_policy(batch_num + 1, save_path=self.training_output_folder) # Plot policy after each batch, save in policy plot folder

        # After all batches, save final combined plots with a different name
        final_reward_plot_filename = f"all_rewards_combined_final_batch_{self.batch_count + num_batches}_avg_reward_{np.mean(all_episode_rewards):.2f}.png" # Final combined plot
        final_loss_plot_filename = f"all_losses_combined_final_batch_{self.batch_count + num_batches}.png" # Final combined plot

        final_plot_rewards_path = os.path.join(self.training_output_folder, final_reward_plot_filename)
        final_plot_losses_path = os.path.join(self.training_output_folder, final_loss_plot_filename)

        plot_rewards(all_episode_rewards, save_path=final_plot_rewards_path, clear_plot=False) # Save final reward plot
        plot_losses(all_actor_losses, all_critic_losses, save_path=final_plot_losses_path, clear_plot=False) # Save final loss plot


        self.batch_count += num_batches # update batch count for next training loop
        self.episode_rewards_history = all_episode_rewards # Update history with all rewards for continuous plotting in Gradio
        self.actor_losses_history = all_actor_losses # Update loss histories
        self.critic_losses_history = all_critic_losses

    async def train(self, num_episodes: int, batch_number: int): # modified train function to be called by train_loop
        episode_rewards = []
        actor_losses_history = []
        critic_losses_history = []
        # No batch-specific folder creation anymore
        time.sleep(1)
        logger.info(f"train - START: Batch {batch_number}, Episodes: {num_episodes}") # log batch number and episodes
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes} - START") # Log episode start
            query = self.get_user_query()
            logger.info(f"Episode {episode + 1}/{num_episodes} - Got user query: {query}") # Log query
            state = await self.get_state(query)
            logger.info(f"Episode {episode + 1}/{num_episodes} - Got state") # Log state retrieval

            action, probs, val = self.rl_algorithm.choose_action(state['query_embedding']) # get action, prob, val from updated PPO Agent # get action, prob, val from updated PPO Agent
            logger.info(f"Episode {episode + 1}/{num_episodes} - Chosen action: {action}") # Log action
            code = await self.generate_code_from_action(action, state, query, episode + 1, episode_rewards=episode_rewards)  # generate the code from the action, pass episode number and episode_rewards
            logger.info(f"Episode {episode + 1}/{num_episodes} - Generated code (first 100 chars): {code[:100]}...") # Log generated code
            # Save the generated code in the training_output folder
            if code:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                reward_value = "N/A"
                if episode_rewards:
                    reward_value = episode_rewards[-1]
                filename = f"episode_{episode + 1}_batch_{batch_number}_reward_{reward_value}_{timestamp}.pinescript" # Include batch number in filename
                code_save_path = os.path.join(self.training_output_folder, filename) # Save in training_output

                try:
                    with open(code_save_path, "w") as f:
                        f.write(code)
                    logger.info(f"Episode {episode + 1}/{num_episodes} - Generated code saved to: {code_save_path}")  # Log the file path
                except Exception as e:
                    logger.error(f"Episode {episode + 1}/{num_episodes} - Failed to save generated code: {e}")

            logger.info(f"Episode {episode + 1}/{num_episodes} - Testing strategy...") # Log strategy testing start
            success, results_or_errors = await self.test_strategy_async(code) # results_or_errors can be results or compilation errors
            logger.info(f"Episode {episode + 1}/{num_episodes} - Strategy tested, Success: {success}") # Log strategy testing end
            reward = self.reward_function(success, results_or_errors) # Pass errors to reward function
            episode_rewards.append(reward) 
            logger.info(f"Episode {episode + 1}/{num_episodes} - Calculated reward: {reward}")

            next_state = await self.get_state(query)
            logger.info(f"Episode {episode + 1}/{num_episodes} - Got next state") 
            self.rl_algorithm.remember(state['query_embedding'], action, probs, val, reward)
            logger.info(f"Episode {episode + 1}/{num_episodes} - Memory remembered") 

            if episode % self.batch_size == 0 and episode > 0: # update policy every batch size episodes
                logger.info(f"Episode {episode + 1}/{num_episodes} - Starting PPO Learn...") 
                actor_losses, critic_losses = self.rl_algorithm.learn() 
                actor_losses_history.extend(actor_losses)
                critic_losses_history.extend(critic_losses)
                logger.info(f"Episode {episode + 1}/{num_episodes} - PPO Learn completed. Actor Loss: {actor_losses[-1]}, Critic Loss: {critic_losses[-1]}, Policy Updated") # Log PPO learn end


            logger.info(f"Episode {episode + 1}/{num_episodes} - END, Reward: {reward}, Avg Reward: {np.mean(episode_rewards):.2f}") # Log episode end

        logger.info(f"train - END: Batch {batch_number} Episodes finished: Reward = {reward}, Avg Reward: {np.mean(episode_rewards):.2f}") # Log last episode reward and average reward - corrected episode number

        return episode_rewards, actor_losses_history, critic_losses_history  


    async def test_strategy_async(self, code: str) -> Tuple[bool, Any]:
        logger.info(f"test_strategy_async - START")
        try:
            success, results = await self.strategy_tester.test_strategy_all_timeframes(code)
            logger.info(f"test_strategy_async - Strategy test completed, Success: {success}")
            return success, results
        except CompilationError as ce: 
            logger.error(f"test_strategy_async - Compilation Error: {ce.errors}")
            return False, ce.errors 
        except Exception as e:
            logger.error(f"test_strategy_async - Error during strategy testing: {e}")
            return False, str(e)
        finally:
            logger.info(f"test_strategy_async - END")

    def get_user_query(self) -> str:
        try:
            with open("queries.txt", "r") as f:
                queries = f.readlines()
                return np.random.choice(queries).strip()
        except FileNotFoundError:
            return "Create a basic Pine Script v6 strategy."

    async def generate_code(self, query: str) -> str:
        state = await self.get_state(query)
        action, probs, val = self.rl_algorithm.choose_action(state['query_embedding'])
        return await self.generate_code_from_action(action, state, query, episode_num=0)

    async def generate_code_from_action(self, action: int, state: np.ndarray, query: str, episode_num: int, episode_rewards: list = None) -> str: # state is now numpy array
        """Generates code using the LLM based on the given action and state."""
        logger.info(f"generate_code_from_action - START, Action: {action}, Query: {query}")
        retrieved_codes = state["retrieved_codes"] 
        code_context = "\n".join(c["code"] for c in retrieved_codes)
        prompt_template = self.get_prompt_template_from_action(action)
        chain_input = {
            "context": state["context"],
            "code_context": code_context,
            "question": query,  
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

        logger.info(f"generate_code_from_action - END, Code generated (first 100 chars): {code[:100]}...")
        return code


    def get_prompt_template_from_action(self, action: int):
        # Define your prompt templates here, mapped to actions
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

    def reward_function(self, success: bool, results_or_errors: Any) -> float:
        logger.info(f"Reward Function - Success: {success}, Results/Errors Type: {type(results_or_errors)}") 
        if success:
            logger.info("Reward Function - Returning reward: 1 (Success)") 
            return 1  # Positive reward for successful strategy
        else:
            if isinstance(results_or_errors, CompilationError): # Check if results_or_errors is a CompilationError object
                logger.info("Reward Function - Compilation Errors Detected") 
                num_errors = len(results_or_errors.errors) if results_or_errors.errors else 1 # Default to 1 if errors is None or empty list
                reward = - (num_errors + 1) # Negative reward based on number of errors
                logger.info(f"Reward Function - Returning reward: {reward} (Compilation Error - {num_errors} errors)") 
                return reward 
            else:
                logger.info("Reward Function - Strategy Failure (No Compilation Error)")
                logger.info(f"Reward Function - Returning reward: -1 (Strategy Failure)") 
                return -1  

    async def plot_policy(self, episode_num, save_path=None):
            logger.info(f"plot_policy - START, Episode: {episode_num}")
            # Define a set of representative queries to generate states
            queries = [
                "Create a basic strategy.",
                "Add risk management to the strategy.",
                "Create a strategy for ranging market.",
                "Create a strategy for trending market."
            ]
            states = []
            for query in queries:
                state_dict = await(self.get_state(query)) 
                states.append(state_dict['query_embedding'])

            # Get action probabilities for each state
            action_probs_list = []
            for state in states:
                dist, _, _, _ = self.rl_algorithm.choose_action(state, return_distribution=True) 
                action_probs = dist.probs.detach().numpy() 
                action_probs_list.append(action_probs)

            # Prepare plot data
            num_states = len(queries)
            num_actions = self.action_dim
            action_indices = np.arange(num_actions)

            fig, axes = plt.subplots(num_states, 1, figsize=(10, 5 * num_states)) 

            for i, query in enumerate(queries):
                ax = axes[i] 
                ax.bar(action_indices, action_probs_list[i][0]) 
                ax.set_title(f"Policy for Query: '{query}'")
                ax.set_ylabel("Probability")
                ax.set_xticks(action_indices)
                ax.set_xticklabels([f"Action {i}" for i in action_indices]) 

            plt.tight_layout()
            if save_path:
                full_save_path = os.path.join(save_path, f"policy_plot_episode_{episode_num}.png") 
                plt.savefig(full_save_path)
                logger.info(f"Policy plot saved to: {full_save_path}")
            else:
                plt.show() 
            plt.close()

            logger.info(f"plot_policy - END, Episode: {episode_num}")
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



def plot_rewards(episode_rewards, save_path="reward_plot.png", clear_plot=True):  # Added clear_plot argument
    if clear_plot:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(num="Reward Plot", figsize=(10, 5), clear=False) \

    plt.plot(episode_rewards, label='Episode Reward') 
    if not clear_plot:
        plt.plot(np.cumsum(episode_rewards), label='Cumulative Reward') 

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode and Cumulative Reward") 
    plt.grid(True)
    plt.legend() 
    plt.savefig(save_path)  
    plt.close() if clear_plot else None


def plot_losses(actor_losses, critic_losses, save_path="loss_plot.png", clear_plot=True): 
    if clear_plot:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(num="Loss Plot", figsize=(10, 5), clear=False)

    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Update Step") 
    plt.ylabel("Loss")
    plt.title("Actor and Critic Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  
    plt.close() if clear_plot else None 