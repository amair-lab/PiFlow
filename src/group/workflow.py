import math
import logging
import warnings
import uuid
from typing import (
    Any,
    Dict,
    List,
    Sequence, Tuple,
)
import os
import time

from dataclasses import dataclass
import numpy as np
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_agentchat.messages import ChatMessage, ToolCallSummaryMessage, TextMessage
from autogen_core import EVENT_LOGGER_NAME
from autogen_core.models import SystemMessage, UserMessage


event_logger = logging.getLogger(EVENT_LOGGER_NAME)


@dataclass
class Hypothesis:
    content: str

@dataclass
class Experiment:
    input: str | List[float] | float
    output: float

class Principle:
    def __init__(self, hypothesis: Hypothesis, experiment: Experiment, llm_claimed_principle: str):
        self.id: str = str(uuid.uuid4())
        self.reward: float = np.inf # equal to experiment's result.

        self.hypothesis: Hypothesis = hypothesis
        self.experiment: Experiment = experiment
        self.llm_claimed_principle: str = llm_claimed_principle

    def parse_experiment_result(self) -> float:
        return self.experiment.output


class PrincipleFlow:
    def __init__(
            self,
            task: str,
            objective: str,
            is_sas: bool,
            is_mas: bool,
            is_principled: bool,
            model_client: OpenAIChatCompletionClient,
            save_dir: None,
    ):
        self.task: str = task
        self.objective: str = objective
        self.model_client = model_client

        self.flow: List[Principle] = []

        self.cached_embeddings: Dict[str, List] = {}

        self.all_evidences: List[Experiment] = []
        self.experiments: List[Dict] = []
        self.action_pathways: List[Dict] = []

        self.recent_rewards = []
        self.plateau_threshold = 0.1
        self.plateau_count_threshold = 3

        self.current_principle_index = -1
        self.current_hypothesis: str = ""
        self.current_candidate: str = ""
        self.current_result: float = 0.0

        self.processed_hypothesis_contents: set = set()
        self.processed_experiment_contents: set = set()

        self.is_sas = is_sas,
        self.is_mas = is_mas,
        self.is_principle = is_principled,

    async def llm_assign_principle(self, hypothesis: str, experiment_result: float) -> str:
        prompt = f"""
    Based on the following Rational of proposing hypothesis, extract or re-formulate a clear scientific principle grounded in physics or chemical mechanisms. 
    Focus on the causal relationship or pattern that explains the observed result. 
    The experiment result is a feedback to let you know whether the hypothesis is validated or not. 

    ### Hypothesis: \n{hypothesis}

    ### Experimental Result: \n{self.objective} = {experiment_result}

    Remember, you MUST: formulate a scientific principle with declarative sentence in custom voice, shortly and concisely (1-2 sentences) but include all rationale of hypothesizing, it is strongly recommended that using analyzing methods with (1) major premises, (2) minor premises, and using bullet points. Any other unrelated response will be strongly rejected. """

        try:
            response = await self.model_client.create([
                SystemMessage(content="You are a scientific principle extractor. You identify underlying scientific principles from hypotheses and experimental data. Formulate a scientific principle with declarative sentence in custom voice, shortly and concisely (1-2 sentences) but include all rationale of hypothesizing, it is strongly recommended that using analyzing methods with (1) major premises, (2) minor premises, and using bullet points. "),
                UserMessage(content=prompt, source="user")
            ])

            principle = response.content.strip()
            return principle
        except Exception as e:
            event_logger.warning(f"Error in LLM principle assignment: {e}")
            return "Unable to determine principle from available data."

    def _reset_curr_state(self):
        self.current_hypothesis = ""
        self.current_candidate = ""
        self.current_result = 0.0

    async def _add_principle_node(self) -> Principle:
        hypothesis_obj = Hypothesis(content=self.current_hypothesis)
        experiment_obj = Experiment(input=self.current_candidate, output=self.current_result)

        self.all_evidences.append(experiment_obj)
        principle_text = await self.llm_assign_principle(hypothesis_obj.content, experiment_obj.output)
        new_principle = Principle(
            hypothesis=hypothesis_obj,
            experiment=experiment_obj,
            llm_claimed_principle=principle_text
        )

        self.flow.append(new_principle)
        return new_principle

    def _is_current_hypo_valid_complete(self) -> bool:
        return (
                self.current_hypothesis and
                self.current_candidate and
                isinstance(self.current_result, (int, float))
        )


    async def _judge_hypothesis(self, message) -> bool:
        judgement = await self.model_client.create(
            messages=[
            SystemMessage(content="You are dealing with a text classification task. Only response with `YES` or `NO`. Other responses will be strongly rejected. Testable hypothesis means it contains specific molecule or parameters. [Meta Instruct] Only ONE word are allowed to say. "),
            UserMessage(content=f"Is the following text containing a scientific hypothesis that can be experimentally tested? Answer YES if it is, else NO. \n\n {message.content}", source="user")
        ])
        judge = judgement.content.strip()
        return "yes" in judgement.content.lower()

    async def _judge_experiment(self, message) -> bool:
        judgement = await self.model_client.create([
            SystemMessage(content="You are dealing with a scientific text classification task. Only response with `YES` or `NO`. Other responses will be strongly rejected. [Meta Instruct] Only ONE word are allowed to say. "),
            UserMessage(content=f"Is the following text an experiment result with JSON format (typically include fields such as tool_name, success, error)? Answer YES if it is, else NO. \n\n {message.content}", source="user")
        ])
        judge = judgement.content.strip()
        return "yes" in judgement.content.lower()


    def _report_to_experiment(self, experiment_dict: Dict[str, Any]) -> None:
        self.experiments.append({
            "input": experiment_dict["input"],
            "output": experiment_dict["output"],
        })

    def embed_hypothesis(self, sentence):
        from openai import OpenAI
        if sentence in self.cached_embeddings.keys():
            embedding = self.cached_embeddings[sentence]
        else:
            embedding = OpenAI(
                base_url=os.environ["PIFLOW_EMBEDDING_MODEL_URL"],
                api_key=os.environ["PIFLOW_EMBEDDING_MODEL_API_KEY"],
            ).embeddings.create(
                model=os.environ["PIFLOW_EMBEDDING_MODEL_NAME"],
                input=sentence,
                dimensions=int(os.environ["PIFLOW_EMBEDDING_MODEL_DIMENSIONS"]),
                encoding_format="float",
            ).data[0].embedding
        return embedding

    async def listen_messages(self, messages: Sequence[ChatMessage]):
        is_new_hypothesis_found = False
        is_new_experiment_found = False

        event_logger.info(f"Listening to recent new {len(messages)} messages (other agents' exploration)...")
        for message in messages:
            if message.source == "hypothesis" and isinstance(message, TextMessage):
                is_valid_hypothesis = await self._judge_hypothesis(message)
                if is_valid_hypothesis:
                    self.current_hypothesis = message.content
                    is_new_hypothesis_found = True


            elif message.source == "experiment" and isinstance(message, ToolCallSummaryMessage):
                is_valid_experiment = await self._judge_experiment(message)
                if is_valid_experiment:
                    if len(message.content.strip().split("\n")) > 1:
                        experiment_data = eval(message.content.strip().split("\n")[0])
                        warnings.warn(
                            "Multiple tools calling detected. Only the first tool call summary collected for experiments. ",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        experiment_data = eval(message.content.strip())
                    if experiment_data:
                        self.current_candidate = experiment_data["input"]
                        self.current_result = experiment_data["output"]
                        self._report_to_experiment(experiment_data)
                        is_new_experiment_found = True

        if is_new_hypothesis_found and is_new_experiment_found and self._is_current_hypo_valid_complete():
            principle = await self._add_principle_node()
            self._reset_curr_state()

        return None

    async def run_principled_reasoning(self, messages: Sequence[ChatMessage]) -> str:
        # Author comment: message here is suggested to be kept for completeness
        suggestion = await self.suggest_action()
        return suggestion



    def _extract_principles_data(self) -> List[Dict]:
        principles_data = []
        for i, principle in enumerate(self.flow):
            data = {
                "index": i,
                "principle_text": principle.llm_claimed_principle,
                "hypothesis": principle.hypothesis.content,
                "experiment_input": principle.experiment.input,
                "experiment_output": principle.experiment.output,
                "reward": principle.parse_experiment_result()
            }
            principles_data.append(data)
        return principles_data


    @staticmethod
    def _compute_reward_statistics(principles_data: List[Dict]) -> Dict:
        rewards = [p["reward"] for p in principles_data]
        if not rewards:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

        return {
            "count": len(rewards),
            "min": min(rewards),
            "max": max(rewards),
            "mean": sum(rewards) / len(rewards),
            "std": (sum((r - (sum(rewards) / len(rewards))) ** 2 for r in rewards) / len(rewards)) ** 0.5 if len(rewards) > 1 else 0
        }


    def _detect_reward_plateau(self) -> bool:
        if len(self.recent_rewards) < self.plateau_count_threshold:
            return False

        # Consider only the most recent rewards
        recent = self.recent_rewards[-self.plateau_count_threshold:]

        # Check if all consecutive differences are within the threshold
        for i in range(1, len(recent)):
            if abs(recent[i] - recent[i - 1]) > self.plateau_threshold:
                return False

        # If we got here, all differences are within threshold
        return True


    def _compute_exploration_scores(self, principles_data: List[Dict]) -> tuple[dict[int | Any, float | int | Any], None] | tuple[dict[int | Any, float | int | Any] | Any, Any]:

        exploration_scores = {}
        if len(principles_data) <= 3:
            for i in range(len(principles_data)):
                exploration_scores[i] = 1.0
            return exploration_scores, None

        embeddings = []
        for principle in principles_data:
            embedding = self.embed_hypothesis(principle["principle_text"])
            embeddings.append(embedding)

        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i, embedding_i in enumerate(embeddings):
            for j, embedding_j in enumerate(embeddings):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                    continue

                dot_product = sum(a * b for a, b in zip(embedding_i, embedding_j))
                norm_i = sum(a * a for a in embedding_i) ** 0.5
                norm_j = sum(b * b for b in embedding_j) ** 0.5

                if norm_i > 0 and norm_j > 0:
                    similarity_matrix[i][j] = dot_product / (norm_i * norm_j)
                else:
                    similarity_matrix[i][j] = 0.0

        for i in range(len(embeddings)):
            similarities = [similarity_matrix[i][j] for j in range(len(embeddings)) if i != j]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            dissimilarity = 1 - avg_similarity

            exploration_scores[i] = dissimilarity

        min_score = min(exploration_scores.values()) if exploration_scores else 0
        max_score = max(exploration_scores.values()) if exploration_scores else 1
        score_range = max_score - min_score

        if score_range > 0:
            for i in exploration_scores:
                exploration_scores[i] = (exploration_scores[i] - min_score) / score_range

        return exploration_scores, similarity_matrix.tolist()


    @staticmethod
    def _compute_exploitation_scores(principles_data: List[Dict], stats: Dict) -> Dict[int, float]:
        exploitation_scores = {}
        if not principles_data:
            return exploitation_scores

        if stats["max"] == stats["min"]:
            for i in range(len(principles_data)):
                exploitation_scores[i] = 0.5  # Neutral score when all rewards are equal
            return exploitation_scores

        for i, principle in enumerate(principles_data):
            if stats["std"] > 0:
                z_score = (principle["reward"] - stats["mean"]) / stats["std"]
            else:
                z_score = 0 if principle["reward"] == stats["mean"] else 1

            reward_sigmoid = 1 / (1 + math.exp(-z_score))
            exploitation_scores[i] = reward_sigmoid

        return exploitation_scores

    @staticmethod
    def _compute_final_scores(exploration_scores: Dict[int, float], exploitation_scores: Dict[int, float], lambda_factor) -> Dict[int, float]:
        final_scores = {}

        for i in exploration_scores.keys():
            explore_exploit_score = \
                lambda_factor * exploration_scores[i] + (1 - lambda_factor) * exploitation_scores[i]
            final_scores[i] = explore_exploit_score

        return final_scores

    @staticmethod
    def _determine_action_type(best_idx: int, best_principle: Principle, exploitation_scores: Dict[int, float]) -> Tuple[str, str]:
        best_exploitation_score = exploitation_scores[best_idx]

        # High exploitation score -> refine the principle
        if best_exploitation_score > 0.7:
            action_type = "refine"
            suggestion = (
                f"Focus on refining the principle: '{best_principle.llm_claimed_principle}'. "
                f"This principle has shown promising results with a reward of {best_principle.experiment.output}. "
                f"Consider exploring variations or extensions of this principle to further improve results."
            )
        # Medium exploitation score -> validate the principle
        elif best_exploitation_score > 0.4:
            action_type = "validate"
            suggestion = (
                f"Validate the principle: '{best_principle.llm_claimed_principle}'. "
                f"This principle shows moderate promise with a reward of {best_principle.experiment.output}. "
                f"Design experiments to confirm its reliability and identify conditions where it applies best."
            )
        # Low exploitation score -> explore new areas
        else:
            action_type = "explore"
            suggestion = (
                f"Explore alternative hypotheses that diverge from the current principle: "
                f"'{best_principle.llm_claimed_principle}'. "
                f"Current results (reward: {best_principle.experiment.output}) suggest we should "
                f"investigate different mechanisms or factors that might yield better outcomes."
            )

        return action_type, suggestion

    async def suggest_action(self) -> str:
        action_info = {
            "timestamp": time.time(),
            "iteration": len(self.action_pathways) + 1,
            "knowledge_state": {},
            "decision_process": {},
            "recommendation": {}
        }

        if not self.flow or len(self.flow) < 3:
            return "[PrincipleFlow Suggestion] Initialize one hypothesis to explore as an attempt. Diverse information is crucial for determining the selection. "


        principles_data = self._extract_principles_data()
        action_info["knowledge_state"]["principles"] = principles_data

        stats = self._compute_reward_statistics(principles_data)
        action_info["knowledge_state"]["statistics"] = stats


        exploration_scores, similarity_matrix = self._compute_exploration_scores(principles_data)
        action_info["decision_process"]["similarity_matrix"] = similarity_matrix
        action_info["decision_process"]["exploration_scores"] = exploration_scores

        exploitation_scores = self._compute_exploitation_scores(principles_data, stats)
        action_info["decision_process"]["exploitation_scores"] = exploitation_scores


        final_scores = self._compute_final_scores(
            exploration_scores,
            exploitation_scores,
            lambda_factor=0.5
        )
        action_info["decision_process"]["final_scores"] = final_scores


        # ======== DECISION MAKING ========
        # Select best principle based on comprehensive scoring
        best_idx = max(final_scores, key=final_scores.get)
        best_principle = self.flow[best_idx]

        self.recent_rewards.append(best_principle.experiment.output)

        # Check if we're stuck in a reward plateau
        is_plateau = self._detect_reward_plateau()
        action_info["decision_process"]["plateau_detected"] = is_plateau

        if is_plateau:
            # Force breakthrough exploration to escape local optimum
            action_type = "explore"
            suggestion = (
                f"We appear to be stuck in a local optimum with reward variations less than "
                f"{self.plateau_threshold} over the last {self.plateau_count_threshold} iterations. "
                f"It's time to explore **new hypothesis/principles in the scope of the task**, rather than refining the current principle or strategy. In the current system, consider investigating different mechanisms that have not been explored yet, possibly combining insights from all principles discovered so far. [Important Note: No matter how complex a mechanism you can think of, be sure to think **within the scope of the given definitions scope**. You can ONLY give solutions within the existing **definitions**, rather than fancifully discussing more unknown situations. Recall of our task: \n{self.task}]"
            )

        else:
            action_type, suggestion = self._determine_action_type(
                best_idx,
                best_principle,
                final_scores,
            )

        action_info["recommendation"] = {
            "action_type": action_type,
            "selected_principle_index": best_idx,
            "selected_principle": best_principle.llm_claimed_principle,
            "selected_principle_reward": best_principle.experiment.output,
            "plateau_detected": is_plateau,
            "suggestion": suggestion
        }

        self.action_pathways.append(action_info)
        return suggestion