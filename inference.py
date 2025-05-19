import asyncio
import os
import argparse
import logging
from typing import Dict, Any, Optional, Union, Sequence

from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache

from src.agents import (
    UserProxy,
    Planner,
    HypothesisAgent,
    ExperimentAgent
)
from src.group.manage import (
    HypoValidGroupChat
)
import src.tools as tools
from src.utils.config import load_config, init_results
from src.tools.tools_registry import collect_tools
from src.utils.console import Console
from src.group.workflow import PrincipleFlow


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PriM:
    def __init__(
            self,
            args,
            task_cfg_path,
            model_cfg_path,
            is_sas,
            is_mas,
            is_principled,
            is_prompted,
            cache_dir: Optional[str] = None,
            save_dir: Optional[str] = "./",
    ) -> None:
        self.args = args

        self.model_config = load_config(model_cfg_path)
        self.task_config = load_config(task_cfg_path)

        self.cache_dir = cache_dir
        self.save_dir = save_dir

        init_results(self.save_dir, model_cfg_path, self.model_config, task_cfg_path, self.task_config)

        self.is_sas = is_sas
        self.is_mas = is_mas
        self.is_principled = is_principled
        self.is_prompted = is_prompted

        self.agents: Dict[str, Union[AssistantAgent, UserProxyAgent]] = {}
        self.principle_flow: Optional[PrincipleFlow] = None

        self.cache_storage = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(directory=cache_dir))

        for key in self.task_config.get("environment").keys():
            os.environ[key] = self.task_config.get("environment")[key]

        self.available_tools = collect_tools(
            FunctionTool,
            modules=[
                tools.characterize_pchembl_value,
                tools.characterize_nanohelix_gfactor,
                tools.characterize_Tc_value
            ]
        )

        self._set_util_client(cache_dir=self.cache_dir)
        self._create_agents()

        logger.info("Hypothesis-Validation mode enabled. ")
        def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
            if self.is_sas:
                handoff_map = {
                    "user": "planner",  # dummy planer for single agent.
                    "planner": "hypothesis",
                    "hypothesis": "experiment",
                    "experiment": "planner",
                }
            else:
                handoff_map = {
                    "user": "planner",
                    "hypothesis": "experiment",
                    "experiment": "analysis",
                    "analysis": "planner",
                    "planner": "hypothesis"
                }
            _last_speaker = messages[-1].source
            return handoff_map[_last_speaker]

        self.team = HypoValidGroupChat(
            [_ for _ in self.agents.values()],
            max_turns=self.args.max_turn,
            model_client=self.util_client,
            termination_condition=TextMentionTermination("TERMINATE"),
            selector_prompt=self.task_config.get("selector_prompt"),
            allow_repeated_speaker=True,
            selector_func=selector_func,
            note_taker_output_file=os.path.join(self.save_dir, "running_notes.json"),
        )

    @property
    def task(self):
        return self.task_config.get("task")

    def _set_util_client(self, cache_dir: Optional[str] = None) -> None:
        # Util client serve as a processing tool for analysis with language.
        openai_model_client = OpenAIChatCompletionClient(
            api_key=os.getenv("UTIL_LLM_CONFIG_API_KEY"),
            base_url=os.getenv("UTIL_LLM_CONFIG_BASE_URL"),
            model=os.getenv("UTIL_LLM_CONFIG_NAME"),
            temperature=float(os.getenv("UTIL_LLM_CONFIG_TEMPERATURE")),
            max_tokens=int(os.getenv("UTIL_LLM_CONFIG_MAX_TOKENS")),
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": ModelFamily.GPT_4,
                "structured_output": False,
            }
        )

        if cache_dir is not None:
            self.util_client: OpenAIChatCompletionClient | ChatCompletionCache = ChatCompletionCache(openai_model_client, self.cache_storage)
            logger.debug("Cached client opened for util-model. ")
        else:
            self.util_client: OpenAIChatCompletionClient | ChatCompletionCache = openai_model_client


    def _create_client(self, llm_config: Dict[str, Any], cache_dir: Optional[str] = None, model_type:str = "openai") -> OpenAIChatCompletionClient | ChatCompletionCache:
        """Create an OpenAIChatCompletionClient instance based on LLM configuration."""

        api_key = llm_config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError("API key is required for OpenAIChatCompletionClient")

        openai_client:OpenAIChatCompletionClient = OpenAIChatCompletionClient(
            api_key=api_key,
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            model=llm_config.get("model_name", "gpt-4o"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2048),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": ModelFamily.R1 if llm_config.get("is_reasoning", False) else ModelFamily.CLAUDE_3_7_SONNET,
                "structured_output": False,
            }
        )
        if cache_dir is not None:
            return ChatCompletionCache(openai_client, self.cache_storage)
        else:
            return openai_client



    def _create_agents(self) -> None:
        """Create all agents based on the configuration."""
        agent_classes = {
            "user_proxy": UserProxy,
            "hypothesis": HypothesisAgent,
            "experiment": ExperimentAgent,
            "planner": Planner,
        }
        # Iterate over agent classes to instantiate them
        for agent_name, agent_class in agent_classes.items():
            if "user_proxy" in agent_name:
                agent_config = self.model_config.get("agents", {}).get(agent_name, {})

                if not agent_config or not agent_config.get("enabled"):
                    continue

                self.agents[agent_name] = agent_class(
                    name=agent_name,
                    description="Human user",
                    input_func=None,
                )
            elif "planner" in agent_name and self.model_config.get("agents", {}).get(agent_name, {}).get("enabled", False):
                agent_config = self.model_config.get("agents", {}).get(agent_name, {})
                llm_config = agent_config.get("api_config", {})

                self.principle_flow = PrincipleFlow(
                    task=self.task_config.get("task"),
                    objective=self.task_config.get("objective_value"),
                    model_client=self._create_client(
                        llm_config=llm_config,
                        cache_dir=self.cache_dir,
                        model_type=llm_config.get("model_type", "openai")
                    ),
                    save_dir=self.save_dir,
                    is_sas=self.is_sas,                 # Dummy value here.
                    is_mas=self.is_mas,                 # Dummy value here.
                    is_principled=self.is_principled    # Only for the Planner Agent.
                )

                # Setting of the Planner.
                self.agents[agent_name] = agent_class(
                    name=agent_name,
                    description=agent_config.get("description", ""),
                    system_message=agent_config.get("system_prompt", None),
                    model_client=self._create_client(
                        llm_config=llm_config,
                        cache_dir=self.cache_dir
                    ),
                    model_client_stream=agent_config.get("streaming", False),
                    tools=[self.available_tools[_]
                           for _ in agent_config.get("tools", [])],
                    model_context=BufferedChatCompletionContext(
                        buffer_size=self.task_config.get("memory_buffer_size")
                    ),
                    flow=self.principle_flow,
                    is_sas=self.is_sas,
                    is_mas=self.is_mas,
                    is_principled=self.is_principled,
                    is_prompted=self.is_prompted,
                )

            elif self.model_config.get("agents", {}).get(agent_name, {}).get("enabled", False):
                agent_config = self.model_config.get("agents", {}).get(agent_name, {})
                llm_config = agent_config.get("api_config", {})

                self.agents[agent_name] = agent_class(
                    name=agent_name,
                    description=agent_config.get("description", ""),
                    system_message=agent_config.get("system_prompt", None),
                    model_client=self._create_client(
                        llm_config=llm_config,
                        cache_dir=self.cache_dir,
                    ),
                    model_client_stream=agent_config.get("streaming", False),
                    tools=[self.available_tools[_] for _ in agent_config.get("tools", [])],
                    model_context=BufferedChatCompletionContext(
                        buffer_size=self.task_config.get("memory_buffer_size")
                    ),
                )

        # Show the created agents.
        for name, agent in self.agents.items():
            logger.info(f"{name.capitalize()} Agent: ")
            for tool in agent._tools:
                logger.info(f"\t- tool name: {tool.name}")
                logger.info(f"\t\t* tool desc: {tool.description}")

        logger.info("Agent system prepared successfully. ")


async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="PriM: Multi-Agent System")

    # Example configuration
    parser.add_argument(
        "--task_config",
        default="configs/nanomaterial-task-QwenMax-PiFlow-mini.yaml"
    )

    # Example configuration
    parser.add_argument(
        "--model_config",
        default="configs/nanomaterial-mode-QwenMax-PiFlow-mini.yaml"
    )

    parser.add_argument("--max_turn", required=True, type=int)
    parser.add_argument("--output", default="run_demo")
    parser.add_argument("--principled", action="store_true", help="Enable principle flow. ")
    parser.add_argument("--prompted", action="store_true", help="Enable Planner's prompt, as the PiFlow is always serving as Plug-and-Play module, Planner can also direct guide the Hypothesis without reasoning/interpreting on the principle (off this), but in practice, we suggest using reasoning over suggested principle for better guidance. ")
    parser.add_argument("--sas",  action="store_true", help="Single agent only. Must be True. ")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ================== RUN ==================
    prim = PriM(
        args,
        task_cfg_path=args.task_config,
        model_cfg_path=args.model_config,
        save_dir=args.output,
        is_sas=args.sas,
        is_mas=False, # Set this False when using Hypothesis, Experiment and Analysis Agent only.
        is_principled=args.principled,
        is_prompted=args.prompted,
        cache_dir=None
    )

    stream = prim.team.run_stream(
        task=prim.task,
    )

    await Console(stream, output_stats=True)




if __name__ == "__main__":
    asyncio.run(main())