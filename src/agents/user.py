from autogen_agentchat.agents import UserProxyAgent


class UserProxy(UserProxyAgent):
    """Agent responsible for analyzing data and results."""
    def __init__(
            self,
            name: str = "UserProxy",
            **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )