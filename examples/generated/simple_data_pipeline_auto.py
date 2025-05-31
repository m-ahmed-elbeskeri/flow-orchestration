# Auto-generated workflow: simple_data_pipeline
# Version: 1.0 - Generated 2025-05-30T20:03:17.105300

import asyncio
from core.agent import Agent, Context


class SimpledatapipelineWorkflow:
    def __init__(self):
        self.agent = Agent("simple_data_pipeline")
        self._setup_states()

    def _setup_states(self):
        self.agent.add_state(name="fetch_data", func=self.fetch_data)
        self.agent.add_state(name="validate_data", func=self.validate_data)
        self.agent.add_state(name="process_data", func=self.process_data)
        self.agent.add_state(name="store_data", func=self.store_data)
        self.agent.add_state(name="handle_errors", func=self.handle_errors)
        self.agent.add_state(name="notify_complete", func=self.notify_complete)

    async def fetch_data(self, context: Context):
        return "validate_data"

    async def validate_data(self, context: Context):
        return "process_data"

    async def process_data(self, context: Context):
        return "store_data"

    async def store_data(self, context: Context):
        return "notify_complete"

    async def handle_errors(self, context: Context):
        return "notify_complete"

    async def notify_complete(self, context: Context):
        print("Data pipeline completed")
        pass

    async def run(self):
        await self.agent.run()


async def main():
    workflow = SimpledatapipelineWorkflow()
    await workflow.run()


if __name__ == "__main__":
    asyncio.run(main())
