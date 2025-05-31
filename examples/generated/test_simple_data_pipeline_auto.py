import pytest
import asyncio
from simple_data_pipeline_auto import SimpledatapipelineWorkflow


@pytest.mark.asyncio
async def test_successful_execution():
    workflow = SimpledatapipelineWorkflow()
    await workflow.run()
