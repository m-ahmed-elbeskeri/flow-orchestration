#!/usr/bin/env python3
"""
email_processor - Workflow Runner
Process urgent emails and send notifications

Generated from: email_processor
Author: Team Workflow <team@example.com>
Version: 1.0.0
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from workflow.main import EmailProcessorWorkflow


async def main():
    """Main entry point for the workflow."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting email_processor workflow...")
    
    try:
        # Initialize and run workflow
        workflow = EmailProcessorWorkflow()
        await workflow.run()
        
        logger.info("Workflow completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())