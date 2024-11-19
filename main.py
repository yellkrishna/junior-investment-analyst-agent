from agents.coordination_agent import team
import asyncio

async def main():
    # Assume 'team' has been properly initialized and configured
    result = await team.run(task="Write a financial report on American Airlines")
    print(result)

# Run the asynchronous function
asyncio.run(main())
