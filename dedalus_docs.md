import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from dedalus_labs.utils.streaming import stream_async

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    response = await runner.run(
        input="What was the score of the 2025 Wimbledon final?",
        model="openai/gpt-4o-mini"
    )

    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())


import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from dedalus_labs.utils.streaming import stream_async

load_dotenv()

async def main():
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    response = await runner.run(
        input="What was the score of the 2025 Wimbledon final?",
        model="openai/gpt-4o-mini"
    )

    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())


# Streaming

> Streaming responses with Agent system

This example demonstrates streaming agent output using the built-in streaming support with the Agent system.

<CodeGroup>
  ```python Python
  import asyncio
  from dedalus_labs import AsyncDedalus, DedalusRunner
  from dotenv import load_dotenv
  from dedalus_labs.utils.streaming import stream_async

  load_dotenv()

  async def main():
      client = AsyncDedalus()
      runner = DedalusRunner(client)

      result = runner.run(
          input="What do you think of Mulligan?",
          model="openai/gpt-4o-mini",
          stream=True
      )

      # use stream parameter and stream_async function to stream output
      await stream_async(result)

  if __name__ == "__main__":
      asyncio.run(main())
  ```

  ```typescript TypeScript
  Coming *Very* Soon
  ```
</CodeGroup>


# Streaming

> Streaming responses with Agent system

This example demonstrates streaming agent output using the built-in streaming support with the Agent system.

<CodeGroup>
  ```python Python
  import asyncio
  from dedalus_labs import AsyncDedalus, DedalusRunner
  from dotenv import load_dotenv
  from dedalus_labs.utils.streaming import stream_async

  load_dotenv()

  async def main():
      client = AsyncDedalus()
      runner = DedalusRunner(client)

      result = runner.run(
          input="What do you think of Mulligan?",
          model="openai/gpt-4o-mini",
          stream=True
      )

      # use stream parameter and stream_async function to stream output
      await stream_async(result)

  if __name__ == "__main__":
      asyncio.run(main())
  ```

  ```typescript TypeScript
  Coming *Very* Soon
  ```
</CodeGroup>


# Streaming

> Streaming responses with Agent system

This example demonstrates streaming agent output using the built-in streaming support with the Agent system.

<CodeGroup>
  ```python Python
  import asyncio
  from dedalus_labs import AsyncDedalus, DedalusRunner
  from dotenv import load_dotenv
  from dedalus_labs.utils.streaming import stream_async

  load_dotenv()

  async def main():
      client = AsyncDedalus()
      runner = DedalusRunner(client)

      result = runner.run(
          input="What do you think of Mulligan?",
          model="openai/gpt-4o-mini",
          stream=True
      )

      # use stream parameter and stream_async function to stream output
      await stream_async(result)

  if __name__ == "__main__":
      asyncio.run(main())
  ```

  ```typescript TypeScript
  Coming *Very* Soon
  ```
</CodeGroup>

# Streaming

> Streaming responses with Agent system

This example demonstrates streaming agent output using the built-in streaming support with the Agent system.

<CodeGroup>
  ```python Python
  import asyncio
  from dedalus_labs import AsyncDedalus, DedalusRunner
  from dotenv import load_dotenv
  from dedalus_labs.utils.streaming import stream_async

  load_dotenv()

  async def main():
      client = AsyncDedalus()
      runner = DedalusRunner(client)

      result = runner.run(
          input="What do you think of Mulligan?",
          model="openai/gpt-4o-mini",
          stream=True
      )

      # use stream parameter and stream_async function to stream output
      await stream_async(result)

  if __name__ == "__main__":
      asyncio.run(main())
  ```

  ```typescript TypeScript
  Coming *Very* Soon
  ```
</CodeGroup>
