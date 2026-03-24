# VertoxQuant

> Building a market making engine and all the other components that go into it
**URL:** https://www.vertoxquant.com/p/how-to-start-a-market-making-business
**Nota:** Artigo com paywall - conteúdo parcial

---

The crypto market may be the only big market where even retail can compete in market making. You have High-frequency firms that pay tons of money for colocated servers and microwave links to gain microsecond advantages in traditional markets which isn’t a thing in crypto yet. While we won’t quite reach the highest level speeds in this article, building your own market-making engine lets you be flexible and control a lot of important components: Ingesting real-time data, customizing strategies, managing risk thightly. It’s also an amazing learning experience!

The code in this article uses Python, Docker, NATS and the Bybit API so it’s best if you get familiar with them, although it’s not absolutely necessary.

Table of Content

Architecture Overview

BybitFeedAdapter - Streaming Market Data

BaseStrategy - A Flexible Strategy Interface

MidSpreadStrategy - A Simple Market Maker

OrderManagerBybit - Safe and Efficient Order Execution

Risk Management and Position Control

Back-Testing and Post-Trade-Analysis

Observability and Monitoring

Security Checklist

Deployment

Running the Market Making Engine

Final Remarks

Architecture Overview

Before drilling into code, let’s map out the overall design. Our engine follows a modular, decoupled architecture – each part has a single responsibility and communicates with others via a lightweight publish/subscribe bus (NATS). Here’s the high-level flow:

Data Flow

The Bybit Feed Adapter maintains a live WebSocket connection to the exchange’s market data.
As each order book update arrives, it publishes the update to a NATS subject (topic) named for the instrument (e.g. crypto.book.BTCUSDT ). The Strategy component subscribes to that subject, receives the update, and decides what (if any) orders to place. It then hands off desired orders to the Order Manager, which in turn calls Bybit’s REST API to create or cancel orders.
This separation means our strategy logic is completely decoupled from the exchange API specifics – it just publishes intentions and the Order Manager handles the rest.
NATS acts as the real-time bus gluing everything together: it’s extremely fast (millions of messages per second), and uses a simple pub/sub model (a publisher sends on a subject, and any subscribers on that subject get the message). By using NATS, we make it easy to plug in additional components if needed (multiple strategies, external monitors, etc.) without hard-coding connections.

Why this design?

It yields a plug-and-play feel. Want to add a new strategy? Just subscribe it to the same NATS feed and give it a separate Order Manager – no need to rewrite the feed logic. Need to switch exchanges? Implement a new feed adapter and order manager, keep the strategy interface the same. The message-bus architecture decouples modules so you can mix and match.

Let’s visualize the project structure to see how these pieces live in code:

project-root/
├── scripts/
│ └── grab_symbol_precisions.py # Creates json with qty and price precisions
├── src/
│ ├── adapters/
│ │ └── bybit_feed.py # BybitFeedAdapter: connects to Bybit WS, publishes to NATS
│ ├── strategies/
│ │ ├── strategy.py # BaseStrategy: abstract strategy interface
│ │ └── mid_spread.py # MidSpreadStrategy: a sample strategy implementation
│ ├── infra/
│ │ ├── order_manager_bybit.py # OrderManagerBybit: handles Bybit REST APIcalls
│ │ ├── bybit_signer.py # Bybit v5 signature helper
│ │ └──nats_client.py # Lightweight NATS JetStream wrapper.
│ └── engine/
│   └── main.py # Main live loop: ties everything together and runs event loop
├── docker-compose.yml # Docker Compose for deploying NATS
├── config.env.example # Environment variable definitions (API keys, etc.)
├── requirements.txt # Dependencies
└── run.ps1 # PowerShell-Skript for easy start and termination

Each module is self-contained. For example, BybitFeedAdapter knows how to connect to Bybit’s WebSocket and nothing about trading logic. The strategies know how to consume market data and generate orders, but nothing about networking or how orders are sent. This separation not only aids claritybut is essential for testability – e.g. we can swap the real feed with a simulated one in a back-test, or swap the real order manager with a stub that just prints orders.

We also include a few supporting pieces for production use. Docker Compose helps us to bring up a NATS server. The .env file holds secrets like API keys. We use asynchronous Python ( asyncio ) throughout to handle concurrent tasks (network reads, message publishing, etc.) efficiently in one process.

Now, let’s dive into each part of the system in detail.