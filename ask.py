"""Hỏi hệ thống Legal Multi-Agent câu hỏi của bạn.

Cách dùng:
    # Cách 1: truyền câu hỏi trực tiếp
    uv run python ask.py "Công ty trốn thuế thì bị xử lý thế nào?"

    # Cách 2: chạy không tham số -> nó sẽ hỏi bạn nhập câu hỏi
    uv run python ask.py
"""

import asyncio
import os
import sys
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")


async def ask(question: str) -> None:
    print(f"\n📨 Câu hỏi: {question}")
    print("⏳ Đang xử lý qua các agent (Customer → Law → Tax + Compliance)...\n")

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
        try:
            card_resp = await http_client.get(card_url)
            card_resp.raise_for_status()
        except Exception as e:
            print(f"❌ Không kết nối được Customer Agent tại {card_url}")
            print(f"   {e}")
            print("   → Hãy chắc chắn 5 service đang chạy (registry + 4 agent).")
            sys.exit(1)

        from a2a.client import A2AClient
        from a2a.types import (
            AgentCard,
            Message,
            MessageSendParams,
            Part,
            Role,
            SendMessageRequest,
            TextPart,
        )

        agent_card = AgentCard.model_validate(card_resp.json())
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message=Message(
                    role=Role.user,
                    parts=[Part(root=TextPart(text=question))],
                    message_id=str(uuid4()),
                )
            ),
        )

        response = await client.send_message(request)

        # Trích phần text trả về
        result_text = ""
        root = getattr(response, "root", None)
        result = getattr(root, "result", None)
        if result is not None:
            if getattr(result, "artifacts", None):
                for artifact in result.artifacts:
                    for part in artifact.parts:
                        p = getattr(part, "root", part)
                        if hasattr(p, "text"):
                            result_text += p.text
            elif getattr(result, "parts", None):
                for part in result.parts:
                    p = getattr(part, "root", part)
                    if hasattr(p, "text"):
                        result_text += p.text

        print("=" * 70)
        print(result_text or f"(Không có text. Raw: {response})")
        print("=" * 70)


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Nhập câu hỏi pháp lý của bạn: ").strip()
    if not question:
        print("Bạn chưa nhập câu hỏi.")
        sys.exit(1)
    asyncio.run(ask(question))


if __name__ == "__main__":
    main()
