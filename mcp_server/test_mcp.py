#!/usr/bin/env python3
"""Test script to verify MCP server functionality."""

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8080"


async def test_mcp_server():
    """Test the MCP server endpoints."""
    async with httpx.AsyncClient() as client:
        # Test 1: Initialize
        print("Testing initialize...")
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = await client.post(BASE_URL, json=init_payload)
        print(f"Initialize response: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print()

        # Test 2: Tools list
        print("Testing tools/list...")
        tools_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = await client.post(BASE_URL, json=tools_payload)
        print(f"Tools list response: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print()

        # Test 3: Prompts list
        print("Testing prompts/list...")
        prompts_payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "prompts/list",
            "params": {},
        }

        response = await client.post(BASE_URL, json=prompts_payload)
        print(f"Prompts list response: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print()

        # Test 4: Resources list
        print("Testing resources/list...")
        resources_payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list",
            "params": {},
        }

        response = await client.post(BASE_URL, json=resources_payload)
        print(f"Resources list response: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        print()

        # Test 5: Notification (should get 200 with no response body)
        print("Testing notification...")
        notification_payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}

        response = await client.post(BASE_URL, json=notification_payload)
        print(f"Notification response: {response.status_code}")
        print(f"Response body: {response.text}")
        print()

        # Test 6: Test the complete MCP handshake sequence
        print("Testing complete MCP handshake sequence...")

        # Step 1: Initialize
        init_response = await client.post(
            BASE_URL,
            json={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            },
        )
        print(f"1. Initialize: {init_response.status_code}")

        # Step 2: Send initialized notification
        notif_response = await client.post(
            BASE_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        print(f"2. Notification: {notif_response.status_code}")

        # Step 3: Now try to list tools
        tools_response = await client.post(
            BASE_URL,
            json={"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}},
        )
        print(f"3. Tools/list: {tools_response.status_code}")
        if tools_response.status_code == 200:
            print("Tools found!")
            print(json.dumps(tools_response.json(), indent=2))

        # Step 4: Try to list prompts
        prompts_response = await client.post(
            BASE_URL,
            json={"jsonrpc": "2.0", "id": 12, "method": "prompts/list", "params": {}},
        )
        print(f"4. Prompts/list: {prompts_response.status_code}")
        if prompts_response.status_code == 200:
            print("Prompts found!")
            print(json.dumps(prompts_response.json(), indent=2))


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
