import json
import asyncio
import websockets
import datetime
import requests

# Function to retrieve market data
def get_current_markets(max_num_events=20):       
    r = requests.get("https://gamma-api.polymarket.com/events?closed=false")
    response = r.json()

    token_ids = []
    index = 0
    for event in response:
        if index >= max_num_events:
            break
        markets = event.get("markets", [])
        for market in markets:
            clob_token_ids = market.get("clobTokenIds", [])
            clob_token_ids = json.loads(clob_token_ids)
            if len(clob_token_ids) > 0:  # Ensure it's not empty
                for token_id in clob_token_ids:
                  token_ids.append(token_id)
        index += 1
    return token_ids

# WebSocket connection and subscription logic
async def subscribe_to_market(url, token_ids, output_file):
    last_time_pong = datetime.datetime.now()

    async with websockets.connect(url) as websocket:
        # Send initial subscription message
        await websocket.send(json.dumps({"assets_ids": token_ids, "type": "market"}))
        print("Subscribed to market updates for tokens:", token_ids)

        while True:
            # Receive a message
            m = await websocket.recv()

            if m != "PONG":
                # Update last_time_pong for non-pong messages
                last_time_pong = datetime.datetime.now()

                # Parse message and save to file
                try:
                    data = json.loads(m)
                    with open(output_file, "a") as f:
                        f.write(json.dumps(data) + "\n")  # Write each JSON object as a new line
                    print(f"Saved message: {data}")
                except json.JSONDecodeError:
                    print(f"Failed to decode message: {m}")

            # Send PING every 60 seconds
            if last_time_pong + datetime.timedelta(seconds=60) < datetime.datetime.now():
                await websocket.send("PING")
                print("Sent PING")

# Main entry point
if __name__ == "__main__":
    url = 'wss://ws-subscriptions-clob.polymarket.com/ws/market'
    output_file = "polymarket_data.json"

    # Retrieve token IDs
    token_ids = get_current_markets()
    if not token_ids:
        print("No token IDs found. Exiting.")
    else:
        # Run the WebSocket subscription logic
        asyncio.run(subscribe_to_market(url, token_ids, output_file))
