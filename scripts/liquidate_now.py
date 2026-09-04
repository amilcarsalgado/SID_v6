import os
import time
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, ClosePositionRequest
from alpaca.trading.enums import QueryOrderStatus

# 1. Locate and load credentials
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

trading_client = TradingClient(api_key, secret_key, paper=True)
tickers_to_close = ["BITO", "IBIT", "MSFT", "MSTR"]

print("⚡ Forcing liquidation of target positions...")

for t in tickers_to_close:
    try:
        print(f"\n🔍 Processing {t}...")

        # 2. Find all open stop-loss orders for this specific ticker
        open_orders_req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[t])
        open_orders = trading_client.get_orders(open_orders_req)

        # 3. Explicitly cancel any open orders found
        if open_orders:
            for order in open_orders:
                trading_client.cancel_order_by_id(order.id)
                print(f"   🗑️ Canceled blocking order: {order.id}")

            # 4. Wait to ensure Alpaca's servers unlock the shares
            print(f"   ⏳ Waiting 3 seconds for shares to unlock...")
            time.sleep(3)

        # 5. Liquidate the freed shares
        close_req = ClosePositionRequest(percentage="100")
        trading_client.close_position(t, close_options=close_req)
        print(f"   ✅ Successfully closed position for {t}")

    except Exception as e:
        print(f"   ❌ Could not close {t}: {e}")

print("\n🏁 Done.")