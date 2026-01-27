import os
from common.r2 import upload_csv_file_to_r2
from dotenv import load_dotenv
load_dotenv()

async def upload_r2():
    csv_file = os.path.join(os.path.dirname(__file__), 'data', 'instacart', 'order_products__prior.csv')
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_content = f.read()
    
    await upload_csv_file_to_r2(
        bucket=os.getenv('R2_BUCKET_NAME'),
        access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
        secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('R2_ENDPOINT_URL'),
        path='instacart/order_products__prior.csv',
        text=csv_content
    )


if __name__ == "__main__":
    import asyncio
    print("Uploading instacart/order_products__prior.csv to R2...")
    asyncio.run(upload_r2())
    print("Upload complete.")