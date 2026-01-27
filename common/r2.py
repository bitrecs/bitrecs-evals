import logging
import aioboto3

logger = logging.getLogger(__name__)


def create_r2_client(bucket: str, access_key_id: str, secret_access_key: str, endpoint_url: str):
    """Factory to create an R2 client on demand."""
    session = aioboto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )
    return session.client('s3', endpoint_url=endpoint_url, region_name='auto')

async def validate_r2_bucket_connection(bucket: str, access_key_id: str, secret_access_key: str, endpoint_url: str) -> bool:
    """
    Test R2 connection by attempting to access the bucket.
    Returns True if successful, raises exception on failure.
    Useful for startup validation to catch config issues early.
    """
    try:
        async with create_r2_client(bucket, access_key_id, secret_access_key, endpoint_url) as s3_client:
            logger.info(f"Testing R2 connection to bucket: {bucket}")
            # Use head_bucket to check bucket existence and access (lightweight)
            await s3_client.head_bucket(Bucket=bucket)
            logger.info(f"R2 connection test successful for bucket: {bucket}")
            return True
    except Exception as e:
        logger.error(f"R2 connection test failed for bucket {bucket}: {e}")
        raise RuntimeError(f"R2 connection test failed: {e}") from e

async def upload_text_file_to_r2(bucket: str, access_key_id: str, secret_access_key: str, endpoint_url: str, path: str, text: str):
    try:
        async with create_r2_client(bucket, access_key_id, secret_access_key, endpoint_url) as s3_client:
            logger.info(f"Uploading text file to r2://{bucket}/{path}")
            await s3_client.put_object(
                Bucket=bucket, 
                Key=path, 
                Body=text.encode('utf-8'), 
                ContentType='text/plain'
            )
            logger.info(f"Successfully uploaded text file to r2://{bucket}/{path}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

async def upload_csv_file_to_r2(bucket: str, access_key_id: str, secret_access_key: str, endpoint_url: str, path: str, text: str):
    try:
        async with create_r2_client(bucket, access_key_id, secret_access_key, endpoint_url) as s3_client:
            logger.info(f"Uploading CSV file to r2://{bucket}/{path}")
            await s3_client.put_object(
                Bucket=bucket, 
                Key=path, 
                Body=text.encode('utf-8'), 
                ContentType='text/csv'
            )
            logger.info(f"Successfully uploaded CSV file to r2://{bucket}/{path}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


async def download_text_file_from_r2(bucket: str, access_key_id: str, secret_access_key: str, endpoint_url: str, path: str) -> str:
    try:
        async with create_r2_client(bucket, access_key_id, secret_access_key, endpoint_url) as s3_client:
            logger.info(f"Downloading text file from r2://{bucket}/{path}")
            response = await s3_client.get_object(Bucket=bucket, Key=path)
            body = await response['Body'].read()
            content = body.decode('utf-8')
            logger.info(f"Successfully downloaded text file from r2://{bucket}/{path}")
            return content
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise