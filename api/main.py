import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from api.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required environment variables
required_env_vars = ['GOOGLE_API_KEY', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("Some functionality may not work correctly without these variables.")

# Configure Google Generative AI
import google.generativeai as genai
from api.config import GOOGLE_API_KEY

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not configured")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8001))

    # Import the app here to ensure environment variables are set first
    from api.api import app

    logger.info(f"Starting Streaming API on port {port}")

    # Run the FastAPI app with uvicorn
    # 为了避免 watchfiles 的噪音日志，在开发环境中也禁用 reload
    # 如果需要重启服务器，请手动重启
    is_development = os.environ.get("NODE_ENV") != "production"
    enable_reload = os.environ.get("ENABLE_RELOAD", "false").lower() == "true"

    if enable_reload:
        logger.info("🔄 启用了文件监控和自动重载")
        logger.info("⚠️  这会产生 watchfiles 日志，如需禁用请设置 ENABLE_RELOAD=false")
    else:
        logger.info("🔧 已禁用文件监控以减少日志噪音")
        logger.info("💡 如需启用自动重载，请设置 ENABLE_RELOAD=true")

    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=enable_reload,
        log_level="info"
    )
