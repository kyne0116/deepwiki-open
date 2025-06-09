from typing import Sequence, List
from copy import deepcopy
from tqdm import tqdm
import logging
import time
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class OllamaDocumentProcessor(DataComponent):
    """
    Process documents for Ollama embeddings by processing one document at a time.
    Adalflow Ollama Client does not support batch embedding, so we need to process each document individually.
    """
    def __init__(self, embedder: adal.Embedder, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        super().__init__()
        self.embedder = embedder
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_embedding_with_retry(self, text: str, doc_info: str) -> List[float]:
        """
        Get embedding for text with retry mechanism.

        Args:
            text: Text to embed
            doc_info: Document info for logging

        Returns:
            List[float]: Embedding vector or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Truncate text if too long (Ollama has token limits)
                if len(text) > 8000:  # Conservative limit
                    text = text[:8000]
                    logger.warning(f"Truncated text for {doc_info} to 8000 characters")

                result = self.embedder(input=text)

                if result and hasattr(result, 'data') and result.data and len(result.data) > 0:
                    embedding = result.data[0].embedding

                    # Validate embedding
                    if embedding and len(embedding) > 0:
                        logger.debug(f"Successfully generated embedding for {doc_info} (size: {len(embedding)})")
                        return embedding
                    else:
                        logger.warning(f"Empty embedding returned for {doc_info} on attempt {attempt + 1}")
                else:
                    logger.warning(f"No embedding data returned for {doc_info} on attempt {attempt + 1}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {doc_info}: {str(e)}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        logger.error(f"Failed to generate embedding for {doc_info} after {self.max_retries} attempts")
        return None

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info(f"🔄 开始处理 {len(output)} 个文档的Ollama嵌入向量")

        successful_docs = []
        failed_docs = []
        expected_embedding_size = None

        # Progress tracking
        processed_count = 0

        for i, doc in enumerate(tqdm(output, desc="生成嵌入向量")):
            processed_count += 1

            # Log progress every 50 documents
            if processed_count % 50 == 0:
                logger.info(f"📊 已处理 {processed_count}/{len(output)} 个文档，成功 {len(successful_docs)} 个")

            file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')

            # Skip empty documents
            if not doc.text or len(doc.text.strip()) == 0:
                logger.warning(f"⚠️  跳过空文档: {file_path}")
                failed_docs.append(file_path)
                continue

            # Get embedding with retry
            embedding = self._get_embedding_with_retry(doc.text, file_path)

            if embedding is None:
                failed_docs.append(file_path)
                continue

            # Validate embedding size consistency
            if expected_embedding_size is None:
                expected_embedding_size = len(embedding)
                logger.info(f"✅ 设置预期嵌入向量大小: {expected_embedding_size}")
            elif len(embedding) != expected_embedding_size:
                logger.warning(f"⚠️  文档 '{file_path}' 嵌入向量大小不一致 {len(embedding)} != {expected_embedding_size}，跳过")
                failed_docs.append(file_path)
                continue

            # Assign the embedding to the document
            output[i].vector = embedding
            successful_docs.append(output[i])

        # Log final results
        total_docs = len(output)
        success_rate = (len(successful_docs) / total_docs * 100) if total_docs > 0 else 0
        logger.info(f"✅ 嵌入向量处理完成: {len(successful_docs)}/{total_docs} 个文档成功 ({success_rate:.1f}%)")

        if failed_docs:
            logger.warning(f"⚠️  失败的文档数量: {len(failed_docs)}")
            if len(failed_docs) <= 10:  # Show first 10 failed documents
                logger.warning(f"失败的文档: {', '.join(failed_docs)}")
            else:
                logger.warning(f"失败的文档 (前10个): {', '.join(failed_docs[:10])}")

        if len(successful_docs) == 0:
            logger.error("❌ 没有任何文档成功生成嵌入向量！请检查Ollama服务和文档内容。")

        return successful_docs