"""
自定义 Ollama 客户端，修复嵌入向量生成问题
"""

import logging
import requests
from typing import Dict, Any, List
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput, Embedding

logger = logging.getLogger(__name__)


class CustomOllamaClient(ModelClient):
    """
    自定义 Ollama 客户端，修复嵌入向量生成问题
    """
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        """
        初始化 Ollama 客户端
        
        Args:
            host: Ollama 服务地址
            timeout: 请求超时时间
        """
        super().__init__()
        self.host = host.rstrip('/')
        self.timeout = timeout
        logger.info(f"初始化自定义 Ollama 客户端，服务地址: {self.host}")
    
    def convert_inputs_to_api_kwargs(
        self, 
        input: Any = None, 
        model_kwargs: Dict = None, 
        model_type: ModelType = None
    ) -> Dict:
        """
        转换输入为 API 参数
        
        Args:
            input: 输入文本
            model_kwargs: 模型参数
            model_type: 模型类型
            
        Returns:
            Dict: API 参数
        """
        model_kwargs = model_kwargs or {}
        
        if model_type == ModelType.EMBEDDER:
            return {
                "model": model_kwargs.get("model", "nomic-embed-text"),
                "prompt": input
            }
        elif model_type == ModelType.LLM:
            return {
                "model": model_kwargs.get("model", "qwen3:1.7b"),
                "prompt": input,
                "stream": model_kwargs.get("stream", False),
                "options": model_kwargs.get("options", {})
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def call(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """
        调用 Ollama API
        
        Args:
            api_kwargs: API 参数
            model_type: 模型类型
            
        Returns:
            API 响应
        """
        api_kwargs = api_kwargs or {}
        
        if model_type == ModelType.EMBEDDER:
            return self._call_embeddings(api_kwargs)
        elif model_type == ModelType.LLM:
            return self._call_generate(api_kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _call_embeddings(self, api_kwargs: Dict) -> EmbedderOutput:
        """
        调用嵌入向量 API
        
        Args:
            api_kwargs: API 参数
            
        Returns:
            EmbedderOutput: 嵌入向量输出
        """
        try:
            url = f"{self.host}/api/embeddings"
            
            logger.debug(f"调用嵌入向量 API: {url}")
            logger.debug(f"请求参数: {api_kwargs}")
            
            response = requests.post(
                url, 
                json=api_kwargs, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                
                if embedding:
                    logger.debug(f"成功获取嵌入向量，大小: {len(embedding)}")
                    
                    # 构造 EmbedderOutput
                    embedding_data = Embedding(
                        embedding=embedding,
                        index=0
                    )
                    
                    return EmbedderOutput(
                        data=[embedding_data],
                        raw_response=result
                    )
                else:
                    logger.error("嵌入向量为空")
                    return EmbedderOutput(
                        data=[],
                        error="嵌入向量为空",
                        raw_response=result
                    )
            else:
                error_msg = f"API 调用失败，状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                return EmbedderOutput(
                    data=[],
                    error=error_msg,
                    raw_response=response.text
                )
                
        except Exception as e:
            error_msg = f"调用嵌入向量 API 异常: {str(e)}"
            logger.error(error_msg)
            return EmbedderOutput(
                data=[],
                error=error_msg,
                raw_response=None
            )
    
    def _call_generate(self, api_kwargs: Dict) -> Any:
        """
        调用文本生成 API
        
        Args:
            api_kwargs: API 参数
            
        Returns:
            生成结果
        """
        try:
            url = f"{self.host}/api/generate"
            
            logger.debug(f"调用文本生成 API: {url}")
            logger.debug(f"请求参数: {api_kwargs}")
            
            response = requests.post(
                url, 
                json=api_kwargs, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API 调用失败，状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"调用文本生成 API 异常: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def acall(self, api_kwargs: Dict = None, model_type: ModelType = None) -> Any:
        """
        异步调用 API（暂时使用同步实现）
        
        Args:
            api_kwargs: API 参数
            model_type: 模型类型
            
        Returns:
            API 响应
        """
        # 暂时使用同步实现
        return self.call(api_kwargs, model_type)
    
    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """
        解析嵌入向量响应
        
        Args:
            response: API 响应
            
        Returns:
            EmbedderOutput: 解析后的嵌入向量输出
        """
        # 响应已经在 _call_embeddings 中处理过了
        return response
