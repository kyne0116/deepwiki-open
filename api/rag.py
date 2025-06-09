import logging
import re
from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
from uuid import uuid4

import adalflow as adal

from api.tools.embedder import get_embedder


# Create our own implementation of the conversation classes
@dataclass
class UserQuery:
    query_str: str

@dataclass
class AssistantResponse:
    response_str: str

@dataclass
class DialogTurn:
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse

class CustomConversation:
    """Custom implementation of Conversation to fix the list assignment index out of range error"""

    def __init__(self):
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn):
        """Safely append a dialog turn to the conversation"""
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)

# Import other adalflow components
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from api.config import configs
from api.data_pipeline import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit

class Memory(adal.core.component.DataComponent):
    """Simple conversation management with a list of dialog turns."""

    def __init__(self):
        super().__init__()
        # Use our custom implementation instead of the original Conversation class
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        """Return the conversation history as a dictionary."""
        all_dialog_turns = {}
        try:
            # Check if dialog_turns exists and is a list
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"Memory content: {len(self.current_conversation.dialog_turns)} turns")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"Added turn {i+1} with ID {turn.id} to memory")
                        else:
                            logger.warning(f"Skipping invalid turn object in memory: {turn}")
                else:
                    logger.info("Dialog turns list exists but is empty")
            else:
                logger.info("No dialog_turns attribute in current_conversation")
                # Try to initialize it
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            # Try to recover
            try:
                self.current_conversation = CustomConversation()
                logger.info("Recovered by creating new conversation")
            except Exception as e2:
                logger.error(f"Failed to recover: {str(e2)}")

        logger.info(f"Returning {len(all_dialog_turns)} dialog turns from memory")
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        Add a dialog turn to the conversation history.

        Args:
            user_query: The user's query
            assistant_response: The assistant's response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a new dialog turn using our custom implementation
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )

            # Make sure the current_conversation has the append_dialog_turn method
            if not hasattr(self.current_conversation, 'append_dialog_turn'):
                logger.warning("current_conversation does not have append_dialog_turn method, creating new one")
                # Initialize a new conversation if needed
                self.current_conversation = CustomConversation()

            # Ensure dialog_turns exists
            if not hasattr(self.current_conversation, 'dialog_turns'):
                logger.warning("dialog_turns not found, initializing empty list")
                self.current_conversation.dialog_turns = []

            # Safely append the dialog turn
            self.current_conversation.dialog_turns.append(dialog_turn)
            logger.info(f"Successfully added dialog turn, now have {len(self.current_conversation.dialog_turns)} turns")
            return True

        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            # Try to recover by creating a new conversation
            try:
                self.current_conversation = CustomConversation()
                dialog_turn = DialogTurn(
                    id=str(uuid4()),
                    user_query=UserQuery(query_str=user_query),
                    assistant_response=AssistantResponse(response_str=assistant_response),
                )
                self.current_conversation.dialog_turns.append(dialog_turn)
                logger.info("Recovered from error by creating new conversation")
                return True
            except Exception as e2:
                logger.error(f"Failed to recover from error: {str(e2)}")
                return False

system_prompt = r"""
You are a code assistant which answers user questions on a Github Repo.
You will receive user query, relevant context, and past conversation history.

LANGUAGE DETECTION AND RESPONSE:
- Detect the language of the user's query
- Respond in the SAME language as the user's query
- IMPORTANT:If a specific language is requested in the prompt, prioritize that language over the query language

FORMAT YOUR RESPONSE USING MARKDOWN:
- Use proper markdown syntax for all formatting
- For code blocks, use triple backticks with language specification (```python, ```javascript, etc.)
- Use ## headings for major sections
- Use bullet points or numbered lists where appropriate
- Format tables using markdown table syntax when presenting structured data
- Use **bold** and *italic* for emphasis
- When referencing file paths, use `inline code` formatting

IMPORTANT FORMATTING RULES:
1. DO NOT include ```markdown fences at the beginning or end of your answer
2. Start your response directly with the content
3. The content will already be rendered as markdown, so just provide the raw markdown content

Think step by step and ensure your answer is well-structured and visually organized.
"""

# Template for RAG
RAG_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
{{output_format_str}}
<END_OF_SYS_PROMPT>
{# OrderedDict of DialogTurn #}
{% if conversation_history %}
<START_OF_CONVERSATION_HISTORY>
{% for key, dialog_turn in conversation_history.items() %}
{{key}}.
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index }}.
File Path: {{context.meta_data.get('file_path', 'unknown')}}
Content: {{context.text}}
{% endfor %}
<END_OF_CONTEXT>
{% endif %}
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""

from dataclasses import dataclass, field

@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Chain of thoughts for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query, formatted in markdown for beautiful rendering with react-markdown. DO NOT include ``` triple backticks fences at the beginning or end of your answer."})

    __output_fields__ = ["rationale", "answer"]

class RAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, provider="google", model=None, use_s3: bool = False):  # noqa: F841 - use_s3 is kept for compatibility
        """
        Initialize the RAG component.

        Args:
            provider: Model provider to use (google, openai, openrouter, ollama)
            model: Model name to use with the provider
            use_s3: Whether to use S3 for database storage (default: False)
        """
        super().__init__()

        self.provider = provider
        self.model = model

        # Import the helper functions
        from api.config import get_embedder_config, is_ollama_embedder

        # Determine if we're using Ollama embedder based on configuration
        self.is_ollama_embedder = is_ollama_embedder()

        # Initialize components
        self.memory = Memory()
        self.embedder = get_embedder()

        # Patch: ensure query embedding is always single string for Ollama
        def single_string_embedder(query):
            # Accepts either a string or a list, always returns embedding for a single string
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ollama embedder only supports a single string")
                query = query[0]
            return self.embedder(input=query)

        # Use single string embedder for Ollama, regular embedder for others
        self.query_embedder = single_string_embedder if self.is_ollama_embedder else self.embedder

        self.initialize_db_manager()

        # Set up the output parser
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)

        # Format instructions to ensure proper output structure
        format_instructions = data_parser.get_output_format_str() + """

IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
4. DO NOT wrap your response in any kind of fences
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them"""

        # Get model configuration based on provider and model
        from api.config import get_model_config
        generator_config = get_model_config(self.provider, self.model)

        # Set up the main generator
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=generator_config["model_client"](),
            model_kwargs=generator_config["model_kwargs"],
            output_processors=data_parser,
        )


    def initialize_db_manager(self):
        """Initialize the database manager with local storage"""
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched embedding sizes.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning("📄 没有提供文档进行嵌入向量验证")
            return []

        logger.info(f"🔍 开始验证 {len(documents)} 个文档的嵌入向量...")

        valid_documents = []
        embedding_sizes = {}
        invalid_count = 0
        empty_count = 0

        # First pass: collect all embedding sizes and count occurrences
        for i, doc in enumerate(documents):
            file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')

            # 详细检查文档结构（仅在前几个文档中记录详细信息）
            if i < 5:  # 只记录前5个文档的详细信息
                logger.debug(f"🔍 检查文档 {i}: {file_path}")
                logger.debug(f"   文档类型: {type(doc)}")
                logger.debug(f"   有vector属性: {hasattr(doc, 'vector')}")
                if hasattr(doc, 'vector'):
                    logger.debug(f"   vector类型: {type(doc.vector)}")
                    logger.debug(f"   vector是否为None: {doc.vector is None}")
                    if doc.vector is not None:
                        try:
                            if isinstance(doc.vector, list):
                                logger.debug(f"   vector长度: {len(doc.vector)}")
                            elif hasattr(doc.vector, 'shape'):
                                logger.debug(f"   vector形状: {doc.vector.shape}")
                            elif hasattr(doc.vector, '__len__'):
                                logger.debug(f"   vector长度: {len(doc.vector)}")
                        except Exception as e:
                            logger.debug(f"   检查vector时出错: {e}")

            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"📄 文档 '{file_path}' 没有嵌入向量，跳过")
                invalid_count += 1
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(f"📄 文档 '{file_path}' 嵌入向量类型无效: {type(doc.vector)}，跳过")
                    invalid_count += 1
                    continue

                if embedding_size == 0:
                    logger.warning(f"📄 文档 '{file_path}' 嵌入向量为空，跳过")
                    empty_count += 1
                    continue

                embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

            except Exception as e:
                logger.warning(f"📄 检查文档 '{file_path}' 嵌入向量大小时出错: {str(e)}，跳过")
                invalid_count += 1
                continue

        # Log validation statistics
        total_docs = len(documents)
        valid_docs_count = sum(embedding_sizes.values())
        logger.info(f"📊 嵌入向量验证统计:")
        logger.info(f"   📄 总文档数: {total_docs}")
        logger.info(f"   ✅ 有效嵌入向量: {valid_docs_count}")
        logger.info(f"   ❌ 无效/空向量: {invalid_count}")
        logger.info(f"   🔍 空嵌入向量: {empty_count}")

        if not embedding_sizes:
            logger.error("❌ 没有找到任何有效的嵌入向量！")
            logger.error("可能的原因:")
            logger.error("1. Ollama服务连接问题")
            logger.error("2. 文档内容格式问题")
            logger.error("3. 嵌入模型配置错误")
            logger.error("4. 仓库下载或读取失败")
            logger.error("5. 文档处理过程中出现异常")

            # 提供更详细的诊断信息
            logger.error("🔍 详细诊断信息:")
            logger.error(f"   📄 总文档数: {total_docs}")
            logger.error(f"   ❌ 无效向量数: {invalid_count}")
            logger.error(f"   🔍 空向量数: {empty_count}")

            # 建议检查步骤
            logger.error("💡 建议检查步骤:")
            logger.error("1. 验证Ollama服务: curl http://localhost:11434/api/tags")
            logger.error("2. 测试嵌入功能: curl -X POST http://localhost:11434/api/embeddings -d '{\"model\":\"nomic-embed-text\",\"prompt\":\"test\"}'")
            logger.error("3. 检查仓库URL是否可访问")
            logger.error("4. 确认仓库包含有效的文档内容")

            # 🔧 临时修复：如果没有有效的嵌入向量，创建一个默认文档
            logger.warning("🔧 应用临时修复：创建默认文档以避免系统崩溃")
            try:
                from adalflow.core.types import Document

                default_doc = Document(
                    text="这是一个默认文档，用于处理嵌入向量生成失败的情况。请检查您的仓库内容和Ollama服务配置。",
                    meta_data={
                        "file_path": "default_fallback.txt",
                        "type": "txt",
                        "is_code": False,
                        "title": "默认回退文档"
                    }
                )

                # 为默认文档生成嵌入向量
                from api.tools.embedder import get_embedder
                embedder = get_embedder()
                result = embedder(input=default_doc.text)

                if result and hasattr(result, 'data') and result.data:
                    embedding = result.data[0].embedding
                    if embedding and len(embedding) > 0:
                        default_doc.vector = embedding
                        logger.warning(f"✅ 默认文档嵌入成功，向量大小: {len(embedding)}")
                        return [default_doc]

                logger.error("❌ 默认文档嵌入也失败了")

            except Exception as e:
                logger.error(f"❌ 创建默认文档时出错: {str(e)}")

            return []

        # Find the most common embedding size (this should be the correct one)
        target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
        logger.info(f"🎯 目标嵌入向量大小: {target_size} (在 {embedding_sizes[target_size]} 个文档中找到)")

        # Log all embedding sizes found
        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f"⚠️  发现 {count} 个文档的嵌入向量大小不正确 ({size})，将被过滤")

        # Second pass: filter documents with the target embedding size
        filtered_count = 0
        for i, doc in enumerate(documents):
            file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')

            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    continue

                if embedding_size == target_size:
                    valid_documents.append(doc)
                else:
                    # Log which document is being filtered out
                    logger.debug(f"🔍 过滤文档 '{file_path}': 嵌入向量大小不匹配 {embedding_size} != {target_size}")
                    filtered_count += 1

            except Exception as e:
                logger.warning(f"⚠️  验证文档 '{file_path}' 嵌入向量时出错: {str(e)}，跳过")
                filtered_count += 1
                continue

        # Final validation results
        success_rate = len(valid_documents) / total_docs * 100 if total_docs > 0 else 0
        logger.info(f"✅ 嵌入向量验证完成: {len(valid_documents)}/{total_docs} 个文档有效 ({success_rate:.1f}%)")

        if len(valid_documents) == 0:
            logger.error("❌ 过滤后没有有效的嵌入向量文档！")
            logger.error("建议检查:")
            logger.error("1. Ollama服务状态: curl http://localhost:11434/api/tags")
            logger.error("2. 嵌入模型是否正确安装: ollama list")
            logger.error("3. 文档内容是否为空或格式异常")
        elif filtered_count > 0:
            logger.warning(f"⚠️  过滤了 {filtered_count} 个文档由于嵌入向量问题")

        return valid_documents

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
        """
        Prepare the retriever for a repository.
        Will load database from local storage if available.

        Args:
            repo_url_or_path: URL or local path to the repository
            access_token: Optional access token for private repositories
            excluded_dirs: Optional list of directories to exclude from processing
            excluded_files: Optional list of file patterns to exclude from processing
            included_dirs: Optional list of directories to include exclusively
            included_files: Optional list of file patterns to include exclusively
        """
        logger.info(f"🔧 初始化数据库管理器...")
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path

        logger.info(f"📊 准备数据库和文档索引...")
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path,
            type,
            access_token,
            is_ollama_embedder=self.is_ollama_embedder,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        logger.info(f"📄 加载了 {len(self.transformed_docs)} 个文档用于检索")

        # Validate and filter embeddings to ensure consistent sizes
        logger.info(f"🔍 验证和过滤嵌入向量...")
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found. Cannot create retriever.")

        logger.info(f"✅ 使用 {len(self.transformed_docs)} 个有效嵌入文档进行检索")

        try:
            logger.info(f"🔧 创建 FAISS 检索器...")
            logger.info(f"🤖 使用嵌入器: {'Ollama' if self.is_ollama_embedder else 'OpenAI'}")
            # Use the appropriate embedder for retrieval
            retrieve_embedder = self.query_embedder if self.is_ollama_embedder else self.embedder
            self.retriever = FAISSRetriever(
                **configs["retriever"],
                embedder=retrieve_embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info("✅ FAISS 检索器创建成功")
        except Exception as e:
            logger.error(f"Error creating FAISS retriever: {str(e)}")
            # Try to provide more specific error information
            if "All embeddings should be of the same size" in str(e):
                logger.error("Embedding size validation failed. This suggests there are still inconsistent embedding sizes.")
                # Log embedding sizes for debugging
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):  # Check first 10 docs
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            if isinstance(doc.vector, list):
                                size = len(doc.vector)
                            elif hasattr(doc.vector, 'shape'):
                                size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                            elif hasattr(doc.vector, '__len__'):
                                size = len(doc.vector)
                            else:
                                size = "unknown"
                            sizes.append(f"doc_{i}: {size}")
                        except:
                            sizes.append(f"doc_{i}: error")
                logger.error(f"Sample embedding sizes: {', '.join(sizes)}")
            raise

    def call(self, query: str, language: str = "en") -> Tuple[List]:
        """
        Process a query using RAG.

        Args:
            query: The user's query

        Returns:
            Tuple of (RAGAnswer, retrieved_documents)
        """
        try:
            retrieved_documents = self.retriever(query)

            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]

            return retrieved_documents

        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")

            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer=f"I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."
            )
            return error_response, []
