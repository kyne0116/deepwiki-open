# DeepWiki 配置更改记录 - 从 Ollama 切换到 OPENROUTER

## 更改日期
$(Get-Date)

## 更改概述
已将 DeepWiki 配置从使用本地 Ollama 服务切换到使用 OPENROUTER 云服务。

## 备份文件
以下文件已创建备份：

1. **`.env.ollama.backup`** - 原始 .env 文件的备份
2. **`api/config/generator.ollama.backup.json`** - 原始生成器配置的备份
3. **`api/config/embedder.ollama.backup.json`** - 原始嵌入器配置的备份

## 配置更改详情

### 1. 环境变量更改 (.env)
- **启用 OPENROUTER API 密钥**：`OPENROUTER_API_KEY=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`
- **注释掉 Ollama 配置**：`OLLAMA_HOST` 已被注释并移到备份部分
- **更新 API 密钥说明**：将 Google 和 OpenAI 密钥标记为 dummy keys for openrouter

### 2. 生成器配置更改 (api/config/generator.json)
- **默认提供商**：从 `"google"` 更改为 `"openrouter"`
- **默认模型**：现在使用 `"openai/gpt-4o"` (OPENROUTER 的 GPT-4o)

### 3. 嵌入器配置更改 (api/config/embedder.json)
- **客户端类**：从 `"OllamaClient"` 更改为 `"OpenAIClient"`
- **模型**：从 `"nomic-embed-text"` 更改为 `"text-embedding-3-small"`
- **移除 Ollama 特定配置**：删除了 `initialize_kwargs` 中的 `host` 配置

## 如何恢复到 Ollama 配置

如果需要恢复到原来的 Ollama 配置，请执行以下步骤：

1. **恢复环境变量**：
   ```bash
   cp .env.ollama.backup .env
   ```

2. **恢复生成器配置**：
   ```bash
   cp api/config/generator.ollama.backup.json api/config/generator.json
   ```

3. **恢复嵌入器配置**：
   ```bash
   cp api/config/embedder.ollama.backup.json api/config/embedder.json
   ```

4. **确保 Ollama 服务运行**：
   ```bash
   ollama serve
   ```

## 验证配置

要验证新配置是否正常工作：

1. **启动后端服务**：
   ```bash
   python -m api.main
   ```

2. **启动前端服务**：
   ```bash
   npm run dev
   ```

3. **访问应用**：打开 http://localhost:3000

4. **测试生成**：选择一个仓库并生成 Wiki，确认使用的是 OPENROUTER 服务

## 注意事项

- OPENROUTER 是付费服务，请监控 API 使用量
- 确保网络连接正常，因为现在依赖云服务而不是本地 Ollama
- 如果遇到 API 限制，可能需要调整请求频率或升级 OPENROUTER 计划

## 支持的模型

当前配置支持以下 OPENROUTER 模型：
- openai/gpt-4o (默认)
- deepseek/deepseek-r1
- openai/gpt-4.1
- openai/o1
- openai/o3
- openai/o4-mini
- anthropic/claude-3.7-sonnet
- anthropic/claude-3.5-sonnet
