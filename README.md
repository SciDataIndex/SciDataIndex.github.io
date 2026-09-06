# SciDataIndex

## 任务语义检索

检索使用 `Xenova/paraphrase-multilingual-MiniLM-L12-v2`（384 维）。数据集描述向量在构建阶段批量生成；用户 query 由浏览器使用同一个模型本地编码，再计算余弦相似度并展示 Top-5。整个流程不使用 API，也不需要 API Key。

### 1. 本地生成数据集向量

安装生成工具并构建索引：

```bash
npm install
npm run build:search-index
```

命令会生成并覆盖以下两个可公开提交的文件：

- `search/dataset_embeddings.json`：数据集名称、描述和索引元数据。
- `search/dataset_embeddings.f32`：归一化后的 Float32 向量。

两个生成文件都可以公开提交。数据集描述发生变化后，需要重新运行生成命令。

### 2. 浏览器查询

打开数据列表页后，第一次执行任务检索时，浏览器会从 Hugging Face 下载量化模型并缓存。后续 query 仅在浏览器中编码，不会发送到 API 或项目服务器。

### 3. 本地预览

必须通过 HTTP 服务访问，不能直接双击 HTML：

```bash
python3 -m http.server 8000
```

打开 `http://localhost:8000/dataset_list.html`。
