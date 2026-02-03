import json
import os
import time

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from utils import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rating2 import build_dataset_profile
import asyncio
from logger import *
app = FastAPI()

@app.get("/api/log-stream")
def log_stream_api(since: int = 0):
    # 返回的是一个 HTTP Response，
    # 但这个 Response 的 body 是“流式的”，
    # 而不是一次性数据。
    return StreamingResponse(log_stream(since), media_type="text/event-stream")


# 语义向量相似度模型预热
model = SentenceTransformer("all-MiniLM-L6-v2")
# text1 = "气候变化如何影响东亚降水模式"
# text2 = "研究全球变暖对东亚地区降雨结构的影响"
# emb1 = model.encode([text1])
# emb2 = model.encode([text2])
# sim = cosine_similarity(emb1, emb2)[0][0]



# 允许前端访问（本地开发必备）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 示例数据（你之后可以换成数据库）
DATASETS = {
    "地球科学": [
        {
            "name": "Coupled Model Intercomparison Project",
            "desc": "耦合气候模型比较项目",
            "detail": "基于第一性原理计算，提供大规模材料结构与能带性质数据",
            "rating": {
                "openness": 8,
                "quality": 10,
                "scale": 10,
                "impact": 6,
                "safety": 10,
                "ai_readiness": 6,
                "total_score": 8.0
            }
        },
        {
            "name": "Hubble Space Telescope Data",
            "desc": "哈勃太空望远镜图像观测数据",
            "detail": "用于研究催化反应路径、吸附能与反应能垒的高质量数据集",
            "rating": {
                "openness": 8,
                "quality": 10,
                "scale": 8,
                "impact": 6,
                "safety": 10,
                "ai_readiness": 6,
                "total_score": 7.4
            }
        },
        {
            "name": "ECMWF reanalysis",
            "desc": "欧洲气象中心全球气象再分析资料",
            "detail": "包含 13 万余种小分子的量子化学性质计算结果",
            "rating": {
                "openness": 8,
                "quality": 10,
                "scale": 8,
                "impact": 0,
                "safety": 7,
                "ai_readiness": 7,
                "total_score": 5.95
            }
        },
{
            "name": "Sloan Digital Sky Survey",
            "desc": "斯隆数字天空巡天",
            "detail": "包含 13 万余种小分子的量子化学性质计算结果",
            "rating": {
                "openness": 8,
                "quality": 9,
                "scale": 8,
                "impact": 6,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 7.15
            }
        },
{
            "name": "Event Horizon Telescope Observations",
            "desc": "事件视界望远镜观测数据",
            "detail": "包含 13 万余种小分子的量子化学性质计算结果",
            "rating": {
                "openness": 10,
                "quality": 0,
                "scale": 9,
                "impact": 6,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 6.75
            }
        }
    ],

    "神经科学": [
        {
            "name": "Hemibrain Connectome Dataset",
            "desc": "果蝇半脑连接组图谱",
            "detail": "高分辨率人类脑结构与功能连接数据",
            "rating": {
                "openness": 8,
                "quality": 9,
                "scale": 7,
                "impact": 2,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 5.85
            }
        },
        {
            "name": "HMC Sleep Staging Dataset",
            "desc": "五分类睡眠分期脑电数据集",
            "detail": "提供多尺度脑区基因表达与解剖结构信息",
            "rating": {
                "openness": 8,
                "quality": 9,
                "scale": 4,
                "impact": 4,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 5.45
            }
        },
        {
            "name": "BOLD5000",
            "desc": "一个大规模、慢速事件相关的fMRI数据集",
            "detail": "基于高通道电极的神经元放电记录数据",
            "rating": {
                "openness": 9,
                "quality": 9,
                "scale": 5,
                "impact": 2,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 5.35
            }
        },
{
            "name": "RSNA Intracranial Hemorrhage Detection",
            "desc": "RSNA颅内出血检测/分类数据集",
            "detail": "基于高通道电极的神经元放电记录数据",
            "rating": {
                "openness": 0,
                "quality": 0,
                "scale": 5,
                "impact": 6,
                "safety": 10,
                "ai_readiness": 4,
                "total_score": 4.3
            }
        },
{
            "name": "APT-36K",
            "desc": "用于动物姿态估计与跟踪的大规模基准数据集",
            "detail": "基于高通道电极的神经元放电记录数据",
            "rating": {
                "openness": 8,
                "quality": 9,
                "scale": 4,
                "impact": 2,
                "safety": 7,
                "ai_readiness": 9,
                "total_score": 5.55
            }
        }
    ],

    "生命科学": [
        {
            "name": "AlphaFold Protein Structure Database",
            "desc": "AlphaFold数据库",
            "detail": "全球最大的核酸序列存储与共享数据库之一",
            "rating": {
                "openness": 8,
                "quality": 10,
                "scale": 7,
                "impact": 6,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 6.95
            }
        },
        {
            "name": "STRING — Search Tool for the Retrieval of Interacting Genes/Proteins",
            "desc": "STRING蛋白交互网络数据库",
            "detail": "提供高质量的蛋白质序列、功能与结构注释信息",
            "rating": {
                "openness": 8,
                "quality": 9,
                "scale": 5,
                "impact": 8,
                "safety": 7,
                "ai_readiness": 5,
                "total_score": 6.55
        }
        },
        {
            "name": "Universal Protein Resource",
            "desc": "蛋白质统一资源库",
            "detail": "通过单细胞测序技术构建人体细胞参考图谱",
            "rating": {
                "openness": 8,
                "quality": 10,
                "scale": 4,
                "impact": 8,
                "safety": 7,
                "ai_readiness": 6,
                "total_score": 6.55
            }
        },
        {
            "name": "InterPro",
            "desc": "综合蛋白质家族、结构域和功能位点注释数据库",
            "detail": "通过单细胞测序技术构建人体细胞参考图谱",
            "rating": {
                "openness": 0,
                "quality": 10,
                "scale": 6,
                "impact": 6,
                "safety": 7,
                "ai_readiness": 8,
                "total_score": 6.25
            }
        },
        {
            "name": "SPICE: Substituted, Polar, Intermolecular, Conformational, and Electronic dataset",
            "desc": "SPICE分子力场机器学习数据集",
            "detail": "通过单细胞测序技术构建人体细胞参考图谱",
            "rating": {
                "openness": 9,
                "quality": 9,
                "scale": 4,
                "impact": 6,
                "safety": 7,
                "ai_readiness": 8,
                "total_score": 6.45
            }
        },
    ],
}

# 启动时加载 alias 映射（只加载一次）
with open("dataset_alias_map.json", encoding="utf-8") as f:
    ALIAS_MAP = json.load(f)

def resolve_dataset_id(input_id: str) -> str:
    """
    将前端传入的 id / 搜索词 映射为规范 dataset_id
    """
    if not input_id:
        return ""

    key = input_id.strip()

    # 先尝试直接映射
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]

    # 再尝试大小写不敏感
    key_lower = key.lower()
    for alias, dataset_id in ALIAS_MAP.items():
        if alias.lower() == key_lower:
            return dataset_id

    # 如果找不到，兜底：做一次 safe 化
    return (
        key.replace(" ", "_")
           .replace("/", "_")
    )

# -------------------------
# 获取学科列表
# -------------------------
@app.get("/api/topics")
def get_topics():
    return list(DATASETS.keys())


# -------------------------
# 获取某个学科的数据集
# -------------------------
@app.get("/api/datasets")
def get_datasets(topic: str):
    return DATASETS.get(topic, [])

@app.get("/dataset", response_class=HTMLResponse)
def dataset_page():
    return ("../dataset.html").read_text(encoding="utf-8")


# 获取数据集文件内容
@app.get("/api/dataset-path")
def get_dataset_path(id: str):
    """
    根据 dataset_id 返回对应的数据集内容
    """
    # 🔑 新增：id 映射
    dataset_id = resolve_dataset_id(id)
    # 假设数据集文件名是 dataset_id_ratings.json
    safe_id = dataset_id.replace(" ", "_")
    dataset_path = os.path.join("groundtruth", f"{safe_id}_ratings1.json")
    print(f"dataset_path: {dataset_path}")
    # 检查文件是否存在
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset_content = json.load(file)  # 解析 JSON 文件内容
            print("找到数据文件")
        return JSONResponse(content=dataset_content)  # 返回文件内容作为 JSON
    else:
        print("未找到数据文件，现场搜索")
        # profile = asyncio.run(build_dataset_profile(
        #     dataset_name="Hubble Space Telescope Data",
        #     openai_api_key="sk-N2YmdRCjPQKKN01BoDjvrWW1yU8YwidaZQ9X0mkYI5QdqQRo",
        #     serper_api_key="809a347173275d7cfe4a5a6f4497ad3e38b45a0a",
        #     template_path="results/CMIP_ratings.json"  # 提前准备字段结构
        # ))
        # return JSONResponse(content=profile)  # 返回文件内容作为 JSON
        return JSONResponse(content={"error": "Dataset file not found"}, status_code=404)

# 获取数据集文件评分
@app.get("/api/dataset-score")
def get_dataset_score(id: str):
    """
    根据 dataset_id 返回对应的数据集内容
    """
    # 🔑 新增：id 映射
    dataset_id = resolve_dataset_id(id)
    # 假设数据集文件名是 dataset_id_ratings.json
    safe_id = dataset_id.replace(" ", "_").replace("/", "_")
    dataset_path = os.path.join("scores", f"{safe_id}_scores.json")
    print(f"dataset_score_path: {dataset_path}")
    # 检查文件是否存在
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset_content = json.load(file)  # 解析 JSON 文件内容
            print("找到评分文件")
        return JSONResponse(content=dataset_content)  # 返回文件内容作为 JSON
    else:
        print("未找到数据文件")
        return JSONResponse(content={"error": "Dataset file not found"}, status_code=404)

# 判断用户输入数据集是否存在
@app.get("/api/dataset-exists")
async def dataset_exists(query: str, type: str):
    """
    检查搜索词是否在 dataset_alias_map.json 中
    """
    print("query:", query)
    if not query:
        return {"type": "unknown","exists": False}

    # catalog = analyze_input_catalog(query)
    # 1️⃣ 科学数据
    if type == "dataset":
        key = query.strip()
        key_lower = key.lower()
        for alias, dataset_id in ALIAS_MAP.items():
            if alias.lower() == key_lower:
                print("在映射文件中找到数据集")
                return {
                    "type": "dataset",
                    "exists": True,
                    "dataset_id": dataset_id
                }
        print("映射文件中不存在，现场搜索")
        # search_logger.info("映射文件中不存在，现场搜索")
        profile = await build_dataset_profile(
            dataset_name=query,
            openai_api_key="sk-N2YmdRCjPQKKN01BoDjvrWW1yU8YwidaZQ9X0mkYI5QdqQRo",
            serper_api_key="809a347173275d7cfe4a5a6f4497ad3e38b45a0a",
            template_path="groundtruth/AirfRANS_ratings.json"  # 提前准备字段结构
        )
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        return {
            "type": "dataset",
            "exists": True,
            "dataset_id": query
        }
    # 2️⃣ 科学问题
    elif type == "task":
        return {
            "type": "text",
            "exists": True
        }

    return {"type": "dataset","exists": False}

# -------------------------
# 🔍 搜索与科学问题相关的数据集
# -------------------------
@app.get("/api/question_analysis")
def question_analysis(query: str):
    return analyze_user_input(query)

@app.get("/api/search_question_datasets")
def search_question_datasets(query: str):
    # time.sleep(3)
    print("query:", query)
    emb1 = model.encode([query])
    excel_path = "/Users/liuxiang/Desktop/scidata.xlsx"
    df = pd.read_excel(excel_path) if excel_path.endswith("xlsx") else pd.read_csv(excel_path)
    results = []
    for _, row in df.iterrows():
        dataset_name = row["数据名称"]  # 假设第一列是数据集名称
        file_name = f"{dataset_name.replace(' ', '_').replace('/', '-')}_ratings1.json"
        with open(f"groundtruth/{file_name}", "r", encoding="utf-8") as f:
            data = json.load(f)
            # print(data["intro"]["detailed_description"])
            # print("\n")
            emb2 = model.encode([data["intro"]["detailed_description"]])
        # sim = cosine_similarity(emb1, emb2)[0][0]
        sim = similarity_retrival(query, data["intro"]["detailed_description"])
        print(file_name, sim)  # 0.7+ 就很像
        results.append({
            "dataset_name": dataset_name,
            "similarity": float(sim)
        })
    # 6️⃣ 按相似度排序 & 取 top-k
    scores = []
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    for result in results[:3]:
        print(result["dataset_name"])
        score_file = f"{result["dataset_name"].replace(' ', '_').replace('/', '-')}_scores.json"
        with open(f"scores/{score_file}", "r", encoding="utf-8") as f:
            score = json.load(f)
            scores.append(score)
            print(score)
    print("搜索与科学问题相关的数据集")
    return {"datasets": scores}
    # return  {
    #     "datasets": [
    #         {
    #             "name": "ERA5",
    #             "desc": "Global atmospheric reanalysis dataset providing hourly atmospheric variables",
    #             "rating": {
    #                 "openness": 5,
    #                 "scale": 5,
    #                 "impact": 5,
    #                 "safety": 5,
    #                 "ai_readiness": 4,
    #                 "quality": 5,
    #                 "total_score": 4.9
    #             }
    #         },
    #         {
    #             "name": "CMIP6",
    #             "desc": "Coupled Model Intercomparison Project Phase 6 climate model outputs",
    #             "rating": {
    #                 "openness": 4,
    #                 "scale": 5,
    #                 "impact": 5,
    #                 "safety": 4,
    #                 "ai_readiness": 3,
    #                 "quality": 4,
    #                 "total_score": 4.3
    #             }
    #         },
    #         {
    #             "name": "Landsat 8",
    #             "desc": "Multispectral satellite imagery for Earth observation",
    #             "rating": {
    #                 "openness": 5,
    #                 "scale": 4,
    #                 "impact": 4,
    #                 "safety": 5,
    #                 "ai_readiness": 4,
    #                 "quality": 4,
    #                 "total_score": 4.2
    #             }
    #         }
    #     ]
    #
    # }

# -------------------------
# 🔍 搜索接口（你真正关心的）
# -------------------------
@app.get("/api/search")
def search_datasets(
    query: str = Query(..., description="搜索关键词"),
    topic: str | None = None,
):
    """
    🔥 你以后只需要改这个函数里的逻辑
    """
    results = []

    for t, datasets in DATASETS.items():
        if topic and t != topic:
            continue
        for ds in datasets:
            if query.lower() in ds["name"].lower():
                results.append(ds)

    return results
