import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "@huggingface/transformers";

const MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const DIMENSIONS = 384;
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA_ROOT = path.join(ROOT, "data_json");
const OUTPUT_DIR = path.join(ROOT, "search");
const METADATA_OUTPUT = path.join(OUTPUT_DIR, "dataset_embeddings.json");
const VECTOR_OUTPUT = path.join(OUTPUT_DIR, "dataset_embeddings.f32");
const BATCH_SIZE = 16;

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function collectDatasets() {
  const entries = await fs.readdir(DATA_ROOT, { withFileTypes: true });
  const subjects = entries.filter((entry) => entry.isDirectory()).sort((a, b) => a.name.localeCompare(b.name));
  const datasets = [];

  for (const subject of subjects) {
    const subjectDir = path.join(DATA_ROOT, subject.name);
    const indexPath = path.join(subjectDir, "dataset_index.json");
    let files;
    try {
      files = await readJson(indexPath);
    } catch {
      continue;
    }

    for (const fileName of files) {
      const filePath = path.join(subjectDir, fileName);
      try {
        const data = await readJson(filePath);
        const name = String(data.dataset || fileName.replace(/_ratings1\.json$/i, ""));
        const brief = String(data.intro?.brief_description || "").trim();
        const detailed = String(data.intro?.detailed_description || "").trim();
        const description = brief || detailed || name;
        datasets.push({
          name,
          description,
          category: subject.name,
          fileName,
          text: `${name}。${description}`
        });
      } catch (error) {
        console.warn(`Skip unreadable dataset: ${path.relative(ROOT, filePath)} (${error.message})`);
      }
    }
  }

  return datasets;
}

const datasets = await collectDatasets();
if (!datasets.length) throw new Error("No datasets were found under data_json.");

console.log(`Loading ${MODEL}...`);
const extractor = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
const items = [];
const vectors = new Float32Array(datasets.length * DIMENSIONS);

for (let start = 0; start < datasets.length; start += BATCH_SIZE) {
  const batch = datasets.slice(start, start + BATCH_SIZE);
  const output = await extractor(batch.map((item) => item.text), {
    pooling: "mean",
    normalize: true
  });
  const batchVectors = output.tolist();
  batch.forEach((item, index) => {
    items.push({
      name: item.name,
      description: item.description,
      category: item.category,
      fileName: item.fileName
    });
    if (batchVectors[index].length !== DIMENSIONS) {
      throw new Error(`Expected ${DIMENSIONS} dimensions, received ${batchVectors[index].length}.`);
    }
    vectors.set(batchVectors[index], (start + index) * DIMENSIONS);
  });
  console.log(`Encoded ${Math.min(start + BATCH_SIZE, datasets.length)}/${datasets.length}`);
}

await fs.mkdir(OUTPUT_DIR, { recursive: true });
await fs.writeFile(METADATA_OUTPUT, JSON.stringify({
  model: MODEL,
  dimensions: DIMENSIONS,
  normalized: true,
  generatedAt: new Date().toISOString(),
  count: items.length,
  items
}, null, 2));
await fs.writeFile(VECTOR_OUTPUT, Buffer.from(vectors.buffer));

console.log(`Wrote ${items.length} embeddings to search/ (${vectors.byteLength} vector bytes).`);
