import {
  generateText,
  GenerationOptions,
  GenerationResponse,
  listModels,
  ModelInfo,
  trainNGramModel,
} from "./api.ts";

const fallbackModels: ModelInfo[] = [
  {
    id: "tiny-shakespeare-demo-ngram",
    name: "Tiny Shakespeare Demo n-gram",
    architecture: "ngram",
    dataset: "tiny-shakespeare",
    artifact_path: "models/artifacts/tiny-shakespeare-demo-ngram/model.json",
    tokenizer: "word-level",
    parameters: 0,
    context_window: 128,
    max_output_tokens: 80,
    supported_options: ["max_tokens", "context_window", "top_k", "seed"],
    notes: "Hand-authored demo artifact for exercising the UI and API; not a trained model.",
  },
];

type RunResult = GenerationResponse & {
  prompt: string;
  modelName: string;
};

let models = fallbackModels;
let selectedModel = fallbackModels[0];
let history: RunResult[] = [];

const modelSelect = requireElement("model-select") as HTMLSelectElement;
const modelDetails = requireElement("model-details");
const modelNotes = requireElement("model-notes");
const promptInput = requireElement("prompt") as HTMLTextAreaElement;
const maxTokensInput = requireElement("max-tokens") as HTMLInputElement;
const contextWindowInput = requireElement("context-window") as HTMLInputElement;
const seedInput = requireElement("seed") as HTMLInputElement;
const generateButton = requireElement("generate");
const output = requireElement("output");
const metrics = requireElement("metrics");
const comparisonList = requireElement("comparison-list");
const statusText = requireElement("status");
const errorText = requireElement("error");
const corpusFileInput = requireElement("corpus-file") as HTMLInputElement;
const trainModelIdInput = requireElement("train-model-id") as HTMLInputElement;
const trainNameInput = requireElement("train-name") as HTMLInputElement;
const trainDatasetInput = requireElement("train-dataset") as HTMLInputElement;
const trainOrderInput = requireElement("train-order") as HTMLInputElement;
const trainContextWindowInput = requireElement("train-context-window") as HTMLInputElement;
const trainMaxOutputInput = requireElement("train-max-output") as HTMLInputElement;
const trainOverwriteInput = requireElement("train-overwrite") as HTMLInputElement;
const trainButton = requireElement("train-ngram");
const trainingStatus = requireElement("training-status");

generateButton.addEventListener("click", handleGenerate);
trainButton.addEventListener("click", handleTrainNGram);
corpusFileInput.addEventListener("change", handleCorpusFileChange);
modelSelect.addEventListener("change", handleModelSelect);

loadModels();
render();

function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element: ${id}`);
  }
  return element;
}

async function loadModels(preferredModelId?: string) {
  try {
    const availableModels = await listModels();
    if (availableModels.length > 0) {
      models = availableModels;
      selectedModel =
        availableModels.find((model) => model.id === preferredModelId) ??
        availableModels.find((model) => model.id === selectedModel.id) ??
        availableModels[0];
      setStatus("Ready");
      render();
    }
  } catch {
    setStatus("Using local placeholder data");
  }
}

function handleCorpusFileChange() {
  const file = corpusFileInput.files?.[0];
  if (!file) {
    return;
  }

  const baseName = file.name.replace(/\.[^.]+$/, "");
  if (!trainModelIdInput.value) {
    trainModelIdInput.value = `${slugify(baseName)}-ngram`;
  }
  if (!trainNameInput.value) {
    trainNameInput.value = `${baseName} n-gram`;
  }
  if (!trainDatasetInput.value) {
    trainDatasetInput.value = slugify(baseName);
  }
}

async function handleTrainNGram() {
  const file = corpusFileInput.files?.[0];
  if (!file) {
    setTrainingStatus("Choose a plain text corpus file first.");
    return;
  }

  const modelId = trainModelIdInput.value.trim();
  const name = trainNameInput.value.trim();
  const dataset = trainDatasetInput.value.trim();

  if (!modelId || !name || !dataset) {
    setTrainingStatus("Model ID, name, and dataset are required.");
    return;
  }

  setStatus("Training...");
  setTrainingStatus("Reading corpus file...");

  try {
    const corpusText = await file.text();
    const response = await trainNGramModel({
      model_id: modelId,
      name,
      dataset,
      corpus_text: corpusText,
      order: Number(trainOrderInput.value),
      context_window: Number(trainContextWindowInput.value),
      max_output_tokens: Number(trainMaxOutputInput.value),
      overwrite: trainOverwriteInput.checked,
    });

    await loadModels(response.model.id);
    setStatus("Ready");
    setTrainingStatus(
      `Saved ${response.model.name}: ${response.stats.tokens} tokens, ${response.stats.contexts} contexts.`,
    );
  } catch (error) {
    setStatus("Ready");
    setTrainingStatus(error instanceof Error ? error.message : "Training failed.");
  }
}

async function handleGenerate() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setError("Prompt is required.");
    return;
  }

  setStatus("Generating...");
  setError("");

  try {
    const options = readOptions();
    const response = await generateText(selectedModel.id, prompt, options);
    const result = {
      ...response,
      prompt,
      modelName: selectedModel.name,
    };

    history = [result, ...history].slice(0, 4);
    renderOutput(result);
    renderHistory();
    setStatus("Ready");
  } catch (error) {
    setStatus("Ready");
    setError(error instanceof Error ? error.message : "Generation failed.");
  }
}

function readOptions(): GenerationOptions {
  return {
    max_tokens: Number(maxTokensInput.value),
    context_window: Number(contextWindowInput.value),
    seed: Number(seedInput.value),
  };
}

function render() {
  renderModelList();
  renderModelDetails();
  renderHistory();
}

function renderModelList() {
  modelSelect.replaceChildren(
    ...models.map((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.name} (${model.architecture} / ${model.dataset})`;
      return option;
    }),
  );
  modelSelect.value = selectedModel.id;
  syncModelLimits();
}

function handleModelSelect() {
  const model = models.find((candidate) => candidate.id === modelSelect.value);
  if (!model) {
    return;
  }

  selectedModel = model;
  syncModelLimits();
  renderModelDetails();
}

function syncModelLimits() {
  maxTokensInput.max = String(selectedModel.max_output_tokens);
  contextWindowInput.max = String(selectedModel.context_window);
}

function renderModelDetails() {
  const details = [
    ["Architecture", selectedModel.architecture],
    ["Dataset", selectedModel.dataset],
    ["Tokenizer", selectedModel.tokenizer],
    ["Parameters", selectedModel.parameters.toLocaleString()],
    ["Context Window", String(selectedModel.context_window)],
    ["Max Output", String(selectedModel.max_output_tokens)],
  ];

  modelDetails.replaceChildren(
    ...details.map(([label, value]) => {
      const item = document.createElement("div");
      item.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`;
      return item;
    }),
  );
  modelNotes.textContent = selectedModel.notes;
}

function renderOutput(result: RunResult) {
  output.textContent = result.output;
  metrics.replaceChildren(
    metric(`Input: ${result.usage.input_tokens ?? 0}`),
    metric(`Output: ${result.usage.output_tokens ?? 0}`),
    metric(`Generation: ${result.timing.generation_ms ?? 0} ms`),
  );
}

function renderHistory() {
  if (history.length === 0) {
    comparisonList.innerHTML = '<p class="empty">Recent model outputs will appear here for side-by-side review.</p>';
    return;
  }

  comparisonList.replaceChildren(
    ...history.map((result) => {
      const item = document.createElement("article");
      item.className = "comparison-item";
      item.innerHTML = `
        <div>
          <strong>${result.modelName}</strong>
          <span>${result.timing.generation_ms ?? 0} ms</span>
        </div>
        <p>${result.output}</p>
      `;
      return item;
    }),
  );
}

function metric(text: string): HTMLSpanElement {
  const element = document.createElement("span");
  element.textContent = text;
  return element;
}

function setStatus(message: string) {
  statusText.textContent = message;
}

function setError(message: string) {
  errorText.textContent = message;
}

function setTrainingStatus(message: string) {
  trainingStatus.textContent = message;
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || "uploaded-corpus";
}
