import { generateText, GenerationOptions, GenerationResponse, listModels, ModelInfo } from "./api.ts";

const fallbackModels: ModelInfo[] = [
  {
    id: "tiny-shakespeare-ngram",
    name: "Tiny Shakespeare n-gram",
    architecture: "ngram",
    dataset: "tiny-shakespeare",
    artifact_path: "models/artifacts/tiny-shakespeare-ngram/model.json",
    tokenizer: "word-level",
    parameters: 0,
    context_window: 128,
    max_output_tokens: 80,
    supported_options: ["max_tokens", "context_window", "top_k", "seed"],
    notes: "Local placeholder metadata while backend wiring is in progress.",
  },
];

type RunResult = GenerationResponse & {
  prompt: string;
  modelName: string;
};

let models = fallbackModels;
let selectedModel = fallbackModels[0];
let history: RunResult[] = [];

const modelList = requireElement("model-list");
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

generateButton.addEventListener("click", handleGenerate);

loadModels();
render();

function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element: ${id}`);
  }
  return element;
}

async function loadModels() {
  try {
    const availableModels = await listModels();
    if (availableModels.length > 0) {
      models = availableModels;
      selectedModel = availableModels[0];
      setStatus("Ready");
      render();
    }
  } catch {
    setStatus("Using local placeholder data");
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
  modelList.replaceChildren(
    ...models.map((model) => {
      const button = document.createElement("button");
      button.className = model.id === selectedModel.id ? "model-button active" : "model-button";
      button.type = "button";
      button.innerHTML = `
        <strong>${model.name}</strong>
        <span>${model.architecture} / ${model.dataset}</span>
      `;
      button.addEventListener("click", () => {
        selectedModel = model;
        maxTokensInput.max = String(model.max_output_tokens);
        contextWindowInput.max = String(model.context_window);
        render();
      });
      return button;
    }),
  );
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
