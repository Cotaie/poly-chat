export type ModelInfo = {
  id: string;
  name: string;
  architecture: string;
  dataset: string;
  artifact_path: string;
  tokenizer: string;
  parameters: number;
  context_window: number;
  max_output_tokens: number;
  supported_options: string[];
  notes: string;
};

export type GenerationOptions = {
  temperature?: number;
  max_tokens?: number;
  context_window?: number;
  top_k?: number;
  top_p?: number;
  seed?: number;
};

export type GenerationResponse = {
  model_id: string;
  output: string;
  usage: Record<string, number>;
  timing: Record<string, number>;
  metadata: Record<string, unknown>;
};

export type NGramTrainingRequest = {
  model_id: string;
  name: string;
  dataset: string;
  corpus_text: string;
  order: number;
  context_window: number;
  max_output_tokens: number;
  overwrite: boolean;
};

export type NGramTrainingResponse = {
  model: ModelInfo;
  artifact_path: string;
  registry_path: string;
  stats: Record<string, number>;
};

const API_BASE_URL = "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function listModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/api/models");
}

export function generateText(
  modelId: string,
  prompt: string,
  options: GenerationOptions,
): Promise<GenerationResponse> {
  return request<GenerationResponse>("/api/generate", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      prompt,
      options,
    }),
  });
}

export function trainNGramModel(
  requestBody: NGramTrainingRequest,
): Promise<NGramTrainingResponse> {
  return request<NGramTrainingResponse>("/api/ngram/train", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });
}
