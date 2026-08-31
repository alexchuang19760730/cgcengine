/**
 * gemma4 本地 OpenAI 兼容 provider（prime-agent extension）。
 *
 * 会被 tb_loop 的 prime-agent adapter 随 harness 一起注入任务容器
 * （$PRIME_AGENT_CODING_AGENT_DIR/extensions/ 自动发现）。
 * host 侧 refine 时同样生效（refine_harness.sh 设置 PRIME_AGENT_CODING_AGENT_DIR=tb_loop/harness）。
 *
 * 端点从环境变量读取，避免把密钥写进代码：
 *   TB_GEMMA4_BASE_URL  http://host.docker.internal:1234/v1
 *   TB_GEMMA4_API_KEY   sk-local
 *   TB_GEMMA4_MODEL     gemma-4-26b-a4b-it
 *
 * 用法（prime-agent 侧）：
 *   prime-agent --model local-gemma4/gemma-4-26b-a4b-it
 */

export default async function (pi: any) {
  const baseUrl = process.env.TB_GEMMA4_BASE_URL ?? "http://host.docker.internal:1234/v1";
  const apiKey = process.env.TB_GEMMA4_API_KEY ?? "sk-local";
  const modelId = process.env.TB_GEMMA4_MODEL ?? "gemma-4-26b-a4b-it";

  pi.registerProvider("local-gemma4", {
    baseUrl,
    apiKey, // 字面值（或 env 变量名）
    api: "openai-completions",
    models: [
      {
        id: modelId,
        name: modelId,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32768,
        maxTokens: 4096,
      },
    ],
  });
}
