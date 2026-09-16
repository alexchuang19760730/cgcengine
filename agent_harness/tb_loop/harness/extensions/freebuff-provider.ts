/**
 * freebuff2api → codebuff 云端模型 provider（prime-agent extension）。
 *
 * 用途：host 侧 /refine 蒸馏——用 codebuff（deepseek-v4-flash）从失败轨迹提炼
 * harness 经验，蒸馏质量高于端侧 gemma4 自蒸馏（refine_harness.sh 默认用它）。
 * 随 harness 一起注入容器时也自动被发现（与 gemma4-provider.ts 同机制）。
 *
 * 端点从环境变量读取（config.env 已定义）：
 *   SFT_API_BASE_URL  http://127.0.0.1:8000/v1   （Windows 本机跑 freebuff2api；
 *                            M4 上跑 refine 时填那台机器的局域网 IP，如 http://192.168.1.10:8000/v1）
 *   SFT_API_KEY       = freebuff2api .env 的 FREEBUFF_API_KEY（本地调用鉴权）
 *   SFT_MODEL         deepseek/deepseek-v4-flash
 *
 * 用法（prime-agent 侧）：
 *   prime-agent --model freebuff-codebuff/deepseek/deepseek-v4-flash
 */

export default async function (pi: any) {
  const baseUrl = process.env.SFT_API_BASE_URL ?? "http://127.0.0.1:8000/v1";
  const apiKey = process.env.SFT_API_KEY ?? "";
  const modelId = process.env.SFT_MODEL ?? "deepseek/deepseek-v4-flash";

  pi.registerProvider("freebuff-codebuff", {
    baseUrl,
    apiKey, // 字面值（或 env 变量名）
    api: "openai-completions",
    models: [
      {
        id: modelId,
        name: modelId,
        reasoning: true, // codebuff 会输出 reasoning_content
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128000,
        maxTokens: 32768,
      },
    ],
  });
}
