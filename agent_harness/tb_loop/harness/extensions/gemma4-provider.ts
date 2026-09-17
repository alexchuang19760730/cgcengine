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
        // 2026-09-17: 32768 -> 128000。原值来自端侧 gemma4 时代；换成云端模型后
        //   （DeepSeek-V4.1-Flash 实为 1M 上下文）harness 会以为只有 32k ⇒ **过早压缩上下文**。
        //   取 128k 而非 1M 是**保守**选择：这里声明的是「允许一次送多大」，
        //   声明成 1M 等于放开一次送超大 context ⇒ 成本与延迟都不可控。
        //   （对照：freebuff-provider.ts 的 contextWindow 本来就是 128000。）
        // ★ 以下两个是**刻意未改**的字段，不是漏改 —— 改它们属于决策而非修 bug：
        //   reasoning: false —— 而 V4.1-Flash 是深度思考模型、会回 reasoning_content。
        //     「声明无 reasoning、实际有」时 harness 的行为未验证；另外官方文档写
        //     「支持非思考与思考模式(默认)」⇒ 默认就是思考模式，每轮更慢更贵，
        //     但切到非思考的参数我还没查证，不猜。
        //   maxTokens: 4096 —— 这是**单次回复上限**，直接决定成本曲线（V4.1-Flash 最大输出 384k）。
        contextWindow: 128000,
        maxTokens: 4096,
      },
    ],
  });
}
