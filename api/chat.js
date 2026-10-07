// Vercel serverless function: POST /api/chat
// Holds the API key and all three prompts so the browser never sees either.
//
// Env vars (Vercel -> Project -> Settings -> Environment Variables):
//   GROQ_API_KEY   required (free key from console.groq.com)
//   ROSTER_MODEL   optional, defaults to llama-3.3-70b-versatile

const MODEL = process.env.ROSTER_MODEL || "llama-3.3-70b-versatile";

/* ---------------- Stage 1: understand the text ---------------- */
const UNDERSTAND_SYSTEM = `You are step 1 of a contact-manager pipeline. You do NOT manage contacts.
Read the user's latest message (and the short chat history for pronouns like "him" or "that one") and rewrite it as clean structured intent.
The user may write in English, Urdu, Roman Urdu, or a mix, and may make typos.

Reply with ONE JSON object and nothing else:
{
  "intent": "create" | "update" | "delete" | "search" | "list" | "other",
  "instruction": "one clear English sentence describing what the user wants",
  "entities": {
    "name": string | null,        // person to add, or the existing person to edit/delete
    "phone": string | null,       // phone number exactly as the user typed it
    "new_name": string | null,    // only for update: the replacement name
    "new_phone": string | null,   // only for update: the replacement number
    "query": string | null        // only for search
  }
}
Use null for anything the user did not say. Never invent a name or number.`;

/* ---------------- Stage 2: tool calling ---------------- */
const TOOL_SYSTEM = `You are step 2 of a contact-manager pipeline: the tool-calling step.
You receive a structured intent. Call exactly ONE tool that carries it out.
Copy names and numbers exactly as given; never invent or guess values.
If required information is missing (for example a create request with no phone number), or the request is not about contacts, do not call a tool. Reply with one short plain-text sentence saying what is missing.`;

const TOOLS = [
  {
    name: "create_contact",
    description: "Add a new contact.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Full name" },
        phone: { type: "string", description: "Phone number as typed" }
      },
      required: ["name", "phone"]
    }
  },
  {
    name: "update_contact",
    description: "Change the name and/or number of an existing contact.",
    input_schema: {
      type: "object",
      properties: {
        target: { type: "string", description: "Current name (or number) of the contact to change" },
        new_name: { type: "string", description: "Replacement name, only if changing it" },
        new_phone: { type: "string", description: "Replacement number, only if changing it" }
      },
      required: ["target"]
    }
  },
  {
    name: "delete_contact",
    description: "Remove a contact.",
    input_schema: {
      type: "object",
      properties: {
        target: { type: "string", description: "Name (or number) of the contact to remove" }
      },
      required: ["target"]
    }
  },
  {
    name: "search_contacts",
    description: "Find contacts whose name or number matches a query.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"]
    }
  },
  {
    name: "list_contacts",
    description: "Show all saved contacts.",
    input_schema: {
      type: "object",
      properties: { sort: { type: "string", enum: ["recent", "name", "oldest"] } }
    }
  }
];

/* ---------------- Stage 3: user-readable reply ---------------- */
const REPLY_SYSTEM = `You are step 3 of a contact-manager pipeline: the voice the user actually hears.
You receive the user's message, the tool that was run, and its result. Write a short, friendly reply.
Rules:
- Use ONLY facts in the result. Never claim something happened if result.ok is false.
- On failure, say plainly what went wrong and what the user can do next.
- If result.ambiguous is present, list the candidates and ask which one they mean.
- For lists, show one contact per line as "Name - number". Give at most 15, then say how many more.
- If no tool ran, answer briefly from the note and gently say what you can do: add, edit, delete, search or list contacts.
- Reply in the same language the user wrote in.
- Plain text only: no markdown, no headings, no emoji. Keep it under 70 words unless listing.`;

/* ---------------- plumbing (Groq, OpenAI-compatible) ---------------- */
async function llm({ system, user, tools, max_tokens }) {
  const body = {
    model: MODEL,
    max_tokens,
    temperature: 0.2,
    messages: [{ role: "system", content: system }, { role: "user", content: user }]
  };
  if (tools) {
    body.tools = tools.map(t => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.input_schema }
    }));
    body.tool_choice = "auto";
  }
  const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer " + process.env.GROQ_API_KEY
    },
    body: JSON.stringify(body)
  });
  const data = await r.json();
  if (!r.ok) throw new Error((data.error && data.error.message) || "Groq API error " + r.status);
  return data.choices && data.choices[0] ? data.choices[0].message : {};
}

function clip(s, n) { return String(s == null ? "" : s).slice(0, n); }

async function stageUnderstand({ message, history }) {
  const hist = (Array.isArray(history) ? history : []).slice(-6)
    .map(h => (h.role === "assistant" ? "assistant" : "user") + ": " + clip(h.text, 500));
  const msg = await llm({
    system: UNDERSTAND_SYSTEM,
    max_tokens: 300,
    user: (hist.length ? "Chat history:\n" + hist.join("\n") + "\n\n" : "") + "Latest message:\n" + clip(message, 600)
  });
  const raw = String(msg.content || "").replace(/^```(?:json)?|```$/gm, "").trim();
  try { return JSON.parse(raw); }
  catch (e) { return { intent: "other", instruction: clip(message, 300), entities: {} }; }
}

async function stageTool({ intent }) {
  const msg = await llm({
    system: TOOL_SYSTEM,
    max_tokens: 300,
    tools: TOOLS,
    user: JSON.stringify(intent)
  });
  const call = msg.tool_calls && msg.tool_calls[0];
  if (call) {
    let args = {};
    try { args = JSON.parse(call.function.arguments || "{}"); } catch (e) {}
    return { tool: call.function.name, args };
  }
  return { tool: null, args: {}, note: String(msg.content || "").trim() };
}

async function stageReply({ message, tool, args, result, note }) {
  const msg = await llm({
    system: REPLY_SYSTEM,
    max_tokens: 400,
    user: JSON.stringify({ user_message: clip(message, 600), tool, args, result, note })
  });
  return { text: String(msg.content || "").trim() };
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") { res.status(405).json({ error: "POST only" }); return; }
  if (!process.env.GROQ_API_KEY) { res.status(500).json({ error: "GROQ_API_KEY is not set on the server." }); return; }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    let out;
    if (body.stage === "understand") out = await stageUnderstand(body);
    else if (body.stage === "tool") out = await stageTool(body);
    else if (body.stage === "reply") out = await stageReply(body);
    else { res.status(400).json({ error: "Unknown stage" }); return; }
    res.status(200).json(out);
  } catch (e) {
    res.status(502).json({ error: e.message || "Request failed" });
  }
};
