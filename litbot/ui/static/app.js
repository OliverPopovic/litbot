const state = {
  currentView: "chat",
  lastResponse: null,
  messages: [],
  activeNoteId: null,
  retrievedNoteIds: [],
};

const els = {
  root: document.documentElement,
  form: document.querySelector("#chatForm"),
  messages: document.querySelector("#messages"),
  question: document.querySelector("#question"),
  workFilter: document.querySelector("#workFilter"),
  topK: document.querySelector("#topK"),
  status: document.querySelector("#status"),
  sendButton: document.querySelector("#sendButton"),
  developerPanel: document.querySelector("#developerPanel"),
  developerEmpty: document.querySelector("#developerEmpty"),
  inspector: document.querySelector("#inspector"),
  themeToggle: document.querySelector("#themeToggle"),
  viewButtons: document.querySelectorAll("[data-view]"),
};

const savedTheme = localStorage.getItem("litbot-theme");
if (savedTheme) {
  els.root.dataset.theme = savedTheme;
}

els.themeToggle.addEventListener("click", () => {
  const nextTheme = els.root.dataset.theme === "dark" ? "light" : "dark";
  els.root.dataset.theme = nextTheme;
  localStorage.setItem("litbot-theme", nextTheme);
});

els.viewButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.currentView = button.dataset.view;
    updateView();
  });
});

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (!question) {
    setStatus("Enter a message before sending.", true);
    els.question.focus();
    return;
  }

  const request = buildRequest(question);
  appendMessage("user", question);
  els.question.value = "";
  setLoading(true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(formatApiError(payload, response.status));
    }

    state.lastResponse = payload;
    updateNoteContext(payload);
    appendMessage("assistant", formatChatAnswer(payload), chatMetadata(payload), pendingActions(payload));
    renderInspector(payload);
    setStatus("");
  } catch (error) {
    appendMessage("assistant", `Something went wrong: ${error.message}`);
    setStatus(error.message, true);
  } finally {
    setLoading(false);
  }
});

function buildRequest(question) {
  const request = { question };
  const work = els.workFilter.value.trim();
  const topK = Number.parseInt(els.topK.value, 10);

  if (work) {
    request.filters = { work };
  }
  if (Number.isInteger(topK)) {
    request.top_k = topK;
  }
  const noteContext = mutationNeedsNoteContext(question) ? buildNoteContext() : null;
  if (noteContext) {
    request.note_context = noteContext;
  }
  return request;
}

function buildNoteContext() {
  if (!state.activeNoteId && !state.retrievedNoteIds.length) {
    return null;
  }
  return {
    active_note_id: state.activeNoteId,
    retrieved_note_ids: state.retrievedNoteIds,
  };
}

function mutationNeedsNoteContext(question) {
  const normalized = question.toLowerCase();
  return /\b(edit|change|update|delete|erase|remove)\b/.test(normalized);
}

function updateView() {
  els.viewButtons.forEach((button) => {
    const isActive = button.dataset.view === state.currentView;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  els.developerPanel.classList.toggle("is-hidden", state.currentView !== "developer");
}

function setLoading(isLoading) {
  els.sendButton.disabled = isLoading;
  els.sendButton.textContent = isLoading ? "Sending" : "Send";
  if (isLoading) {
    setStatus("Thinking...");
  }
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

function appendMessage(role, text, metadata = [], actions = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (metadata.length) {
    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    metadata.forEach((item) => {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = item;
      meta.appendChild(pill);
    });
    bubble.appendChild(meta);
  }

  if (actions.length) {
    const actionRow = document.createElement("div");
    actionRow.className = "bubble-actions";
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      button.addEventListener("click", () => sendPendingAction(action.id, action.confirm));
      actionRow.appendChild(button);
    });
    bubble.appendChild(actionRow);
  }

  article.appendChild(bubble);
  els.messages.appendChild(article);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function formatChatAnswer(response) {
  if (response.note_operation_status) {
    return response.answer || "Note operation updated.";
  }
  if (response.note_status === "saved") {
    return response.note || response.answer || "Saved note.";
  }
  if (response.note_status === "not_saved") {
    return response.note_rejection_reason || response.answer || "The note was not saved.";
  }
  return response.answer || "No answer returned.";
}

function pendingActions(response) {
  if (response.note_operation_status !== "pending_confirmation" || !response.pending_note_action_id) {
    return [];
  }
  return [
    { label: "Confirm", id: response.pending_note_action_id, confirm: true },
    { label: "Cancel", id: response.pending_note_action_id, confirm: false },
  ];
}

function chatMetadata(response) {
  const metadata = [];
  if (response.intent) {
    metadata.push(`Intent: ${response.intent}`);
  }
  if (response.citations?.length) {
    metadata.push(`${response.citations.length} citations`);
  }
  if (response.retrieved_chunks?.length) {
    metadata.push(`${response.retrieved_chunks.length} chunks`);
  }
  if (response.retrieved_notes?.length) {
    metadata.push(`${response.retrieved_notes.length} notes`);
  }
  return metadata;
}

function renderInspector(response) {
  els.developerEmpty.classList.add("is-hidden");
  els.inspector.classList.remove("is-hidden");
  els.inspector.replaceChildren(
    section("Run", keyValues(runDetails(response))),
    section("Citations", listItems(response.citations, renderCitation)),
    section("Retrieved Chunks", listItems(response.retrieved_chunks, renderChunk)),
    section("Retrieved Notes", listItems(response.retrieved_notes, renderNote)),
    section("Unsupported", listItems(response.unsupported, (item) => textItem(item))),
    section("Raw JSON", rawJson(response)),
  );
}

function section(title, bodyNode) {
  const wrapper = document.createElement("section");
  wrapper.className = "detail-section";

  const heading = document.createElement("h2");
  heading.textContent = title;

  const body = document.createElement("div");
  body.className = "detail-body";
  body.appendChild(bodyNode);

  wrapper.append(heading, body);
  return wrapper;
}

function keyValues(entries) {
  const wrapper = document.createElement("div");
  entries.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "kv";

    const label = document.createElement("strong");
    label.textContent = key;

    const content = document.createElement("span");
    content.textContent = value || "-";

    row.append(label, content);
    wrapper.appendChild(row);
  });
  return wrapper;
}

function runDetails(response) {
  return [
    ["Trace ID", response.trace_id],
    ["Prompt", response.prompt_version],
    ["Intent", response.intent],
    ["Confidence", formatScore(response.intent_confidence)],
    ["Note status", response.note_status],
    ["Note query", response.note_query_status],
    ["Note operation", response.note_operation],
    ["Operation status", response.note_operation_status],
    ["Pending action", response.pending_note_action_id],
    ["Created", response.created_at],
  ];
}

function listItems(items, render) {
  const wrapper = document.createElement("div");
  if (!items?.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "None";
    wrapper.appendChild(empty);
    return wrapper;
  }

  items.forEach((item) => wrapper.appendChild(render(item)));
  return wrapper;
}

function renderCitation(citation) {
  return textItem(`[${citation.label}] ${citation.reference}`, [
    citation.source_id,
    citation.chunk_id,
  ]);
}

function renderChunk(chunk) {
  const work = chunk.metadata?.work || chunk.metadata?.title || chunk.source_id;
  return textItem(`[${chunk.label}] ${work}`, [
    chunk.chunk_id,
    `score ${formatScore(chunk.combined_score)}`,
    chunk.reason,
    preview(chunk.text),
  ]);
}

function renderNote(note) {
  const item = textItem(`[${note.label}] ${note.inferred_work}`, [
    note.note_id,
    `score ${formatScore(note.combined_score)}`,
    note.rewritten_note,
  ]);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "item-action";
  button.textContent = state.activeNoteId === note.note_id ? "Active" : "Use";
  button.addEventListener("click", () => {
    state.activeNoteId = note.note_id;
    setStatus(`Active note: ${note.label}`);
    renderInspector(state.lastResponse);
  });
  item.appendChild(button);
  return item;
}

function textItem(title, details = []) {
  const item = document.createElement("article");
  item.className = "item";

  const titleNode = document.createElement("div");
  titleNode.className = "item-title";
  titleNode.textContent = title;
  item.appendChild(titleNode);

  details.filter(Boolean).forEach((detail) => {
    const node = document.createElement("div");
    node.className = "item-meta";
    node.textContent = detail;
    item.appendChild(node);
  });

  return item;
}

function rawJson(response) {
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(response, null, 2);
  return pre;
}

function preview(text, limit = 260) {
  const collapsed = (text || "").replace(/\s+/g, " ").trim();
  if (collapsed.length <= limit) {
    return collapsed;
  }
  return `${collapsed.slice(0, limit - 1).trim()}...`;
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(3) : "";
}

function formatApiError(payload, status) {
  if (typeof payload?.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => {
      const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
      return location ? `${location}: ${item.msg}` : item.msg;
    }).join("; ");
  }
  return payload?.error || `Request failed with status ${status}`;
}

async function sendPendingAction(actionId, confirm) {
  const request = {
    question: confirm ? "Confirm note action." : "Cancel note action.",
    pending_note_action_id: actionId,
    confirm_note_action: confirm,
    cancel_note_action: !confirm,
  };
  const noteContext = buildNoteContext();
  if (noteContext) {
    request.note_context = noteContext;
  }

  appendMessage("user", confirm ? "Confirm note action." : "Cancel note action.");
  setLoading(true);
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(formatApiError(payload, response.status));
    }
    state.lastResponse = payload;
    updateNoteContext(payload);
    appendMessage("assistant", formatChatAnswer(payload), chatMetadata(payload), pendingActions(payload));
    renderInspector(payload);
    setStatus("");
  } catch (error) {
    appendMessage("assistant", `Something went wrong: ${error.message}`);
    setStatus(error.message, true);
  } finally {
    setLoading(false);
  }
}

function updateNoteContext(response) {
  const noteIds = (response.retrieved_notes || []).map((note) => note.note_id);
  if (noteIds.length) {
    state.retrievedNoteIds = noteIds;
    state.activeNoteId = noteIds.length === 1 ? noteIds[0] : null;
  }
  if (response.note_id) {
    state.activeNoteId = response.note_id;
    state.retrievedNoteIds = [response.note_id];
  }
  if (
    response.note_operation_status === "completed" &&
    ["delete", "delete_all"].includes(response.note_operation)
  ) {
    state.activeNoteId = null;
    state.retrievedNoteIds = [];
  }
}

updateView();
