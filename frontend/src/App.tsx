import { useEffect, useRef, useState } from "react";
import axios from "axios";
import clsx from "clsx";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

dayjs.extend(relativeTime);

const SESSION_ID = "default";

type Role = "user" | "assistant" | "system";

interface Specialist {
  name: string;
  confidence: number;
  summary: string;
}

interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  domain?: string;
  difficulty?: string;
  confidence?: number;
  specialists?: Specialist[];
  retrievedContext?: string[];
}

interface CurriculumStatus {
  iteration: number;
  max_iterations: number;
  difficulty_distribution: Record<string, number>;
  scheduler_type: string;
}

interface ChatPayload {
  answer: string;
  domain: string;
  difficulty_level: string;
  confidence: number;
  specialists: Specialist[];
  retrieved_context: string[];
  timestamp: string;
}

interface VqaPayload {
  answer: string;
  domain: string;
  difficulty_level: string;
  retrieved_context: Array<{
    title: string;
    snippet: string;
    similarity: number;
    domain: string;
  }>;
}

type VqaDoc = VqaPayload["retrieved_context"][number];

const createId = () => crypto.randomUUID();

const initialSystemMessage: Message = {
  id: createId(),
  role: "system",
  content: "Welcome! Ask a medical question or attach an image for MedGemma to inspect.",
  createdAt: new Date().toISOString(),
};

const roleLabel: Record<Role, string> = {
  user: "You",
  assistant: "Assistant",
  system: "System",
};

const toneForSpecialist = (name: string): string => {
  const normalized = name.toLowerCase();
  if (normalized.includes("cardio")) return "cardiology";
  if (normalized.includes("radio")) return "radiology";
  if (normalized.includes("emerg")) return "emergency";
  return "general";
};

const formatTime = (iso: string) => dayjs(iso).format("HH:mm");

const orderedDifficulties = ["trivial", "easy", "medium", "hard"] as const;

export default function App() {
  const [messages, setMessages] = useState<Message[]>([initialSystemMessage]);
  const [textInput, setTextInput] = useState<string>("");
  const [isSending, setIsSending] = useState<boolean>(false);
  const [curriculum, setCurriculum] = useState<CurriculumStatus | null>(null);
  const [vqaQuestion, setVqaQuestion] = useState<string>("");
  const [vqaFile, setVqaFile] = useState<File | null>(null);
  const [vqaSending, setVqaSending] = useState<boolean>(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  const pushMessage = (message: Message) => {
    setMessages((prev) => [...prev, message]);
  };

  const fetchCurriculum = async () => {
    try {
      const { data } = await axios.get<CurriculumStatus>("/curriculum/status");
      setCurriculum(data);
    } catch (error) {
      console.error("Failed to load curriculum status", error);
    }
  };

  useEffect(() => {
    void fetchCurriculum();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const trimmed = textInput.trim();
    if (!trimmed || isSending) return;

    const userMessage: Message = {
      id: createId(),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    };

    pushMessage(userMessage);
    setTextInput("");

    try {
      setIsSending(true);
      const { data } = await axios.post<ChatPayload>("/chat", {
        message: trimmed,
        session_id: SESSION_ID,
      });

      const assistantMessage: Message = {
        id: createId(),
        role: "assistant",
        content: data.answer,
        createdAt: data.timestamp,
        domain: data.domain,
        difficulty: data.difficulty_level,
        confidence: data.confidence,
        specialists: data.specialists,
        retrievedContext: data.retrieved_context,
      };

      pushMessage(assistantMessage);
      void fetchCurriculum();
    } catch (error) {
      console.error("Chat error", error);
      pushMessage({
        id: createId(),
        role: "system",
        content: "We couldn't send that message. Please try again.",
        createdAt: new Date().toISOString(),
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleVqa = async () => {
    const trimmedQuestion = vqaQuestion.trim();
    if (!vqaFile || !trimmedQuestion || vqaSending) return;

    pushMessage({
      id: createId(),
      role: "user",
      content: `[Image] ${trimmedQuestion}`,
      createdAt: new Date().toISOString(),
    });

    const formData = new FormData();
    formData.append("question", trimmedQuestion);
    formData.append("image", vqaFile);
    formData.append("session_id", SESSION_ID);

    try {
      setVqaSending(true);
      const { data } = await axios.post<VqaPayload>("/vqa", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const assistantMessage: Message = {
        id: createId(),
        role: "assistant",
        content: data.answer,
        createdAt: new Date().toISOString(),
        domain: data.domain,
        difficulty: data.difficulty_level,
        retrievedContext: data.retrieved_context.map(
          (doc: VqaDoc) => `${doc.title}: ${doc.snippet}`,
        ),
      };

      pushMessage(assistantMessage);
      setVqaQuestion("");
      setVqaFile(null);
      void fetchCurriculum();
    } catch (error) {
      console.error("VQA error", error);
      pushMessage({
        id: createId(),
        role: "system",
        content: "Image question failed. Please try again.",
        createdAt: new Date().toISOString(),
      });
    } finally {
      setVqaSending(false);
    }
  };

  const resetCurriculum = async () => {
    try {
      await axios.post("/curriculum/reset");
      await fetchCurriculum();
      pushMessage({
        id: createId(),
        role: "system",
        content: "Curriculum progression has been reset.",
        createdAt: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Reset failed", error);
      pushMessage({
        id: createId(),
        role: "system",
        content: "Could not reset the curriculum. Check the server logs.",
        createdAt: new Date().toISOString(),
      });
    }
  };

  return (
    <div className="page">
      <div className="chat-card">
        <header className="card-header">
          <div className="header-copy">
            <h1>Curriculum Medical Assistant</h1>
            <p>
              Curriculum-aware reasoning with specialist hand-offs and MedGemma visual answers.
            </p>
          </div>
          <div className="header-status">
            {curriculum ? (
              <>
                <span className="status-line">
                  Iteration {curriculum.iteration} / {curriculum.max_iterations}
                </span>
                <div className="difficulty-tags">
                  {orderedDifficulties.map((level) => {
                    const prob = curriculum.difficulty_distribution[level];
                    if (prob === undefined) return null;
                    return (
                      <span key={level} className="difficulty-tag">
                        {level}: {Math.round(prob * 100)}%
                      </span>
                    );
                  })}
                </div>
                <button className="link-button" type="button" onClick={resetCurriculum}>
                  Reset curriculum
                </button>
              </>
            ) : (
              <span className="muted">Loading curriculum…</span>
            )}
          </div>
        </header>

        <main className="messages-area">
          {messages.map((message) => (
            <article
              key={message.id}
              className={clsx("message", `message-${message.role}`)}
            >
              <div className="message-header">
                <span>{roleLabel[message.role]}</span>
                <time>{formatTime(message.createdAt)}</time>
              </div>
              <div className="message-body">{message.content}</div>

              {message.role === "assistant" && (
                <div className="meta-row">
                  {message.domain && (
                    <span className="meta-chip domain">{message.domain}</span>
                  )}
                  {message.difficulty && (
                    <span className="meta-chip difficulty">{message.difficulty}</span>
                  )}
                  {typeof message.confidence === "number" && (
                    <span className="meta-chip confidence">
                      Confidence {Math.round(message.confidence * 100)}%
                    </span>
                  )}
                </div>
              )}

              {message.specialists && message.specialists.length > 0 && (
                <div className="specialist-row">
                  {message.specialists.map((specialist) => (
                    <span
                      key={specialist.name}
                      className={clsx("specialist-chip", toneForSpecialist(specialist.name))}
                      title={specialist.summary}
                    >
                      {specialist.name} · {Math.round(specialist.confidence * 100)}%
                    </span>
                  ))}
                </div>
              )}

              {message.retrievedContext && message.retrievedContext.length > 0 && (
                <details className="context-details">
                  <summary>Retrieved context</summary>
                  <ul>
                    {message.retrievedContext.map((entry, index) => (
                      <li key={index}>{entry}</li>
                    ))}
                  </ul>
                </details>
              )}
            </article>
          ))}
          <div ref={bottomRef} />
        </main>

        <section className="composer">
          <div>
            <label htmlFor="chat-question">Your question</label>
            <textarea
              id="chat-question"
              placeholder="Ask anything medical. Press Enter to send, Shift+Enter for a new line."
              value={textInput}
              onChange={(event) => setTextInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              disabled={isSending}
            />
          </div>
          <div className="composer-actions">
            <span className="muted">Shift + Enter for newline</span>
            <button
              className="primary-button"
              type="button"
              onClick={sendMessage}
              disabled={isSending || !textInput.trim()}
            >
              {isSending ? "Sending…" : "Send"}
            </button>
          </div>

          <div className="vqa-card">
            <div>
              <h2>Image question</h2>
              <p>Attach an image when you want MedGemma to ground the answer visually.</p>
            </div>
            <div className="vqa-grid">
              <div className="field">
                <label htmlFor="vqa-file">Medical image</label>
                <input
                  id="vqa-file"
                  type="file"
                  accept="image/*"
                  onChange={(event) => setVqaFile(event.target.files?.[0] ?? null)}
                />
                {vqaFile && <p className="file-info">{vqaFile.name}</p>}
              </div>
              <div className="field">
                <label htmlFor="vqa-question">Question</label>
                <input
                  id="vqa-question"
                  type="text"
                  placeholder="What should the model look for?"
                  value={vqaQuestion}
                  onChange={(event) => setVqaQuestion(event.target.value)}
                />
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={handleVqa}
                disabled={vqaSending || !vqaFile || !vqaQuestion.trim()}
              >
                {vqaSending ? "Submitting…" : "Send image question"}
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
