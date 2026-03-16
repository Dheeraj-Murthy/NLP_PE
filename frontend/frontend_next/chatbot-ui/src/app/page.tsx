"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: string[];
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Good day. I am your LegalRAG assistant. How may I assist you with your legal research or document analysis today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // TODO: Connect to backend
      // const response = await fetch("http://localhost:8000/chat", {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify({ message: input, history: messages }),
      // });
      // const data = await response.json();

      await new Promise((r) => setTimeout(r, 1500));
      const responses = [
        `I've analyzed your query against the Karnataka High Court judgments database. Based on the precedents set in Vijay Kumar v. State of Karnataka (1987) and State of Karnataka v. Karnataka Industrial Areas Development Board (1992), the principles of natural justice require that: (1) no person shall be condemned unheard, (2) the decision-maker must be free from bias, and (3) adequate reasons must be provided for any adverse decision.`,
        `Your question regarding the Karnataka Education Act raises important points about regulatory compliance. The relevant provisions under Section 12 and Section 15 establish the framework for institutional accountability. Would you like me to elaborate on any specific aspect?`,
        `I've found several relevant judgments on this matter. The ratio decidendi in Madhukari v. University of Mysore (1995) establishes that institutional autonomy must be balanced with regulatory oversight. Shall I provide more details on the specific precedents?`,
      ];
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: responses[Math.floor(Math.random() * responses.length)],
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "I apologize, but I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const handleFiles = (fileList: File[]) => {
    const validTypes = [".pdf", ".docx", ".txt"];
    const newFiles: UploadedFile[] = fileList
      .filter((file) => {
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        return validTypes.includes(ext) && file.size <= 50 * 1024 * 1024;
      })
      .map((file) => ({
        id: Date.now().toString() + Math.random(),
        name: file.name,
        size: file.size,
      }));
    setFiles((prev) => [...prev, ...newFiles]);
  };

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="flex-1 flex max-w-[1800px] mx-auto w-full overflow-hidden">
      <aside className="w-[35%] min-w-[300px] border-r border-[#E5E2D9] bg-white p-6 flex flex-col overflow-y-auto">
          <div className="mb-6">
            <h2 className="font-serif text-lg font-semibold text-[#111827] mb-1">
              Document Context
            </h2>
            <p className="text-sm text-[#6B7280]">
              Upload legal documents, case files, or contracts to analyze.
            </p>
          </div>

          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
              isDragging
                ? "border-[#C5A880] bg-[#C5A880]/5"
                : "border-[#E5E2D9] hover:border-[#C5A880]"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input
              type="file"
              id="file-input"
              className="hidden"
              multiple
              accept=".pdf,.docx,.txt"
              onChange={handleFileSelect}
            />
            <div className="text-[#C5A880] text-4xl mb-2">📄</div>
            <p className="font-medium text-[#111827]">Drag & drop files here</p>
            <p className="text-xs text-[#6B7280] mt-1">
              Supports PDF, DOCX, TXT (Max 50MB)
            </p>
          </div>

          {files.length > 0 && (
            <div className="mt-6 space-y-2">
              {files.map((file) => (
                <div
                  key={file.id}
                  className="flex items-center gap-3 p-3 bg-[#F7F6F2] border border-[#E5E2D9] rounded text-sm"
                >
                  <span className="text-[#8B1E28]">📄</span>
                  <span className="flex-1 truncate font-medium">
                    {file.name}
                  </span>
                  <span className="text-xs text-[#6B7280]">
                    {formatFileSize(file.size)}
                  </span>
                  <button
                    onClick={() => removeFile(file.id)}
                    className="text-[#6B7280] hover:text-[#8B1E28] transition-colors"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </aside>

        <main className="flex-1 flex flex-col bg-[#F7F6F2]">
          <div className="flex-1 overflow-y-auto p-8 space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex max-w-[80%] ${
                  message.role === "user" ? "ml-auto flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 mx-3 font-semibold text-sm ${
                    message.role === "user"
                      ? "bg-[#111827] text-white"
                      : "bg-white border border-[#E5E2D9] text-[#8B1E28] font-serif text-xl"
                  }`}
                >
                  {message.role === "user" ? "You" : "§"}
                </div>
                <div
                  className={`px-5 py-4 rounded-xl shadow-sm ${
                    message.role === "user"
                      ? "bg-[#111827] text-white rounded-tr-none"
                      : "bg-white text-[#374151] rounded-tl-none"
                  }`}
                  style={{
                    borderLeftWidth: message.role === "assistant" ? "3px" : 0,
                    borderLeftColor: message.role === "assistant" ? "#8B1E28" : "transparent",
                  }}
                >
                  <p className={message.role === "assistant" ? "font-serif text-lg" : ""}>
                    {message.content}
                  </p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex max-w-[80%]">
                <div className="w-9 h-9 rounded-full bg-white border border-[#E5E2D9] flex items-center justify-center mx-3">
                  <span className="text-[#8B1E28] font-serif text-xl">§</span>
                </div>
                <div
                  className="bg-white px-5 py-4 rounded-xl rounded-tl-none shadow-sm"
                  style={{ borderLeftWidth: "3px", borderLeftColor: "#8B1E28" }}
                >
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-[#6B7280] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 bg-[#6B7280] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 bg-[#6B7280] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-6 bg-gradient-to-t from-[#F7F6F2] to-transparent">
            <div className="flex items-end bg-white border border-[#E5E2D9] rounded-xl px-4 py-2 shadow-md focus-within:border-[#D1CDBF] focus-within:shadow-lg transition-all">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a legal question or query your documents..."
                className="flex-1 py-3 bg-transparent outline-none resize-none max-h-[150px] text-[#111827] placeholder:text-[#6B7280]"
                rows={1}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="ml-2 w-10 h-10 bg-[#111827] text-white rounded-lg flex items-center justify-center hover:bg-[#8B1E28] disabled:bg-[#E5E2D9] disabled:text-[#6B7280] transition-colors"
              >
                ➤
              </button>
            </div>
          </div>
        </main>
    </div>
  );
}
