import { useRef, useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, Send, Paperclip, X, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types";
import { CHAT_TEXTAREA_MAX_HEIGHT } from "@/constants/chatConfig";

interface PendingFile {
  base64: string;
  mimeType: string;
  name: string;
}

interface CoderSidebarProps {
  messages: ChatMessage[];
  loading: boolean;
  onSend: (message: string, fileBase64?: string, fileMimeType?: string, fileName?: string) => void;
}

export function CoderSidebar({ messages, loading, onSend }: CoderSidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [input, setInput] = useState("");
  const [pendingFile, setPendingFile] = useState<PendingFile | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFile = (file: File) => {
    const allowed = file.type.startsWith("image/") || file.type === "application/pdf";
    if (!allowed) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      const base64 = result.split(",")[1];
      setPendingFile({ base64, mimeType: file.type, name: file.name });
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const item = Array.from(e.clipboardData.items).find((i) =>
      i.type.startsWith("image/")
    );
    if (item) {
      const file = item.getAsFile();
      if (file) handleFile(file);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if ((!trimmed && !pendingFile) || loading) return;
    const defaultPrompt = pendingFile?.mimeType === "application/pdf"
      ? "Analyse this PDF and generate a robot flow."
      : "Analyse this image and generate a robot flow.";
    onSend(trimmed || defaultPrompt, pendingFile?.base64, pendingFile?.mimeType, pendingFile?.name);
    setInput("");
    setPendingFile(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const canSubmit = !loading && (!!input.trim() || !!pendingFile);
  const isImage = pendingFile?.mimeType.startsWith("image/");
  const isPdf = pendingFile?.mimeType === "application/pdf";

  return (
    <div
      className={cn(
        "relative flex flex-col border-l transition-[width] duration-250 ease-in-out overflow-hidden",
        collapsed ? "w-9" : "w-80"
      )}
      style={{ background: "var(--sidebar-bg)", borderColor: "var(--sidebar-border)" }}
    >
      {/* Toggle */}
      <button
        className="absolute top-2 left-1 z-10 flex items-center justify-center w-7 h-7 rounded bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground cursor-pointer border-none"
        onClick={() => setCollapsed((c) => !c)}
      >
        {collapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
      </button>

      {!collapsed && (
        <div className="flex flex-col pt-10 px-3 pb-0 overflow-hidden h-full">
          <h2 className="forgis-text-title font-normal uppercase text-[var(--gunmetal-50)] leading-none font-forgis-digit mb-3">
            Chat
          </h2>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto flex flex-col gap-2 pb-2">
            <div
              className="self-start text-foreground max-w-[90%] px-3 py-2 rounded-lg rounded-bl-sm forgis-text-body leading-snug font-forgis-body"
              style={{ background: "var(--chat-assistant-bg)" }}
            >
              Welcome! Describe a robot task and I'll generate an execution flow for you. You can also attach an image or PDF.
            </div>
            {messages.map((msg) => (
              <div key={msg.id} className={cn("flex flex-col gap-1", msg.role === "user" ? "items-end" : "items-start")}>
                {/* Image attachment */}
                {msg.fileBase64 && msg.fileMimeType?.startsWith("image/") && (
                  <img
                    src={`data:${msg.fileMimeType};base64,${msg.fileBase64}`}
                    alt="attached"
                    className="max-w-[90%] rounded-lg border border-border object-cover"
                    style={{ maxHeight: 140 }}
                  />
                )}
                {/* PDF attachment */}
                {msg.fileBase64 && msg.fileMimeType === "application/pdf" && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-border bg-muted max-w-[90%]">
                    <FileText size={13} className="text-[var(--tiger)] shrink-0" />
                    <span className="forgis-text-body text-foreground truncate font-forgis-body">{msg.fileName ?? "document.pdf"}</span>
                  </div>
                )}
                <div
                  className={cn(
                    "max-w-[90%] px-3 py-2 rounded-lg forgis-text-body leading-snug break-words text-foreground font-forgis-body",
                    msg.role === "user"
                      ? "self-end rounded-br-sm border"
                      : "self-start rounded-bl-sm"
                  )}
                  style={
                    msg.role === "user"
                      ? { background: "var(--chat-user-bg)", borderColor: "var(--chat-user-border)" }
                      : { background: "var(--chat-assistant-bg)" }
                  }
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && (
              <div
                className="self-start text-[var(--gunmetal-50)] max-w-[90%] px-3 py-2 rounded-lg rounded-bl-sm forgis-text-body italic opacity-70 font-forgis-body"
                style={{ background: "var(--chat-assistant-bg)" }}
              >
                Thinking...
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* File preview strip */}
          {pendingFile && (
            <div className="relative self-start mb-1.5">
              {isImage ? (
                <img
                  src={`data:${pendingFile.mimeType};base64,${pendingFile.base64}`}
                  alt="pending"
                  className="h-16 w-auto rounded-md border border-border object-cover"
                />
              ) : isPdf ? (
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-[var(--tiger)] bg-muted">
                  <FileText size={13} className="text-[var(--tiger)] shrink-0" />
                  <span className="forgis-text-body text-foreground truncate font-forgis-body max-w-[180px]">{pendingFile.name}</span>
                </div>
              ) : null}
              <button
                type="button"
                onClick={() => setPendingFile(null)}
                className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-foreground text-background flex items-center justify-center hover:bg-muted-foreground border-none cursor-pointer"
              >
                <X size={9} />
              </button>
            </div>
          )}

          {/* Input form */}
          <form className="flex items-end gap-1.5 py-2.5 border-t border-border" onSubmit={handleSubmit}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,application/pdf"
              className="hidden"
              onChange={handleFileChange}
            />

            <button
              type="button"
              title="Attach image or PDF"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className={cn(
                "flex items-center justify-center w-8 h-8 rounded-md border border-border bg-input text-muted-foreground cursor-pointer hover:text-foreground hover:border-[var(--tiger)] transition-colors shrink-0 disabled:opacity-40 disabled:cursor-not-allowed",
                pendingFile && "border-[var(--tiger)] text-[var(--tiger)]"
              )}
            >
              <Paperclip size={14} />
            </button>

            <textarea
              ref={textareaRef}
              className="flex-1 px-2.5 py-2 bg-input border border-border rounded-md text-foreground forgis-text-body outline-none focus:border-[var(--tiger)] placeholder:text-muted-foreground font-forgis-body resize-none overflow-hidden"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, CHAT_TEXTAREA_MAX_HEIGHT)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              onPaste={handlePaste}
              placeholder={pendingFile ? "Add a description (optional)…" : "Describe the task..."}
              disabled={loading}
              rows={1}
              style={{ maxHeight: CHAT_TEXTAREA_MAX_HEIGHT }}
            />
            <button
              className="px-3 py-2 bg-[var(--tiger)] text-white rounded-md forgis-text-body font-normal cursor-pointer border-none hover:bg-[var(--tiger)]/90 disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
              type="submit"
              disabled={!canSubmit}
            >
              <Send size={14} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
