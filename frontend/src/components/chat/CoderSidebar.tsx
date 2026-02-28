import { useRef, useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, Send, Paperclip, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types";
import { CHAT_TEXTAREA_MAX_HEIGHT } from "@/constants/chatConfig";

interface CoderSidebarProps {
  messages: ChatMessage[];
  loading: boolean;
  onSend: (message: string, fileBase64?: string, fileMimeType?: string, fileName?: string) => void;
}

export function CoderSidebar({ messages, loading, onSend }: CoderSidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [input, setInput] = useState("");
  const [attachedFile, setAttachedFile] = useState<{ base64: string; mimeType: string; name: string } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const readFileAsBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(",")[1]); // strip data:...;base64, prefix
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const attachFile = async (file: File) => {
    const base64 = await readFileAsBase64(file);
    setAttachedFile({ base64, mimeType: file.type, name: file.name });
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) await attachFile(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handlePaste = async (e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) await attachFile(file);
        return;
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    onSend(trimmed, attachedFile?.base64, attachedFile?.mimeType, attachedFile?.name);
    setInput("");
    setAttachedFile(null);
  };

  const isImage = attachedFile?.mimeType.startsWith("image/");

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
            {/* Welcome message always shown first */}
            <div
              className="self-start text-foreground max-w-[90%] px-3 py-2 rounded-lg rounded-bl-sm forgis-text-body leading-snug font-forgis-body"
              style={{ background: "var(--chat-assistant-bg)" }}
            >
              Welcome! Describe a robot task and I'll generate an execution flow for you. You can also attach an image or PDF.
            </div>
            {messages.map((msg) => (
              <div
                key={msg.id}
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
                {msg.fileBase64 && msg.fileMimeType?.startsWith("image/") && (
                  <img
                    src={`data:${msg.fileMimeType};base64,${msg.fileBase64}`}
                    alt="attached"
                    className="max-w-full rounded mb-1.5"
                    style={{ maxHeight: 160 }}
                  />
                )}
                {msg.fileBase64 && msg.fileMimeType === "application/pdf" && msg.fileName && (
                  <div className="text-xs bg-muted px-2 py-1 rounded mb-1.5 inline-block">
                    PDF: {msg.fileName}
                  </div>
                )}
                {msg.content}
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

          {/* Attachment preview */}
          {attachedFile && (
            <div className="flex items-center gap-2 px-1 py-1.5">
              {isImage ? (
                <img
                  src={`data:${attachedFile.mimeType};base64,${attachedFile.base64}`}
                  alt="preview"
                  className="h-10 w-10 rounded object-cover"
                />
              ) : (
                <span className="text-xs bg-muted px-2 py-1 rounded truncate max-w-[180px]">
                  {attachedFile.name}
                </span>
              )}
              <button
                className="p-0.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground border-none bg-transparent cursor-pointer"
                onClick={() => setAttachedFile(null)}
              >
                <X size={14} />
              </button>
            </div>
          )}

          {/* Input form */}
          <form className="flex items-end gap-1.5 py-2.5 border-t border-border" onSubmit={handleSubmit}>
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,application/pdf"
              className="hidden"
              onChange={handleFileChange}
            />
            {/* Paperclip button */}
            <button
              type="button"
              className="px-1.5 py-2 bg-transparent text-muted-foreground hover:text-foreground rounded-md cursor-pointer border-none shrink-0"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              title="Attach image or PDF"
            >
              <Paperclip size={16} />
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
              placeholder="Describe the task..."
              disabled={loading}
              rows={1}
              style={{ maxHeight: CHAT_TEXTAREA_MAX_HEIGHT }}
            />
            <button
              className="px-3 py-2 bg-[var(--tiger)] text-white rounded-md forgis-text-body font-normal cursor-pointer border-none hover:bg-[var(--tiger)]/90 disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
              type="submit"
              disabled={loading || !input.trim()}
            >
              <Send size={14} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
