import { useRef, useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types";
import { CHAT_TEXTAREA_MAX_HEIGHT } from "@/constants/chatConfig";

interface CoderSidebarProps {
  messages: ChatMessage[];
  loading: boolean;
  onSend: (message: string) => void;
}

export function CoderSidebar({ messages, loading, onSend }: CoderSidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;
    onSend(trimmed);
    setInput("");
  };

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
              Welcome! Describe a robot task and I'll generate an execution flow for you.
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

          {/* Input form */}
          <form className="flex items-end gap-1.5 py-2.5 border-t border-border" onSubmit={handleSubmit}>
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
