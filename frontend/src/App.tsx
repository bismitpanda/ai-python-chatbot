import React, { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User } from "lucide-react";

interface Message {
    role: "user" | "bot";
    content: string;
}

interface ChatResponse {
    response: string;
}

export default function App() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        // Add user message
        const userMessage: Message = { role: "user", content: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        try {
            const response = await fetch("http://localhost:8000/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: input }),
            });

            const data: ChatResponse = await response.json();

            // Add bot response
            setMessages((prev) => [
                ...prev,
                { role: "bot", content: data.response },
            ]);
        } catch (error) {
            console.error("Error:", error);
            setMessages((prev) => [
                ...prev,
                {
                    role: "bot",
                    content:
                        "Sorry, I'm having trouble connecting to the server.",
                },
            ]);
        }

        setIsLoading(false);
    };

    const MessageBubble: React.FC<{ message: Message }> = ({ message }) => (
        <div
            className={`flex items-start gap-2 ${
                message.role === "user" ? "flex-row-reverse" : ""
            }`}
        >
            {message.role === "user" ? (
                <User className="w-6 h-6 mt-1 flex-shrink-0" />
            ) : (
                <Bot className="w-6 h-6 mt-1 flex-shrink-0" />
            )}
            <div
                className={`rounded-lg px-4 py-2 max-w-[80%] ${
                    message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                }`}
            >
                {message.content}
            </div>
        </div>
    );

    return (
        <Card className="w-full max-w-2xl mx-auto h-[600px] flex flex-col">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Bot className="w-6 h-6" />
                    Chatbot
                </CardTitle>
            </CardHeader>

            <CardContent className="flex-1 flex flex-col gap-4">
                <ScrollArea className="flex-1 pr-4 max-h-[450px]">
                    <div className="flex flex-col gap-4">
                        {messages.map((message, index) => (
                            <MessageBubble key={index} message={message} />
                        ))}
                        {isLoading && (
                            <div className="flex items-start gap-2">
                                <Bot className="w-6 h-6 mt-1 flex-shrink-0" />
                                <div className="rounded-lg px-4 py-2 bg-muted animate-pulse">
                                    Thinking...
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                </ScrollArea>

                <form onSubmit={handleSubmit} className="flex gap-2">
                    <Input
                        value={input}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setInput(e.target.value)
                        }
                        placeholder="Type your message..."
                        disabled={isLoading}
                        className="flex-1"
                    />
                    <Button type="submit" disabled={isLoading}>
                        <Send className="w-4 h-4" />
                    </Button>
                </form>
            </CardContent>
        </Card>
    );
}
