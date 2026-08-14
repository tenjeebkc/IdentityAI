const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const welcomeScreen = document.getElementById("welcome-screen");

function scrollToLatest() {
    chatBox.scrollTop = chatBox.scrollHeight;
}

function resizeInput() {
    userInput.style.height = "auto";
    userInput.style.height = `${Math.min(userInput.scrollHeight, 130)}px`;
}

function updateSendButton() {
    sendButton.disabled = !userInput.value.trim() || sendButton.dataset.loading === "true";
}

function addMessage(text, role) {
    const message = document.createElement("article");
    message.className = `message ${role}-message`;

    if (role === "bot") {
        const avatar = document.createElement("span");
        avatar.className = "message-avatar";
        avatar.textContent = "✦";
        avatar.setAttribute("aria-hidden", "true");
        message.append(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;
    message.append(bubble);
    chatBox.append(message);
    scrollToLatest();
}

function addTypingIndicator() {
    const indicator = document.createElement("article");
    indicator.className = "message bot-message typing-message";
    indicator.id = "typing-indicator";
    indicator.innerHTML = '<span class="message-avatar" aria-hidden="true">✦</span><div class="message-bubble">IdentityAI is thinking <span class="typing-dots" aria-label="Loading"><i></i><i></i><i></i></span></div>';
    chatBox.append(indicator);
    scrollToLatest();
    return indicator;
}

async function sendMessage(messageOverride) {
    const message = (messageOverride ?? userInput.value).trim();
    if (!message || sendButton.dataset.loading === "true") return;

    welcomeScreen?.remove();
    addMessage(message, "user");
    userInput.value = "";
    resizeInput();
    sendButton.dataset.loading = "true";
    updateSendButton();
    const typingIndicator = addTypingIndicator();

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message })
        });

        if (!response.ok) throw new Error("The chat service could not respond.");
        const data = await response.json();
        addMessage(data.reply || "I’m sorry, I couldn’t generate a response just now.", "bot");
    } catch (error) {
        addMessage("I’m having trouble connecting right now. Please check that the server is running and try again.", "bot");
        console.error("Chat request failed:", error);
    } finally {
        typingIndicator.remove();
        delete sendButton.dataset.loading;
        updateSendButton();
        userInput.focus();
        scrollToLatest();
    }
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
});

userInput.addEventListener("input", () => {
    resizeInput();
    updateSendButton();
});

userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => sendMessage(card.dataset.prompt));
});
