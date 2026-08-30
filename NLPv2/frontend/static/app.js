// LegalRAG Frontend - Client-side Logic
// Prepared for backend connection to localhost:8000

const API_BASE_URL = 'http://localhost:8000';

// State
let uploadedFiles = [];
let chatHistory = [];
let isProcessing = false;

// DOM Elements
const chatHistoryEl = document.getElementById('chat-history');
const chatInputEl = document.getElementById('chat-input');
const sendButtonEl = document.getElementById('send-button');
const dropZoneEl = document.getElementById('drop-zone');
const fileInputEl = document.getElementById('file-input');
const fileListEl = document.getElementById('file-list');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Chat input
    chatInputEl.addEventListener('input', handleInputChange);
    chatInputEl.addEventListener('keydown', handleKeyDown);
    sendButtonEl.addEventListener('click', handleSendMessage);

    // File upload
    dropZoneEl.addEventListener('click', () => fileInputEl.click());
    dropZoneEl.addEventListener('dragover', handleDragOver);
    dropZoneEl.addEventListener('dragleave', handleDragLeave);
    dropZoneEl.addEventListener('drop', handleDrop);
    fileInputEl.addEventListener('change', handleFileSelect);
}

// ==================== Chat Functions ====================

function handleInputChange() {
    // Auto-resize textarea
    chatInputEl.style.height = 'auto';
    chatInputEl.style.height = Math.min(chatInputEl.scrollHeight, 150) + 'px';
    
    // Enable/disable send button
    sendButtonEl.disabled = !chatInputEl.value.trim();
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendButtonEl.disabled) {
            handleSendMessage();
        }
    }
}

async function handleSendMessage() {
    const message = chatInputEl.value.trim();
    if (!message || isProcessing) return;

    // Add user message to UI
    addMessage(message, 'user');
    chatInputEl.value = '';
    handleInputChange();

    // Show typing indicator
    showTypingIndicator();

    try {
        // TODO: Connect to backend when ready
        // const response = await fetch(`${API_BASE_URL}/chat`, {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({ message, history: chatHistory })
        // });
        // const data = await response.json();
        
        // Simulated response for now
        await simulateBotResponse(message);
        
    } catch (error) {
        console.error('Error sending message:', error);
        removeTypingIndicator();
        addMessage('I apologize, but I encountered an error. Please try again.', 'bot');
    }
}

function addMessage(content, sender) {
    const messageEl = document.createElement('div');
    messageEl.className = `message message-${sender}`;
    
    const avatar = sender === 'user' ? 'You' : '§';
    
    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${formatMessage(content)}</div>
    `;
    
    chatHistoryEl.appendChild(messageEl);
    scrollToBottom();
    
    // Store in history
    chatHistory.push({ role: sender, content });
}

function formatMessage(text) {
    // Escape HTML
    let formatted = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Highlight citations [Citation: case name]
    formatted = formatted.replace(/\[Citation: ([^\]]+)\]/g, 
        '<span class="citation" title="Click to view source">[$1]</span>');
    
    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

function showTypingIndicator() {
    const typingEl = document.createElement('div');
    typingEl.className = 'message message-bot';
    typingEl.id = 'typing-indicator';
    typingEl.innerHTML = `
        <div class="message-avatar">§</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatHistoryEl.appendChild(typingEl);
    scrollToBottom();
}

function removeTypingIndicator() {
    const typingEl = document.getElementById('typing-indicator');
    if (typingEl) typingEl.remove();
}

function scrollToBottom() {
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

// Simulated bot response (replace with actual API call)
async function simulateBotResponse(userMessage) {
    removeTypingIndicator();
    
    // This is placeholder logic - connect to /chat endpoint when backend ready
    const responses = [
        "I've analyzed your query against the Karnataka High Court judgments database. Based on the precedents set in *Vijay Kumar v. State of Karnataka* (1987) and *State of Karnataka v. Karnataka Industrial Areas Development Board* (1992), the principles of natural justice require that: (1) no person shall be condemned unheard, (2) the decision-maker must be free from bias, and (3) adequate reasons must be provided for any adverse decision.",
        "Your question regarding the Karnataka Education Act raises important points about regulatory compliance. The relevant provisions under Section 12 and Section 15 establish the framework for institutional accountability. Would you like me to elaborate on any specific aspect?",
        "I've found several relevant judgments on this matter. The ratio decidendi in *Madhukari v. University of Mysore* (1995) establishes that institutional autonomy must be balanced with regulatory oversight. Shall I provide more details on the specific precedents?"
    ];
    
    const response = responses[Math.floor(Math.random() * responses.length)];
    addMessage(response, 'bot');
}

// ==================== File Upload Functions ====================

function handleDragOver(e) {
    e.preventDefault();
    dropZoneEl.classList.add('drag-active');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZoneEl.classList.remove('drag-active');
}

function handleDrop(e) {
    e.preventDefault();
    dropZoneEl.classList.remove('drag-active');
    
    const files = e.dataTransfer.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    const validTypes = ['.pdf', '.docx', '.txt'];
    
    Array.from(files).forEach(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(ext)) {
            alert(`Invalid file type: ${file.name}. Please upload PDF, DOCX, or TXT files.`);
            return;
        }
        
        if (file.size > 50 * 1024 * 1024) {
            alert(`File too large: ${file.name}. Maximum size is 50MB.`);
            return;
        }
        
        // Add to uploaded files
        uploadedFiles.push({
            name: file.name,
            size: file.size,
            type: ext,
            file: file
        });
        
        renderFileList();
        
        // TODO: Upload to backend when ready
        // const formData = new FormData();
        // formData.append('file', file);
        // await fetch(`${API_BASE_URL}/document`, { method: 'POST', body: formData });
    });
}

function renderFileList() {
    fileListEl.innerHTML = '';
    
    uploadedFiles.forEach((file, index) => {
        const fileEl = document.createElement('div');
        fileEl.className = 'file-item';
        
        const icon = file.type === '.pdf' ? 'ph-file-pdf' : 
                     file.type === '.docx' ? 'ph-file-doc' : 'ph-file-text';
        
        fileEl.innerHTML = `
            <i class="ph ${icon} file-item-icon"></i>
            <span class="file-item-name" title="${file.name}">${file.name}</span>
            <button class="file-item-remove" data-index="${index}">
                <i class="ph-bold ph-x"></i>
            </button>
        `;
        
        fileEl.querySelector('.file-item-remove').addEventListener('click', () => {
            uploadedFiles.splice(index, 1);
            renderFileList();
        });
        
        fileListEl.appendChild(fileEl);
    });
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
