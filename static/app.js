document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const videoInput = document.getElementById('videoInput');
    const browseBtn = document.getElementById('browseBtn');
    const dropzonePrompt = document.getElementById('dropzonePrompt');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadStatusText = document.getElementById('uploadStatusText');

    const videoPreviewContainer = document.getElementById('videoPreviewContainer');
    const videoPlayer = document.getElementById('videoPlayer');
    const metaFileName = document.getElementById('metaFileName');
    const metaFileSize = document.getElementById('metaFileSize');
    const metaDuration = document.getElementById('metaDuration');
    const removeVideoBtn = document.getElementById('removeVideoBtn');

    const analyzeBtn = document.getElementById('analyzeBtn');
    const customPrompt = document.getElementById('customPrompt');
    const numFramesSelect = document.getElementById('numFramesSelect');
    const presetBtns = document.querySelectorAll('.preset-btn');
    const statusBadge = document.getElementById('statusBadge');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const loaderTitle = document.getElementById('loaderTitle');
    const loaderSub = document.getElementById('loaderSub');
    const dot1 = document.getElementById('dot1');
    const dot2 = document.getElementById('dot2');
    const dot3 = document.getElementById('dot3');

    const resultsContainer = document.getElementById('resultsContainer');
    const keyframeGrid = document.getElementById('keyframeGrid');
    const keyframeQty = document.getElementById('keyframeQty');
    const reportModelTag = document.getElementById('reportModelTag');
    const reportMarkdown = document.getElementById('reportMarkdown');
    const copyReportBtn = document.getElementById('copyReportBtn');

    let currentUploadedFilename = null;
    let currentReportText = "";

    // 1. Drag & Drop Handlers
    browseBtn.addEventListener('click', () => videoInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // 2. Upload File to Backend
    async function handleSelectedFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please select a valid MP4 or video file.');
            return;
        }

        // Show uploading UI
        dropzonePrompt.classList.add('hidden');
        uploadProgress.classList.remove('hidden');
        uploadStatusText.textContent = `Uploading ${file.name}...`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload-video', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || 'Upload failed');
            }

            // Success
            currentUploadedFilename = data.filename;
            metaFileName.textContent = data.original_name;
            metaFileSize.textContent = `${data.file_size_mb} MB`;
            metaDuration.textContent = `${data.duration_seconds} sec`;

            videoPlayer.src = data.video_url;

            dropzone.classList.add('hidden');
            videoPreviewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
            statusBadge.textContent = 'Video Uploaded';
            statusBadge.style.color = '#34d399';

        } catch (err) {
            alert(`Upload Error: ${err.message}`);
            dropzonePrompt.classList.remove('hidden');
            uploadProgress.classList.add('hidden');
        }
    }

    // Remove Video Handler
    removeVideoBtn.addEventListener('click', () => {
        currentUploadedFilename = null;
        videoPlayer.pause();
        videoPlayer.src = '';
        videoInput.value = '';

        videoPreviewContainer.classList.add('hidden');
        dropzone.classList.remove('hidden');
        dropzonePrompt.classList.remove('hidden');
        uploadProgress.classList.add('hidden');

        analyzeBtn.disabled = true;
        statusBadge.textContent = 'Ready';
        statusBadge.style.color = '';

        resultsContainer.classList.add('hidden');
        emptyState.classList.remove('hidden');
    });

    // Preset Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            customPrompt.value = btn.getAttribute('data-prompt');
        });
    });

    // 3. Analyze Video Button Handler
    analyzeBtn.addEventListener('click', async () => {
        if (!currentUploadedFilename) return;

        // UI State -> Loading
        emptyState.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');
        analyzeBtn.disabled = true;

        statusBadge.textContent = 'Analyzing with Groq Vision...';
        statusBadge.style.color = '#fbbf24';

        // Step 1 Loading Indicator
        loaderTitle.textContent = 'Extracting Video Keyframes...';
        loaderSub.textContent = 'OpenCV is capturing evenly spaced video frames';
        dot1.classList.add('active');
        dot2.classList.remove('active');
        dot3.classList.remove('active');

        // Step 2 Loading Indicator Timer
        setTimeout(() => {
            loaderTitle.textContent = 'Calling Groq Vision API...';
            loaderSub.textContent = 'Processing keyframes through Llama 3.2 Multimodal Vision';
            dot2.classList.add('active');
        }, 1500);

        const formData = new FormData();
        formData.append('filename', currentUploadedFilename);
        formData.append('custom_prompt', customPrompt.value);
        formData.append('num_frames', numFramesSelect.value);

        try {
            const res = await fetch('/api/analyze-video', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || 'Analysis failed');
            }

            // Step 3 Finish
            dot3.classList.add('active');
            displayResults(data);

        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
            statusBadge.textContent = 'Error';
            statusBadge.style.color = '#f87171';
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    // Chat Memory Elements
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');
    const chatThread = document.getElementById('chatThread');
    const chipBtns = document.querySelectorAll('.chip-btn');

    let chatHistory = [];

    // Reset Chat State on New Video Analysis
    function resetChatState() {
        chatHistory = [];
        chatThread.innerHTML = `
            <div class="chat-msg ai-msg">
                <div class="msg-avatar">⚡</div>
                <div class="msg-bubble">
                    I have loaded this video report into memory context! Ask me any follow-up questions, like <em>"Draft a video prompt for scene 2"</em>, <em>"Elaborate on audio cues"</em>, or <em>"Explain what happens at 0:10"</em>.
                </div>
            </div>
        `;
    }

    // Display Results
    function displayResults(data) {
        loadingState.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        statusBadge.textContent = 'Analysis Complete';
        statusBadge.style.color = '#34d399';

        // Render Keyframe Thumbnails
        keyframeQty.textContent = data.keyframe_count;
        keyframeGrid.innerHTML = '';

        data.keyframes.forEach((url, i) => {
            const card = document.createElement('div');
            card.className = 'keyframe-card';
            card.innerHTML = `<img src="${url}" alt="Keyframe ${i+1}" title="Keyframe ${i+1}">`;
            keyframeGrid.appendChild(card);
        });

        // Render Markdown Report
        reportModelTag.textContent = `Model: ${data.model_used}`;
        currentReportText = data.analysis;

        if (window.marked) {
            reportMarkdown.innerHTML = marked.parse(data.analysis);
        } else {
            reportMarkdown.textContent = data.analysis;
        }

        resetChatState();
    }

    // Copy Report Button
    copyReportBtn.addEventListener('click', () => {
        if (!currentReportText) return;
        navigator.clipboard.writeText(currentReportText);
        copyReportBtn.textContent = '✓ Copied!';
        setTimeout(() => {
            copyReportBtn.textContent = '📋 Copy Report';
        }, 2000);
    });

    // 4. Chat Follow-up Q&A Handler
    async function sendFollowupQuestion(questionText) {
        if (!questionText || !currentUploadedFilename || !currentReportText) return;

        // Add User Message Bubble
        appendChatMessage('user', questionText);
        chatInput.value = '';
        sendChatBtn.disabled = true;

        // Add Loading AI Bubble
        const loadingMsgId = appendChatMessage('ai', '<em>Thinking with video memory context... ⚡</em>');

        try {
            const res = await fetch('/api/chat-followup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: currentUploadedFilename,
                    question: questionText,
                    initial_analysis: currentReportText,
                    history: chatHistory
                })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Follow-up query failed');

            // Update Loading AI Bubble with final response
            updateChatMessage(loadingMsgId, data.reply);

            // Update local dialogue memory history
            chatHistory.push({ role: 'user', content: questionText });
            chatHistory.push({ role: 'assistant', content: data.reply });

        } catch (err) {
            updateChatMessage(loadingMsgId, `<span style="color: #f87171">❌ Error: ${err.message}</span>`);
        } finally {
            sendChatBtn.disabled = false;
        }
    }

    sendChatBtn.addEventListener('click', () => sendFollowupQuestion(chatInput.value.trim()));
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendFollowupQuestion(chatInput.value.trim());
        }
    });

    chipBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const q = btn.getAttribute('data-question');
            if (q) sendFollowupQuestion(q);
        });
    });

    function appendChatMessage(sender, text) {
        const msgDiv = document.createElement('div');
        const id = 'msg-' + Date.now();
        msgDiv.id = id;
        msgDiv.className = `chat-msg ${sender}-msg`;
        msgDiv.innerHTML = `
            <div class="msg-avatar">${sender === 'user' ? '👤' : '⚡'}</div>
            <div class="msg-bubble">${window.marked && sender === 'ai' ? marked.parse(text) : text}</div>
        `;
        chatThread.appendChild(msgDiv);
        chatThread.scrollTop = chatThread.scrollHeight;
        return id;
    }

    function updateChatMessage(msgId, text) {
        const msgDiv = document.getElementById(msgId);
        if (msgDiv) {
            const bubble = msgDiv.querySelector('.msg-bubble');
            if (bubble) {
                bubble.innerHTML = window.marked ? marked.parse(text) : text;
            }
        }
        chatThread.scrollTop = chatThread.scrollHeight;
    }
});

