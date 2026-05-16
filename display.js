const app = element.querySelector('#chip-ocr-app');
const video = app.querySelector('[data-role="video"]');
const canvas = app.querySelector('[data-role="canvas"]');
const statusEl = app.querySelector('[data-role="status"]');
const summaryEl = app.querySelector('[data-role="summary"]');
const resultsEl = app.querySelector('[data-role="results"]');
const historyEl = app.querySelector('[data-role="history"]');
const historySummaryEl = app.querySelector('[data-role="history-summary"]');
const historyEmpty = app.querySelector('[data-role="history-empty"]');
const annotatedEl = app.querySelector('[data-role="annotated"]');
const cameraEmpty = app.querySelector('[data-role="camera-empty"]');
const previewEmpty = app.querySelector('[data-role="preview-empty"]');
const cameraButton = app.querySelector('[data-role="camera-button"]');
const startButton = app.querySelector('[data-role="start-button"]');
const pauseButton = app.querySelector('[data-role="pause-button"]');
const stopButton = app.querySelector('[data-role="stop-button"]');
const constraints = __WEBCAM_CONSTRAINTS__;
const intervalMs = __CAPTURE_INTERVAL_MS__;
const jpegQuality = __JPEG_QUALITY__;
const maxHistoryRecords = 20;
const historyDedupeMs = 3000;
let stream = null;
let recognizeTimer = null;
let recognizing = false;
let requestInFlight = false;
let historyRecordCount = 0;
let lastHistorySignature = '';
let lastHistoryAt = 0;

function setStatus(message) {
    statusEl.textContent = message;
}

function normalizeServerPayload(value, fallback = {}) {
    if (value == null) {
        return {...fallback};
    }
    if (Array.isArray(value)) {
        if (value.length === 0) return {...fallback};
        if (value.length === 1) return normalizeServerPayload(value[0], fallback);
        return {...fallback, data: value};
    }
    if (typeof value === 'object') {
        if ('data' in value) return normalizeServerPayload(value.data, fallback);
        if ('value' in value) return normalizeServerPayload(value.value, fallback);
        return {...fallback, ...value};
    }
    return {...fallback, status: String(value)};
}

async function callServer(fnName, args = [], fallback = {}) {
    if (!server || typeof server[fnName] !== 'function') {
        return {...fallback};
    }
    const value = await server[fnName](...args);
    return normalizeServerPayload(value, fallback);
}

function setPreview(src) {
    if (src) {
        annotatedEl.src = src;
        annotatedEl.classList.remove('is-empty');
        previewEmpty.style.display = 'none';
    } else {
        annotatedEl.removeAttribute('src');
        annotatedEl.classList.add('is-empty');
        previewEmpty.style.display = 'block';
    }
}

function clearResults(message = '暂无结果') {
    summaryEl.textContent = message;
    resultsEl.replaceChildren();
}

function addText(parent, tag, text, className = '') {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    parent.appendChild(node);
    return node;
}

function matchedModels(payload) {
    const models = [];
    for (const chip of payload.chips || []) {
        const partNumber = chip.match && chip.match.part_number;
        if (partNumber && !models.includes(partNumber)) {
            models.push(partNumber);
        }
    }
    return models;
}

function formatRecordTime(date) {
    return date.toLocaleString('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });
}

function updateHistorySummary() {
    const count = historyEl.querySelectorAll('.history-record').length;
    historySummaryEl.textContent = `${count} 条记录`;
    if (historyEmpty) {
        historyEmpty.style.display = count ? 'none' : 'block';
    }
}

function appendMatchedHistory(payload) {
    const models = matchedModels(payload);
    if (!models.length || !payload.annotated_image) return;

    const now = Date.now();
    const signature = models.join('|');
    if (signature === lastHistorySignature && now - lastHistoryAt < historyDedupeMs) {
        return;
    }
    lastHistorySignature = signature;
    lastHistoryAt = now;

    const record = document.createElement('article');
    record.className = 'history-record';
    record.dataset.historyId = String(++historyRecordCount);

    const image = document.createElement('img');
    image.className = 'history-thumb';
    image.alt = '识别框画面';
    image.src = payload.annotated_image;
    record.appendChild(image);

    const modelWrap = document.createElement('div');
    modelWrap.className = 'history-models';
    for (const model of models) {
        addText(modelWrap, 'span', model, 'model-pill');
    }
    record.appendChild(modelWrap);

    addText(record, 'time', formatRecordTime(new Date(now)), 'history-time');
    historyEl.prepend(record);

    const records = historyEl.querySelectorAll('.history-record');
    for (let index = maxHistoryRecords; index < records.length; index += 1) {
        records[index].remove();
    }
    updateHistorySummary();
}

function renderResults(payload) {
    payload = normalizeServerPayload(payload, {
        status: '未收到识别结果',
        summary: '暂无结果',
        chips: [],
        annotated_image: null,
    });
    if (payload.annotated_image) setPreview(payload.annotated_image);
    if (payload.status) setStatus(payload.status);
    summaryEl.textContent = payload.summary || '暂无结果';
    resultsEl.replaceChildren();

    for (const chip of payload.chips || []) {
        const card = document.createElement('article');
        card.className = 'chip-card';
        addText(card, 'h3', `芯片 ${chip.index}`);

        const match = chip.match || {};
        if (match.part_number) {
            addText(card, 'div', `匹配型号：${match.part_number}`, 'match');
            if (match.description) addText(card, 'div', match.description, 'muted');
        } else {
            addText(card, 'div', '未匹配到型号', 'muted');
        }

        if (chip.error) {
            addText(card, 'div', chip.error, 'muted');
        } else if (chip.texts && chip.texts.length) {
            for (const item of chip.texts) {
                const score = item.score == null ? '' : `  (${Number(item.score).toFixed(4)})`;
                addText(card, 'div', `${item.text}${score}`, 'ocr-line');
            }
        } else {
            addText(card, 'div', '未识别到文本', 'muted');
        }

        if (match.normalized_text) {
            addText(card, 'div', `清洗文本：${match.normalized_text}`, 'muted');
        }
        resultsEl.appendChild(card);
    }
    appendMatchedHistory(payload);
}

async function startCamera() {
    if (stream) return;
    cameraButton.disabled = true;
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        await video.play();
        cameraEmpty.style.display = 'none';
        setStatus('摄像头原始流已启动');
    } catch (error) {
        stream = null;
        setStatus(`摄像头启动失败：${error.message || error}`);
    } finally {
        cameraButton.disabled = false;
    }
}

function stopRecognitionLoop() {
    recognizing = false;
    if (recognizeTimer) {
        window.clearInterval(recognizeTimer);
        recognizeTimer = null;
    }
}

async function captureAndRecognize() {
    if (!recognizing || requestInFlight || !video.videoWidth || !video.videoHeight) {
        return;
    }

    requestInFlight = true;
    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL('image/jpeg', jpegQuality);
        const payload = await callServer(
            'recognize_multi_chip_frame',
            [frame, recognizing],
            {
                status: '未收到识别结果',
                summary: '暂无结果',
                chips: [],
                annotated_image: null,
            }
        );
        renderResults(payload);
    } catch (error) {
        setStatus(`识别失败：${error.message || error}`);
    } finally {
        requestInFlight = false;
    }
}

async function startRecognition() {
    startButton.disabled = true;
    try {
        await startCamera();
        const service = await callServer(
            'start_recognition_service',
            [],
            {status: 'OCR 服务接口未返回状态'}
        );
        if (service.status) setStatus(service.status);
        recognizing = true;
        if (!recognizeTimer) {
            recognizeTimer = window.setInterval(captureAndRecognize, intervalMs);
        }
        await captureAndRecognize();
    } catch (error) {
        setStatus(`启动失败：${error.message || error}`);
    } finally {
        startButton.disabled = false;
    }
}

function pauseRecognition() {
    stopRecognitionLoop();
    setStatus('实时识别已暂停');
}

async function stopService() {
    stopRecognitionLoop();
    stopButton.disabled = true;
    try {
        const payload = await callServer(
            'stop_ocr_service_for_ui',
            [],
            {status: 'OCR 服务已停止'}
        );
        setStatus(payload.status || 'OCR 服务已停止');
    } catch (error) {
        setStatus(`停止失败：${error.message || error}`);
    } finally {
        stopButton.disabled = false;
    }
}

cameraButton.addEventListener('click', startCamera);
startButton.addEventListener('click', startRecognition);
pauseButton.addEventListener('click', pauseRecognition);
stopButton.addEventListener('click', stopService);

setStatus('OCR 服务未启动，点击“开启实时识别”后启动');

