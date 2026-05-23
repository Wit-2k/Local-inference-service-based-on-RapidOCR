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
const frameDiffWidth = __FRAME_DIFF_WIDTH__;
const frameDiffMeanThreshold = __FRAME_DIFF_MEAN_THRESHOLD__;
const frameDiffPixelThreshold = __FRAME_DIFF_PIXEL_THRESHOLD__;
const frameDiffChangedRatioThreshold = __FRAME_DIFF_CHANGED_RATIO_THRESHOLD__;
const maxHistoryRecords = 20;
const historyDedupeMs = 3000;
const motionCanvas = document.createElement('canvas');
const motionContext = motionCanvas.getContext('2d', {willReadFrequently: true});
let stream = null;
let recognizeTimer = null;
let recognizing = false;
let requestInFlight = false;
let historyRecordCount = 0;
let lastHistorySignature = '';
let lastHistoryAt = 0;
let previousMotionLuma = null;
let hasCompleteRecognitionResult = false;

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

function resetFrameDiffState() {
    previousMotionLuma = null;
    hasCompleteRecognitionResult = false;
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

function formatDurationMs(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '--';
    return `${number.toFixed(2)} ms`;
}

function addHistoryTiming(parent, payload) {
    const timings = payload.timings || {};
    const timingWrap = document.createElement('div');
    timingWrap.className = 'history-timing';

    const summary = document.createElement('div');
    summary.className = 'history-timing-summary';
    addText(summary, 'span', `总耗时 ${formatDurationMs(timings.total_ms)}`);
    addText(summary, 'span', `分割 ${formatDurationMs(timings.segment_ms)}`);
    addText(summary, 'span', `OCR ${formatDurationMs(timings.ocr_ms)}`);
    timingWrap.appendChild(summary);

    const chips = payload.chips || [];
    if (chips.length) {
        const chipTimes = document.createElement('div');
        chipTimes.className = 'history-chip-times';
        for (const chip of chips) {
            const chipTimings = chip.timings || {};
            const match = chip.match || {};
            const model = match.part_number ? ` ${match.part_number}` : '';
            let text = `芯片 ${chip.index}${model}：`;
            if (chip.cached) {
                text += '复用';
                if (chipTimings.cache_age_ms != null) {
                    text += `，缓存 ${formatDurationMs(chipTimings.cache_age_ms)} 前`;
                }
            } else {
                text += `OCR ${formatDurationMs(chipTimings.ocr_wall_ms)}`;
            }
            addText(chipTimes, 'div', text, 'history-chip-time');
        }
        timingWrap.appendChild(chipTimes);
    }

    parent.appendChild(timingWrap);
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

    const details = document.createElement('div');
    details.className = 'history-details';

    const modelWrap = document.createElement('div');
    modelWrap.className = 'history-models';
    for (const model of models) {
        addText(modelWrap, 'span', model, 'model-pill');
    }
    details.appendChild(modelWrap);
    addHistoryTiming(details, payload);
    record.appendChild(details);

    addText(record, 'time', formatRecordTime(new Date(now)), 'history-time');
    historyEl.prepend(record);

    const records = historyEl.querySelectorAll('.history-record');
    for (let index = maxHistoryRecords; index < records.length; index += 1) {
        records[index].remove();
    }
    updateHistorySummary();
}

function canSkipStableFrame(payload) {
    if (!payload || !payload.annotated_image) return false;
    const chips = payload.chips || [];
    if (!chips.length) return false;
    if (!chips.every((chip) => chip.match && chip.match.part_number)) return false;

    const status = payload.status || '';
    return !(
        status.includes('摄像头帧无效') ||
        status.includes('分割失败') ||
        status.includes('OCR 服务不可用') ||
        status.includes('OCR 服务已手动停止') ||
        status.includes('OCR 服务启动失败') ||
        status.includes('未收到识别结果') ||
        status.includes('未完成识别')
    );
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
        if (chip.cached && chip.timings && chip.timings.cache_age_ms != null) {
            addText(card, 'div', `OCR 结果复用：${chip.timings.cache_age_ms} ms 前`, 'muted');
        } else if (chip.timings && chip.timings.ocr_wall_ms != null) {
            addText(card, 'div', `OCR 请求用时：${chip.timings.ocr_wall_ms} ms`, 'muted');
        }
        resultsEl.appendChild(card);
    }
    appendMatchedHistory(payload);
    hasCompleteRecognitionResult = canSkipStableFrame(payload);
}

function sampleFrameDiff() {
    if (!motionContext || !video.videoWidth || !video.videoHeight) {
        return {stable: false, meanDelta: null, changedRatio: null};
    }

    const width = Math.max(1, Math.min(frameDiffWidth, video.videoWidth));
    const height = Math.max(1, Math.round(video.videoHeight * width / video.videoWidth));
    if (motionCanvas.width !== width || motionCanvas.height !== height) {
        motionCanvas.width = width;
        motionCanvas.height = height;
        previousMotionLuma = null;
    }

    motionContext.drawImage(video, 0, 0, width, height);
    const pixels = motionContext.getImageData(0, 0, width, height).data;
    const current = new Uint8Array(width * height);

    for (let source = 0, target = 0; source < pixels.length; source += 4, target += 1) {
        current[target] = Math.round(
            pixels[source] * 0.299 +
            pixels[source + 1] * 0.587 +
            pixels[source + 2] * 0.114
        );
    }

    if (!previousMotionLuma || previousMotionLuma.length !== current.length) {
        previousMotionLuma = current;
        return {stable: false, meanDelta: null, changedRatio: null};
    }

    let totalDelta = 0;
    let changedPixels = 0;
    for (let index = 0; index < current.length; index += 1) {
        const delta = Math.abs(current[index] - previousMotionLuma[index]);
        totalDelta += delta;
        if (delta > frameDiffPixelThreshold) changedPixels += 1;
    }
    previousMotionLuma = current;

    const meanDelta = totalDelta / current.length;
    const changedRatio = changedPixels / current.length;
    return {
        stable: meanDelta < frameDiffMeanThreshold &&
            changedRatio < frameDiffChangedRatioThreshold,
        meanDelta,
        changedRatio,
    };
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
        const frameDiff = sampleFrameDiff();
        if (hasCompleteRecognitionResult && frameDiff.stable) {
            setStatus('画面稳定，全部芯片已识别，跳过识别（沿用上次结果）');
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL('image/jpeg', jpegQuality);
        const payload = await callServer(
            'recognize_multi_chip_frame',
            [frame, recognizing, Boolean(frameDiff.stable)],
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
        resetFrameDiffState();
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
    resetFrameDiffState();
    setStatus('实时识别已暂停');
}

async function stopService() {
    stopRecognitionLoop();
    resetFrameDiffState();
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

