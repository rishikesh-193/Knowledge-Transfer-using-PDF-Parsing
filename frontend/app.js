// Production Render backend base URL with local development fallback
const API_BASE_URL = (
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1' ||
  window.location.hostname === ''
)
  ? 'http://127.0.0.1:8000'
  : 'https://knowledge-transfer-using-pdf-parsing.onrender.com';

// DOM Element References
const backendStatusText = document.getElementById('backend-status-text');
const backendStatusDot = document.querySelector('#backend-status .status-dot');
const backendUrlLink = document.getElementById('backend-url-link');

const dropZone = document.getElementById('drop-zone');
const pdfInput = document.getElementById('pdf-input');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadBtn = document.getElementById('upload-btn');
const uploadSpinner = document.getElementById('upload-spinner');
const uploadAlert = document.getElementById('upload-alert');

const indexedSummary = document.getElementById('indexed-summary');
const summaryFilename = document.getElementById('summary-filename');
const summaryPages = document.getElementById('summary-pages');
const summaryChunks = document.getElementById('summary-chunks');

const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const askSpinner = document.getElementById('ask-spinner');
const qaAlert = document.getElementById('qa-alert');

const answerContainer = document.getElementById('answer-container');
const answerText = document.getElementById('answer-text');
const sourcesList = document.getElementById('sources-list');

let selectedFile = null;

// Initialize Footer & Health Check
if (backendUrlLink) {
  backendUrlLink.href = API_BASE_URL;
  backendUrlLink.textContent = API_BASE_URL;
}

checkBackendHealth();

// 1. Health Check Function
async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (response.ok) {
      const data = await response.json();
      if (data.status === 'ok') {
        backendStatusText.textContent = 'Render Backend Online';
        backendStatusDot.classList.add('online');
        backendStatusDot.classList.remove('offline');
        return;
      }
    }
    throw new Error('Health check returned non-ok status');
  } catch (err) {
    backendStatusText.textContent = 'Backend Offline / Reconnecting...';
    backendStatusDot.classList.remove('online');
    backendStatusDot.classList.add('offline');
  }
}

// 2. Drag & Drop File Event Listeners
dropZone.addEventListener('click', () => pdfInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

pdfInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  hideAlert(uploadAlert);
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showAlert(uploadAlert, 'Error: Invalid file type. Please select a .pdf file.', 'error');
    selectedFile = null;
    fileNameDisplay.textContent = 'No file selected';
    uploadBtn.disabled = true;
    return;
  }

  selectedFile = file;
  fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  uploadBtn.disabled = false;
}

// 3. Upload & Index PDF Handler
uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  hideAlert(uploadAlert);
  setLoading(uploadBtn, uploadSpinner, true, 'Uploading & Indexing...');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data.detail || 'Failed to process PDF upload.';
      throw new Error(errorMsg);
    }

    // Display Success Summary
    showAlert(uploadAlert, `Successfully indexed "${data.filename}"!`, 'success');
    summaryFilename.textContent = data.filename;
    summaryPages.textContent = data.pages;
    summaryChunks.textContent = data.chunks_indexed;
    indexedSummary.classList.remove('hidden');

    checkBackendHealth();

  } catch (err) {
    showAlert(uploadAlert, `Upload Error: ${err.message}`, 'error');
  } finally {
    setLoading(uploadBtn, uploadSpinner, false, 'Upload & Index PDF');
  }
});

// 4. Grounded Question Answering Handler
askBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  hideAlert(qaAlert);
  answerContainer.classList.add('hidden');

  if (!question) {
    showAlert(qaAlert, 'Please enter a valid question before asking.', 'error');
    return;
  }

  setLoading(askBtn, askSpinner, true, 'Querying RAG Agent...');

  try {
    const response = await fetch(`${API_BASE_URL}/rag/invoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        input: { question: question }
      })
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = data.detail || 'Failed to query RAG agent.';
      throw new Error(errorMsg);
    }

    const output = data.output || {};
    const answer = output.answer || 'No response returned from RAG chain.';
    const sources = output.sources || [];

    // Render Answer & Sources
    answerText.textContent = answer;
    renderSources(sources);
    answerContainer.classList.remove('hidden');

  } catch (err) {
    showAlert(qaAlert, `Q&A Error: ${err.message}`, 'error');
  } finally {
    setLoading(askBtn, askSpinner, false, 'Ask Grounded Question');
  }
});

// Helper Functions
function renderSources(sources) {
  sourcesList.innerHTML = '';
  if (!sources || sources.length === 0) {
    const emptyBadge = document.createElement('span');
    emptyBadge.className = 'source-badge';
    emptyBadge.textContent = 'No source pages cited';
    sourcesList.appendChild(emptyBadge);
    return;
  }

  sources.forEach((src) => {
    const badge = document.createElement('span');
    badge.className = 'source-badge';
    badge.textContent = `📄 ${src.source} — Page ${src.page}`;
    sourcesList.appendChild(badge);
  });
}

function setLoading(btn, spinner, isLoading, defaultText) {
  const btnText = btn.querySelector('.btn-text');
  if (isLoading) {
    btn.disabled = true;
    spinner.classList.remove('hidden');
    if (btnText) btnText.textContent = defaultText;
  } else {
    btn.disabled = false;
    spinner.classList.add('hidden');
    if (btnText) btnText.textContent = defaultText;
  }
}

function showAlert(alertEl, message, type) {
  alertEl.textContent = message;
  alertEl.className = `alert alert-${type}`;
  alertEl.classList.remove('hidden');
}

function hideAlert(alertEl) {
  alertEl.classList.add('hidden');
  alertEl.textContent = '';
}
