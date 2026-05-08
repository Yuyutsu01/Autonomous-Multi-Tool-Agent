import './style.css';

const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const traceLogs = document.getElementById('trace-logs');
const statusText = document.getElementById('status-text');
const statusDot = document.querySelector('.dot');

let socket = null;

function connectWebSocket() {
  socket = new WebSocket('ws://localhost:8000/ws/chat');

  socket.onopen = () => {
    statusText.textContent = 'Connected';
    statusDot.classList.add('connected');
  };

  socket.onclose = () => {
    statusText.textContent = 'Disconnected';
    statusDot.classList.remove('connected');
    setTimeout(connectWebSocket, 3000); // Reconnect attempt
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleSocketMessage(data);
  };
}

function handleSocketMessage(data) {
  if (data.type === 'status') {
    updateTraceStatus(data.message);
  } else if (data.type === 'plan') {
    clearTraceLogs();
    data.plan.forEach((stepDesc, index) => {
      createTraceItem(index, stepDesc);
    });
  } else if (data.type === 'step_start') {
    updateTraceItemStatus(data.step_index, 'running', 'Running...');
  } else if (data.type === 'step_validating') {
    updateTraceItemStatus(data.step_index, 'running', 'Validating...');
  } else if (data.type === 'step_retry') {
    updateTraceItemStatus(data.step_index, 'error', `Retry: ${data.reason}`);
  } else if (data.type === 'step_complete') {
    updateTraceItemStatus(data.step_index, 'success', 'Complete');
    updateTraceItemOutput(data.step_index, data.output);
  } else if (data.type === 'result') {
    appendMessage('agent', data.output);
  } else if (data.type === 'error') {
    appendMessage('agent', `Error: ${data.message}`);
  }
}

function appendMessage(role, content) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'Agent';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;

  msgDiv.appendChild(label);
  msgDiv.appendChild(contentDiv);
  chatHistory.appendChild(msgDiv);
  
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function clearTraceLogs() {
  traceLogs.innerHTML = '';
}

function updateTraceStatus(msg) {
  if (traceLogs.innerHTML.includes('empty-trace')) {
    clearTraceLogs();
  }
  const el = document.createElement('div');
  el.className = 'trace-item-desc';
  el.style.color = 'var(--text-muted)';
  el.textContent = `> ${msg}`;
  traceLogs.appendChild(el);
  traceLogs.scrollTop = traceLogs.scrollHeight;
}

function createTraceItem(index, description) {
  const item = document.createElement('div');
  item.className = 'trace-item';
  item.id = `trace-${index}`;

  const header = document.createElement('div');
  header.className = 'trace-item-header';

  const title = document.createElement('span');
  title.textContent = `Step ${index + 1}`;

  const status = document.createElement('span');
  status.className = 'trace-item-status';
  status.id = `trace-status-${index}`;
  status.textContent = 'Pending';

  header.appendChild(title);
  header.appendChild(status);

  const desc = document.createElement('div');
  desc.className = 'trace-item-desc';
  desc.textContent = description;

  const output = document.createElement('div');
  output.className = 'trace-item-output';
  output.id = `trace-output-${index}`;
  output.style.display = 'none';

  item.appendChild(header);
  item.appendChild(desc);
  item.appendChild(output);

  traceLogs.appendChild(item);
}

function updateTraceItemStatus(index, statusClass, statusText) {
  const statusEl = document.getElementById(`trace-status-${index}`);
  if (statusEl) {
    statusEl.className = `trace-item-status ${statusClass}`;
    statusEl.textContent = statusText;
  }
}

function updateTraceItemOutput(index, text) {
  const outputEl = document.getElementById(`trace-output-${index}`);
  if (outputEl) {
    outputEl.textContent = text;
    outputEl.style.display = 'block';
  }
  traceLogs.scrollTop = traceLogs.scrollHeight;
}

chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  appendMessage('user', text);
  chatInput.value = '';
  
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ request: text }));
    clearTraceLogs();
  } else {
    appendMessage('agent', 'Error: Not connected to backend.');
  }
});

// Init
connectWebSocket();
