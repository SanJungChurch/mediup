// === 화면 관리 ===
const screens = {
    A: document.getElementById("screenA"),
    B: document.getElementById("screenB"),
    C: document.getElementById("screenC"),
    D: document.getElementById("screenD"),
    E: document.getElementById("screenE"),
    F: document.getElementById("screenF"),
    G: document.getElementById("screenG"),
    H: document.getElementById("screenH"),
    I: document.getElementById("screenI"),
    J: document.getElementById("screenJ")
};

const buttons = {
    btnA: document.getElementById("pushButtonContainer"),
    btnB: document.getElementById("pushButtonContainerB"),
    btnC: document.getElementById("pushButtonContainerC"),
    btnD: document.getElementById("pushButtonContainerD"),
    btnE: document.getElementById("pushButtonContainerE"),
    btnF: document.getElementById("pushButtonContainerF"),
    btnF_Reject: document.getElementById("pushButtonRejectF"),
    btnG: document.getElementById("pushButtonContainerG"),
    btnStop: document.getElementById("pushButtonStop"),
    btnStats: document.getElementById("text1"),
    btnBack: document.getElementById("pushButtonContainerI"),
    btnBackToMonitor: document.getElementById("btnBackToMonitor"),
    btnNewChat: document.getElementById("btnNewChat"),
    btnAnalyze: document.getElementById("btnAnalyze"),
    btnSend: document.getElementById("btnSend")
};

// === WebSocket 상태 ===
let ws = null;
let detectEnabled = true;
let latestFeatures = null;
let latestIndices = null;
let cumulativeStats = null;

// 통계 차트 인스턴스
let statsCharts = {};

// 히스토리 데이터 저장
let historyData = {
    perclos: [],
    headpose: [],
    fatigue: [],
    stress: [],
    yawnRate: [],
    gaze: [],
    near: [],
    timestamps: []
};

// 대화 히스토리
let conversationHistory = [];

// === 유틸리티 ===
function switchScreen(currentScreen, nextScreen) {
    if (currentScreen) currentScreen.style.display = 'none';
    if (nextScreen) nextScreen.style.display = 'block';
}

function fmt(v, digits=2){
    if(v===undefined || v===null || Number.isNaN(v)) return '—';
    return (typeof v==='number')? v.toFixed(digits) : String(v);
}

// === WebSocket 연결 ===
function openWS(){
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => { console.log('WS 연결됨'); };
    ws.onclose = () => { console.log('WS 종료'); setTimeout(openWS, 1500); };
    ws.onerror = (e) => { console.warn('WS 에러', e); };

    ws.onmessage = (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            
            // 데이터 저장
            latestFeatures = msg.features || {};
            latestIndices = msg.indices || {};
            cumulativeStats = msg.cumulative || {};

            // 히스토리 데이터 저장 (최근 100개)
            if (msg.features && msg.indices) {
                historyData.perclos.push(latestFeatures.perclos || 0);
                historyData.headpose.push(latestFeatures.headpose_var || 0);
                historyData.fatigue.push(latestIndices.fatigue || 0);
                historyData.stress.push(latestIndices.stress || 0);
                historyData.yawnRate.push(latestFeatures.yawn_rate_min || 0);
                historyData.gaze.push(latestFeatures.gaze_on_pct || 0);
                historyData.near.push(latestFeatures.near_work || 0);
                historyData.timestamps.push(new Date().toLocaleTimeString());
                
                // 100개 제한
                const maxHistory = 100;
                if (historyData.perclos.length > maxHistory) {
                    Object.keys(historyData).forEach(key => {
                        historyData[key].shift();
                    });
                }
            }

            // Screen H에서 UI 업데이트
            if (screens.H && screens.H.style.display !== 'none') {
                updateDashboardUI(msg);
            }

            // 버튼 상태 동기화
            detectEnabled = !!msg.detect_enabled;
        } catch(e){
            console.warn('WS 파싱 실패', e);
        }
    };
}

// === Dashboard UI 업데이트 ===
function updateDashboardUI(msg) {
    const features = msg.features || {};
    const indices = msg.indices || {};
    const cumulative = msg.cumulative || {};
    
    // 지표 & 지수
    const elFatigue = document.getElementById("val-fatigue");
    const elStress = document.getElementById("val-stress");
    if (elFatigue) elFatigue.innerText = fmt(indices.fatigue, 1);
    if (elStress) elStress.innerText = fmt(indices.stress, 1);
    
    // 상세 분석
    const elPerclos = document.getElementById("val-perclos");
    const elYawnRate = document.getElementById("val-yawn-rate");
    const elPosture = document.getElementById("val-posture");
    const elHeadpose = document.getElementById("val-headpose");
    const elGaze = document.getElementById("val-gaze");
    const elNear = document.getElementById("val-near");
    
    if (elPerclos) elPerclos.innerText = fmt(features.perclos, 3);
    if (elYawnRate) elYawnRate.innerText = fmt(features.yawn_rate_min, 2);
    if (elPosture) elPosture.innerText = fmt(features.posture_angle_norm, 2);
    if (elHeadpose) elHeadpose.innerText = fmt(features.headpose_var, 2);
    if (elGaze) elGaze.innerText = fmt(features.gaze_on_pct, 2);
    if (elNear) elNear.innerText = fmt(features.near_work, 2);
    
    // 상태 감지 (누적)
    const elBlink = document.getElementById("val-blink");
    const elYawn = document.getElementById("val-yawn");
    const elNodding = document.getElementById("val-nodding");
    
    if (elBlink) elBlink.innerText = cumulative.blink_count || 0;
    if (elYawn) elYawn.innerText = cumulative.yawn_count || 0;
    if (elNodding) elNodding.innerText = cumulative.nodding_count || 0;
    
    // 비디오 프리뷰 (base64)
    if (msg.frame_b64) {
        const videoEl = document.getElementById('webcam');
        if (videoEl && videoEl.tagName === 'VIDEO') {
            // video를 img로 교체
            const container = videoEl.parentElement;
            let img = container.querySelector('img.preview-img');
            if (!img) {
                img = document.createElement('img');
                img.className = 'preview-img';
                img.style.width = '100%';
                img.style.height = '100%';
                img.style.objectFit = 'contain';
                container.appendChild(img);
                videoEl.style.display = 'none';
            }
            img.src = `data:image/jpeg;base64,${msg.frame_b64}`;
        }
    }
}

// === 리포트 요청 ===
async function requestReport() {
    const adviceBox = document.getElementById('adviceBox');
    const btnAdvice = document.getElementById('btnGetAdvice');
    
    if (!latestIndices || !cumulativeStats) {
        if (adviceBox) {
            adviceBox.innerHTML = '<p style="color: #ffd166;">⏳ 데이터 수집 중... 잠시 후 다시 시도해주세요.</p>';
        }
        return;
    }
    
    try {
        // 버튼 비활성화 및 로딩 표시
        if (btnAdvice) {
            btnAdvice.disabled = true;
            btnAdvice.innerText = '⏳ 생성 중...';
        }
        if (adviceBox) {
            adviceBox.innerHTML = '<p style="color: #ffd166;">🤖 AI가 분석 중입니다...</p>';
        }
        
        const stats = {
            avg_fatigue: (latestIndices?.fatigue ?? 0) * 1.0,
            avg_stress: (latestIndices?.stress ?? 0) * 1.0,
            perclos: (latestFeatures?.perclos ?? 0),
            blink_count: cumulativeStats.blink_count || 0,
            yawn_count: cumulativeStats.yawn_count || 0
        };
        const docs = [{ title: '세션 로그', path: 'local' }];

        const res = await fetch('/report', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({stats, docs})
        });
        
        if (!res.ok) {
            throw new Error(`요청 실패 (${res.status})`);
        }
        
        const data = await res.json();
        
        // 조언 박스에 LLM 결과 표시
        if (adviceBox) {
            adviceBox.innerHTML = `<p style="white-space: pre-wrap; font-size: 13px; line-height: 1.6; color: #eaeaea;">${data.text || '결과 없음'}</p>`;
        }
        
        // 알림 표시
        if ("Notification" in window && Notification.permission === "granted") {
            const firstLine = data.text.split('\n')[0] || "새로운 조언이 도착했습니다!";
            new Notification("💡 코칭 조언", {
                body: firstLine,
                icon: "/static/user_icon_placeholder.png"
            });
        }
        
        console.log('✅ 리포트 생성 완료:', data.text);
        
    } catch(err) {
        console.error('❌ 리포트 실패:', err);
        if (adviceBox) {
            adviceBox.innerHTML = `<p style="color: #ff6b6b;">⚠️ 생성 실패: ${err.message}</p>`;
        }
    } finally {
        // 버튼 다시 활성화
        if (btnAdvice) {
            btnAdvice.disabled = false;
            btnAdvice.innerText = '💡 조언 받기';
        }
    }
}

// === 화면 전환 이벤트 ===
window.onload = function() {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('reset') === 'true') {
        localStorage.removeItem("onboardingComplete");
        window.history.replaceState({}, document.title, window.location.pathname);
    }
    if (localStorage.getItem("onboardingComplete") === "true") {
        screens.A.style.display = 'none';
        screens.G.style.display = 'block';
    }
};

if (buttons.btnA) buttons.btnA.addEventListener("click", () => switchScreen(screens.A, screens.B));
if (buttons.btnB) buttons.btnB.addEventListener("click", () => switchScreen(screens.B, screens.C));
if (buttons.btnC) buttons.btnC.addEventListener("click", () => switchScreen(screens.C, screens.D));

if (buttons.btnD) {
    buttons.btnD.addEventListener("click", () => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
                s.getTracks().forEach(t => t.stop());
                switchScreen(screens.D, screens.E);
            }).catch(() => switchScreen(screens.D, screens.E));
        } else { switchScreen(screens.D, screens.E); }
    });
}

if (buttons.btnE) buttons.btnE.addEventListener("click", () => switchScreen(screens.E, screens.F));

if (buttons.btnF) {
    buttons.btnF.addEventListener("click", () => {
        if ("Notification" in window) Notification.requestPermission();
        localStorage.setItem("onboardingComplete", "true");
        switchScreen(screens.F, screens.G);
    });
}

if (buttons.btnF_Reject) {
    buttons.btnF_Reject.addEventListener("click", () => {
        localStorage.setItem("onboardingComplete", "true");
        switchScreen(screens.F, screens.G);
    });
}

// START 버튼 (G -> H)
if (buttons.btnG) {
    buttons.btnG.addEventListener("click", () => {
        switchScreen(screens.G, screens.H);
        openWS(); // WebSocket 연결
    });
}

// 조언 받기 버튼 → 대화 화면으로 전환
const btnGetAdvice = document.getElementById('btnGetAdvice');
if (btnGetAdvice) {
    btnGetAdvice.addEventListener('click', () => {
        switchScreen(screens.H, screens.J);
    });
}

// STOP 버튼 (H -> G)
if (buttons.btnStop) {
    buttons.btnStop.addEventListener("click", () => {
        if (ws) {
            ws.close();
            ws = null;
        }
        switchScreen(screens.H, screens.G);
    });
}

// 통계 보기 (G -> I)
if (buttons.btnStats) {
    buttons.btnStats.addEventListener("click", () => {
        switchScreen(screens.G, screens.I);
        console.log("누적 통계:", cumulativeStats);
        console.log("히스토리 데이터:", historyData);
        
        // 차트 생성
        setTimeout(() => {
            createStatsCharts();
        }, 100);
    });
}

// Back (I -> G)
if (buttons.btnBack) {
    buttons.btnBack.addEventListener("click", () => {
        switchScreen(screens.I, screens.G);
        
        // 차트 정리
        Object.values(statsCharts).forEach(chart => {
            if (chart) chart.destroy();
        });
        statsCharts = {};
    });
}

// === 통계 차트 생성 ===
function createStatsCharts() {
    // 기존 차트 정리
    Object.values(statsCharts).forEach(chart => {
        if (chart) chart.destroy();
    });
    statsCharts = {};
    
    const maxPoints = 50;
    const recentData = (arr) => arr.slice(-maxPoints);
    const labels = recentData(historyData.timestamps).map((_, i) => i + 1);
    
    const chartConfig = (label, data, color, yMax = null) => ({
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: recentData(data),
                borderColor: color,
                backgroundColor: color + '20',
                tension: 0.3,
                fill: true,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: { 
                    display: true,
                    max: yMax,
                    ticks: { color: '#9aa0a6', font: { size: 10 } }
                }
            }
        }
    });
    
    // 차트 생성
    const createChart = (id, config) => {
        const canvas = document.getElementById(id);
        if (canvas) {
            statsCharts[id] = new Chart(canvas, config);
        }
    };
    
    createChart('chart-perclos', chartConfig('PERCLOS', historyData.perclos, '#75B9E6', 1));
    createChart('chart-headpose', chartConfig('Headpose Var', historyData.headpose, '#9D7FE8'));
    createChart('chart-fatigue', chartConfig('피로도', historyData.fatigue, '#FF6B6B', 100));
    createChart('chart-stress', chartConfig('스트레스', historyData.stress, '#FFA07A', 100));
    createChart('chart-yawn-rate', chartConfig('하품/분', historyData.yawnRate, '#FFD700'));
    createChart('chart-gaze', chartConfig('시선온', historyData.gaze, '#90EE90', 1));
    createChart('chart-near', chartConfig('근거리작업', historyData.near, '#DDA0DD'));
    
    // 누적 카운트 표시 (숫자로)
    if (cumulativeStats) {
        const displayCount = (id, value) => {
            const canvas = document.getElementById(id);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
                ctx.fillStyle = '#eaeaea';
                ctx.font = 'bold 48px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(value || 0, canvas.width / 2, canvas.height / 2);
                ctx.font = '14px Inter';
                ctx.fillStyle = '#9aa0a6';
                ctx.fillText('회', canvas.width / 2, canvas.height / 2 + 35);
            }
        };
        
        displayCount('chart-blink', cumulativeStats.blink_count);
        displayCount('chart-yawn', cumulativeStats.yawn_count);
        displayCount('chart-nodding', cumulativeStats.nodding_count);
    }
}

// === 대화 기능 ===
function addMessage(role, content) {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;
    
    // 웰컴 메시지 제거
    const welcomeMsg = messagesDiv.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = role === 'user' ? '👤 나' : '🤖 AI 코치';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    
    // 스크롤 아래로
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator';
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = '🤖 AI 코치';
    
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'typing-indicator';
    indicatorDiv.innerHTML = '<span></span><span></span><span></span>';
    
    typingDiv.appendChild(headerDiv);
    typingDiv.appendChild(indicatorDiv);
    messagesDiv.appendChild(typingDiv);
    
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

async function sendMessage(userMessage, isAnalysis = false) {
    const btnSend = document.getElementById('btnSend');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const chatInput = document.getElementById('chatInput');
    
    try {
        // 버튼 비활성화
        if (btnSend) btnSend.disabled = true;
        if (btnAnalyze) btnAnalyze.disabled = true;
        if (chatInput) chatInput.disabled = true;
        
        // 사용자 메시지 표시
        if (userMessage) {
            addMessage('user', userMessage);
            conversationHistory.push({
                role: 'user',
                content: userMessage
            });
        }
        
        // 타이핑 인디케이터 표시
        showTypingIndicator();
        
        // API 요청 준비
        let requestBody;
        
        if (isAnalysis) {
            // 상태 분석 요청
            requestBody = {
                stats: {
                    avg_fatigue: (latestIndices?.fatigue ?? 0) * 1.0,
                    avg_stress: (latestIndices?.stress ?? 0) * 1.0,
                    perclos: (latestFeatures?.perclos ?? 0),
                    blink_count: cumulativeStats?.blink_count || 0,
                    yawn_count: cumulativeStats?.yawn_count || 0
                },
                docs: [{ title: '세션 로그', path: 'local' }],
                conversation_history: conversationHistory
            };
        } else {
            // 일반 대화 요청
            requestBody = {
                stats: {
                    avg_fatigue: (latestIndices?.fatigue ?? 0) * 1.0,
                    avg_stress: (latestIndices?.stress ?? 0) * 1.0,
                    perclos: (latestFeatures?.perclos ?? 0),
                    blink_count: cumulativeStats?.blink_count || 0,
                    yawn_count: cumulativeStats?.yawn_count || 0
                },
                docs: [],
                conversation_history: conversationHistory,
                user_message: userMessage
            };
        }
        
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestBody)
        });
        
        if (!res.ok) {
            throw new Error(`요청 실패 (${res.status})`);
        }
        
        const data = await res.json();
        
        // 타이핑 인디케이터 제거
        removeTypingIndicator();
        
        // AI 응답 표시
        addMessage('assistant', data.text || '응답을 생성할 수 없습니다.');
        
        // 대화 히스토리에 추가
        conversationHistory.push({
            role: 'assistant',
            content: data.text || ''
        });
        
        console.log('✅ 응답 완료:', data.text);
        
    } catch(err) {
        console.error('❌ 요청 실패:', err);
        removeTypingIndicator();
        addMessage('assistant', `⚠️ 오류가 발생했습니다: ${err.message}`);
    } finally {
        // 버튼 다시 활성화
        if (btnSend) btnSend.disabled = false;
        if (btnAnalyze) btnAnalyze.disabled = false;
        if (chatInput) {
            chatInput.disabled = false;
            chatInput.value = '';
            chatInput.focus();
        }
    }
}

function clearChat() {
    conversationHistory = [];
    const messagesDiv = document.getElementById('chatMessages');
    if (messagesDiv) {
        messagesDiv.innerHTML = `
            <div class="welcome-message">
                <p>안녕하세요! 👋</p>
                <p>저는 여러분의 디지털 웰빙을 도와드리는 AI 코치입니다.</p>
                <p>아래 버튼을 눌러 현재 상태 분석을 시작하거나, 궁금한 점을 물어보세요!</p>
            </div>
        `;
    }
}

// === 대화 화면 이벤트 ===
// 모니터링으로 돌아가기
if (buttons.btnBackToMonitor) {
    buttons.btnBackToMonitor.addEventListener('click', () => {
        switchScreen(screens.J, screens.H);
    });
}

// 새 대화 시작
if (buttons.btnNewChat) {
    buttons.btnNewChat.addEventListener('click', () => {
        if (confirm('현재 대화를 초기화하시겠습니까?')) {
            clearChat();
        }
    });
}

// 상태 분석 버튼
if (buttons.btnAnalyze) {
    buttons.btnAnalyze.addEventListener('click', () => {
        sendMessage('현재 나의 상태를 분석해주세요.', true);
    });
}

// 메시지 전송 버튼
if (buttons.btnSend) {
    buttons.btnSend.addEventListener('click', () => {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput?.value?.trim();
        if (message) {
            sendMessage(message, false);
        }
    });
}

// Enter 키로 전송
const chatInput = document.getElementById('chatInput');
if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (message) {
                sendMessage(message, false);
            }
        }
    });
}
