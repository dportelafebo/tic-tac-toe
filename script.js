// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────
const GAMMA  = 0.9;
const EPOCHS = 5;
const LR     = 0.001;

const model = tf.sequential();
model.add(tf.layers.dense({ units: 36, inputShape: [9], activation: 'relu' }));
model.add(tf.layers.dense({ units: 36, activation: 'relu' }));
model.add(tf.layers.dense({ units: 9,  activation: 'linear' }));
model.compile({ optimizer: tf.train.adam(LR), loss: 'meanSquaredError' });

// ─────────────────────────────────────────────────────────────────────────────
// Game state
// ─────────────────────────────────────────────────────────────────────────────
let gameActive   = false;
let moveHistory  = [];
let isAutoTraining = false;
let stopRequested  = false;

// Stats
let totalGames = 0;
let cpuWins    = 0;
let rngWins    = 0;
let ties       = 0;
const gameResults = []; // rolling history for chart: 1=cpu, -1=rng, 0=tie
const ROLLING_N   = 20;

// ─────────────────────────────────────────────────────────────────────────────
// Board helpers
// ─────────────────────────────────────────────────────────────────────────────
const WIN_COMBOS = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6],
];

function getBoardState() {
    return Array.from(document.querySelectorAll('.cell'))
        .map(c => c.innerHTML === 'X' ? 1 : c.innerHTML === 'O' ? -1 : 0);
}

function perspective(board, player) {
    const sign = player === 'X' ? 1 : -1;
    return board.map(v => v * sign);
}

function checkWinnerFromBoard(board) {
    for (const [a,b,c] of WIN_COMBOS) {
        if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c])
            return board[a]; // 1=X, -1=O
    }
    if (board.every(v => v !== 0)) return 0; // tie
    return null;
}

function emptyCells(board) {
    return board.reduce((acc, v, i) => { if (v === 0) acc.push(i); return acc; }, []);
}

// ─────────────────────────────────────────────────────────────────────────────
// Heatmap
// ─────────────────────────────────────────────────────────────────────────────
// Maps a score to a CSS color: negative → red, zero → neutral, positive → green
function scoreToColor(score, minScore, maxScore) {
    // Normalise to [0, 1]
    const range = maxScore - minScore;
    const t = range < 1e-6 ? 0.5 : (score - minScore) / range;

    // Interpolate: red(240,80,80) → neutral(245,245,245) → green(80,190,100)
    let r, g, b;
    if (t < 0.5) {
        const s = t * 2;
        r = Math.round(240 + s * (245 - 240));
        g = Math.round(80  + s * (245 - 80));
        b = Math.round(80  + s * (245 - 80));
    } else {
        const s = (t - 0.5) * 2;
        r = Math.round(245 + s * (80  - 245));
        g = Math.round(245 + s * (190 - 245));
        b = Math.round(245 + s * (100 - 245));
    }
    return `rgb(${r},${g},${b})`;
}

async function updateHeatmap() {
    const board  = getBoardState();
    const boardO = perspective(board, 'O'); // O's perspective (CPU)
    const inp    = tf.tensor2d([boardO]);
    const pred   = model.predict(inp);
    const vals   = Array.from(await pred.data());
    inp.dispose();
    pred.dispose();

    const cells = document.querySelectorAll('.cell');

    // Only colour empty cells
    const emptyScores = vals.filter((_, i) => board[i] === 0);
    const minS = emptyScores.length ? Math.min(...emptyScores) : 0;
    const maxS = emptyScores.length ? Math.max(...emptyScores) : 1;

    cells.forEach((cell, i) => {
        if (board[i] === 0) {
            cell.style.background = scoreToColor(vals[i], minS, maxS);
        } else {
            cell.style.background = '';
        }
    });
}

function clearHeatmap() {
    document.querySelectorAll('.cell').forEach(c => { c.style.background = ''; });
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart
// ─────────────────────────────────────────────────────────────────────────────
let winChart;

function initChart() {
    const ctx = document.getElementById('winChart').getContext('2d');
    winChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: `CPU win % (last ${ROLLING_N})`,
                data: [],
                borderColor: '#4caf50',
                backgroundColor: 'rgba(76,175,80,0.12)',
                borderWidth: 2.5,
                pointRadius: 2,
                tension: 0.35,
                fill: true,
            }],
        },
        options: {
            animation: { duration: 150 },
            scales: {
                x: {
                    title: { display: true, text: 'Games played', color: '#888' },
                    ticks: { color: '#888', maxTicksLimit: 8 },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                },
                y: {
                    min: 0, max: 100,
                    title: { display: true, text: 'Win %', color: '#888' },
                    ticks: {
                        color: '#888',
                        callback: v => v + '%',
                    },
                    grid: { color: 'rgba(255,255,255,0.06)' },
                },
            },
            plugins: {
                legend: { labels: { color: '#ccc' } },
                tooltip: {
                    callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)}%` },
                },
            },
        },
    });
}

function recordResult(winner) {
    // winner: 1=X(rng), -1=O(cpu), 0=tie   (board encoding)
    // or strings 'X','O','tie' from human game
    let code;
    if (winner === -1 || winner === 'O') code =  1;  // cpu win
    else if (winner === 1 || winner === 'X') code = -1; // rng/human win
    else code = 0;

    totalGames++;
    gameResults.push(code);
    if (code ===  1) cpuWins++;
    else if (code === -1) rngWins++;
    else ties++;

    // Rolling win rate
    const window = gameResults.slice(-ROLLING_N);
    const pct = (window.filter(r => r === 1).length / window.length) * 100;

    winChart.data.labels.push(totalGames);
    winChart.data.datasets[0].data.push(parseFloat(pct.toFixed(1)));
    winChart.update('none');

    // Stat bar
    document.getElementById('stat-cpu').textContent = cpuWins;
    document.getElementById('stat-rng').textContent = rngWins;
    document.getElementById('stat-tie').textContent = ties;
    document.getElementById('game-counter').textContent = `${totalGames} game${totalGames !== 1 ? 's' : ''}`;
    document.getElementById('stat-pct').textContent =
        gameResults.length >= 5 ? pct.toFixed(1) + '%' : '—';
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural net helpers (shared by human game + auto-train)
// ─────────────────────────────────────────────────────────────────────────────
async function getModelMove(board) {
    const boardO = perspective(board, 'O');
    const inp    = tf.tensor2d([boardO]);
    const pred   = model.predict(inp);
    const vals   = Array.from(await pred.data());
    inp.dispose();
    pred.dispose();

    const free = emptyCells(board);
    let bestIdx = free[0], bestVal = -Infinity;
    for (const i of free) {
        if (vals[i] > bestVal) { bestVal = vals[i]; bestIdx = i; }
    }
    return bestIdx;
}

async function trainModel(history, winner) {
    if (!history.length) return;

    const xMoves = history.filter(m => m.player === 'X');
    const oMoves = history.filter(m => m.player === 'O');

    let xReward, oReward;
    if (winner === 'X')        { xReward =  1;    oReward = -1;   }
    else if (winner === 'O')   { xReward = -1;    oReward =  1;   }
    else                       { xReward = -0.1;  oReward = -0.1; }

    const boards  = [];
    const targets = [];

    for (const [moves, baseReward] of [[xMoves, xReward], [oMoves, oReward]]) {
        for (let t = 0; t < moves.length; t++) {
            const { board, moveIdx } = moves[t];
            const discounted = baseReward * Math.pow(GAMMA, moves.length - 1 - t);

            const inp  = tf.tensor2d([board]);
            const pred = model.predict(inp);
            const cur  = Array.from(await pred.data());
            inp.dispose();
            pred.dispose();

            cur[moveIdx] = discounted;
            boards.push(board);
            targets.push(cur);
        }
    }

    if (!boards.length) return;
    const xT = tf.tensor2d(boards);
    const yT = tf.tensor2d(targets);
    await model.fit(xT, yT, { epochs: EPOCHS, shuffle: true, verbose: 0 });
    xT.dispose();
    yT.dispose();
}

// ─────────────────────────────────────────────────────────────────────────────
// Human vs CPU game
// ─────────────────────────────────────────────────────────────────────────────
function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function startHumanGame() {
    if (isAutoTraining) return;
    moveHistory = [];
    gameActive  = true;
    document.querySelectorAll('.cell').forEach(c => {
        c.innerHTML = '';
        c.disabled  = false;
        c.style.background = '';
        c.classList.remove('win-cell');
    });
    document.getElementById('newGameButton').style.display = 'none';
    setStatus('Your turn — you are X');
    updateHeatmap();
}

function highlightWin(board, winner) {
    const cells = document.querySelectorAll('.cell');
    for (const [a,b,c] of WIN_COMBOS) {
        if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
            [a,b,c].forEach(i => cells[i].classList.add('win-cell'));
            return;
        }
    }
}

async function handleHumanClick(idx) {
    if (!gameActive || isAutoTraining) return;
    const cells = document.querySelectorAll('.cell');
    const cell  = cells[idx];
    if (cell.innerHTML !== '') return;

    // Human plays X
    const boardBefore = getBoardState();
    moveHistory.push({
        board: perspective(boardBefore, 'X'),
        moveIdx: idx,
        player: 'X',
    });
    cell.innerHTML = 'X';

    // Disable board while processing
    gameActive = false;
    cells.forEach(c => c.disabled = true);

    let board = getBoardState();
    let w     = checkWinnerFromBoard(board);
    if (w !== null) {
        const label = w === 1 ? 'X' : w === -1 ? 'O' : 'tie';
        await finishHumanGame(board, label);
        return;
    }

    setStatus('CPU is thinking…');
    await updateHeatmap();

    // Small visual pause so the heatmap is visible
    await sleep(180);

    // CPU plays O
    const cpuIdx = await getModelMove(board);
    const cpuBoard = getBoardState();
    moveHistory.push({
        board: perspective(cpuBoard, 'O'),
        moveIdx: cpuIdx,
        player: 'O',
    });
    cells[cpuIdx].innerHTML = 'O';

    board = getBoardState();
    w     = checkWinnerFromBoard(board);
    if (w !== null) {
        const label = w === 1 ? 'X' : w === -1 ? 'O' : 'tie';
        await finishHumanGame(board, label);
        return;
    }

    // Game continues
    gameActive = true;
    cells.forEach(c => { if (c.innerHTML === '') c.disabled = false; });
    setStatus('Your turn — you are X');
    await updateHeatmap();
}

async function finishHumanGame(board, winner) {
    highlightWin(board, winner);
    clearHeatmap();

    if (winner === 'X')   setStatus('You win! 🎉');
    else if (winner === 'O') setStatus('CPU wins!');
    else                  setStatus("It's a tie!");

    await trainModel(moveHistory, winner);
    recordResult(winner === 'X' ? 1 : winner === 'O' ? -1 : 0);

    document.getElementById('newGameButton').style.display = 'inline-block';
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-training (CPU vs random, no DOM board updates for fast mode)
// ─────────────────────────────────────────────────────────────────────────────
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function randomChoice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

// Plays one full simulated game off-DOM, returns winner code (-1=O/cpu, 1=X/rng, 0=tie)
async function simulateGame() {
    const board   = Array(9).fill(0);
    const history = [];

    while (true) {
        // X = random
        const xFree = emptyCells(board);
        if (!xFree.length) break;
        const xIdx = randomChoice(xFree);
        history.push({ board: perspective(board.slice(), 'X'), moveIdx: xIdx, player: 'X' });
        board[xIdx] = 1;
        let w = checkWinnerFromBoard(board);
        if (w !== null) {
            await trainModel(history, w === 1 ? 'X' : w === -1 ? 'O' : 'tie');
            return w;
        }

        // O = model
        const oIdx = await getModelMove(board);
        history.push({ board: perspective(board.slice(), 'O'), moveIdx: oIdx, player: 'O' });
        board[oIdx] = -1;
        w = checkWinnerFromBoard(board);
        if (w !== null) {
            await trainModel(history, w === 1 ? 'X' : w === -1 ? 'O' : 'tie');
            return w;
        }
    }
    return 0;
}

// Plays one game WITH animation on the DOM board (slow mode)
async function simulateGameAnimated(delayMs) {
    // Reset board visually
    const cells = document.querySelectorAll('.cell');
    cells.forEach(c => {
        c.innerHTML = '';
        c.disabled  = true;
        c.style.background = '';
        c.classList.remove('win-cell');
    });

    const board   = Array(9).fill(0);
    const history = [];

    while (true) {
        // X = random
        const xFree = emptyCells(board);
        if (!xFree.length) break;
        const xIdx = randomChoice(xFree);
        history.push({ board: perspective(board.slice(), 'X'), moveIdx: xIdx, player: 'X' });
        board[xIdx] = 1;
        cells[xIdx].innerHTML = 'X';
        await updateHeatmap();
        await sleep(delayMs);

        let w = checkWinnerFromBoard(board);
        if (w !== null) {
            highlightWin(board, w === 1 ? 'X' : 'O');
            await trainModel(history, w === 1 ? 'X' : w === -1 ? 'O' : 'tie');
            return w;
        }

        // O = model
        const oIdx = await getModelMove(board);
        history.push({ board: perspective(board.slice(), 'O'), moveIdx: oIdx, player: 'O' });
        board[oIdx] = -1;
        cells[oIdx].innerHTML = 'O';
        await updateHeatmap();
        await sleep(delayMs);

        w = checkWinnerFromBoard(board);
        if (w !== null) {
            highlightWin(board, w === 1 ? 'X' : 'O');
            await trainModel(history, w === 1 ? 'X' : w === -1 ? 'O' : 'tie');
            return w;
        }
    }
    return 0;
}

async function runAutoTrain(n) {
    if (isAutoTraining) return;
    isAutoTraining  = true;
    stopRequested   = false;
    gameActive      = false;

    // Lock human board & buttons
    document.querySelectorAll('.cell').forEach(c => c.disabled = true);
    document.querySelectorAll('.batch-btn').forEach(b => b.disabled = true);
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('newGameButton').style.display = 'none';

    const progFill  = document.getElementById('progress-fill');
    const progLabel = document.getElementById('progress-label');

    let lastAnimated = false;

    for (let i = 0; i < n; i++) {
        if (stopRequested) break;

        const delayMs = parseInt(document.getElementById('speedSlider').value, 10);
        const animated = delayMs > 0;
        lastAnimated = animated;

        let winner;
        if (animated) {
            setStatus(`Auto-training… game ${i + 1} / ${n}`);
            winner = await simulateGameAnimated(delayMs);
        } else {
            winner = await simulateGame();
            // Yield to browser every 10 games so UI stays responsive
            if (i % 10 === 9) {
                setStatus(`Auto-training… game ${i + 1} / ${n}`);
                await sleep(0);
            }
        }

        recordResult(winner);

        // Progress bar
        const pct = ((i + 1) / n) * 100;
        progFill.style.width = pct + '%';
        progLabel.textContent = `${i + 1} / ${n}`;
    }

    // Done
    isAutoTraining = false;
    document.querySelectorAll('.batch-btn').forEach(b => b.disabled = false);
    document.getElementById('stopBtn').disabled = true;

    if (!lastAnimated) {
        // Refresh board heatmap after fast batch
        document.querySelectorAll('.cell').forEach(c => {
            c.innerHTML = '';
            c.style.background = '';
            c.classList.remove('win-cell');
        });
        await updateHeatmap();
    } else {
        clearHeatmap();
    }

    const pct = gameResults.length >= ROLLING_N
        ? ((gameResults.slice(-ROLLING_N).filter(r => r === 1).length / ROLLING_N) * 100).toFixed(1)
        : '—';
    setStatus(`Training done! CPU win rate (last ${ROLLING_N}): ${pct}% — play a game!`);

    // Re-enable human game cells
    document.querySelectorAll('.cell').forEach(c => c.disabled = false);
    gameActive = false; // player needs to click New Game or start fresh
    document.getElementById('newGameButton').style.display = 'inline-block';
    document.getElementById('newGameButton').textContent = 'Play Now';
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initChart();

    // Cell clicks
    document.querySelectorAll('.cell').forEach(cell => {
        const idx = parseInt(cell.dataset.idx, 10);
        cell.addEventListener('click', () => handleHumanClick(idx));
    });

    // New Game button
    const ngBtn = document.getElementById('newGameButton');
    ngBtn.addEventListener('click', () => {
        ngBtn.textContent = 'New Game';
        startHumanGame();
    });

    // Batch train buttons
    document.querySelectorAll('.batch-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const n = parseInt(btn.dataset.n, 10);
            runAutoTrain(n);
        });
    });

    // Stop button
    document.getElementById('stopBtn').addEventListener('click', () => {
        stopRequested = true;
    });

    // Initial heatmap
    updateHeatmap();
    setStatus('Press Train or play a game!');
});
